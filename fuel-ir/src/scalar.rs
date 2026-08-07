//! Scalar values matching one of the supported [`DType`] variants.
use crate::DType;
use float8::F8E4M3 as f8e4m3;
use half::{bf16, f16};

/// A typed scalar value matching one of the supported [`DType`] variants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    U8(u8),
    I8(i8),
    U32(u32),
    I16(i16),
    I32(i32),
    I64(i64),
    BF16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
    F8E4M3(f8e4m3),
}

impl Scalar {
    /// Returns the zero value for the given [`DType`].
    ///
    /// Real arithmetic dtypes yield `Ok`. The block-scale dtypes
    /// (`F8E8M0`/`F8E6M2`) have no exact zero (they encode `2^(x−bias)`), so
    /// they return `Err(NoZeroScalar)` — permanently, not merely as an interim
    /// (later tasks give scales real `one`/`from_f64`, but never a zero). The
    /// packed element dtypes (`F4`/`F6E2M3`/`F6E3M2`) have no scalar
    /// representation and return `Err(PackedElementHasNoScalar)`. Never panics.
    pub fn zero(dtype: DType) -> crate::Result<Self> {
        Ok(match dtype {
            DType::U8 => Scalar::U8(0),
            DType::I8 => Scalar::I8(0),
            DType::U32 => Scalar::U32(0),
            DType::I16 => Scalar::I16(0),
            DType::I32 => Scalar::I32(0),
            DType::I64 => Scalar::I64(0),
            DType::BF16 => Scalar::BF16(bf16::ZERO),
            DType::F16 => Scalar::F16(f16::ZERO),
            DType::F32 => Scalar::F32(0.0),
            DType::F64 => Scalar::F64(0.0),
            DType::F8E4M3 => Scalar::F8E4M3(f8e4m3::ZERO),
            // Scales: no exact zero (this stays Err even once scales are real).
            DType::F8E8M0 | DType::F8E6M2 => {
                return Err(crate::Error::NoZeroScalar(dtype));
            }
            // Packed element formats: no scalar representation.
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 => {
                return Err(crate::Error::PackedElementHasNoScalar(dtype));
            }
        })
    }

    /// Returns the one value for the given [`DType`].
    ///
    /// Real arithmetic dtypes yield `Ok`. The block-scale dtypes
    /// (`F8E8M0`/`F8E6M2`) interim-return `Err(NoOneScalar)` (later tasks give
    /// them the real `2^0` byte). The packed element dtypes return
    /// `Err(PackedElementHasNoScalar)`. Never panics.
    pub fn one(dtype: DType) -> crate::Result<Self> {
        Ok(match dtype {
            DType::U8 => Scalar::U8(1),
            DType::I8 => Scalar::I8(1),
            DType::U32 => Scalar::U32(1),
            DType::I16 => Scalar::I16(1),
            DType::I32 => Scalar::I32(1),
            DType::I64 => Scalar::I64(1),
            DType::BF16 => Scalar::BF16(bf16::ONE),
            DType::F16 => Scalar::F16(f16::ONE),
            DType::F32 => Scalar::F32(1.0),
            DType::F64 => Scalar::F64(1.0),
            DType::F8E4M3 => Scalar::F8E4M3(f8e4m3::ONE),
            // Scales: interim Err (Task 2/3 give the real 2^0 byte).
            DType::F8E8M0 | DType::F8E6M2 => {
                return Err(crate::Error::NoOneScalar(dtype));
            }
            // Packed element formats: no scalar representation.
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 => {
                return Err(crate::Error::PackedElementHasNoScalar(dtype));
            }
        })
    }

    /// Reconstruct a scalar of the given [`DType`] from an `f64` value — the
    /// inverse of [`Scalar::to_f64`] paired with a target dtype. Integer dtypes
    /// truncate toward zero (`v as iN`); float dtypes round to nearest via the
    /// type's `from_f64`. Sibling of [`Scalar::zero`]/[`Scalar::one`]: the
    /// value-carrying reconstructor a recipe re-emit needs when a fill scalar
    /// (e.g. `MaskedFill`) rides an `f64` + a separately-carried dtype.
    ///
    /// Real arithmetic dtypes yield `Ok`. The block-scale dtypes
    /// (`F8E8M0`/`F8E6M2`) interim-return `Err(ScalarUnrepresentable)` (later
    /// tasks give them real rounding). The packed element dtypes return
    /// `Err(PackedElementHasNoScalar)`. Never panics.
    pub fn from_f64(v: f64, dtype: DType) -> crate::Result<Self> {
        Ok(match dtype {
            DType::U8 => Scalar::U8(v as u8),
            DType::I8 => Scalar::I8(v as i8),
            DType::U32 => Scalar::U32(v as u32),
            DType::I16 => Scalar::I16(v as i16),
            DType::I32 => Scalar::I32(v as i32),
            DType::I64 => Scalar::I64(v as i64),
            DType::BF16 => Scalar::BF16(bf16::from_f64(v)),
            DType::F16 => Scalar::F16(f16::from_f64(v)),
            DType::F32 => Scalar::F32(v as f32),
            DType::F64 => Scalar::F64(v),
            DType::F8E4M3 => Scalar::F8E4M3(f8e4m3::from_f64(v)),
            // Scales: interim Err (Task 2/3 give the real nearest-representable).
            DType::F8E8M0 | DType::F8E6M2 => {
                return Err(crate::Error::ScalarUnrepresentable(dtype, v));
            }
            // Packed element formats: no scalar representation.
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 => {
                return Err(crate::Error::PackedElementHasNoScalar(dtype));
            }
        })
    }

    /// Returns the [`DType`] of this scalar value.
    pub fn dtype(&self) -> DType {
        match self {
            Scalar::U8(_) => DType::U8,
            Scalar::I8(_) => DType::I8,
            Scalar::U32(_) => DType::U32,
            Scalar::I16(_) => DType::I16,
            Scalar::I32(_) => DType::I32,
            Scalar::I64(_) => DType::I64,
            Scalar::BF16(_) => DType::BF16,
            Scalar::F16(_) => DType::F16,
            Scalar::F32(_) => DType::F32,
            Scalar::F64(_) => DType::F64,
            Scalar::F8E4M3(_) => DType::F8E4M3,
        }
    }

    /// Converts the scalar value to `f64`.
    pub fn to_f64(&self) -> f64 {
        match self {
            Scalar::U8(v) => *v as f64,
            Scalar::I8(v) => *v as f64,
            Scalar::U32(v) => *v as f64,
            Scalar::I16(v) => *v as f64,
            Scalar::I32(v) => *v as f64,
            Scalar::I64(v) => *v as f64,
            Scalar::BF16(v) => v.to_f64(),
            Scalar::F16(v) => v.to_f64(),
            Scalar::F32(v) => *v as f64,
            Scalar::F64(v) => *v,
            Scalar::F8E4M3(v) => v.to_f64(),
        }
    }
}

impl<T: crate::WithDType> From<T> for Scalar {
    fn from(value: T) -> Self {
        value.to_scalar()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctors_return_err_not_panic_for_subbyte_dtypes() {
        for dt in [DType::F6E2M3, DType::F6E3M2, DType::F4] {
            assert!(matches!(Scalar::zero(dt), Err(_)), "{dt:?} zero");
            assert!(matches!(Scalar::one(dt), Err(_)), "{dt:?} one");
            assert!(matches!(Scalar::from_f64(1.0, dt), Err(_)), "{dt:?} from_f64");
        }
    }

    #[test]
    fn ctors_ok_for_real_dtypes() {
        assert_eq!(Scalar::zero(DType::F32).unwrap(), Scalar::F32(0.0));
        assert_eq!(Scalar::one(DType::I64).unwrap(), Scalar::I64(1));
        assert_eq!(Scalar::from_f64(-1.0, DType::F16).unwrap(),
                   Scalar::F16(f16::from_f64(-1.0)));
    }
}
