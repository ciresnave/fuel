// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scalar values matching one of the supported [`DType`] variants.
use crate::DType;
use float8::F8E4M3 as f8e4m3;
use float8::F8E5M2 as f8e5m2;
use half::{bf16, f16};

/// A typed scalar value matching one of the supported [`DType`] variants.
// EXHAUSTIVE-BY-DESIGN: `Scalar` is a closed set — every consumer matches it
// exhaustively (e.g. `scalar_to_bytes` -> `Vec<u8>`, `push_scalar_arg`), and no
// wildcard arm can encode an unknown variant correctly (it could only panic or
// emit wrong bytes). Adding a variant is a breaking change: gate it
// workspace-wide (`cargo check` across every consumer, never just `-p fuel-ir`)
// and fix each match the compiler names. Do NOT add `#[non_exhaustive]` here —
// it would silence exactly those errors. See GAP-049.
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
    /// 8-bit float, 5-bit exponent, 2-bit mantissa (OCP FP8 gradient format).
    F8E5M2(f8e5m2),
    /// OCP-MX `E8M0` block scale: raw byte `X` decodes to `2^(X − 127)` for
    /// `X ∈ 0..=254`; `X == 255` is NaN. Unsigned; no zero, no subnormals.
    F8E8M0(u8),
    /// Boolean truth value (the scalar companion of [`DType::Bool`]). One byte;
    /// `false`/`true`. Not an arithmetic scalar — `to_f64` maps it to `0.0`/`1.0`
    /// for uniformity, but a mask used AS A NUMBER is rejected upstream.
    Bool(bool),
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
            DType::F8E5M2 => Scalar::F8E5M2(f8e5m2::ZERO),
            // Bool: the `false` value.
            DType::Bool => Scalar::Bool(false),
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
            DType::F8E5M2 => Scalar::F8E5M2(f8e5m2::ONE),
            // Bool: the `true` value.
            DType::Bool => Scalar::Bool(true),
            // OCP-MX E8M0: 2^0 => X = 127.
            DType::F8E8M0 => Scalar::F8E8M0(127),
            // F8E6M2: a scale, but its exact bit-encoding is a Fuel-local
            // invention with no citable spec — deferred as token-only. Err until
            // the encoding is authored. GAP(GAP-045)
            DType::F8E6M2 => {
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
            DType::F8E5M2 => Scalar::F8E5M2(f8e5m2::from_f64(v)),
            // Bool is a CONSTRUCTOR target, not a cast: reconstruct the exact
            // truth value, never invent one. `0.0` -> false, `1.0` -> true; any
            // other number (`0.5`, `2.0`, NaN) is not a truth value and is
            // declined as `ScalarUnrepresentable`. The `!= 0` truthiness
            // coercion is a CAST concern, never a constructor's — conflating the
            // two is the silent coercion the `Bool` dtype exists to remove
            // (GAP-168(c)). Chosen over PyTorch's `!= 0` deliberately: `from_f64`
            // rebuilds a value that WAS a bool, it does not decide truthiness.
            // (Whether any backend yet implements a float→Bool cast is a
            // separate question; this constructor simply refuses to BE one.)
            DType::Bool => {
                if v == 0.0 {
                    Scalar::Bool(false)
                } else if v == 1.0 {
                    Scalar::Bool(true)
                } else {
                    return Err(crate::Error::ScalarUnrepresentable(dtype, v));
                }
            }
            // OCP-MX E8M0: nearest power of two, X = round(log2(v)) + 127.
            DType::F8E8M0 => {
                if !v.is_finite() || v <= 0.0 {
                    return Err(crate::Error::ScalarUnrepresentable(dtype, v));
                }
                let x = v.log2().round() as i32 + 127;
                if !(0..=254).contains(&x) {
                    return Err(crate::Error::ScalarUnrepresentable(dtype, v));
                }
                Scalar::F8E8M0(x as u8)
            }
            // F8E6M2: a scale, but its exact bit-encoding is a Fuel-local
            // invention with no citable spec — deferred as token-only. Err until
            // the encoding is authored. GAP(GAP-045)
            DType::F8E6M2 => {
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
            Scalar::F8E5M2(_) => DType::F8E5M2,
            Scalar::F8E8M0(_) => DType::F8E8M0,
            Scalar::Bool(_) => DType::Bool,
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
            Scalar::F8E5M2(v) => v.to_f64(),
            Scalar::F8E8M0(x) => {
                if *x == 255 {
                    f64::NAN
                } else {
                    2f64.powi(*x as i32 - 127)
                }
            }
            Scalar::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
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
            assert!(Scalar::zero(dt).is_err(), "{dt:?} zero");
            assert!(Scalar::one(dt).is_err(), "{dt:?} one");
            assert!(Scalar::from_f64(1.0, dt).is_err(), "{dt:?} from_f64");
        }
    }

    #[test]
    fn ctors_ok_for_real_dtypes() {
        assert_eq!(Scalar::zero(DType::F32).unwrap(), Scalar::F32(0.0));
        assert_eq!(Scalar::one(DType::I64).unwrap(), Scalar::I64(1));
        assert_eq!(
            Scalar::from_f64(-1.0, DType::F16).unwrap(),
            Scalar::F16(f16::from_f64(-1.0))
        );
    }

    #[test]
    fn f8e8m0_decode_matches_ocp() {
        // value = 2^(X - 127); X = 255 => NaN. No zero, no negatives.
        assert_eq!(Scalar::F8E8M0(127).to_f64(), 1.0); // 2^0
        assert_eq!(Scalar::F8E8M0(128).to_f64(), 2.0); // 2^1
        assert_eq!(Scalar::F8E8M0(126).to_f64(), 0.5); // 2^-1
        assert!(Scalar::F8E8M0(255).to_f64().is_nan());
    }

    #[test]
    fn f8e8m0_roundtrip_all_finite_bytes() {
        for x in 0u8..=254 {
            let v = Scalar::F8E8M0(x).to_f64();
            assert_eq!(
                Scalar::from_f64(v, DType::F8E8M0).unwrap(),
                Scalar::F8E8M0(x),
                "byte {x}"
            );
        }
    }

    #[test]
    fn f8e8m0_one_and_no_zero() {
        assert_eq!(Scalar::one(DType::F8E8M0).unwrap(), Scalar::F8E8M0(127));
        assert!(matches!(
            Scalar::zero(DType::F8E8M0),
            Err(crate::Error::NoZeroScalar(DType::F8E8M0))
        ));
    }

    #[test]
    fn scalar_ctors_never_unwind_over_all_dtypes() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        // DType::ALL is kept complete by a REMINDER anchored to `all_variants_witness`
        // (adding a variant is a compile error there, but nothing forces it into ALL —
        // GAP-248), not compiler-derived. So this sweep covers every variant IN ALL,
        // which is measured complete at head — not a compiler proof against a vacuous pass.
        for &dt in DType::ALL {
            assert!(
                catch_unwind(AssertUnwindSafe(|| Scalar::zero(dt))).is_ok(),
                "zero() unwound for {dt:?}"
            );
            assert!(
                catch_unwind(AssertUnwindSafe(|| Scalar::one(dt))).is_ok(),
                "one() unwound for {dt:?}"
            );
            assert!(
                catch_unwind(AssertUnwindSafe(|| Scalar::from_f64(1.0, dt))).is_ok(),
                "from_f64() unwound for {dt:?}"
            );
        }
    }

    /// GAP-168(c): `Scalar` carries a real `Bool` value — `zero`/`one` are
    /// `false`/`true`, never an `Err` (Fuel fully supports the dtype).
    #[test]
    fn bool_zero_and_one_are_real_values() {
        assert_eq!(Scalar::zero(DType::Bool).unwrap(), Scalar::Bool(false));
        assert_eq!(Scalar::one(DType::Bool).unwrap(), Scalar::Bool(true));
        assert_eq!(Scalar::Bool(true).dtype(), DType::Bool);
        assert_eq!(Scalar::Bool(false).to_f64(), 0.0);
        assert_eq!(Scalar::Bool(true).to_f64(), 1.0);
    }

    /// GAP-168(c): `from_f64` is a CONSTRUCTOR, not a cast. It rebuilds a value
    /// that was a bool (`0.0`/`1.0`) and DECLINES a number that never was one
    /// (`0.5`) rather than silently coercing via `!= 0`. This is the assertion
    /// that separates the constructor semantics from PyTorch's cast semantics —
    /// under `Scalar::Bool(v != 0.0)` the `0.5` case would be `Ok(true)` and
    /// this test would fail.
    #[test]
    fn bool_from_f64_is_a_constructor_not_a_cast() {
        assert_eq!(
            Scalar::from_f64(0.0, DType::Bool).unwrap(),
            Scalar::Bool(false)
        );
        assert_eq!(
            Scalar::from_f64(1.0, DType::Bool).unwrap(),
            Scalar::Bool(true)
        );
        // The whole point: a non-truth-value number is refused, not coerced.
        assert!(
            matches!(
                Scalar::from_f64(0.5, DType::Bool),
                Err(crate::Error::ScalarUnrepresentable(DType::Bool, _)),
            ),
            "0.5 is not a truth value — from_f64 must decline it, not coerce to true"
        );
        assert!(matches!(
            Scalar::from_f64(2.0, DType::Bool),
            Err(crate::Error::ScalarUnrepresentable(DType::Bool, _)),
        ));
    }
}
