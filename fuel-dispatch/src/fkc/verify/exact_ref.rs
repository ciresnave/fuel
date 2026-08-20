// SPDX-License-Identifier: MIT OR Apache-2.0
//! In-process EXACT references for the ops whose contracts declare
//! `max_ulp: 0`.
//!
//! **Why this is not an oracle problem.** `max_ulp: 0` means bit-exact
//! against the correctly-rounded result, and for this population that result
//! is computable here, exactly, with no reference implementation and no
//! device:
//!
//! * a `Cast` has a defined correctly-rounded value — most of these are
//!   widening conversions the contracts already annotate as exact;
//! * `Gather` / `IndexSelect` perform no arithmetic at all: the output must
//!   reproduce input bytes;
//! * `Maximum` / `Minimum` are selects;
//! * a single IEEE `+ - * /` computed in f64 rounds to the f32 result without
//!   double rounding (f64's 53 bits exceed the 2p+2 = 48 the hazard needs for
//!   f32). **That holds for ONE operation and not for a chain**, which is why
//!   every reference here is single-op.
//!
//! **Independence.** These references are written from the dtype semantics —
//! IEEE rounding and Rust's conversion rules — not from Fuel's kernels. A
//! reference derived from the thing under test measures agreement with
//! itself; that distinction is the whole reason `agrees_with_<backend>_to_ulp`
//! is a different claim from `max_ulp` (GAP-225).
//!
//! **A disagreement here is a finding, not a failure of the harness.** If
//! Fuel's conversion differs from the correctly-rounded one, the sweep
//! records `fail` and names the element — it never records a pass.

use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;

use crate::fkc::verify::bit_stability::{HostTensor, KernelInvoker, VerifyError};
use crate::kernel::{BindingEntry, OpParams};

/// One decoded scalar, in the widest lossless carrier for its dtype class.
///
/// Two carriers rather than one `f64`: an `I64` value above 2^53 does not
/// survive `f64`, so an integer path that went through `f64` would silently
/// lose exactly the elements a bit-exactness claim is about.
#[derive(Debug, Clone, Copy)]
enum Scalar {
    F(f64),
    I(i128),
}

fn width_of(dt: DType) -> Option<usize> {
    Some(match dt {
        DType::Bool | DType::U8 | DType::I8 => 1,
        DType::F16 | DType::BF16 | DType::I16 => 2,
        DType::F32 | DType::U32 | DType::I32 => 4,
        DType::F64 | DType::I64 => 8,
        _ => return None,
    })
}

fn decode(dt: DType, b: &[u8]) -> Option<Scalar> {
    Some(match dt {
        DType::Bool | DType::U8 => Scalar::I(i128::from(b[0])),
        DType::I8 => Scalar::I(i128::from(b[0] as i8)),
        DType::I16 => Scalar::I(i128::from(i16::from_le_bytes([b[0], b[1]]))),
        DType::U32 => Scalar::I(i128::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
        DType::I32 => Scalar::I(i128::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
        DType::I64 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(b);
            Scalar::I(i128::from(i64::from_le_bytes(a)))
        }
        DType::F16 => Scalar::F(f64::from(
            half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
        )),
        DType::BF16 => Scalar::F(f64::from(
            half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
        )),
        DType::F32 => Scalar::F(f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
        DType::F64 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(b);
            Scalar::F(f64::from_le_bytes(a))
        }
        _ => return None,
    })
}

/// Encode `s` into `dt`, rounding to nearest-even where the target is
/// narrower. Integer targets follow Rust's saturating `as` conversion, which
/// is the semantics Fuel's CPU casts are written against.
fn encode(dt: DType, s: Scalar) -> Option<Vec<u8>> {
    let as_f = |s: Scalar| match s {
        Scalar::F(v) => v,
        Scalar::I(v) => v as f64,
    };
    let as_i = |s: Scalar| -> i128 {
        match s {
            Scalar::I(v) => v,
            // Rust's float -> int `as` saturates and truncates toward zero.
            Scalar::F(v) => {
                if v.is_nan() {
                    0
                } else {
                    v.trunc() as i128
                }
            }
        }
    };
    Some(match dt {
        // Bool is NOT `truncate then compare`: 0.5 is a true value, and
        // truncating first would make it false. Tests the VALUE against zero.
        //
        // The first version of this arm did truncate, and the sweep caught it
        // as four `Cast [T, Bool]` failures — candidate 0x01 vs reference
        // 0x00. **The kernel was right and this reference was wrong**, which
        // is the outcome a truth-reference must be able to have: a
        // disagreement is a finding about ONE of the two sides, and deciding
        // which is the work.
        DType::Bool => vec![u8::from(match s {
            Scalar::I(v) => v != 0,
            Scalar::F(v) => v != 0.0,
        })],
        DType::U8 => vec![as_i(s).clamp(0, i128::from(u8::MAX)) as u8],
        DType::I8 => {
            vec![(as_i(s).clamp(i128::from(i8::MIN), i128::from(i8::MAX)) as i8) as u8]
        }
        DType::I16 => (as_i(s).clamp(i128::from(i16::MIN), i128::from(i16::MAX)) as i16)
            .to_le_bytes()
            .to_vec(),
        DType::U32 => (as_i(s).clamp(0, i128::from(u32::MAX)) as u32)
            .to_le_bytes()
            .to_vec(),
        DType::I32 => (as_i(s).clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32)
            .to_le_bytes()
            .to_vec(),
        DType::I64 => (as_i(s).clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64)
            .to_le_bytes()
            .to_vec(),
        DType::F16 => half::f16::from_f64(as_f(s)).to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f64(as_f(s)).to_le_bytes().to_vec(),
        DType::F32 => (as_f(s) as f32).to_le_bytes().to_vec(),
        DType::F64 => as_f(s).to_le_bytes().to_vec(),
        _ => return None,
    })
}

/// An exact, in-process reference for one `(op, dtypes, params)`.
///
/// Implements [`KernelInvoker`] so it drops straight into
/// [`super::ulp::verify_precision_bound`] as the reference side — the same
/// seam a device invoker uses, which is what keeps "candidate vs reference"
/// one comparison rather than two code paths.
pub(crate) struct ExactRefInvoker {
    pub(crate) op: OpKind,
    pub(crate) out_dtype: DType,
    pub(crate) out_shape: Vec<usize>,
    pub(crate) params: OpParams,
}

/// Which `(op, dtypes)` this module can reference exactly. Anything else is
/// declined by the caller BEFORE a probe runs, so an unsupported case
/// contributes no record rather than a record earned against nothing.
///
/// **The dtype half is not incidental.** Declining on the op alone let 22
/// `F8E4M3` casts run and fail deep inside `encode` as `invoke error: no
/// width for F8E4M3` — a REFUSAL wearing an ERROR's label. They are the same
/// outcome for the ledger (no record either way) and NOT the same thing for a
/// reader: an error invites debugging, a refusal is a stated limit.
pub(crate) fn has_exact_reference(op: OpKind, dtypes: &[DType]) -> bool {
    if !dtypes.iter().all(|d| width_of(*d).is_some()) {
        return false;
    }
    matches!(
        op,
        OpKind::Cast
            | OpKind::AddElementwise
            | OpKind::SubElementwise
            | OpKind::MulElementwise
            | OpKind::DivElementwise
            | OpKind::MaximumElementwise
            | OpKind::MinimumElementwise
    )
}

impl KernelInvoker for ExactRefInvoker {
    fn invoke(
        &self,
        _entry: &BindingEntry,
        inputs: &[HostTensor],
    ) -> Result<HostTensor, VerifyError> {
        let n: usize = self.out_shape.iter().product();
        let ow = width_of(self.out_dtype)
            .ok_or_else(|| VerifyError::Backend(format!("no width for {:?}", self.out_dtype)))?;
        let mut out = Vec::with_capacity(n * ow);

        let elem = |t: &HostTensor, i: usize| -> Result<Scalar, VerifyError> {
            let w = width_of(t.dtype)
                .ok_or_else(|| VerifyError::Backend(format!("no width for {:?}", t.dtype)))?;
            let b = t
                .bytes
                .get(i * w..(i + 1) * w)
                .ok_or_else(|| VerifyError::Backend(format!("input short at elem {i}")))?;
            decode(t.dtype, b)
                .ok_or_else(|| VerifyError::Backend(format!("cannot decode {:?}", t.dtype)))
        };

        for i in 0..n {
            let s = match self.op {
                OpKind::Cast => {
                    let a = inputs
                        .first()
                        .ok_or_else(|| VerifyError::Backend("cast needs 1 input".into()))?;
                    elem(a, i)?
                }
                OpKind::AddElementwise
                | OpKind::SubElementwise
                | OpKind::MulElementwise
                | OpKind::DivElementwise
                | OpKind::MaximumElementwise
                | OpKind::MinimumElementwise => {
                    let (a, b) = (
                        inputs.first().ok_or_else(|| {
                            VerifyError::Backend("binary op needs 2 inputs".into())
                        })?,
                        inputs.get(1).ok_or_else(|| {
                            VerifyError::Backend("binary op needs 2 inputs".into())
                        })?,
                    );
                    // f64 for a SINGLE operation: exact for the f32 result,
                    // and for f16/bf16 the f64 result rounds to the same value
                    // their own arithmetic would produce, because both are
                    // strict subsets of f64.
                    let (x, y) = (
                        match elem(a, i)? {
                            Scalar::F(v) => v,
                            Scalar::I(v) => v as f64,
                        },
                        match elem(b, i)? {
                            Scalar::F(v) => v,
                            Scalar::I(v) => v as f64,
                        },
                    );
                    Scalar::F(match self.op {
                        OpKind::AddElementwise => x + y,
                        OpKind::SubElementwise => x - y,
                        OpKind::MulElementwise => x * y,
                        OpKind::DivElementwise => x / y,
                        // Fuel's Maximum/Minimum are NaN-PROPAGATING (torch
                        // convention, pinned in the NaN-conventions decision),
                        // which is NOT what `f64::max` does — it returns the
                        // non-NaN operand. Written explicitly so the reference
                        // states the convention rather than inheriting the
                        // standard library's opposite one.
                        OpKind::MaximumElementwise => {
                            if x.is_nan() || y.is_nan() {
                                f64::NAN
                            } else if x >= y {
                                x
                            } else {
                                y
                            }
                        }
                        _ => {
                            if x.is_nan() || y.is_nan() {
                                f64::NAN
                            } else if x <= y {
                                x
                            } else {
                                y
                            }
                        }
                    })
                }
                other => {
                    return Err(VerifyError::Backend(format!(
                        "no exact reference for {other:?}"
                    )));
                }
            };
            out.extend_from_slice(&encode(self.out_dtype, s).ok_or_else(|| {
                VerifyError::Backend(format!("cannot encode {:?}", self.out_dtype))
            })?);
        }

        Ok(HostTensor {
            dtype: self.out_dtype,
            shape: self.out_shape.clone(),
            bytes: out,
        })
    }
}
