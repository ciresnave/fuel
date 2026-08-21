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

/// Decode an `F8E4M3` (OCP **E4M3FN**) bit pattern to its exact value.
///
/// FN = **f**inite, **n**o infinities. Sign in bit 7, 4-bit exponent (bias 7)
/// in bits 6-3, 3-bit mantissa in bits 2-0. `S.1111.111` is the ONLY reserved
/// pattern (two encodings, one per sign); `1111.000`-`1111.110` are ordinary
/// finite values, the largest being 448.
///
/// ⚠️ **Written from the format, deliberately NOT from `float8`** — which is
/// what Fuel's cast kernel delegates to (`byte_kernels.rs:3792`
/// `float8::F8E4M3::from_f32`). A reference built on the same crate would
/// compare that crate against itself and pass unconditionally: `max_ulp` in
/// name, `agrees_with_float8` in fact, which is GAP-227's defect arriving in
/// a new place.
///
/// **The independence this buys is real but bounded, and the bound is worth
/// stating:** because the kernel is a thin delegation, a pass here mostly
/// certifies `float8` against this reading of the E4M3FN spec, plus Fuel's
/// byte plumbing around it. That is a weaker claim than for the ops Fuel
/// implements itself — and it is still worth having, because a disagreement
/// would mean one of the two readings of the format is wrong.
pub(super) fn f8e4m3_decode(bits: u8) -> f64 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 3) & 0x0F) as i32;
    let mant = (bits & 0x07) as f64;
    if (bits & 0x7F) == 0x7F {
        return f64::NAN;
    }
    if exp == 0 {
        // Subnormal: no implicit leading 1, fixed exponent 2^(1-bias).
        sign * (mant / 8.0) * 2f64.powi(1 - 7)
    } else {
        sign * (1.0 + mant / 8.0) * 2f64.powi(exp - 7)
    }
}

/// Encode `v` into `F8E4M3` by choosing the NEAREST representable value,
/// ties to even.
///
/// Brute force over all 256 patterns rather than bit-twiddling an exponent
/// and mantissa. It is 256 comparisons, it is correct by construction for
/// round-to-nearest, and it cannot carry the subnormal / saturation / tie
/// bugs that a hand-rolled shift-and-round is prone to — which matters
/// especially for a reference, where a subtle bug produces a confident wrong
/// verdict about someone else's kernel.
fn f8e4m3_encode(v: f64) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    // ⚠️ SATURATE EXPLICITLY BEFORE THE SEARCH, because the search's own
    // arithmetic loses resolution on large inputs — a precision bug inside a
    // precision harness.
    //
    // `(cand - v).abs()` is computed in f64. For `v` around 4.5e18 the f64
    // ULP is ~512, so the distances to -448 and -384 ROUND TO THE SAME
    // NUMBER; a strict ordering becomes a spurious tie and the tie-break
    // picks the lower bit pattern. That produced `Cast [I64, F8E4M3]`:
    // candidate `0xfe` (-448, Fuel, correct) vs reference `0xfc` (-384).
    //
    // E4M3FN has no infinities, so an out-of-range magnitude saturates to the
    // largest finite value, 448 (`0x7E` / `0xFE`). Handling it here also
    // makes the intent explicit rather than an emergent property of a
    // distance comparison.
    const MAX_FINITE: f64 = 448.0;
    if v.abs() >= MAX_FINITE {
        return if v < 0.0 { 0xFE } else { 0x7E };
    }
    let mut best: Option<(f64, u8)> = None;
    for bits in 0u16..256 {
        let b = bits as u8;
        if (b & 0x7F) == 0x7F {
            continue; // NaN encodings are not candidates for a finite value
        }
        let cand = f8e4m3_decode(b);
        let d = (cand - v).abs();
        best = match best {
            None => Some((d, b)),
            Some((bd, bb)) => {
                if d < bd {
                    Some((d, b))
                } else if d == bd {
                    // Tie -> even mantissa LSB. Also prefers +0 over -0,
                    // whose mantissa bits are both zero, by keeping the
                    // first-seen (lower) pattern when both are even.
                    if (b & 1) == 0 && (bb & 1) != 0 {
                        Some((d, b))
                    } else {
                        Some((bd, bb))
                    }
                } else {
                    Some((bd, bb))
                }
            }
        };
    }
    best.map(|(_, b)| b).unwrap_or(0)
}

fn width_of(dt: DType) -> Option<usize> {
    Some(match dt {
        DType::F8E4M3 => 1,
        DType::Bool | DType::U8 | DType::I8 => 1,
        DType::F16 | DType::BF16 | DType::I16 => 2,
        DType::F32 | DType::U32 | DType::I32 => 4,
        DType::F64 | DType::I64 => 8,
        _ => return None,
    })
}

fn decode(dt: DType, b: &[u8]) -> Option<Scalar> {
    Some(match dt {
        DType::F8E4M3 => Scalar::F(f8e4m3_decode(b[0])),
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
/// narrower.
///
/// ⚠️ **Integer targets follow TWO different rules depending on where the
/// value came from, and conflating them is wrong in a way that only shows up
/// on out-of-range inputs.** Rust's `as`:
///
/// * **float -> int SATURATES** (and maps NaN to 0);
/// * **int -> int TRUNCATES** (keeps the low bits, wrapping).
///
/// The first version of this function clamped both, and the sweep caught it
/// the moment the probe values stopped being all zeros: `Cast [I64, I16]`
/// reported candidate `0xa2` vs reference `0x80` — the reference saturating
/// to `i16::MIN` where the kernel correctly truncated. **Second time the
/// reference was wrong and the kernel was right**, which is what a truth
/// reference is supposed to be able to show.
///
/// Degenerate probes hid this completely: zero is in range for every target,
/// so saturation and truncation agree on it.
fn encode(dt: DType, s: Scalar) -> Option<Vec<u8>> {
    let as_f = |s: Scalar| match s {
        Scalar::F(v) => v,
        Scalar::I(v) => v as f64,
    };
    // Saturating conversion, for a FLOAT source: Rust's `f64 as iN` clamps to
    // the target's range and maps NaN to 0.
    let sat = |s: Scalar, lo: i128, hi: i128| -> i128 {
        match s {
            Scalar::I(v) => v,
            Scalar::F(v) => {
                if v.is_nan() {
                    0
                } else {
                    (v.trunc() as i128).clamp(lo, hi)
                }
            }
        }
    };
    // Whether the value came from an integer source, which selects TRUNCATION
    // over saturation for an integer target.
    let from_int = matches!(s, Scalar::I(_));
    let raw = match s {
        Scalar::I(v) => v,
        Scalar::F(v) => {
            if v.is_nan() {
                0
            } else {
                v.trunc() as i128
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
        DType::U8 => {
            let v = if from_int {
                raw as u8
            } else {
                sat(s, 0, i128::from(u8::MAX)) as u8
            };
            vec![v]
        }
        DType::I8 => {
            let v = if from_int {
                raw as i8
            } else {
                sat(s, i128::from(i8::MIN), i128::from(i8::MAX)) as i8
            };
            vec![v as u8]
        }
        DType::I16 => {
            let v = if from_int {
                raw as i16
            } else {
                sat(s, i128::from(i16::MIN), i128::from(i16::MAX)) as i16
            };
            v.to_le_bytes().to_vec()
        }
        DType::U32 => {
            let v = if from_int {
                raw as u32
            } else {
                sat(s, 0, i128::from(u32::MAX)) as u32
            };
            v.to_le_bytes().to_vec()
        }
        DType::I32 => {
            let v = if from_int {
                raw as i32
            } else {
                sat(s, i128::from(i32::MIN), i128::from(i32::MAX)) as i32
            };
            v.to_le_bytes().to_vec()
        }
        DType::I64 => {
            let v = if from_int {
                raw as i64
            } else {
                sat(s, i128::from(i64::MIN), i128::from(i64::MAX)) as i64
            };
            v.to_le_bytes().to_vec()
        }
        DType::F8E4M3 => vec![f8e4m3_encode(as_f(s))],
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
    /// Test-only: corrupt this reference's output so the comparison against
    /// the kernel MUST fail.
    ///
    /// **This exists to test ATTACHMENT, which no count can.** A reference
    /// bound to the wrong `(op, dtypes)` still produces a perfectly correct
    /// number of earned records — GAP-228(a)'s run-1 defect in a different
    /// costume. Poisoning ONE family and checking that exactly that family's
    /// registrations fail is the control that separates "the evidence exists"
    /// from "the evidence is attached to the kernel it claims to be about".
    #[cfg(test)]
    pub(crate) poison: bool,
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
        OpKind::Gather
            | OpKind::IndexSelect
            | OpKind::Cast
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

        // --- Pure byte movement: no arithmetic, so the reference is the
        // --- MOVE ITSELF and the comparison is exact by construction.
        //
        // Deliberately copies BYTES rather than decode/encode per element.
        // A gather that round-tripped through `f64` would be a claim about
        // the CARRIER's fidelity, not about the move — and would quietly
        // pass for a kernel that had corrupted a NaN payload or an I64
        // beyond 2^53. Copying bytes cannot express that failure mode.
        if matches!(self.op, OpKind::Gather | OpKind::IndexSelect) {
            let src = inputs
                .first()
                .ok_or_else(|| VerifyError::Backend("needs a source input".into()))?;
            let idx_t = inputs
                .get(1)
                .ok_or_else(|| VerifyError::Backend("needs a U32 index input".into()))?;
            if idx_t.dtype != DType::U32 {
                return Err(VerifyError::Backend(format!(
                    "index operand is {:?}, expected U32",
                    idx_t.dtype
                )));
            }
            let w = width_of(src.dtype)
                .ok_or_else(|| VerifyError::Backend(format!("no width for {:?}", src.dtype)))?;
            let idx = |k: usize| -> Result<usize, VerifyError> {
                let b = idx_t
                    .bytes
                    .get(k * 4..(k + 1) * 4)
                    .ok_or_else(|| VerifyError::Backend(format!("index short at {k}")))?;
                Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
            };
            let mut copy_from = |src_elem: usize, out: &mut Vec<u8>| -> Result<(), VerifyError> {
                let b = src
                    .bytes
                    .get(src_elem * w..(src_elem + 1) * w)
                    .ok_or_else(|| {
                        VerifyError::Backend(format!("source short at elem {src_elem}"))
                    })?;
                out.extend_from_slice(b);
                Ok(())
            };

            match (&self.params, self.op) {
                (
                    OpParams::IndexSelect {
                        outer_count,
                        source_dim_size,
                        n_indices,
                        inner_count,
                    },
                    OpKind::IndexSelect,
                ) => {
                    for o in 0..*outer_count {
                        for k in 0..*n_indices {
                            let sel = idx(k)?;
                            for j in 0..*inner_count {
                                let src_elem =
                                    o * source_dim_size * inner_count + sel * inner_count + j;
                                copy_from(src_elem, &mut out)?;
                            }
                        }
                    }
                }
                (
                    OpParams::Gather {
                        source_shape,
                        output_shape,
                        dim,
                    },
                    OpKind::Gather,
                ) => {
                    let total: usize = output_shape.iter().product();
                    for flat in 0..total {
                        // Decompose the output flat index into coords, replace
                        // the gathered dim with the index tensor's value, and
                        // re-flatten over the SOURCE shape. Row-major both
                        // sides; the two shapes agree on every dim but `dim`.
                        let mut coords = vec![0usize; output_shape.len()];
                        let mut rem = flat;
                        for d in (0..output_shape.len()).rev() {
                            coords[d] = rem % output_shape[d];
                            rem /= output_shape[d];
                        }
                        coords[*dim] = idx(flat)?;
                        let mut src_elem = 0usize;
                        for d in 0..source_shape.len() {
                            src_elem = src_elem * source_shape[d] + coords[d];
                        }
                        copy_from(src_elem, &mut out)?;
                    }
                }
                (params, op) => {
                    return Err(VerifyError::Backend(format!(
                        "{op:?} with {params:?} is not the params shape this reference implements"
                    )));
                }
            }
            #[cfg(test)]
            if self.poison {
                if let Some(b) = out.first_mut() {
                    *b ^= 0x01;
                }
            }
            return Ok(HostTensor {
                dtype: self.out_dtype,
                shape: self.out_shape.clone(),
                bytes: out,
            });
        }

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

        #[cfg(test)]
        if self.poison {
            // Flip a bit in the first byte. Enough to fail `MaxUlp(0)` for
            // any dtype, in both the float and the byte-exact modes.
            if let Some(b) = out.first_mut() {
                *b ^= 0x01;
            }
        }

        Ok(HostTensor {
            dtype: self.out_dtype,
            shape: self.out_shape.clone(),
            bytes: out,
        })
    }
}
