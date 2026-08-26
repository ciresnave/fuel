// SPDX-License-Identifier: MIT OR Apache-2.0
//! Empirical precision-bound verification (`V-FKC-9`, Task 4.4).
//!
//! [`verify_precision_bound`] diffs a *candidate* kernel invocation against a
//! *reference*-tagged invocation of the same probe inputs, checking the
//! declared precision [`Bound`] (`max_ulp` / `max_relative` / `max_absolute`
//! from the FKC precision block). Hardware-free: both `cand` and `refr` are
//! [`KernelInvoker`]s, so unit tests here use fake in-process invokers; the
//! real CPU-reference-vs-CUDA-candidate wiring is Task 4.5.

use crate::fkc::verify::bit_stability::{
    HostTensor, KernelInvoker, ProbeInputs, VerifyError, VerifyOutcome,
};
use crate::kernel::BindingEntry;
use fuel_graph::jit::{OpTag, PatternNode};
use fuel_ir::DType;

/// ULP (units-in-the-last-place) distance between two `f32` values.
///
/// **Consumed directly from kiss-ref** ([`kiss_ref_core::ulp_distance_f32`]) —
/// the single cross-project source of truth for float comparison, so Fuel's
/// verify stack and any other "KISS-speaking" project (kiss-ref itself, its
/// conformance corpus, sibling consumers) can never disagree on how far apart
/// two floats are. This used to be a Fuel-internal copy; two copies that must
/// agree by hand is a latent drift bug, and they HAD drifted (see below).
///
/// kiss-ref uses the identical IEEE-754 **total-order** mapping Fuel used
/// (`kiss_ref_core`'s `key_f32` is byte-for-byte the old `total_order_key`), so
/// every non-NaN result is unchanged — the total-order transform is what makes
/// the distance correct across the sign/zero boundary (`-0.0`/`+0.0` are 1 ULP
/// apart, not `2^31`; straddling zero is meaningful). What kiss-ref ADDS is the
/// correct NaN handling the old copy lacked: both-NaN → 0 (a reference NaN
/// matched by a candidate NaN conforms), exactly one NaN → `u32::MAX` (a
/// NaN-vs-number pair is NEVER "within N ULP" — the old raw-key `abs_diff` could
/// wrongly *pass* a `MaxUlp` bound when the NaN and number keys happened to land
/// close). Widened to `u64` at the boundary so the `Bound::MaxUlp(u32)` call
/// sites are unchanged.
pub(crate) fn ulp_distance(x: f32, y: f32) -> u64 {
    u64::from(kiss_ref_core::ulp_distance_f32(x, y))
}

/// A declared precision bound to check a candidate against a reference.
/// Mirrors the FKC precision block's machine-checkable claims
/// (`max_ulp` / `max_relative` / `max_absolute`).
#[derive(Debug, Clone, Copy)]
pub enum Bound {
    /// Maximum allowed ULP (units-in-last-place) distance between candidate
    /// and reference bit patterns.
    MaxUlp(u32),
    /// Maximum allowed `|cand - ref| / |ref|` (reference-relative error).
    MaxRelative(f64),
    /// Maximum allowed `|cand - ref|` (absolute error).
    MaxAbsolute(f64),
}

/// One element of a kernel output, decoded in ITS OWN dtype.
///
/// `key` is a total-order key over that dtype's bit width: within a sign the
/// magnitude field of every IEEE-style binary format is monotone in value, so
/// the signed key orders the whole format and `|k_a - k_b|` counts
/// REPRESENTABLE STEPS — which is what a ULP is.
#[derive(Debug, Clone, Copy)]
struct Elem {
    /// Total-order key in the dtype's own bit width. ULP space.
    key: i128,
    /// The element's numeric value. Value space — what `max_relative` and
    /// `max_absolute` are defined on. Both spaces are carried because the
    /// three bounds are NOT interchangeable: a ULP is a count of
    /// representable steps, an absolute bound is a distance between values,
    /// and answering one with the other is the error this whole rewrite is
    /// about.
    value: f64,
    is_nan: bool,
}

/// Total-order key for any 16-bit IEEE-style float — **F16 and BF16 both**.
///
/// They are NOT conflated by sharing this. The key needs only the sign in bit
/// 15 and a monotone magnitude field below it, which both layouts have. What
/// differs is the REAL-VALUE SIZE of one step, and that difference is
/// preserved exactly BECAUSE the bits are never converted. Widening is what
/// destroys it: two f16 values one ULP apart are 8192 f32-ULPs apart after
/// conversion, so a widened comparison silently measures a different quantity
/// for any bound above zero.
///
/// NaN detection is the part that genuinely differs between the two formats
/// (5-bit vs 8-bit exponent), and it is done per-format in `elem_at`.
fn key16(bits: u16) -> i128 {
    if bits & 0x8000 != 0 {
        -i128::from(bits & 0x7FFF)
    } else {
        i128::from(bits)
    }
}

fn key32(bits: u32) -> i128 {
    if bits & 0x8000_0000 != 0 {
        -i128::from(bits & 0x7FFF_FFFF)
    } else {
        i128::from(bits)
    }
}

fn key64(bits: u64) -> i128 {
    if bits & 0x8000_0000_0000_0000 != 0 {
        -i128::from(bits & 0x7FFF_FFFF_FFFF_FFFF)
    } else {
        i128::from(bits)
    }
}

/// How a dtype's outputs must be compared. Not a preference — a property of
/// the dtype: ULP is a floating-point concept with no meaning for an integer,
/// where the only sensible bound is bit-exactness.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CompareMode {
    /// Count representable steps in the dtype's own space.
    UlpFloat,
    /// An integer dtype. A `max_ulp` bound on one is answered as
    /// BIT-EXACTNESS, because a ULP is a count of representable float steps
    /// and has no meaning here — the claim is real and correctly reasoned,
    /// only its NAME cannot express it, which is a contract-surface question
    /// and not this function's. Value-space bounds (`max_absolute`,
    /// `max_relative`) remain answerable normally, because those ARE defined
    /// on integers.
    ExactInt,
}

/// The comparison this verifier implements for `dt`, or `None` if it refuses.
///
/// **Refusing is the point.** The previous version reinterpreted every output
/// as `f32` and compared whatever fell out; an unhandled dtype must produce a
/// stated refusal, never a verdict about the wrong bytes.
fn compare_mode(dt: DType) -> Option<(CompareMode, usize)> {
    Some(match dt {
        // E4M3FN: sign in bit 7, monotone magnitude in bits 6-0 — the same
        // key construction as f16/bf16, one width down. Only the NaN
        // predicate is per-format, which is exactly where f16 and bf16 differ
        // from each other too.
        DType::F8E4M3 => (CompareMode::UlpFloat, 1),
        DType::F16 | DType::BF16 => (CompareMode::UlpFloat, 2),
        DType::F32 => (CompareMode::UlpFloat, 4),
        DType::F64 => (CompareMode::UlpFloat, 8),
        DType::Bool | DType::U8 => (CompareMode::ExactInt, 1),
        DType::I8 => (CompareMode::ExactInt, 1),
        DType::I16 => (CompareMode::ExactInt, 2),
        DType::U32 | DType::I32 => (CompareMode::ExactInt, 4),
        DType::I64 => (CompareMode::ExactInt, 8),
        // Everything else — the sub-byte and 8-bit float formats — is REFUSED
        // rather than approximated. Adding one means deciding its total order
        // and its NaN encoding, which is a per-format question.
        _ => return None,
    })
}

/// Decode one element at index `i`, in its own dtype.
fn elem_at(dt: DType, bytes: &[u8], i: usize, width: usize) -> Option<Elem> {
    let b = bytes.get(i * width..(i + 1) * width)?;
    Some(match dt {
        DType::F8E4M3 => {
            let bits = b[0];
            Elem {
                key: if bits & 0x80 != 0 {
                    -i128::from(bits & 0x7F)
                } else {
                    i128::from(bits)
                },
                // The REAL value, from the one E4M3FN decoder (in
                // `exact_ref`), not a placeholder. A `f64::NAN` here would
                // have made every value-space bound on an fp8 output fail
                // silently while `is_nan` said the element was finite — a
                // wrong verdict from a field nobody would look at.
                value: super::exact_ref::f8e4m3_decode(bits),
                // E4M3FN reserves EXACTLY `S.1111.111` — two encodings, one
                // per sign. Exponent `1111` is NOT wholly reserved: mantissas
                // 000-110 are ordinary finite values, the largest 448.
                is_nan: (bits & 0x7F) == 0x7F,
            }
        }
        DType::F16 => {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            Elem {
                key: key16(bits),
                value: f64::from(half::f16::from_bits(bits).to_f32()),
                // f16: 5-bit exponent all ones + non-zero mantissa.
                is_nan: (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0,
            }
        }
        DType::BF16 => {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            Elem {
                key: key16(bits),
                value: f64::from(half::bf16::from_bits(bits).to_f32()),
                // bf16: 8-bit exponent all ones + non-zero mantissa.
                is_nan: (bits & 0x7F80) == 0x7F80 && (bits & 0x007F) != 0,
            }
        }
        DType::F32 => {
            let bits = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            Elem {
                key: key32(bits),
                value: f64::from(f32::from_bits(bits)),
                is_nan: f32::from_bits(bits).is_nan(),
            }
        }
        DType::F64 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(b);
            let bits = u64::from_le_bytes(a);
            Elem {
                key: key64(bits),
                value: f64::from_bits(bits),
                is_nan: f64::from_bits(bits).is_nan(),
            }
        }
        DType::Bool | DType::U8 => Elem {
            key: i128::from(b[0]),
            value: f64::from(b[0]),
            is_nan: false,
        },
        DType::I8 => Elem {
            key: i128::from(b[0] as i8),
            value: f64::from(b[0] as i8),
            is_nan: false,
        },
        DType::I16 => {
            let v = i16::from_le_bytes([b[0], b[1]]);
            Elem {
                key: i128::from(v),
                value: f64::from(v),
                is_nan: false,
            }
        }
        DType::U32 => {
            let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            Elem {
                key: i128::from(v),
                value: f64::from(v),
                is_nan: false,
            }
        }
        DType::I32 => {
            let v = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            Elem {
                key: i128::from(v),
                value: f64::from(v),
                is_nan: false,
            }
        }
        DType::I64 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(b);
            let v = i64::from_le_bytes(a);
            Elem {
                key: i128::from(v),
                value: v as f64,
                is_nan: false,
            }
        }
        _ => return None,
    })
}

/// Empirically checks a precision [`Bound`] by invoking `cand` and `refr` on
/// the same probes and comparing their outputs **in the output's own dtype**.
/// Returns the FIRST out-of-bound element as `Fail { detail }`.
///
/// ⚠️ **This used to reinterpret every output as `f32` regardless of dtype.**
/// Its only shape guard was `bytes.len() % 4 != 0`, which a half-precision
/// buffer satisfies (4 x f16 = 8 bytes), so f16 bytes were read as f32s and
/// compared — a verdict about the wrong bytes, with nothing saying so. That
/// mattered for real: of the 84 CPU entries blocked on `max_ulp`, 54 are
/// `Cast`, whose outputs span I8 / U8 / I32 / I64 / U32 / F16 / BF16 / F64.
///
/// Two modes, chosen by the dtype and never by the caller: ULP distance in the
/// dtype's own bit width for floats, byte equality for integers. A dtype the
/// comparison does not implement is REFUSED BY NAME, never approximated.
///
/// NaN follows `kiss_ref_core::ulp_distance_f32`'s convention: both-NaN is
/// distance 0, exactly-one-NaN is the maximum distance, so no bound can be
/// satisfied by one side being NaN.
///
/// Never panics: every decode is bounds-checked, and a short, mismatched or
/// misaligned buffer is reported as a `Fail`.
pub fn verify_precision_bound(
    cand: &dyn KernelInvoker,
    refr: &dyn KernelInvoker,
    entry: &BindingEntry,
    probes: &[ProbeInputs],
    bound: Bound,
) -> Result<VerifyOutcome, VerifyError> {
    for (probe_idx, probe) in probes.iter().enumerate() {
        let a: HostTensor = cand.invoke(entry, probe)?;
        let b: HostTensor = refr.invoke(entry, probe)?;

        if a.dtype != b.dtype {
            return Ok(VerifyOutcome::Fail {
                detail: format!(
                    "probe {probe_idx}: candidate dtype {:?} != reference dtype {:?} — a comparison across dtypes is not a precision measurement",
                    a.dtype, b.dtype
                ),
            });
        }
        let Some((mode, width)) = compare_mode(a.dtype) else {
            return Ok(VerifyOutcome::Fail {
                detail: format!(
                    "probe {probe_idx}: no comparison implemented for output dtype {:?} — refused rather than approximated in another dtype's space",
                    a.dtype
                ),
            });
        };
        if a.bytes.len() != b.bytes.len() {
            return Ok(VerifyOutcome::Fail {
                detail: format!(
                    "probe {probe_idx}: candidate is {} bytes, reference {} — lengths must match or the comparison silently covers only a prefix",
                    a.bytes.len(),
                    b.bytes.len()
                ),
            });
        }
        if !a.bytes.len().is_multiple_of(width) {
            return Ok(VerifyOutcome::Fail {
                detail: format!(
                    "probe {probe_idx}: {} bytes is not a whole number of {:?} elements",
                    a.bytes.len(),
                    a.dtype
                ),
            });
        }

        // A ULP bound on an integer dtype is answered as bit-exactness. Value
        // -space bounds fall through to the elementwise loop, which handles
        // integers like anything else.
        if mode == CompareMode::ExactInt && matches!(bound, Bound::MaxUlp(_)) {
            if let Some(i) = a.bytes.iter().zip(b.bytes.iter()).position(|(x, y)| x != y) {
                return Ok(VerifyOutcome::Fail {
                    detail: format!(
                        "probe {probe_idx} byte {i}: candidate {:#04x} vs reference {:#04x} — {:?} outputs must be bit-exact, and {bound:?} on an integer dtype IS a bit-exactness claim",
                        a.bytes[i], b.bytes[i], a.dtype
                    ),
                });
            }
            continue;
        }

        let n = a.bytes.len() / width;
        for i in 0..n {
            let (Some(x), Some(y)) = (
                elem_at(a.dtype, &a.bytes, i, width),
                elem_at(b.dtype, &b.bytes, i, width),
            ) else {
                return Ok(VerifyOutcome::Fail {
                    detail: format!("probe {probe_idx} elem {i}: could not decode {:?}", a.dtype),
                });
            };
            let dist: u64 = if x.is_nan && y.is_nan {
                0
            } else if x.is_nan || y.is_nan {
                u64::MAX
            } else {
                let d = (x.key - y.key).unsigned_abs();
                if d > u128::from(u64::MAX) {
                    u64::MAX
                } else {
                    d as u64
                }
            };
            // Each bound is answered in ITS OWN space: ULP counts
            // representable steps, absolute and relative are distances
            // between values. A NaN on exactly one side fails every bound.
            let ok = match bound {
                Bound::MaxUlp(m) => dist <= u64::from(m),
                Bound::MaxAbsolute(m) => !(x.is_nan || y.is_nan) && (x.value - y.value).abs() <= m,
                Bound::MaxRelative(m) => {
                    let denom = y.value.abs().max(f64::from(f32::EPSILON));
                    !(x.is_nan || y.is_nan) && ((x.value - y.value).abs() / denom) <= m
                }
            };
            if !ok {
                let detail = match bound {
                    // Include the RAW BYTES, not just the distance. "2 ULP
                    // apart" names the size of a disagreement and not its
                    // content, and for a narrow format the bit patterns are
                    // what identify it — an overflow that saturated one way
                    // and another looks identical to a rounding difference
                    // when only the distance is reported.
                    Bound::MaxUlp(_) => format!(
                        "probe {probe_idx} elem {i}: {:?} candidate {:02x?} ({}) vs reference {:02x?} ({}) are {dist} ULP apart, exceeds {bound:?}",
                        a.dtype,
                        &a.bytes[i * width..(i + 1) * width],
                        x.value,
                        &b.bytes[i * width..(i + 1) * width],
                        y.value,
                    ),
                    _ => format!(
                        "probe {probe_idx} elem {i}: {:?} candidate {} vs reference {} exceeds {bound:?}",
                        a.dtype, x.value, y.value
                    ),
                };
                return Ok(VerifyOutcome::Fail { detail });
            }
        }
    }
    Ok(VerifyOutcome::Pass)
}

/// A transcendental unary atom — one whose hardware value can differ from the
/// wide-precision (§6.5-0007) truth by more than a correctly-rounded op. IEEE
/// requires `Sqrt`/`Recip` to be correctly-rounded, so they are NOT here;
/// `Exp`/`Log`/`Sin`/`Cos`/`Tanh`/`Sigmoid`/`Silu`/`Gelu`/`GeluErf`/`Erf`/
/// `Rsqrt` are. Mirrors `cost.rs`'s `cost_elementwise_unary_transcendental_cpu`
/// classification so the two never drift.
pub(crate) fn is_transcendental(tag: OpTag) -> bool {
    use OpTag::*;
    matches!(
        tag,
        Exp | Log | Sin | Cos | Tanh | Sigmoid | Silu | Gelu | GeluErf | Erf | Rsqrt
    )
}

/// Whether a recipe region contains any transcendental atom. A
/// transcendental-containing region gets the widened comparison band on the
/// live kiss-ref / CPU-oracle path (see [`widen_bound_for_transcendental`]).
/// `Bind`/`Any` are leaves (no op); `SeeThrough` recurses.
pub fn region_contains_transcendental(region: &PatternNode) -> bool {
    match region {
        PatternNode::Op { op, operands, .. } => {
            is_transcendental(*op) || operands.iter().any(region_contains_transcendental)
        }
        PatternNode::SeeThrough { then } => region_contains_transcendental(then),
        PatternNode::Bind { .. } | PatternNode::Any => false,
    }
}

/// Widen a precision [`Bound`] to 2× for a live comparison of a
/// transcendental-containing region. Two implementations each within the ULP
/// ceiling `C` of the wide-precision truth can differ from EACH OTHER by up to
/// `2C` (triangle inequality); kiss-ref and Fuel's CPU oracle are both
/// hardware-precision (neither is the wide-precision truth), so a live
/// candidate-vs-reference check on a transcendental region must allow `2C`.
/// Tight transcendental verification defers to the frozen wide-precision
/// corpus, not to this live path (KISS, 2026-07-18). `MaxUlp` saturates so a
/// huge declared ceiling can never overflow (never-panic).
pub fn widen_bound_for_transcendental(bound: Bound) -> Bound {
    match bound {
        Bound::MaxUlp(m) => Bound::MaxUlp(m.saturating_mul(2)),
        Bound::MaxRelative(m) => Bound::MaxRelative(m * 2.0),
        Bound::MaxAbsolute(m) => Bound::MaxAbsolute(m * 2.0),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Bound, is_transcendental, region_contains_transcendental, ulp_distance,
        widen_bound_for_transcendental,
    };
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};

    /// F16 outputs are compared **in f16 space**, and the distance is the one
    /// f16 actually has.
    ///
    /// ⚠️ **The first version of this test was written as a characterisation
    /// of the OLD wrong behaviour, claimed to be born-red for the fix — and it
    /// was not. It passed before AND after.** Its "stale detector" asserted
    /// `!detail.contains("candidate 1 vs reference 2")`, an exact string from
    /// the old message format; the fixed verifier emits a different sentence,
    /// so the negative assertion held vacuously. **An assertion keyed to a
    /// message STRING cannot detect a behaviour change that rewrites the
    /// message** — which is the same defect as a report line that stopped
    /// depending on its measurement, pointed at a test.
    ///
    /// So this asserts the NUMBER instead. `1.0f16` is `0x3C00` and `2.0f16`
    /// is `0x4000`; as total-order keys that is 15360 and 16384, so the
    /// distance is exactly **1024 f16 ULP**. Nothing about that value is
    /// reachable by reinterpreting the same 8 bytes as two f32s, so it
    /// discriminates the two implementations by construction rather than by
    /// wording.
    #[test]
    fn f16_outputs_are_compared_in_f16_space_not_widened_to_f32() {
        use super::{Bound, verify_precision_bound};
        use crate::fkc::verify::bit_stability::{
            HostTensor, KernelInvoker, VerifyError, VerifyOutcome,
        };
        use crate::kernel::BindingEntry;

        struct ConstInvoker(Vec<u8>);
        impl KernelInvoker for ConstInvoker {
            fn invoke(
                &self,
                _entry: &BindingEntry,
                _inputs: &[HostTensor],
            ) -> Result<HostTensor, VerifyError> {
                Ok(HostTensor {
                    dtype: fuel_ir::DType::F16,
                    shape: vec![4],
                    bytes: self.0.clone(),
                })
            }
        }

        let ones: Vec<u8> = (0..4)
            .flat_map(|_| half::f16::from_f32(1.0).to_le_bytes())
            .collect();
        let twos: Vec<u8> = (0..4)
            .flat_map(|_| half::f16::from_f32(2.0).to_le_bytes())
            .collect();
        assert_eq!(
            ones.len(),
            8,
            "4 x f16 is 8 bytes — the length the old %4 guard let through"
        );

        let entry = BindingEntry {
            kernel: crate::dispatch::add_elementwise_f32_cpu_wrapper,
            caps: crate::kernel::KernelCaps::empty(),
            precision: crate::fused::PrecisionGuarantee::UNAUDITED,
            cost: crate::kernel::unknown_cost,
            kernel_source: "test",
            is_generic: false,
            kernel_revision_hash: 0,
            cost_expr: None,
        };
        let probes = vec![vec![]];

        // 1024 ULP apart in f16 space: inside a bound of 1024, outside 1023.
        // Asserting BOTH sides pins the number rather than just its direction —
        // a verifier that reported any large distance would satisfy only one.
        let inside = verify_precision_bound(
            &ConstInvoker(ones.clone()),
            &ConstInvoker(twos.clone()),
            &entry,
            &probes,
            Bound::MaxUlp(1024),
        )
        .expect("no infrastructure error");
        assert!(
            matches!(inside, VerifyOutcome::Pass),
            "1.0f16 and 2.0f16 are exactly 1024 f16-ULP apart and must pass a \
             1024 bound; got {inside:?}"
        );

        let outside = verify_precision_bound(
            &ConstInvoker(ones.clone()),
            &ConstInvoker(twos.clone()),
            &entry,
            &probes,
            Bound::MaxUlp(1023),
        )
        .expect("no infrastructure error");
        match outside {
            VerifyOutcome::Fail { detail } => assert!(
                detail.contains("1024 ULP apart"),
                "expected the true f16 distance of 1024 in the report; got: {detail}"
            ),
            other => panic!(
                "1.0f16 vs 2.0f16 must exceed a 1023 bound; got {other:?}. A Pass here \
                 is a fabricated agreement, which is worse than a wrong number."
            ),
        }

        // Identical buffers are 0 ULP apart under the strictest bound — the
        // control that stops the two assertions above from being satisfied by
        // a verifier that simply always reports a large distance.
        let same = verify_precision_bound(
            &ConstInvoker(ones.clone()),
            &ConstInvoker(ones),
            &entry,
            &probes,
            Bound::MaxUlp(0),
        )
        .expect("no infrastructure error");
        assert!(
            matches!(same, VerifyOutcome::Pass),
            "identical f16 buffers must be 0 ULP apart; got {same:?}"
        );
    }

    /// Integer outputs compare EXACTLY, because a ULP has no meaning there.
    ///
    /// `max_ulp: 0` on an integer-output `Cast` is a bit-exactness claim
    /// wearing a floating-point word — the claim is real, the name cannot
    /// express it, and the naming is tracked separately. What the comparison
    /// must not do is answer it in some other dtype's space.
    #[test]
    fn integer_outputs_compare_exactly_rather_than_in_ulp_space() {
        use super::{Bound, verify_precision_bound};
        use crate::fkc::verify::bit_stability::{
            HostTensor, KernelInvoker, VerifyError, VerifyOutcome,
        };
        use crate::kernel::BindingEntry;

        struct I8Invoker(Vec<u8>);
        impl KernelInvoker for I8Invoker {
            fn invoke(
                &self,
                _entry: &BindingEntry,
                _inputs: &[HostTensor],
            ) -> Result<HostTensor, VerifyError> {
                Ok(HostTensor {
                    dtype: fuel_ir::DType::I8,
                    shape: vec![4],
                    bytes: self.0.clone(),
                })
            }
        }

        let entry = BindingEntry {
            kernel: crate::dispatch::add_elementwise_f32_cpu_wrapper,
            caps: crate::kernel::KernelCaps::empty(),
            precision: crate::fused::PrecisionGuarantee::UNAUDITED,
            cost: crate::kernel::unknown_cost,
            kernel_source: "test",
            is_generic: false,
            kernel_revision_hash: 0,
            cost_expr: None,
        };
        let probes = vec![vec![]];

        // Four I8 elements are 4 bytes — the length the old `% 4` guard let
        // through as "one f32". A single byte differing by 1 must fail even
        // the most generous ULP bound, because the mode is exactness and the
        // bound's magnitude is irrelevant.
        let a = vec![1u8, 2, 3, 4];
        let b = vec![1u8, 2, 3, 5];
        let out = verify_precision_bound(
            &I8Invoker(a.clone()),
            &I8Invoker(b),
            &entry,
            &probes,
            Bound::MaxUlp(u32::MAX),
        )
        .expect("no infrastructure error");
        match out {
            VerifyOutcome::Fail { detail } => assert!(
                detail.contains("bit-exact"),
                "expected the failure to say integers are compared bit-exactly; got: {detail}"
            ),
            other => panic!(
                "a differing I8 byte must fail even an unbounded ULP claim; got {other:?} — \
                 a generous bound must not launder an integer mismatch"
            ),
        }

        let same = verify_precision_bound(
            &I8Invoker(a.clone()),
            &I8Invoker(a),
            &entry,
            &probes,
            Bound::MaxUlp(0),
        )
        .expect("no infrastructure error");
        assert!(
            matches!(same, VerifyOutcome::Pass),
            "identical I8 buffers must pass; got {same:?}"
        );
    }

    /// A dtype with no implemented comparison is REFUSED by name, never
    /// approximated. This is the property whose absence caused the original
    /// defect: an unhandled dtype fell through to an f32 reinterpretation.
    #[test]
    fn an_unimplemented_output_dtype_is_refused_by_name() {
        use super::{Bound, verify_precision_bound};
        use crate::fkc::verify::bit_stability::{
            HostTensor, KernelInvoker, VerifyError, VerifyOutcome,
        };
        use crate::kernel::BindingEntry;

        struct F8Invoker;
        impl KernelInvoker for F8Invoker {
            fn invoke(
                &self,
                _entry: &BindingEntry,
                _inputs: &[HostTensor],
            ) -> Result<HostTensor, VerifyError> {
                Ok(HostTensor {
                    // ⚠️ This test named F8E4M3 until F8E4M3 was implemented,
                    // at which point it failed — a test whose fixture is
                    // "something unsupported" expires as support grows, the
                    // same way a hand-picked entry and a hand-picked contract
                    // set did. F8E5M2 is unimplemented today; when it is not,
                    // this expires again and the durable fix is a dtype that
                    // is unsupported BY CONSTRUCTION rather than by schedule.
                    dtype: fuel_ir::DType::F8E5M2,
                    shape: vec![4],
                    bytes: vec![0x30, 0x31, 0x32, 0x33],
                })
            }
        }

        let entry = BindingEntry {
            kernel: crate::dispatch::add_elementwise_f32_cpu_wrapper,
            caps: crate::kernel::KernelCaps::empty(),
            precision: crate::fused::PrecisionGuarantee::UNAUDITED,
            cost: crate::kernel::unknown_cost,
            kernel_source: "test",
            is_generic: false,
            kernel_revision_hash: 0,
            cost_expr: None,
        };
        let out =
            verify_precision_bound(&F8Invoker, &F8Invoker, &entry, &[vec![]], Bound::MaxUlp(0))
                .expect("no infrastructure error");
        match out {
            VerifyOutcome::Fail { detail } => {
                assert!(
                    detail.contains("F8E5M2") && detail.contains("refused"),
                    "the refusal must NAME the dtype it refused; got: {detail}"
                );
            }
            other => panic!(
                "two IDENTICAL F8E5M2 buffers returned {other:?}. A Pass here is the \
                 defect: it would be a verdict produced without an implemented \
                 comparison, and identical inputs are exactly how such a verdict \
                 looks correct."
            ),
        }
    }

    #[test]
    fn widen_doubles_each_bound() {
        assert!(matches!(
            widen_bound_for_transcendental(Bound::MaxUlp(4)),
            Bound::MaxUlp(8)
        ));
        match widen_bound_for_transcendental(Bound::MaxRelative(1e-6)) {
            Bound::MaxRelative(m) => assert!((m - 2e-6).abs() < 1e-18),
            other => panic!("expected MaxRelative, got {other:?}"),
        }
        match widen_bound_for_transcendental(Bound::MaxAbsolute(0.25)) {
            Bound::MaxAbsolute(m) => assert_eq!(m, 0.5),
            other => panic!("expected MaxAbsolute, got {other:?}"),
        }
        // Saturates rather than overflowing (never-panic).
        assert!(matches!(
            widen_bound_for_transcendental(Bound::MaxUlp(u32::MAX)),
            Bound::MaxUlp(u32::MAX)
        ));
    }

    #[test]
    fn is_transcendental_classifies_exactly() {
        for t in [
            OpTag::Exp,
            OpTag::Log,
            OpTag::Sin,
            OpTag::Cos,
            OpTag::Tanh,
            OpTag::Sigmoid,
            OpTag::Silu,
            OpTag::Gelu,
            OpTag::GeluErf,
            OpTag::Erf,
            OpTag::Rsqrt,
        ] {
            assert!(is_transcendental(t), "{t:?} should be transcendental");
        }
        // Sqrt/Recip are IEEE correctly-rounded — NOT band-widened.
        for t in [
            OpTag::Sqrt,
            OpTag::Recip,
            OpTag::Relu,
            OpTag::Neg,
            OpTag::Abs,
            OpTag::Sqr,
        ] {
            assert!(!is_transcendental(t), "{t:?} should NOT be transcendental");
        }
    }

    #[test]
    fn region_transcendental_detection_walks_nested() {
        // Op{Neg, [Op{Exp, [Bind0]}]} — a nested transcendental atom.
        let inner = PatternNode::Op {
            op: OpTag::Exp,
            operands: vec![PatternNode::Bind { index: 0 }],
            attrs: OpAttrs::default(),
        };
        let outer = PatternNode::Op {
            op: OpTag::Neg,
            operands: vec![inner],
            attrs: OpAttrs::default(),
        };
        assert!(
            region_contains_transcendental(&outer),
            "nested Exp must be found"
        );

        // Op{Neg, [Op{Sqr, [Bind0]}]} — no transcendental atom.
        let inner2 = PatternNode::Op {
            op: OpTag::Sqr,
            operands: vec![PatternNode::Bind { index: 0 }],
            attrs: OpAttrs::default(),
        };
        let outer2 = PatternNode::Op {
            op: OpTag::Neg,
            operands: vec![inner2],
            attrs: OpAttrs::default(),
        };
        assert!(
            !region_contains_transcendental(&outer2),
            "no transcendental atom present"
        );
    }

    #[test]
    fn ulp_distance_signed_zero_is_one() {
        // -0.0 and +0.0 are adjacent in IEEE-754 total order: 1 ULP apart,
        // NOT 2^31 (the raw sign-magnitude subtraction bug).
        assert_eq!(ulp_distance(-0.0, 0.0), 1);
        assert_eq!(ulp_distance(0.0, -0.0), 1);
    }

    #[test]
    fn ulp_distance_same_value_is_zero() {
        assert_eq!(ulp_distance(1.0, 1.0), 0);
        assert_eq!(ulp_distance(-3.5, -3.5), 0);
    }

    #[test]
    fn ulp_distance_adjacent_same_sign_is_one() {
        let a = 1.0_f32;
        let b = f32::from_bits(a.to_bits() + 1); // next representable above 1.0
        assert_eq!(ulp_distance(a, b), 1);
        let c = -1.0_f32;
        let d = f32::from_bits(c.to_bits() + 1); // next-toward-zero below -1.0
        assert_eq!(ulp_distance(c, d), 1);
    }

    #[test]
    fn ulp_distance_straddling_zero_is_small() {
        // smallest +subnormal -> +0 -> -0 -> smallest -subnormal = 3 steps.
        let pos_min = f32::from_bits(1); // +2^-149
        let neg_min = f32::from_bits(0x8000_0001); // -2^-149
        assert_eq!(ulp_distance(pos_min, neg_min), 3);
    }

    #[test]
    fn ulp_distance_nan_handling_matches_kiss_ref() {
        // Regression for the drift the kiss-ref repoint fixed: exactly one NaN
        // must SATURATE (u32::MAX widened) so a NaN-vs-number pair can never pass
        // a `MaxUlp` bound; both-NaN conforms (0). The old Fuel-internal copy
        // returned a finite raw-key abs_diff for the one-NaN case.
        assert_eq!(ulp_distance(f32::NAN, 1.0), u64::from(u32::MAX));
        assert_eq!(ulp_distance(1.0, f32::NAN), u64::from(u32::MAX));
        assert_eq!(ulp_distance(f32::NAN, f32::NAN), 0);
    }
}
