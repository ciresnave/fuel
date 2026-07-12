//! Empirical precision-bound verification (`V-FKC-9`, Task 4.4).
//!
//! [`verify_precision_bound`] diffs a *candidate* kernel invocation against a
//! *reference*-tagged invocation of the same probe inputs, checking the
//! declared precision [`Bound`] (`max_ulp` / `max_relative` / `max_absolute`
//! from the FKC precision block). Hardware-free: both `cand` and `refr` are
//! [`KernelInvoker`]s, so unit tests here use fake in-process invokers; the
//! real CPU-reference-vs-CUDA-candidate wiring is Task 4.5.

use crate::fkc::verify::bit_stability::{HostTensor, KernelInvoker, ProbeInputs, VerifyError, VerifyOutcome};
use crate::kernel::BindingEntry;

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

/// Empirically checks a precision [`Bound`] by invoking `cand` and `refr` on
/// the same probes and comparing their `f32` outputs elementwise. Returns the
/// FIRST out-of-bound element as `Fail { detail }` (mirrors
/// `verify_bit_stability`'s "report the first divergence" posture) rather
/// than accumulating every mismatch.
///
/// Never panics: a probe whose byte length isn't a multiple of 4 would panic
/// inside `bytemuck::cast_slice`, so this reinterprets defensively — any
/// non-`f32`-aligned output is reported as a `Fail` rather than allowed to
/// panic the process. Mismatched candidate/reference output lengths only
/// compare the overlapping prefix (`zip` stops at the shorter side); that is
/// a conservative pass they'd otherwise need a separate shape-check for, and
/// is out of scope for this Phase-1 numeric-bound verifier.
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

        if a.bytes.len() % 4 != 0 || b.bytes.len() % 4 != 0 {
            return Ok(VerifyOutcome::Fail {
                detail: format!(
                    "probe {probe_idx}: output byte length not a multiple of 4 (cand {}, ref {}) — cannot reinterpret as f32",
                    a.bytes.len(),
                    b.bytes.len()
                ),
            });
        }
        let af: &[f32] = bytemuck::cast_slice(&a.bytes);
        let bf: &[f32] = bytemuck::cast_slice(&b.bytes);

        for (elem_idx, (x, y)) in af.iter().zip(bf.iter()).enumerate() {
            let ok = match bound {
                Bound::MaxAbsolute(m) => (*x as f64 - *y as f64).abs() <= m,
                Bound::MaxRelative(m) => {
                    let denom = (*y as f64).abs().max(f64::from(f32::EPSILON));
                    ((*x as f64 - *y as f64).abs() / denom) <= m
                }
                Bound::MaxUlp(m) => {
                    let ulp = (x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs();
                    ulp <= m as u64
                }
            };
            if !ok {
                return Ok(VerifyOutcome::Fail {
                    detail: format!(
                        "probe {probe_idx} elem {elem_idx}: candidate {x} vs reference {y} exceeds bound {bound:?}"
                    ),
                });
            }
        }
    }
    Ok(VerifyOutcome::Pass)
}
