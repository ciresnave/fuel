//! Accept-coverage smoke verification (`V-FKC-9`, Task 4.4 — Phase-1 stub).
//!
//! A kernel contract's `accept` block declares which `(dtypes, layout,
//! op_params variant)` combinations a kernel claims to handle. The full
//! verifier (Task 4.6) will synthesize probes for every declared combo, invoke
//! the kernel, and cross-check the real output's shape/dtype against the
//! contract's `return` block via Group 3's return-rule interpreter
//! (`crate::fkc::return_check::{eval_dtype_rule, eval_shape_rule}`) — closing
//! the "declared accept coverage was never actually exercised" gap.
//!
//! This slice (Task 4.4) ships only a minimal smoke-check placeholder: it
//! confirms the invoker can be called successfully on every supplied probe,
//! without yet cross-checking output shape/dtype against declared return
//! rules. It exists so callers (and the harness scaffolding landing in later
//! tasks) have a stable, never-panic entry point to build on, rather than
//! wiring the interpreter prematurely against a probe-synthesis surface that
//! doesn't exist yet.

use crate::fkc::verify::bit_stability::{KernelInvoker, ProbeInputs, VerifyError, VerifyOutcome};
use crate::kernel::BindingEntry;

/// Phase-1 accept-coverage smoke-check: invokes `inv` once per probe and
/// requires every call to succeed. Returns [`VerifyOutcome::NoReference`] for
/// an empty probe list — there is nothing declared to smoke-test, which is a
/// clean "not applicable" outcome rather than a vacuous pass.
///
/// Never-panic: every branch returns a value; the only way this function
/// stops early is a propagated [`VerifyError`] from the invoker itself (an
/// infrastructure failure, not a panic), matching the posture of
/// [`super::bit_stability::verify_bit_stability`] and
/// [`super::ulp::verify_precision_bound`].
pub fn verify_accept_coverage(
    inv: &dyn KernelInvoker,
    entry: &BindingEntry,
    probes: &[ProbeInputs],
) -> Result<VerifyOutcome, VerifyError> {
    if probes.is_empty() {
        return Ok(VerifyOutcome::NoReference);
    }
    for probe in probes {
        inv.invoke(entry, probe)?;
    }
    Ok(VerifyOutcome::Pass)
}
