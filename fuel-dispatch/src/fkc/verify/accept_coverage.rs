// SPDX-License-Identifier: MIT OR Apache-2.0
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
/// ⚠️ **GAP-217b trace, 2026-08-20: this function has NO consumer anywhere,
/// under any feature — and it is the ROOT of four other dead-code reports,
/// which is why they are one finding rather than five.**
///
/// It is the ONLY constructor of [`VerifyOutcome::NoReference`] (`:39`
/// below). With it unwired, that variant is unconstructible, and the four
/// `Ok(Ok(VerifyOutcome::NoReference)) => ...` arms in `harness`,
/// `seed_cpu_ledger` (x2), `seed_cuda_ledger` and `seed_vulkan_ledger` are
/// **unreachable**: they name an outcome nothing can produce.
///
/// **Kept, not deleted.** `verify/mod.rs` already records the decision — no
/// in-crate consumer today, re-export alongside the first one — and deleting
/// it would take the variant and the arms with it, destroying the only
/// evidence that "nothing to smoke-test" was ever meant to be a distinct
/// outcome. The arms are harmless: they map to an "unverified" log line.
///
/// **What is NOT harmless is leaving this undated.** It is an expiring
/// decline with no detector — if the consumer never lands, nothing fires and
/// this outlives its reason silently. Tie it to a checkpoint that WILL occur
/// rather than to an event that may not: if `verify_accept_coverage` still
/// has no consumer when the accept-coverage claim is next touched, it and its
/// variant come out together.
#[allow(
    dead_code,
    reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
)]
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
