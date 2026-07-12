//! Empirical verification of FKC precision claims (`V-FKC-9`).
//!
//! A kernel contract's `precision` block *asserts* claims (bit-stability,
//! max ULP, etc.) but nothing has historically checked that those claims
//! were ever actually measured. This module closes that gap: a git-checked-
//! in [`ledger::VerificationLedger`] records which `(kernel_revision_hash,
//! backend, dtypes, claim)` tuples have empirically passed, and a later
//! import-time gate downgrades any precision claim the ledger doesn't cover.
//!
//! ## Status — Task 4.2 (this slice)
//!
//! The ledger foundation (Task 4.1: [`LedgerRecord`], [`VerificationLedger`],
//! its `embedded()` loader, and `has_pass`) plus the import-time gate itself:
//! [`LedgerQuery`] + [`gate_precision`]. `gate_precision` checks each
//! machine-checkable claim in a declared `PrecisionGuarantee` against
//! `has_pass` for the current `kernel_revision_hash`; any unbacked claim
//! collapses the WHOLE guarantee to `UNAUDITED` plus a warning naming it.
//! An audited-none (no machine-checkable bounds) guarantee passes through
//! untouched. Declared here: `mod ledger` only.
//!
//! NOT yet implemented (later tasks in the same program — 4.4/4.5 per
//! `docs/session-prompts/` — extend this file's `mod` list when they land):
//! bit-stability / ULP / accept-coverage verifiers, and the CPU/CUDA/Vulkan
//! kernel invokers that produce ledger entries; and wiring `gate_precision`
//! into the actual import path (this task ships the gate as pure logic
//! only, not yet called from anywhere).

mod ledger;

pub use ledger::{gate_precision, LedgerQuery, LedgerRecord, VerificationLedger};

/// The embedded (compile-time) verification ledger. Thin wrapper so callers
/// outside this module can reach it as `verify::embedded()` without
/// depending on the `ledger` submodule path directly.
pub fn embedded() -> &'static VerificationLedger {
    VerificationLedger::embedded()
}
