//! Empirical verification of FKC precision claims (`V-FKC-9`).
//!
//! A kernel contract's `precision` block *asserts* claims (bit-stability,
//! max ULP, etc.) but nothing has historically checked that those claims
//! were ever actually measured. This module closes that gap: a git-checked-
//! in [`ledger::VerificationLedger`] records which `(kernel_revision_hash,
//! backend, dtypes, claim)` tuples have empirically passed, and a later
//! import-time gate downgrades any precision claim the ledger doesn't cover.
//!
//! ## Status — Task 4.4 (this slice)
//!
//! The ledger foundation (Task 4.1: [`LedgerRecord`], [`VerificationLedger`],
//! its `embedded()` loader, and `has_pass`) plus the import-time gate
//! ([`LedgerQuery`] + [`gate_precision`], Task 4.2). This slice (Task 4.4)
//! adds the empirical VERIFICATION MACHINERY that *produces* ledger
//! entries — still hardware-free, exercised here with a fake in-process
//! [`KernelInvoker`]:
//! - [`bit_stability::verify_bit_stability`] — the `bit_stable_on_same_hardware`
//!   claim, N repeat calls, byte-identical required.
//! - [`ulp::verify_precision_bound`] — the `max_ulp` / `max_relative` /
//!   `max_absolute` claims, candidate-vs-reference diffing.
//! - [`accept_coverage::verify_accept_coverage`] — a Phase-1 smoke-check
//!   stub for the `accept` block's declared combos (full cross-check against
//!   Group 3's return-rule interpreter is Task 4.6).
//!
//! NOT yet implemented (later tasks in the same program — 4.5/4.6 per
//! `docs/session-prompts/` — extend this file's `mod` list when they land):
//! the real CPU/CUDA/Vulkan kernel invokers that call actual device kernels
//! and produce ledger entries; the full accept-coverage cross-check; and
//! wiring `gate_precision` into the actual import path (it ships as pure
//! logic only, not yet called from anywhere).

mod accept_coverage;
mod bit_stability;
mod ledger;
mod ulp;

pub use accept_coverage::verify_accept_coverage;
pub use bit_stability::{
    fill_deterministic, verify_bit_stability, HostTensor, KernelInvoker, ProbeInputs, VerifyError,
    VerifyOutcome,
};
pub use ledger::{gate_precision, LedgerQuery, LedgerRecord, VerificationLedger};
pub use ulp::{verify_precision_bound, Bound};

/// The embedded (compile-time) verification ledger. Thin wrapper so callers
/// outside this module can reach it as `verify::embedded()` without
/// depending on the `ledger` submodule path directly.
pub fn embedded() -> &'static VerificationLedger {
    VerificationLedger::embedded()
}
