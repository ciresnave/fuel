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
//!   Group 3's return-rule interpreter is a later, not-yet-scheduled task —
//!   Task 4.6 turned out to be the rope-apply CUDA harness below, not this).
//!
//! ## Status — Task 4.5 (this slice)
//!
//! Adds the real CPU [`invoker_cpu::CpuInvoker`] — it drives an actual
//! registered `BindingEntry::kernel` fn-pointer against host-resident
//! bytes (no fake), the first invoker that can produce a genuine empirical
//! result. Also adds `#[cfg(feature = "cuda")]` /
//! `#[cfg(feature = "vulkan")]` device-invoker scaffolds
//! ([`invoker_cuda`] / [`invoker_vulkan`]) that upload/download bytes to
//! the respective device storage; their live-hardware tests are
//! `#[ignore]`'d (no device in this build environment to verify against).
//!
//! ## Status — Task 4.6 (this slice)
//!
//! Adds [`harness::run_fkc_verify_harness`] (`#[cfg(feature = "cuda")]`):
//! the rope-apply CUDA acceptance harness. Imports
//! `docs/kernel-contracts/cuda/rope-apply.fkc.md` (baracuda's
//! `rope_apply_<dt>_run` — shipped, never wired into any dispatch path
//! before this task) into a fresh, harness-local `KernelBindingTable`,
//! drives `verify_bit_stability` through a real [`CudaInvoker`], and
//! writes `pass`/`fail`/`no_reference` records keyed on
//! `kernel_revision_hash`. Its acceptance test,
//! `fkc_verify_rope_apply_writes_a_pass_ledger_entry`, is `#[ignore]`'d
//! (needs a live CUDA device) — see `harness.rs`'s module doc for what it
//! does and doesn't check (notably: no `verify_precision_bound` cross-check
//! against a CPU reference yet — a documented gap, not an oversight).
//!
//! ## Status — Task 4.3 (wiring, landed earlier than this file's numbering
//! suggests — commit `461c3bbc`)
//!
//! [`gate_precision`] is wired into the live import path: `register.rs`'s
//! `import_bundle_str` calls it (see `register.rs:363` and `:372`, once per
//! `precision` block on a plain and a fused kernel entry respectively)
//! against the embedded ledger, so every FKC import downgrades any
//! precision claim the ledger doesn't cover. It is no longer pure
//! unwired logic.
//!
//! NOT yet implemented (later tasks in the same program extend this file's
//! `mod` list when they land): the full accept-coverage cross-check; and a
//! `verify_precision_bound`-shaped helper that accepts per-invoker probes
//! (needed for any op whose candidate/reference backends disagree on
//! operand-shape convention, like rope-apply's half-width vs full-width
//! cos/sin tables).

mod accept_coverage;
mod bit_stability;
mod ledger;
mod ulp;
mod invoker_cpu;
mod seed_cpu_ledger;
#[cfg(feature = "cuda")]
mod invoker_cuda;
#[cfg(feature = "vulkan")]
mod invoker_vulkan;
#[cfg(feature = "cuda")]
mod harness;
#[cfg(feature = "cuda")]
mod seed_cuda_ledger;

// ---------------------------------------------------------------------------
// Re-export gating.
//
// `verify` is `pub(crate) mod` (see `fkc/mod.rs`), so a `pub use` here is only
// ever reachable from inside this crate — which means a re-export nobody in the
// crate NAMES is a plain unused import. Each line below is therefore gated to
// exactly the feature set of its real consumers rather than deleted, so the
// `jit` / `cuda` builds keep every name they reach either through `super::`
// (`harness.rs`, `seed_cuda_ledger.rs` — both `cuda`-gated) or through
// `crate::fkc::verify::` (`jit_ingest.rs`, `jit_ingest_probe.rs` — `jit`-gated,
// with a further `cuda`-gated block inside `jit_ingest`).
//
// Consumers that go through a SUBMODULE path instead
// (`crate::fkc::verify::bit_stability::…`, as `invoker_cpu`/`invoker_cuda`/
// `invoker_vulkan`/`ulp`/`seed_cpu_ledger` and their tests do) do not depend on
// any of these lines.
// ---------------------------------------------------------------------------

// `fill_deterministic`/`HostTensor`: `jit_ingest_probe` (jit) and
// `harness`+`seed_cuda_ledger` (cuda, via `super::`).
#[cfg(any(feature = "jit", feature = "cuda"))]
pub use bit_stability::{fill_deterministic, HostTensor};
// The invoker/outcome surface: `jit_ingest`'s cuda block (jit+cuda) and
// `harness`+`seed_cuda_ledger` (cuda, via `super::`) — i.e. `cuda` covers both.
#[cfg(feature = "cuda")]
pub use bit_stability::{
    verify_bit_stability, KernelInvoker, ProbeInputs, VerifyError, VerifyOutcome,
};
pub use ledger::{gate_precision, LedgerQuery, VerificationLedger};
// `LedgerRecord`: `jit_ingest` (jit) and `harness`+`seed_cuda_ledger` (cuda).
#[cfg(any(feature = "jit", feature = "cuda"))]
pub use ledger::LedgerRecord;
// The numeric-bound surface: consumed ONLY by `jit_ingest`'s cuda-gated
// candidate-vs-reference arm, i.e. `jit` AND `cuda`.
#[cfg(all(feature = "jit", feature = "cuda"))]
pub use ulp::{
    region_contains_transcendental, verify_precision_bound, widen_bound_for_transcendental, Bound,
};
// The per-op transcendental classification — the SINGLE source `jit_ingest`'s
// advisory ULP band shares with the region-level check above so the two never
// drift. `jit`-gated: its only crate consumer is `jit_ingest` (an un-gated
// re-export would warn as unused in non-jit builds).
#[cfg(feature = "jit")]
pub(crate) use ulp::is_transcendental;
// `CpuInvoker`: the CPU reference invoker `seed_cuda_ledger` diffs CUDA against
// (cuda, via `super::`). `seed_cpu_ledger` reaches it by submodule path.
#[cfg(feature = "cuda")]
pub use invoker_cpu::CpuInvoker;
// NOTE: `seed_cpu_ledger::{run_cpu_verification, SeedAttempt}` and
// `accept_coverage::verify_accept_coverage` are deliberately NOT re-exported —
// they have no in-crate consumer under any feature today, so the re-export was
// a pure unused import. The items themselves are untouched; re-add the
// `pub use` alongside the first consumer.
//
// `to_bytes` is `pub(crate)` on `seed_cpu_ledger` (not `pub`) — re-exported
// here at crate visibility so `jit_ingest_probe::probe_from_operands` can
// reuse the exact dtype-aware float→bytes encode without duplicating it.
// (`harness`/`seed_cuda_ledger` each carry their own private `to_bytes`.)
#[cfg(feature = "jit")]
pub(crate) use seed_cpu_ledger::to_bytes;
#[cfg(feature = "cuda")]
pub use invoker_cuda::CudaInvoker;
#[cfg(feature = "vulkan")]
pub use invoker_vulkan::VulkanInvoker;
#[cfg(feature = "cuda")]
pub use harness::run_fkc_verify_harness;
#[cfg(feature = "cuda")]
pub use seed_cuda_ledger::{run_cuda_verification, CudaSeedAttempt};

/// The embedded (compile-time) verification ledger. Thin wrapper so callers
/// outside this module can reach it as `verify::embedded()` without
/// depending on the `ledger` submodule path directly.
pub fn embedded() -> &'static VerificationLedger {
    VerificationLedger::embedded()
}
