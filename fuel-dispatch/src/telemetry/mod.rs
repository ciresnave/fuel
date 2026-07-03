//! Baracuda dispatch-telemetry / miss-reporting **emission** layer.
//!
//! This is the emission half of the Fuel↔Baracuda boundary (the tensor-
//! description half is FDX; the kernel-advertisement half is FKC). It does NOT
//! retain timings — the Judge already retains per-`(op, dtype, size_class,
//! backend, kernel_source)` latencies including losers. This module turns that
//! retained data, plus the planner's own contract-matching outcome, into a
//! `DispatchRecord` / `MissRecord` JSONL feed for Baracuda's AOT kernel-
//! specialization matrix.
//!
//! Design: `docs/session-prompts/baracuda-telemetry-plan.md`. Behind the
//! `telemetry` cargo feature; default builds are untouched, and **no record is
//! ever written unless emission is explicitly enabled** (the opt-in flag, a
//! later step). The JSONL *sink* (file writer + on-disk path) lives in
//! `fuel-core` (it has the concrete oracle + cache dir); this crate owns the
//! record types and the key derivation.

pub mod impl_id;
pub mod record;
pub mod structure_key;

pub use impl_id::ImplId;
pub use record::{Candidate, DispatchRecord, HwStamp, MissRecord};
pub use structure_key::StructureKeyToken;
