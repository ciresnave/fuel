// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! later step). The JSONL *sink* (file writer + on-disk path, [`judge_sink`])
//! moved here from `fuel-core` (fuel-core dissolution, plan item 8) once its
//! two preconditions — the `ProfileJudgeOracle` adapter and the report's
//! cache-dir path — moved out of `fuel-core::judge` ahead of it
//! ([`crate::ranker::profile_oracle`], [`crate::judge_report_path`]). This
//! crate now owns the record types, the key derivation, AND the sink.

// ⚠️ TWO GATES, DELIBERATELY DIFFERENT SCOPES. Do not unify them.
//
//   COMPILE-VISIBILITY (here)  -> `baracuda-types`
//   RUNTIME-SELECTION          -> `cuda`: [`judge_sink::TelemetryInstall::new`]
//                                 picks `BaracudaStructureKeyProvider` vs
//                                 `NullStructureKeyProvider` as a cfg-selected
//                                 struct field on THIS crate's `cuda` flag.
//                                 (Moved in from `fuel-core/src/telemetry.rs`,
//                                 fuel-core dissolution plan item 8 — selection
//                                 lived in a different crate from this gate
//                                 when this comment was first written; it now
//                                 lives in the same crate, same reasoning.)
//
// The provider is pure host code — a keying function over operand descriptors
// in `baracuda-kernels-types`, no FFI and no device — so `baracuda-types` is
// exactly what it needs to COMPILE. It was gated on `cuda`, which additionally
// pulls `fuel-cuda-backend` and therefore the kernel forge, so this file could
// only be type-checked at the cost of a full forge build (measured 1h51m cold,
// ~14s narrowed). That is why an `E0004` here survived an entire dtype sweep:
// no gate anyone runs could afford to compile it. (GAP-173.)
//
// Narrowing the COMPILE gate must not widen SELECTION: a CPU-only build that
// started emitting real baracuda structure keys for an arch it is not running
// would be a WRONG signal, and this module's posture is that no signal beats a
// wrong one. That separation is structural rather than careful — selection is
// gated on `cuda` here specifically so widening the COMPILE gate above cannot
// reach it. The `pub use` below stays on `cuda` for the same reason.
#[cfg(feature = "baracuda-types")]
pub mod baracuda_provider;
pub mod config;
pub mod hooks;
pub mod impl_id;
#[cfg(feature = "telemetry")]
pub mod judge_sink;
pub mod miss;
pub mod record;
pub mod sink;
pub mod structure_key;
pub mod structure_key_derive;

#[cfg(feature = "cuda")]
pub use baracuda_provider::BaracudaStructureKeyProvider;
pub use config::{TelemetryConfig, TelemetryMode};
pub use hooks::TelemetryHooks;
pub use impl_id::ImplId;
#[cfg(feature = "telemetry")]
pub use judge_sink::{
    TelemetryInstall, current_mode, default_telemetry_dir, disable, enable, flush, flush_to,
    is_enabled, reset,
};
pub use miss::{AdmittedContract, detect_miss, detect_miss_precomputed, is_generic_contract};
pub use record::{Candidate, DispatchRecord, HwStamp, MissRecord, TELEMETRY_SCHEMA_VERSION};
pub use sink::TelemetrySink;
pub use structure_key::{
    Contiguity, FdxOperandDesc, NullStructureKeyProvider, StructureKeyProvider, StructureKeyToken,
};
