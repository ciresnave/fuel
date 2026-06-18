//! FDX — the Fuel DLPack eXtension: a versioned, optional sidecar over
//! standard DLPack for tensor interchange between Fuel, its kernels, and the
//! ecosystem. Canonical design: `docs/specs/dlpack-extension.md`.
//!
//! Behind the `dlpack` cargo feature. This module is the normative owner of
//! the shared dtype/quant/granularity/substrate/gather/extent codes (the FKC
//! kernel-contract format references them by symbol; see [`codes`]).
//!
//! Build order (per `docs/session-prompts/dlpack-comm-layer-plan.md`):
//! 1. [`abi`] — standard DLPack structs (done).
//! 2. [`codes`] — FDX magic/version/flags/codes/sentinels (done).
//! 3. `sidecar` — the `FDXSidecar` + sub-structs (gather, affine, quant,
//!    dtype-ext, residency, bundle) — next slice.
//! 4. `validate` — the V* checks as `Result` — next slice.
//! 5. `fuel_dlpack_ext.h` — the co-maintained C header — next slice.

pub mod abi;
pub mod codes;
pub mod sidecar;
pub mod validate;
