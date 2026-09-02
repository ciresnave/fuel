// SPDX-License-Identifier: MIT OR Apache-2.0
//! # Fuel
//!
//! Fuel is a DAG-first, lazy-only ML framework for Rust. This crate is the
//! **public facade** for the framework: it re-exports the entire public surface
//! of [`fuel_core`], so `use fuel::…` resolves exactly as it did when `fuel` was
//! a manifest alias for `fuel-core` (restructure Stage 1). The
//! byte-identical-public-API gate (`cargo public-api`, run per feature set)
//! enforces that the surface is unchanged; feature flags are forwarded 1:1 to
//! `fuel-core` (see this crate's manifest).
//!
//! See `docs/restructure-migration-design.md` §Stage 1.

// The entire public surface of fuel-core, in the type/value namespaces — every
// public module (and thus every path beneath it), type, function, and re-export.
// Feature-gated items ride along: they are present in fuel-core only when the
// corresponding feature is on, and this crate forwards those features, so the
// glob carries exactly what fuel-core exposes under the active feature set.
pub use fuel_core::*;

// `#[macro_export]` macros are placed at the CRATE ROOT in the MACRO namespace,
// which a glob re-export (`pub use fuel_core::*`) does NOT carry. `bail!` is the
// framework's one exported macro (used in ~60 consumer files), so it is
// re-exported explicitly here. The public-API gate reports it (`pub macro
// fuel::bail!`), so its presence is verified, not assumed.
pub use fuel_core::bail;

// Stage 2: the model zoo moved from fuel-core into fuel-transformers (Models
// tier). Re-export it so `fuel::lazy_bert` etc. still resolve — the consumer
// path set is unchanged (147 − 1: lazy_latent_cache stays in fuel-core). This
// glob carries exactly the moved models; `fuel_transformers::models` is exactly
// the `lazy_<model>` modules, disjoint from `fuel_core::*` by the stay-list
// (fuel-core keeps `lazy` and `lazy_latent_cache`, neither a `lazy_<model>`
// re-export collision) — asserted in tests/facade_disjoint.rs.
pub use fuel_transformers::models::*;
