// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuel-specific Error and Result
//!
//! The canonical definitions live in [`fuel_ir::error`].
//! This module re-exports them so that `crate::Error`, `crate::Result`,
//! `crate::Context`, etc. continue to resolve within fuel-core.
pub use fuel_ir::error::{Context, Error, MatMulUnexpectedStriding, Result, zip};

// `bail!` consolidation (fuel-core dissolution, GAP-347 PR 4): this used to
// be a local `macro_rules! bail` duplicating `fuel_ir::error::bail!` (the
// canonical copy, since `Error` above is itself a bare re-export of
// `fuel_ir::Error` — not a distinct type). `$crate` inside a `macro_rules!`
// body resolves to the crate that WROTE the macro, not the one that
// re-exports or invokes it, so re-exporting here keeps `fuel_core::bail!`
// (and, transitively, `fuel::bail!` and every bare `bail!` inside this
// crate) producing byte-identical `fuel_ir::Error::Msg` values — proved,
// not just reasoned about, by the before/after tests added across this
// PR's consumer crate families (fuel-datasets, fuel-nn, fuel-model-llama,
// fuel-model-phi, fuel-transformers, fuel-examples) plus fuel-tensor's
// pre-existing `bail_tests::bail_macro_returns_a_typed_err`.
pub use fuel_ir::bail;
