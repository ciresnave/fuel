// SPDX-License-Identifier: MIT OR Apache-2.0
//! Moved to [`fuel_dispatch::telemetry::judge_sink`] (fuel-core dissolution,
//! plan item 8): it had zero coupling to `fuel-core`'s `Tensor`/`Device` or
//! to the rest of `judge/mod.rs`, only to the `ProfileJudgeOracle` adapter
//! and the report cache-dir path, both moved into `fuel-dispatch` ahead of
//! it in the same plan item. This module re-exports it so existing
//! `fuel_core::telemetry` / `fuel::telemetry` call sites keep compiling.
pub use fuel_dispatch::telemetry::judge_sink::*;
