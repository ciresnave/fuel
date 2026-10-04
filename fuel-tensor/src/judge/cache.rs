// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dispatch-table cache — compat shim + the one Tensor-dependent write
//! path.
//!
//! The storage/lookup mechanics ([`cached`], [`cached_oracle`],
//! [`invalidate`], the process-wide slot, and the `DispatchTable`/
//! `ProfileJudgeOracle` types) moved to
//! [`fuel_dispatch::judge_cache`] (board #109 PR-D follow-up) — this
//! file had zero `Tensor` dependency except [`populate_dispatch_table`]
//! below, which runs [`crate::judge::Judge`] (genuinely `Tensor`-dependent
//! — it realizes real tensor ops across backends to profile them, the
//! reason `judge/mod.rs` stayed in `fuel-tensor` at all in #305).
//!
//! `docs/restructure-migration-design.md` §5.1 Row 3 (RULED 2026-09-02)
//! called a bare `cache.rs -> fuel-dispatch` move clean; that was measured
//! before #305 landed `judge/mod.rs` in `fuel-tensor`, and a bare move
//! would have recreated the `fuel-dispatch -> fuel-tensor` cycle #305 was
//! built to avoid (see that doc's amended note). The fix inverts the one
//! write path instead: this file RUNS the judge (needs `Tensor`) and
//! HANDS the finished report down via
//! [`fuel_dispatch::judge_cache::store`] — a downward call, no cycle.
//! Every other symbol below is a straight re-export so
//! `crate::judge::cached()` / `fuel_core::judge::cache::cached()` /
//! `fuel::judge::cached()` etc. are unchanged for callers.

pub use fuel_dispatch::judge_cache::{
    Criterion, DEFAULT_ACCURACY_PENALTY, DispatchOptions, DispatchTable, OpKind, Pick,
    ProfileEntry, ProfileReport, SizeClass, cached, cached_oracle, invalidate,
};

/// Force-populate the dispatch table by running the probe + judge matrix
/// and persisting the result.
///
/// Idempotent: if a table is already cached (in memory or via the lazy
/// disk-load on first access), returns immediately. To force a fresh
/// measurement (driver upgrade, hardware change), call [`invalidate`]
/// first.
///
/// Apps that want zero startup cost should call this from a background
/// thread; the routed-op path falls through to default backends until the
/// populate completes. Apps that prefer determinism should call this on
/// the main thread at startup — blocks for tens of seconds on first-ever
/// run, instant on every subsequent run thanks to disk cache.
pub fn populate_dispatch_table() -> fuel_ir::Result<()> {
    if cached().is_some() {
        return Ok(());
    }
    let probe = fuel_hardware::probe::ProbeReport::probe_all();
    if let Some(p) = fuel_hardware::probe::default_report_path() {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        probe.save(&p)?;
    }
    let report = crate::judge::Judge::default().run(&probe);
    fuel_dispatch::judge_cache::store(&report)
}
