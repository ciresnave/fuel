//! Ranked dispatch tables — Phase 6b's O(1) runtime lookup.
//!
//! The [`Judge`](crate::judge) produces a [`ProfileReport`]
//! ([`crate::judge::cache::ProfileReport`]) of raw measurements. That's
//! not yet useful for realize-time dispatch — the router needs
//! to ask "given this (op, dtype, size_class), which backend wins
//! under criterion X?" and get a constant-time answer. This
//! module hosts the process-wide cache for that table; the table
//! types themselves live in [`fuel_ir::dispatch`] so that
//! `fuel-graph-router`'s `Router` can consume them without
//! depending on `fuel-core`.
//!
//! # Criteria
//!
//! - [`Criterion::Fastest`] — pick the backend with the lowest
//!   median latency. Excludes the reference backend by default
//!   (it's the oracle, not a production path — pathologically
//!   slow by design).
//! - [`Criterion::MostAccurate`] — pick the backend with the lowest
//!   `max_rel_error` against the reference backend. Also excludes
//!   reference (reference vs reference is by definition 0, but
//!   dispatching to it defeats the purpose of having an optimized
//!   backend at all). Ties broken by latency.
//! - [`Criterion::Balanced`] — minimise `latency_ns * (1 +
//!   accuracy_penalty * rel_error)`. Penalty coefficient defaults to
//!   100 so a 1% rel-error bump is equivalent to a 2× latency
//!   penalty — steep enough that numerically-unsound fast paths
//!   don't win by default.
//!
//! # Process-wide cache
//!
//! The dispatch table is hardware-determined: it doesn't depend on
//! the app, only on the CPU/GPU configuration. So a single
//! process-wide instance is the right granularity, with three
//! states:
//!
//! 1. **In-memory** — populated by an explicit
//!    [`populate_dispatch_table`] call this process. Authoritative.
//! 2. **On-disk** — persisted from a prior process; the same
//!    hardware was profiled previously. Lazy-loaded on first
//!    [`cached`] call.
//! 3. **Absent** — no profile yet, or hardware changed since the
//!    persisted profile was taken. Routed ops fall through to the
//!    Router's default backend until [`populate_dispatch_table`]
//!    runs successfully.

use crate::judge::oracle::ProfileJudgeOracle;
use fuel_ir::Result;
pub use fuel_ir::dispatch::{
    Criterion, DispatchOptions, DispatchTable, OpKind, Pick, ProfileEntry, ProfileReport,
    SizeClass, DEFAULT_ACCURACY_PENALTY,
};

use std::sync::{Arc, OnceLock, RwLock};

/// Both artifacts the Judge's profile run produces, derived from the
/// SAME [`ProfileReport`] so they can never disagree about what was
/// measured:
///
/// - `table` — winners-only O(1) lookup for the Router's routed
///   realize path.
/// - `oracle` — every raw measurement (losing alternatives included)
///   for the optimizer ranker's Layer-2 cost refinement
///   ([`fuel_dispatch::plan::PlanOptions::with_judge`]).
#[derive(Clone)]
struct CachedJudge {
    table:  Arc<DispatchTable>,
    oracle: Arc<ProfileJudgeOracle>,
}

impl CachedJudge {
    fn from_report(report: &ProfileReport) -> Self {
        Self {
            table:  Arc::new(DispatchTable::build(report)),
            oracle: Arc::new(ProfileJudgeOracle::from_report(report)),
        }
    }
}

/// Process-wide Judge cache. The outer `OnceLock` is set on
/// first access, with lazy-loaded contents from disk if a prior
/// run persisted a profile for the current hardware. The inner
/// `RwLock` exists so [`populate_dispatch_table`] and [`invalidate`]
/// can update the cache after first access — `OnceLock` alone is
/// write-once, which would prevent re-profiling on driver upgrades.
static DISPATCH_TABLE: OnceLock<RwLock<Option<CachedJudge>>> = OnceLock::new();

fn slot() -> &'static RwLock<Option<CachedJudge>> {
    DISPATCH_TABLE.get_or_init(|| {
        // First access this process: try to populate from disk
        // synchronously. Sub-millisecond if a valid prior profile
        // exists for the same hardware; returns None otherwise.
        let initial = try_load_persisted();
        RwLock::new(initial)
    })
}

/// The currently-active dispatch table, if any.
///
/// Returns `None` when no profile has been computed for this
/// hardware (first run, fresh install, or after [`invalidate`]).
/// Routed ops should fall through to a default backend in that case.
///
/// On the first call this process, lazily attempts to load a prior
/// run's profile from disk — sub-millisecond on cache hit.
/// Subsequent calls return without touching the filesystem.
pub fn cached() -> Option<Arc<DispatchTable>> {
    // A poisoned lock means a panic mid-update elsewhere; treat it
    // as "no profile" rather than propagating the panic — callers
    // all have a default-backend fallback.
    slot()
        .read()
        .ok()
        .and_then(|g| g.as_ref().map(|c| Arc::clone(&c.table)))
}

/// The currently-active [`ProfileJudgeOracle`], if any — same
/// lifecycle and data source as [`cached`], but exposing EVERY raw
/// measurement (losing alternatives included) instead of winners
/// only. The pipelined bridge attaches it to
/// [`fuel_dispatch::plan::PlanOptions::with_judge`] so `compile_plan`
/// refines Layer-1 static costs with measured latencies.
///
/// Returns `None` when no profile has been computed for this
/// hardware — plan compilation then ranks on Layer-1 static costs
/// alone, exactly the pre-oracle behavior.
pub fn cached_oracle() -> Option<Arc<ProfileJudgeOracle>> {
    slot()
        .read()
        .ok()
        .and_then(|g| g.as_ref().map(|c| Arc::clone(&c.oracle)))
}

/// Force-populate the dispatch table by running the probe + judge
/// matrix and persisting the result.
///
/// Idempotent: if a table is already cached (in memory or via the
/// lazy disk-load on first access), returns immediately. To force a
/// fresh measurement (driver upgrade, hardware change), call
/// [`invalidate`] first.
///
/// Apps that want zero startup cost should call this from a
/// background thread; the routed-op path falls through to default
/// backends until the populate completes. Apps that prefer
/// determinism should call this on the main thread at startup —
/// blocks for tens of seconds on first-ever run, instant on every
/// subsequent run thanks to disk cache.
pub fn populate_dispatch_table() -> Result<()> {
    if cached().is_some() { return Ok(()); }
    let probe = crate::probe::ProbeReport::probe_all();
    if let Some(p) = crate::probe::default_report_path() {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        probe.save(&p)?;
    }
    let report = crate::judge::Judge::default().run(&probe);
    if let Some(p) = super::default_report_path() {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        report.save(&p)?;
    }
    let built = CachedJudge::from_report(&report);
    *slot()
        .write()
        .map_err(|_| fuel_ir::Error::Msg("judge cache lock poisoned".into()))? =
        Some(built);
    Ok(())
}

/// Drop the in-memory cache and delete the persisted profile on
/// disk. The next [`populate_dispatch_table`] call will re-run the
/// probe + judge from scratch.
///
/// Use this when an external change has invalidated the existing
/// profile — driver upgrade, BLAS library swap, OS update —
/// and you want the next measurement to reflect the new state.
/// Without this call, [`cached`] would keep returning the stale
/// in-memory table and [`populate_dispatch_table`] would no-op
/// because of its idempotence guard.
pub fn invalidate() -> Result<()> {
    *slot()
        .write()
        .map_err(|_| fuel_ir::Error::Msg("judge cache lock poisoned".into()))? = None;
    if let Some(p) = crate::probe::default_report_path() {
        let _ = std::fs::remove_file(&p);
    }
    if let Some(p) = super::default_report_path() {
        let _ = std::fs::remove_file(&p);
    }
    Ok(())
}

/// Try to load a previously-persisted profile from disk. Returns
/// `None` if anything is missing, the schema versions mismatch, or
/// the current hardware doesn't match what was probed when the
/// profile was last saved.
fn try_load_persisted() -> Option<CachedJudge> {
    let probe_path = crate::probe::default_report_path()?;
    let prior_probe = crate::probe::ProbeReport::load(&probe_path).ok().flatten()?;
    let now_probe = crate::probe::ProbeReport::probe_all();
    if now_probe.diff(&prior_probe).needs_rejudge() {
        return None;
    }
    let judge_path = super::default_report_path()?;
    let report = ProfileReport::load(&judge_path).ok().flatten()?;
    Some(CachedJudge::from_report(&report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::PROFILE_REPORT_VERSION;
    use fuel_ir::probe::BackendId;
    use fuel_ir::DType;

    fn entry(backend: BackendId, op: OpKind, size: u8, latency: u64, err: f32) -> ProfileEntry {
        ProfileEntry {
            op,
            dtype:         DType::F32,
            size_class:    SizeClass(size),
            backend,
            device_index:  0,
            latency_ns:    latency,
            iterations:    7,
            max_rel_error: err,
            kernel_source: String::new(),
        }
    }

    /// Variant that lets a test pin a specific `kernel_source` on the
    /// fixture entry — exercises per-alternative dispatch (e.g. AOCL
    /// vs MKL siblings at one cell).
    #[allow(dead_code)]
    fn entry_with_source(
        backend: BackendId,
        op: OpKind,
        size: u8,
        latency: u64,
        err: f32,
        kernel_source: &str,
    ) -> ProfileEntry {
        ProfileEntry {
            op,
            dtype:         DType::F32,
            size_class:    SizeClass(size),
            backend,
            device_index:  0,
            latency_ns:    latency,
            iterations:    7,
            max_rel_error: err,
            kernel_source: kernel_source.to_string(),
        }
    }

    fn sample_report() -> ProfileReport {
        ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![
                // Size class 12: CUDA wins fastest (2ms < 10ms) but errs more
                entry(BackendId::Cpu,       OpKind::MatMul, 12, 10_000_000, 1e-6),
                entry(BackendId::Cuda,      OpKind::MatMul, 12,  2_000_000, 1e-4),
                // Size class 16: CPU is fastest + most accurate
                entry(BackendId::Cpu,       OpKind::MatMul, 16,   500_000, 1e-6),
                entry(BackendId::Cuda,      OpKind::MatMul, 16, 1_000_000, 1e-3),
            ],
        }
    }

    #[test]
    fn fastest_picks_lowest_latency() {
        let tbl = DispatchTable::build(&sample_report());
        let p = tbl.pick(OpKind::MatMul, DType::F32, SizeClass(12), Criterion::Fastest).unwrap();
        assert_eq!(p, Pick {
            backend: BackendId::Cuda,
            device_index: 0,
            kernel_source: "",
        });
    }

    #[test]
    fn most_accurate_picks_lowest_rel_err() {
        let tbl = DispatchTable::build(&sample_report());
        let p = tbl.pick(OpKind::MatMul, DType::F32, SizeClass(12), Criterion::MostAccurate).unwrap();
        // CPU has 1e-6 rel_err; CUDA has 1e-4. CPU wins MostAccurate.
        assert_eq!(p, Pick {
            backend: BackendId::Cpu,
            device_index: 0,
            kernel_source: "",
        });
    }

    #[test]
    fn pick_nearest_falls_back_to_largest_class() {
        let tbl = DispatchTable::build(&sample_report());
        // Size class 14: not profiled. Nearest are 12 (diff 2) and 16
        // (diff 2). Tie-break prefers larger → 16 → CPU wins fastest.
        let p = tbl.pick_nearest(OpKind::MatMul, DType::F32, SizeClass(14), Criterion::Fastest).unwrap();
        assert_eq!(p, Pick {
            backend: BackendId::Cpu,
            device_index: 0,
            kernel_source: "",
        });
    }

    #[test]
    fn keys_returns_distinct_combinations() {
        let tbl = DispatchTable::build(&sample_report());
        let keys = tbl.keys();
        // Two distinct (op, dtype, size_class) triples in the sample
        assert_eq!(keys.len(), 2);
    }

    /// Per-alternative measurement v2: two CPU kernel siblings at one
    /// `(op, dtype, size_class)` cell produce two separate
    /// `ProfileEntry`s. The picker selects the faster as winner; the
    /// `Pick::kernel_source` surfaces which sibling won, even though
    /// both share `(BackendId::Cpu, device_index=0)`.
    #[test]
    fn two_kernel_sources_at_one_cpu_cell_produce_two_entries_and_picker_names_winner() {
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![
                // Two CPU MatMul kernels at the same cell: AOCL is faster
                // (200 µs) vs portable-cpu (1 ms). They differ ONLY in
                // kernel_source — same backend, same device.
                entry_with_source(
                    BackendId::Cpu, OpKind::MatMul, 12, 1_000_000, 1e-6, "portable-cpu",
                ),
                entry_with_source(
                    BackendId::Cpu, OpKind::MatMul, 12,   200_000, 1e-6, "aocl",
                ),
            ],
        };

        // ProfileEntry shape: both entries are present and the
        // kernel_source field distinguishes them.
        assert_eq!(report.entries.len(), 2);
        let portable = &report.entries[0];
        let aocl     = &report.entries[1];
        assert_eq!(portable.kernel_source, "portable-cpu");
        assert_eq!(aocl.kernel_source,     "aocl");
        assert_eq!(portable.backend, aocl.backend); // same backend
        assert_eq!(portable.device_index, aocl.device_index); // same device

        // DispatchTable picks the winner ACROSS kernel_sources at one
        // cell. Pick.kernel_source identifies it.
        let tbl = DispatchTable::build(&report);
        let p = tbl.pick(OpKind::MatMul, DType::F32, SizeClass(12), Criterion::Fastest).unwrap();
        assert_eq!(p.backend, BackendId::Cpu);
        assert_eq!(p.device_index, 0);
        // AOCL is faster → wins Fastest.
        assert_eq!(p.kernel_source, "aocl");
    }
}
