//! Orchestrator for the Phase 6b probe → judge → dispatch pipeline.
//!
//! Wraps the three pieces so callers get a ready-to-query
//! [`DispatchTable`] from one call. Does the right thing re: reuse
//! across restarts: if the persisted probe matches the current
//! hardware and a persisted profile is present, the Judge is
//! skipped entirely.
//!
//! # One-call API
//!
//! ```no_run
//! use fuel_core::scheduling::{prepare_dispatch_table, ScheduleOptions};
//! let (table, _report) = prepare_dispatch_table(ScheduleOptions::default())
//!     .expect("prepare_dispatch_table");
//! // `table` is now queryable with `.pick(op, dtype, size, criterion)`.
//! ```
//!
//! # Reuse rules
//!
//! 1. Probe the current hardware.
//! 2. If a persisted [`ProbeReport`] exists and `probe.diff(&prior)`
//!    is [`HardwareChange::Unchanged`], look for a persisted
//!    [`ProfileReport`].
//! 3. If a persisted profile is present **and** its schema version
//!    matches, **skip the Judge** and build the dispatch table from
//!    the persisted profile. Save cost on every startup.
//! 4. Otherwise re-run the Judge and persist both reports.
//!
//! # Where the reports live
//!
//! By default, both reports live under the OS cache dir
//! (`%LOCALAPPDATA%\fuel\` on Windows, `$XDG_CACHE_HOME/fuel/` on
//! Linux). Callers that want explicit paths (for CI, containers,
//! per-user config) can override via [`ScheduleOptions`].

use crate::dispatch::{DispatchOptions, DispatchTable};
use crate::judge::{Judge, ProfileReport};
use crate::probe::{HardwareChange, ProbeReport};
use fuel_core_types::Result;
use std::path::PathBuf;

/// Options for [`prepare_dispatch_table`] — lets callers override
/// paths, force a re-Judge, and control dispatch-table construction.
pub struct ScheduleOptions {
    /// Explicit path for the probe report. `None` = OS cache default.
    pub probe_path: Option<PathBuf>,
    /// Explicit path for the profile report. `None` = OS cache default.
    pub profile_path: Option<PathBuf>,
    /// Force the Judge to re-run even if the persisted state is
    /// otherwise reusable. Useful after toolchain upgrades where a
    /// driver version bump didn't happen but compiler codegen
    /// changed.
    pub force_rejudge: bool,
    /// Judge config. Default = `Judge::default()`.
    pub judge: Judge,
    /// Dispatch table construction options.
    pub dispatch: DispatchOptions,
}

impl Default for ScheduleOptions {
    fn default() -> Self {
        Self {
            probe_path:    None,
            profile_path:  None,
            force_rejudge: false,
            judge:         Judge::default(),
            dispatch:      DispatchOptions::default(),
        }
    }
}

/// Probe → [load / re-Judge] → build dispatch. Persists both reports
/// on a fresh Judge run. Returns the dispatch table paired with the
/// profile report it was built from (callers often want the raw
/// measurements too — for logging, debugging, or a custom secondary
/// dispatch table).
pub fn prepare_dispatch_table(
    opts: ScheduleOptions,
) -> Result<(DispatchTable, ProfileReport)> {
    let probe_path = opts.probe_path.clone()
        .or_else(crate::probe::default_report_path);
    let profile_path = opts.profile_path.clone()
        .or_else(crate::judge::default_report_path);

    let current_probe = ProbeReport::probe_all();

    // Step 1: decide if the persisted profile is reusable.
    let mut reuse_profile = false;
    if !opts.force_rejudge {
        if let Some(pp) = probe_path.as_ref() {
            if let Ok(Some(prior)) = ProbeReport::load(pp) {
                if matches!(current_probe.diff(&prior), HardwareChange::Unchanged) {
                    reuse_profile = true;
                }
            }
        }
    }

    let profile = if reuse_profile {
        if let Some(pp) = profile_path.as_ref() {
            match ProfileReport::load(pp)? {
                Some(r) => r,
                None => run_and_persist(&current_probe, &opts, &probe_path, &profile_path)?,
            }
        } else {
            run_and_persist(&current_probe, &opts, &probe_path, &profile_path)?
        }
    } else {
        run_and_persist(&current_probe, &opts, &probe_path, &profile_path)?
    };

    let table = DispatchTable::build_with(&profile, opts.dispatch);
    Ok((table, profile))
}

fn run_and_persist(
    probe: &ProbeReport,
    opts: &ScheduleOptions,
    probe_path: &Option<PathBuf>,
    profile_path: &Option<PathBuf>,
) -> Result<ProfileReport> {
    let profile = opts.judge.run(probe);

    // Best-effort persistence: if the parent dir doesn't exist or a
    // write fails, log to stderr and continue. Dispatch decisions
    // still work in-memory even when we can't write the reports.
    if let Some(p) = probe_path {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = probe.save(p) {
            eprintln!("fuel scheduling: failed to persist probe report to {p:?}: {e}");
        }
    }
    if let Some(p) = profile_path {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = profile.save(p) {
            eprintln!("fuel scheduling: failed to persist profile report to {p:?}: {e}");
        }
    }

    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::{OpKind, OpSize};

    /// Force-rejudge on a fresh scratch dir — verifies the full
    /// orchestrator path end-to-end (no prior state, run everything,
    /// persist, reload).
    #[test]
    fn end_to_end_probe_judge_dispatch() {
        let scratch = std::env::temp_dir().join(format!(
            "fuel-schedule-test-{}", std::process::id()
        ));
        let _ = std::fs::create_dir_all(&scratch);

        let opts = ScheduleOptions {
            probe_path:    Some(scratch.join("probe.json")),
            profile_path:  Some(scratch.join("judge.json")),
            force_rejudge: true,
            judge: Judge {
                iterations: 3,
                warmup: 1,
                size_plan_override: Some(vec![
                    (OpKind::MatMul, OpSize::MatMul { m: 32, n: 32, k: 32 }),
                ]),
            },
            dispatch: Default::default(),
        };

        let (table, profile) = prepare_dispatch_table(opts).expect("prepare");
        assert!(profile.entries.len() >= 2, "profile should have cpu + ref entries");
        assert!(table.len() >= 1, "dispatch table should have at least one entry");

        // Both files should exist now.
        assert!(scratch.join("probe.json").exists());
        assert!(scratch.join("judge.json").exists());

        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// Second call with matching hardware and no force_rejudge should
    /// skip the Judge (observable as "profile entries come from the
    /// persisted file, not a fresh measurement"). We can't assert
    /// exact byte equality because the device's description timing
    /// varies, but entry count should be stable.
    #[test]
    fn reuses_profile_when_hardware_unchanged() {
        let scratch = std::env::temp_dir().join(format!(
            "fuel-schedule-reuse-{}", std::process::id()
        ));
        let _ = std::fs::create_dir_all(&scratch);

        let tiny_judge = || Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(vec![
                (OpKind::MatMul, OpSize::MatMul { m: 16, n: 16, k: 16 }),
            ]),
        };

        let first = prepare_dispatch_table(ScheduleOptions {
            probe_path:    Some(scratch.join("probe.json")),
            profile_path:  Some(scratch.join("judge.json")),
            force_rejudge: true,
            judge:         tiny_judge(),
            dispatch:      Default::default(),
        }).expect("first run");

        let second = prepare_dispatch_table(ScheduleOptions {
            probe_path:    Some(scratch.join("probe.json")),
            profile_path:  Some(scratch.join("judge.json")),
            force_rejudge: false,  // reuse eligible
            judge:         tiny_judge(),
            dispatch:      Default::default(),
        }).expect("second run");

        // Second run's profile should equal first's (loaded from disk,
        // not re-measured). Latency fields would differ on re-measure
        // — identical if loaded from disk.
        assert_eq!(first.1, second.1,
            "hardware unchanged + no force_rejudge should yield identical profile");

        let _ = std::fs::remove_dir_all(&scratch);
    }
}
