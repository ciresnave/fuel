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

use crate::dispatch::{Criterion, DispatchOptions, DispatchTable, Pick};
use crate::judge::{Judge, OpKind, ProfileReport, SizeClass};
use crate::probe::{HardwareChange, ProbeReport};
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, DeviceLocation, Result};
use fuel_graph::{Graph, NodeId, Op};
use std::collections::HashMap;
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

// ---- Placement recommender ----------------------------------------------
//
// Phase 6b's last machinery piece: walk a graph, ask the dispatch
// table where each node would run best, return a per-NodeId placement
// plan. The plan is the input that feeds into
// `fuel_graph::opt::insert_moves` (or future router-level placement
// passes); this function does NOT mutate the graph itself.
//
// Today the dispatch table only knows about MatMul + AddElementwise
// (the Judge's profile matrix). Every other op falls through to
// `fallback_device`. As the Judge's op coverage grows the
// recommender's coverage grows with it for free.

/// Convert a `fuel_graph::Op` to the Judge's `OpKind`. Returns `None`
/// for ops the Judge doesn't profile yet.
fn op_to_kind(op: &Op) -> Option<OpKind> {
    match op {
        Op::MatMul => Some(OpKind::MatMul),
        Op::Add    => Some(OpKind::AddElementwise),
        _          => None,
    }
}

/// Convert a dispatch-table [`Pick`] into the Fuel placement enum.
fn pick_to_location(pick: Pick) -> Option<DeviceLocation> {
    match pick.backend {
        BackendId::Cpu       => Some(DeviceLocation::Cpu),
        BackendId::Reference => Some(DeviceLocation::Cpu),
        BackendId::Cuda      => Some(DeviceLocation::Cuda   { gpu_id: pick.device_index as usize }),
        BackendId::Vulkan    => Some(DeviceLocation::Vulkan { gpu_id: pick.device_index as usize }),
        BackendId::Metal     => Some(DeviceLocation::Metal  { gpu_id: pick.device_index as usize }),
        // CPU-vendor variants (Mkl, Aocl) collapse to plain Cpu —
        // they're picked by the per-CPU-backend selection layer, not
        // by the device-placement layer.
        BackendId::Mkl       => Some(DeviceLocation::Cpu),
        BackendId::Aocl      => Some(DeviceLocation::Cpu),
        // BackendId is #[non_exhaustive]; future variants get the
        // CPU fallback until they have a placement rule of their own.
        _ => Some(DeviceLocation::Cpu),
    }
}

/// Walk every node in `graph` and produce a recommended
/// [`DeviceLocation`] per node. Ops the Judge profiled produce a
/// dispatch-table-driven recommendation; ops it didn't profile fall
/// back to `fallback_device` (typically the router's default).
///
/// The output is a `HashMap<NodeId, DeviceLocation>` ready to feed
/// into `fuel_graph::opt::insert_moves` or any other placement-
/// honouring pass.
///
/// # When to use
///
/// The recommender is intended for **realize-time** placement
/// decisions, after the orchestrator has produced a [`DispatchTable`].
/// It does not mutate the graph; it just produces a plan. Callers
/// can override individual nodes (pin a specific tensor to CPU for
/// debugging, force a backbone onto a GPU regardless of size, etc.)
/// before passing the plan to `insert_moves`.
///
/// # Tolerance for unprofiled ops
///
/// Most ops in any real graph won't be profiled in v1 (Judge today
/// covers MatMul + AddElementwise). Those nodes get
/// `fallback_device`, which means the dispatch table is acting as a
/// "preferred placement for hot ops" overlay rather than a hard
/// scheduling decision. As the Judge's op coverage grows the overlay
/// grows.
pub fn recommend_placement(
    graph: &Graph,
    table: &DispatchTable,
    criterion: Criterion,
    fallback_device: DeviceLocation,
) -> HashMap<NodeId, DeviceLocation> {
    let mut plan = HashMap::with_capacity(graph.len());
    for i in 0..graph.len() {
        let id = NodeId(i);
        let node = graph.node(id);
        let device = recommend_for_node(node, table, criterion).unwrap_or(fallback_device);
        plan.insert(id, device);
    }
    plan
}

/// Single-node recommendation. Public so callers that already have
/// a node in hand (e.g. during a custom graph rewrite pass) can ask
/// the dispatch table directly without building the full plan.
pub fn recommend_for_node(
    node: &fuel_graph::Node,
    table: &DispatchTable,
    criterion: Criterion,
) -> Option<DeviceLocation> {
    let kind = op_to_kind(&node.op)?;
    if node.dtype != DType::F32 {
        // Judge only covers F32 today. Other dtypes get fallback.
        return None;
    }
    let size_class = SizeClass::from_elem_count(node.shape.elem_count());
    let pick = table.pick_nearest(kind, node.dtype, size_class, criterion)?;
    pick_to_location(pick)
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

    use crate::dispatch::Criterion;
    use crate::judge::{ProfileEntry, ProfileReport, PROFILE_REPORT_VERSION};
    use fuel_graph::Tensor;
    use fuel_core_types::Shape;
    use std::sync::Arc;

    /// Hand-build a tiny dispatch table where CUDA wins MatMul at
    /// large sizes and CPU wins at small. Then build a graph with
    /// matmuls of varying sizes plus an unprofiled op (Mul) and
    /// verify the recommender sends each node where the table
    /// predicts.
    #[test]
    fn recommend_placement_routes_per_node() {
        // -- 1) hand-craft a profile report with deliberate winners --
        let mk = |backend: BackendId, size: u8, latency: u64| ProfileEntry {
            op: OpKind::MatMul,
            dtype: DType::F32,
            size_class: SizeClass(size),
            backend,
            device_index: 0,
            latency_ns: latency,
            iterations: 1,
            max_rel_error: 0.0,
        };
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![
                // size 12 (≈ 64×64 matmul): CPU wins (10μs vs 2ms CUDA launch overhead)
                mk(BackendId::Cpu,  12,    10_000),
                mk(BackendId::Cuda, 12, 2_000_000),
                // size 20 (≈ 1024×1024 matmul): CUDA wins (5ms vs 50ms CPU)
                mk(BackendId::Cpu,  20, 50_000_000),
                mk(BackendId::Cuda, 20,  5_000_000),
            ],
        };
        let table = DispatchTable::build(&report);

        // -- 2) build a graph with two matmuls + one unprofiled op
        //       — all derived from a single root so they share a graph.
        let small_a = Tensor::from_f32(vec![0.0_f32; 64 * 64], Shape::from_dims(&[64, 64]));
        let small_b = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 64 * 64]),
            Shape::from_dims(&[64, 64]),
        );
        let small_mm = small_a.matmul(&small_b);  // size_class = 12 (4096 elements)

        // Build the big matmul as constants on the SAME graph as small_a.
        let big_a = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1024 * 1024]),
            Shape::from_dims(&[1024, 1024]),
        );
        let big_b = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1024 * 1024]),
            Shape::from_dims(&[1024, 1024]),
        );
        let big_mm = big_a.matmul(&big_b);  // size_class = 20

        // Unprofiled op (Sub) on small tensors — should fall back.
        let unprofiled = small_mm.sub(&small_b);

        // The graph for the recommender is whichever one any of our
        // tensors point into; they all share the same graph since
        // they were built from the same `from_f32` root.
        let graph = small_a.graph().borrow();

        let plan = recommend_placement(
            &graph,
            &table,
            Criterion::Fastest,
            DeviceLocation::Cpu,
        );

        // small matmul → CPU (size_class 12 winner)
        assert_eq!(plan[&small_mm.id()], DeviceLocation::Cpu);
        // big matmul → CUDA (size_class 20 winner)
        assert_eq!(plan[&big_mm.id()], DeviceLocation::Cuda { gpu_id: 0 });
        // Unprofiled sub-op → fallback (CPU)
        assert_eq!(plan[&unprofiled.id()], DeviceLocation::Cpu);
        // Const nodes → fallback (no Op::Const → OpKind mapping)
        assert!(matches!(plan[&small_a.id()], DeviceLocation::Cpu));
    }

}
