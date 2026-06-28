//! Bridge from fuel-core's user-facing `Tensor::realize_*` API to
//! fuel-storage's `PipelinedExecutor` (Phase 7.6 step 9c, sub-phases
//! E.1 + E.2).
//!
//! Pre-Phase-E, `Tensor::realize_f32` etc. constructed a
//! `fuel-graph-executor::GraphExecutor<B>` and called its typed
//! `realize_f32(&tensor)` method. The legacy executor's
//! `try_adopt_slot` walked the graph's storage map, did D2H, then
//! `B::upload(&buf, shape)` to put the data on the backend.
//!
//! Post-Phase-E, the user-facing API:
//! 1. Walks the graph from the requested targets and **pre-realizes
//!    every reachable `Op::Const`** into a `StorageCache` on the
//!    chosen target device. This is the legacy `try_adopt_slot`
//!    work, now external to the executor.
//! 2. Sets `target_backend` on every reachable computational node
//!    (the legacy executor implicitly used `self.backend`; the
//!    pipelined path reads it from the graph side-table).
//! 3. For non-CPU realize devices, splices an
//!    `Op::Copy { target: Cpu }` at each realize root so D2H runs
//!    as a graph node the optimizer can see (bridge-retirement
//!    Phase 2, post-9c). The Op::Copy node's kernel is registered
//!    at `(OpKind::Copy, [dt, dt], source_backend)`; the executor's
//!    `WorkItemKind::Copy` arm allocates the output on the target
//!    location and runs the source-backend's download wrapper.
//! 4. Calls [`PipelinedExecutor::realize_many`] for multi-target or
//!    `PipelinedExecutor::realize` for single-target on the spliced
//!    targets — the executor returns a `BackendStorage::Cpu` for
//!    each.
//! 5. Reads the CPU bytes into a typed `Vec<T>` via `bytemuck`.
//!
//! This module owns steps 1–5 so [`crate::lazy::LazyTensor`]'s
//! `realize_*` methods stay one-liners.
//!
//! ## Status post-Phase 3
//!
//! Bridge-retirement Phases 2 + 3a + 3b complete:
//! * **Phase 2** (D2H): `realize_*_as` splices `Op::Copy { target: Cpu }`
//!   at every realize root; the executor's `WorkItemKind::Copy` arm
//!   downloads bytes via the binding-table-registered source-backend
//!   wrapper. `BackendStorage::read_to_cpu_bytes` deleted.
//! * **Phase 3a** (zero-alloc): `KvCache::with_capacity` emits
//!   `Op::Alloc → Op::ZeroFill` pairs and realizes via
//!   `PipelinedExecutor::realize_many`. `alloc_zeroed_on` deleted.
//! * **Phase 3b** (H2D Const upload): [`build_const_cache`] (for
//!   non-CPU targets) builds a transient graph of `Op::Const →
//!   Op::Copy { target: device }` pairs and realizes them
//!   multi-target. The executor's `WorkItemKind::Copy` arm allocates
//!   the device-side output (uninit) and the `copy_from_cpu_wrapper`
//!   writes host bytes via per-backend H2D helpers
//!   (`CudaStorageBytes::write_from_host`,
//!   `VulkanBackend::write_bytes`). `upload_host_buffer` deleted.
//!
//! Residual bridge code: [`device_seed_storage`] (~30 LOC, just the
//! 0-byte device-handle anchor per backend) and
//! [`host_buffer_to_bytes`] (per-dtype HostBuffer → bytes
//! conversion — orthogonal to the device-dispatch concern).
//!
//! ## Phase E.3 coverage (complete)
//!
//! Autoregressive decoding (the former `KVCache<B>` /
//! `forward_with_cache_on<B>` / `generate_*` / speculative-decode
//! surfaces) runs through [`crate::inference_context`]: a long-lived
//! `InferenceContext` seeds each realize call's `StorageCache` and
//! `KvCache::with_capacity` + `Op::WriteSlice` keep K/V device-
//! resident across steps. The legacy generic-over-`B` family retired
//! in Unification Session 4 (E.3.4).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use fuel_ir::{
    probe::BackendId, DeviceLocation, Error, HostBuffer, Layout, Result, SymEnv,
};
use fuel_backend_contract::backend::BackendRuntime;
use fuel_ir::backend::FitStatus;
use fuel_backend_contract::dyn_backend::DynBackendDevice;
use fuel_graph::{Graph, Node, NodeId, Op, PickedRoute, topo_order_multi};
use fuel_dispatch::dispatch::global_bindings;
use fuel_dispatch::optimize::{optimize_graph, OptimizedGraph};
use fuel_dispatch::plan::{ExecutionPlan, PlanOptions};
use fuel_dispatch::pipelined::{PipelinedExecutor, StorageCache};
use fuel_dispatch::ranker::{
    pick_route, BackendRuntimeHandle, BackendRuntimeLookup, ChainedSelector, JudgeOracle,
    RuntimeSelector,
};
use fuel_memory::{BackendStorage, Storage};

use crate::Device;
use crate::topology::SystemTopology;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Realize a single tensor by NodeId on the given device, returning
/// the result's host bytes as a typed `Vec<T>` via `bytemuck`.
///
/// Steps:
/// 1. `prepare` — splice the realize-root `Op::Copy { target: Cpu }`
///    (bridge-retirement Phase 2) and build the const cache on
///    `device`. Per-node backends are NOT pinned here (picker-arc
///    step 4a) — the device is the only pin.
/// 2. `build_optimized_graph` — `optimize_graph` transforms the graph
///    in place ("plan IS the graph") against the pinned device: it stamps
///    each winner's backend onto the graph (cleanup A1) AND runs the
///    residency (cross-device `Op::Copy`) + layout-fixup (`Op::Contiguize`)
///    passes (cleanup Step B), then surfaces the `ExecutionPlan`. The bridge
///    no longer makes any of those decisions — the graph arrives fully
///    stamped, copy-stitched, and fixed up.
/// 3. `PipelinedExecutor::realize_with_optimized` — kick the compile +
///    execute pipeline over the run/`lower_run` dispatch order; returns
///    a `BackendStorage::Cpu` for the spliced root.
/// 4. `bytemuck::cast_slice` — reinterpret the CPU bytes as `T`.
pub fn realize_one_as<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    target: NodeId,
    device: &Device,
) -> Result<Vec<T>> {
    realize_one_as_with_initial::<T>(graph, target, device, StorageCache::new())
}

/// Multi-target counterpart of [`realize_one_as`]. Returns parallel
/// `Vec<Vec<T>>` in the order of `targets`.
pub fn realize_many_as<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    device: &Device,
) -> Result<Vec<Vec<T>>> {
    realize_many_as_with_initial::<T>(graph, targets, device, StorageCache::new())
}

/// Realize-one variant that seeds the executor's input cache with
/// `initial` before adding Op::Const slot uploads. Used by
/// [`crate::inference_context::InferenceContext`] to thread its
/// persistent storage Arcs through each realize call without
/// re-uploading them. NodeIds already present in `initial` are
/// not re-fetched from the graph's storage_map; their Arcs survive
/// the call.
pub fn realize_one_as_with_initial<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    target: NodeId,
    device: &Device,
    initial: StorageCache,
) -> Result<Vec<T>> {
    realize_one_as_with_initial_reporting::<T>(graph, target, device, initial, &SymEnv::default())
        .map(|(bytes, _root_kernel_source)| bytes)
}

/// Env-carrying sibling of [`realize_one_as_with_initial`]: threads a
/// per-pass [`SymEnv`] supplying the runtime bindings for `DynScalar`
/// op params (Phase D symbolic extents — e.g. the persistent-decode
/// KV-cache write offset). An **empty** env is byte-identical to
/// [`realize_one_as_with_initial`]. Used by
/// [`crate::inference_context::InferenceContext`] to carry the
/// session's per-token `cached_len` binding into realize.
pub fn realize_one_as_with_initial_env<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    target: NodeId,
    device: &Device,
    initial: StorageCache,
    sym_env: &SymEnv,
) -> Result<Vec<T>> {
    realize_one_as_with_initial_reporting::<T>(graph, target, device, initial, sym_env)
        .map(|(bytes, _root_kernel_source)| bytes)
}

/// [`realize_one_as_with_initial`] sibling that additionally reports
/// the `kernel_source` of the alternative the picker dispatched for
/// the realize ROOT — `target`, the caller's node, not the spliced
/// D2H `Op::Copy` that `prepare()` adds on top of it.
///
/// Executor-unification Session 3 rider: the Judge's bridge realizer
/// (`crate::factories::BridgeRealizer`) consumes this so the
/// realizer-measured `CellRun` records the TRUE dispatched sibling at
/// multi-alternative cells (portable/AOCL/MKL at one CPU key) instead
/// of assuming the first-registered one.
///
/// `None` means the plan carried no `AlternativeSet` for the root —
/// the executor then dispatched the first-registered binding via its
/// `compile_node` fallback, so callers wanting an attribution tag in
/// that case should fall back to the first-registered convention.
pub fn realize_one_as_with_initial_reporting<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    target: NodeId,
    device: &Device,
    initial: StorageCache,
    sym_env: &SymEnv,
) -> Result<(Vec<T>, Option<&'static str>)> {
    let (cache, _backend_id, mut effective_targets) =
        prepare(graph, &[target], device, initial)?;
    let Some(cpu_target) = effective_targets.pop() else {
        return Err(Error::Msg(
            "pipelined_bridge: prepare returned no effective target for a \
             single-target realize — internal bug"
                .into(),
        )
        .bt());
    };
    // Optimize the graph in place (the "plan IS the graph" form) and
    // drive the executor from its run/`lower_run` dispatch order.
    //
    // Retry on `TopologyChanged`. If the live SystemTopology generation
    // advances between optimize and dispatch (a backend
    // registered/unregistered mid-flight), the executor surfaces
    // `Error::TopologyChanged`; we re-optimize against the fresh
    // topology and try again. Retries back off exponentially
    // (`topology_retry_backoff`) so a burst of generation bumps —
    // device hotplug storms, or the test suite's concurrent churn —
    // settles instead of exhausting instant rebuilds; capped at
    // `MAX_PLAN_REBUILDS` to prevent infinite spin under genuinely
    // persistent churn.
    let (storage, root_kernel_source) = dispatch_with_plan_retry(
        graph, cpu_target, cache, device, target, sym_env,
    )?;
    Ok((extract_cpu_bytes_typed::<T>(&storage)?, root_kernel_source))
}

/// Retry-on-stale-plan loop for the single-target path. Pulled out so
/// the multi-target path can reuse the same retry shape.
///
/// Each attempt runs the full optimize → stamp → copy-insert → fixup
/// sequence. `optimize_graph` (via [`build_optimized_graph`]) transforms
/// the graph in place against the pinned DEVICE and surfaces the
/// `ExecutionPlan` it computed; the per-node winners are then stamped
/// onto the graph (`stamp_plan_backends`), the cross-device-copy pass
/// stitches residency against those final placements, and the
/// layout-fixup pass runs last. Re-optimizing after a `TopologyChanged`
/// retry re-runs the whole sequence so stamps stay consistent with the
/// fresh placement.
///
/// Dispatch goes through [`PipelinedExecutor::realize_with_optimized`]
/// — the executor recomputes its run/`lower_run` dispatch order from the
/// (post-stamping) graph and resolves each node's kernel via the
/// binding-table lookup; the surfaced plan is reused only for the
/// stamp/residency/layout passes and the root-attribution report.
/// Settle budget for `TopologyChanged` plan rebuilds, measured from
/// the **last observed generation movement**, not from retry start.
/// A generation bump means registration state is actively
/// reconfiguring; as long as the counter keeps moving, rebuilding
/// and waiting is the correct behavior, so observed movement resets
/// the deadline. The budget therefore bounds the *quiescent-but-
/// still-failing* case: once the topology stops changing, a realize
/// that still can't dispatch within this window fails loudly with
/// the executor's typed error. (History: an attempt-capped ~63ms
/// budget flaked 50% of suite runs — the churn test's 50µs bumper
/// sleeps stretch to ~15ms each under Windows timer resolution,
/// and a fixed-from-start 2s cap could still expire while attempts
/// were burning real plan-build/dispatch time inside the storm.)
const TOPOLOGY_SETTLE_BUDGET: std::time::Duration =
    std::time::Duration::from_secs(2);

/// Belt-and-braces ceiling on rebuild iterations, bounding the
/// perpetual-churn case (a device flapping indefinitely keeps
/// resetting the settle deadline; the iteration ceiling is what
/// finally stops us). With the 16ms backoff cap this is ≥4s of
/// pure waiting plus per-attempt plan/dispatch time — far beyond
/// any legitimate reconfiguration.
const MAX_PLAN_REBUILDS: usize = 256;

/// Capped exponential settle-wait between `TopologyChanged` plan
/// rebuilds: 1, 2, 4, 8, then 16ms per attempt thereafter. A brief
/// sleep is the correct primitive — rebuilding in a tight loop just
/// re-reads the same moving registration state. Not a perf-path
/// concern (this only runs when a rebuild is already required).
fn topology_retry_backoff(attempt: usize) {
    let ms = 1u64 << attempt.min(4);
    std::thread::sleep(std::time::Duration::from_millis(ms));
}

/// Retry bookkeeping for the `TopologyChanged` loops: tracks the
/// last topology generation observed and the time it was last seen
/// to MOVE, so the settle budget measures quiescence rather than
/// total elapsed time.
struct TopologyRetryState {
    attempt: usize,
    last_gen: u64,
    last_movement: std::time::Instant,
}

impl TopologyRetryState {
    fn new() -> Self {
        Self {
            attempt: 0,
            last_gen: fuel_dispatch::dispatch::topology_generation(),
            last_movement: std::time::Instant::now(),
        }
    }

    /// Called on a `TopologyChanged` error. Returns true when the
    /// retry should continue (after sleeping the backoff), false
    /// when the budget is exhausted and the error should escape.
    fn permit_retry(&mut self) -> bool {
        let now_gen = fuel_dispatch::dispatch::topology_generation();
        if now_gen != self.last_gen {
            // The topology is still moving — reset the settle clock.
            self.last_gen = now_gen;
            self.last_movement = std::time::Instant::now();
        }
        if self.attempt + 1 >= MAX_PLAN_REBUILDS
            || self.last_movement.elapsed() >= TOPOLOGY_SETTLE_BUDGET
        {
            return false;
        }
        topology_retry_backoff(self.attempt);
        self.attempt += 1;
        true
    }
}

/// Build the `OptimizedGraph` lowering view for the realize path and
/// surface the transient `ExecutionPlan` it computed.
///
/// `optimize_graph` transforms the graph **in place** into the "plan IS
/// the graph" form (zero `Op::Branch` nodes today — every graph is
/// branchless until A4) and returns the transient [`OptimizedGraph`]
/// whose `dispatch_order` (the runs' `lower_run` sequence) the executor
/// walks, plus the [`ExecutionPlan`] it drove `compile_plan` to produce
/// for placement/cost/validation. PR-A3b-2 reuses that single plan for
/// the bridge's `stamp_plan_backends` / residency / layout passes
/// instead of re-running `compile_plan` — one `compile_plan` per
/// realize. Build-time diagnostics (missing binding, no device context)
/// fire here exactly as they did for the legacy `compile_plan` path.
fn build_optimized_graph(
    graph: &Arc<RwLock<Graph>>,
    roots: &[NodeId],
    pinned_device: DeviceLocation,
    cache: &StorageCache,
) -> Result<(OptimizedGraph, ExecutionPlan)> {
    let topology = SystemTopology::current();
    let placements_for = |dev: DeviceLocation| -> Vec<BackendId> {
        topology.backends_for(dev).to_vec()
    };
    let capabilities_for = |b: BackendId|
        -> Option<&fuel_ir::backend::BackendCapabilities>
    { topology.capabilities(b) };
    let fallback_for = |dev: DeviceLocation|
        -> Vec<(BackendId, DeviceLocation)>
    {
        let mut out = Vec::new();
        for &d in topology.devices() {
            if d == dev {
                continue;
            }
            for &b in topology.backends_for(d) {
                out.push((b, d));
            }
        }
        out
    };
    let judge_oracle = crate::judge::cached_oracle();
    let input_residency = |id: NodeId| -> Option<DeviceLocation> {
        let slot = cache.get(&id)?;
        let guard = slot.read().ok()?;
        cached_storage_location(&guard)
    };

    let mut options = PlanOptions::new()
        .with_placements_for_device(&placements_for)
        .with_capabilities_for(&capabilities_for)
        .with_pinned_device(pinned_device)
        .with_fallback_placements_for(&fallback_for)
        .with_transfer_estimator(&*topology)
        .with_input_residency(&input_residency);
    if let Some(oracle) = judge_oracle.as_deref() {
        options = options.with_judge(oracle);
    }

    let bindings_guard = global_bindings();
    let mut g = graph
        .write()
        .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
    optimize_graph(&mut g, roots, &bindings_guard, &options)
}

fn dispatch_with_plan_retry(
    graph: &Arc<RwLock<Graph>>,
    cpu_target: NodeId,
    cache: StorageCache,
    device: &Device,
    report_node: NodeId,
    sym_env: &SymEnv,
) -> Result<(Arc<RwLock<Storage>>, Option<&'static str>)> {
    let pinned_loc = device.location();
    let mut retry = TopologyRetryState::new();
    // Hold a clone of `cache` outside the loop; the inner clone per-
    // attempt shares the Arcs (cheap) while keeping a fresh
    // structurally-empty layer for the executor to consume.
    loop {
        // Optimize the graph in place — `optimize_graph` runs the ONE
        // `compile_plan` per realize, **stamps each node's winning backend
        // onto the graph** (cleanup A1: the executor needs `target_backend`
        // on every kernel node, and that's an optimizer concern), and
        // surfaces its `ExecutionPlan` (PR-A3b-2 de-dup) for the residency
        // + layout passes below. Build-time validation (missing binding /
        // no device) fires inside `optimize_graph`.
        let (optimized, plan) =
            build_optimized_graph(graph, &[cpu_target], pinned_loc, &cache)?;
        // Residency (cross-device `Op::Copy`) and layout-fixup
        // (`Op::Contiguize`) are now optimizer passes inside `optimize_graph`
        // (cleanup Step B) — driven by the graph stamps + the `input_residency`
        // provider threaded through `build_optimized_graph`. The graph arrives
        // here already copy-stitched and fixed up; the bridge no longer runs
        // either pass.
        // PR-C1: resolve the runtime route — Picker 2 chooses one arm per
        // `Op::Branch` by live telemetry (VRAM-pressure guard + Judge
        // rank over the production `ChainedSelector`). A branchless graph
        // (CPU-only build, or no genuine fork) yields an empty route, so
        // the executor falls back to the arm-0 lowering — realize
        // unchanged from Phase B.
        let route = resolve_runtime_route(graph, &[cpu_target], &plan, device)?;
        let cache_for_attempt = cache.clone();
        // Dispatch the "plan IS the graph" form: the executor recomputes
        // its run/`lower_run` dispatch order from the (now fully-stamped)
        // graph, following the picker's chosen arms (or arm-0 when there
        // is no route), and resolves each node's kernel via the
        // binding-table lookup.
        let result = match &route {
            Some(route) => PipelinedExecutor::realize_with_optimized_route_env(
                graph.clone(), cpu_target, cache_for_attempt, &optimized, route,
                sym_env.clone(),
            ),
            None => PipelinedExecutor::realize_with_optimized_env(
                graph.clone(), cpu_target, cache_for_attempt, &optimized,
                sym_env.clone(),
            ),
        };
        match result {
            Ok((storage, _layout)) => {
                // Session 3 rider: report which sibling dispatched for
                // `report_node`, from the SAME plan the successful
                // attempt ran with. The executor dispatches via the
                // first-registered binding-table lookup, so the static
                // `set.winner()` here is the matching attribution.
                let dispatched = dispatched_kernel_source(&plan, report_node);
                return Ok((storage, dispatched));
            }
            Err(e) if matches!(e, Error::TopologyChanged { .. })
                && retry.permit_retry() =>
            {
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Replicate the executor's `resolve_compiled` pick for one node
/// against the plan that just dispatched: the static
/// `AlternativeSet::winner`. `None` when the plan has no
/// `AlternativeSet` for the node — the executor then dispatched the
/// first-registered binding via its `compile_node` fallback.
///
/// The optimized realize path dispatches via the binding-table lookup
/// (no runtime route-picker), so the static winner is the matching
/// attribution.
fn dispatched_kernel_source(
    plan: &ExecutionPlan,
    node: NodeId,
) -> Option<&'static str> {
    let set = plan.alternatives(node)?;
    let pick = set.winner()?;
    Some(pick.kernel_source)
}

/// Multi-target counterpart of [`realize_one_as_with_initial`].
pub fn realize_many_as_with_initial<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    device: &Device,
    initial: StorageCache,
) -> Result<Vec<Vec<T>>> {
    realize_many_as_with_initial_env::<T>(graph, targets, device, initial, &SymEnv::default())
}

/// Env-carrying sibling of [`realize_many_as_with_initial`]: threads a
/// per-pass [`SymEnv`] for `DynScalar` op params (Phase D symbolic
/// extents). An **empty** env is byte-identical to
/// [`realize_many_as_with_initial`].
pub fn realize_many_as_with_initial_env<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    device: &Device,
    initial: StorageCache,
    sym_env: &SymEnv,
) -> Result<Vec<Vec<T>>> {
    if targets.is_empty() {
        return Ok(Vec::new());
    }
    let (cache, _, effective_targets) = prepare(graph, targets, device, initial)?;
    let results = dispatch_many_with_plan_retry(
        graph, &effective_targets, cache, device, sym_env,
    )?;
    let mut out = Vec::with_capacity(results.len());
    for (storage, _layout) in results {
        out.push(extract_cpu_bytes_typed::<T>(&storage)?);
    }
    Ok(out)
}

/// Realize-split: realize `targets` in ONE executor pass, download
/// only the first `n_host` results to host bytes, and return the
/// remaining results as device-resident storage Arcs + layouts.
///
/// The pipelined replacement for the legacy
/// `GraphExecutor::realize_split` (executor-unification Session 5):
/// the Trainer realizes `[loss, new_params…, new_opt_state…]` per
/// step with `n_host = 1` — the loss scalar comes back as host
/// `Vec<T>` while updated parameters and optimizer moments stay
/// where the picker placed them, ready to seed the next step's
/// `StorageCache` without a D2H/H2D round-trip.
///
/// Resident results are `(storage, layout)` pairs in `targets` order
/// (offset by `n_host`); the storage Arc's `BackendStorage` variant
/// carries the device identity.
pub fn realize_split_as_with_initial<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    n_host: usize,
    device: &Device,
    initial: StorageCache,
) -> Result<(Vec<Vec<T>>, Vec<(Arc<RwLock<Storage>>, Layout)>)> {
    if n_host > targets.len() {
        return Err(Error::Msg(format!(
            "realize_split_as_with_initial: n_host ({n_host}) exceeds \
             target count ({})",
            targets.len(),
        ))
        .bt());
    }
    if targets.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let (cache, _, effective_targets) =
        prepare_split(graph, targets, n_host, device, initial)?;
    let results = dispatch_many_with_plan_retry(
        graph, &effective_targets, cache, device, &SymEnv::default(),
    )?;
    let mut host_out = Vec::with_capacity(n_host);
    let mut resident_out = Vec::with_capacity(results.len().saturating_sub(n_host));
    for (i, (storage, layout)) in results.into_iter().enumerate() {
        if i < n_host {
            host_out.push(extract_cpu_bytes_typed::<T>(&storage)?);
        } else {
            resident_out.push((storage, layout));
        }
    }
    Ok((host_out, resident_out))
}

/// Multi-target dispatch with topology-stale retry, post-optimize
/// winner stamping, cross-device-copy stitching, and layout-fixup
/// insertion. See [`dispatch_with_plan_retry`] for the single-target
/// version of the loop; this multi-target sibling shares the same shape
/// and serves both [`realize_many_as_with_initial`] and
/// [`realize_split_as_with_initial`].
fn dispatch_many_with_plan_retry(
    graph: &Arc<RwLock<Graph>>,
    effective_targets: &[NodeId],
    cache: StorageCache,
    device: &Device,
    sym_env: &SymEnv,
) -> Result<Vec<(Arc<RwLock<Storage>>, Layout)>> {
    let pinned_loc = device.location();
    let mut retry = TopologyRetryState::new();
    loop {
        // See `dispatch_with_plan_retry`: one `optimize_graph` runs the
        // single `compile_plan`, stamps the winning backends onto the graph
        // (cleanup A1), AND runs the residency + layout-fixup passes (cleanup
        // Step B); the executor recomputes its run/`lower_run` order from the
        // stamped, copy-stitched graph.
        let (optimized, plan) =
            build_optimized_graph(graph, effective_targets, pinned_loc, &cache)?;
        // PR-C1: resolve the runtime route (Picker 2) over the branches;
        // empty for a branchless graph ⇒ arm-0 lowering ⇒ Phase B.
        let route =
            resolve_runtime_route(graph, effective_targets, &plan, device)?;
        let cache_for_attempt = cache.clone();
        let result = match &route {
            Some(route) => PipelinedExecutor::realize_many_with_optimized_route_env(
                graph.clone(), effective_targets, cache_for_attempt, &optimized,
                route, sym_env.clone(),
            ),
            None => PipelinedExecutor::realize_many_with_optimized_env(
                graph.clone(), effective_targets, cache_for_attempt, &optimized,
                sym_env.clone(),
            ),
        };
        match result {
            Ok(r) => break Ok(r),
            Err(e) if matches!(e, Error::TopologyChanged { .. })
                && retry.permit_retry() =>
            {
                continue;
            }
            Err(e) => break Err(e),
        }
    }
}

/// Read a realize result's CPU bytes and reinterpret them as `Vec<T>`.
///
/// Post bridge-retirement Phase 2: the executor produced this Storage
/// through the spliced `Op::Copy { target: Cpu }` node (for non-CPU
/// devices) or directly on CPU (for CPU realizes). Either way, this
/// is a `BackendStorage::Cpu` — extract its bytes via the
/// CPU-variant pattern.
fn extract_cpu_bytes_typed<T: bytemuck::Pod>(
    storage: &Arc<RwLock<Storage>>,
) -> Result<Vec<T>> {
    let guard = storage
        .read()
        .map_err(|_| Error::Msg("storage lock poisoned".into()).bt())?;
    let bytes: &[u8] = match &guard.inner {
        BackendStorage::Cpu(s) => s.bytes(),
        // The other arms are feature-gated; on default-features-only
        // builds CPU is the sole variant and this arm is unreachable
        // — but suppress the lint so it still parses with `--features
        // cuda` / `--features vulkan`.
        #[allow(unreachable_patterns)]
        other => {
            return Err(Error::Msg(format!(
                "pipelined_bridge: realize root produced non-CPU storage \
                 ({other:?}) — the Op::Copy splice in `prepare()` should \
                 have made the root CPU-resident. This is a bug.",
            ))
            .bt());
        }
    };
    Ok(bytemuck::cast_slice::<u8, T>(bytes).to_vec())
}

// ---------------------------------------------------------------------------
// Prep — internal
// ---------------------------------------------------------------------------

/// One-shot prep: derive a `BackendId` from `device`, build a
/// `StorageCache` containing every reachable `Op::Const`, and
/// (post-9c Phase 2 of bridge-retirement) splice an
/// `Op::Copy { target: Cpu }` at each realize root so the executor
/// produces a CPU storage at the returned `effective_targets`.
///
/// Picker-arc step 4a: per-node `target_backend` pinning moved OUT
/// of this function — `optimize_graph` stamps each node's plan winner
/// onto the graph (cleanup A1; the optimizer owns placement).
///
/// Returns `(cache, backend_id, effective_targets)`:
/// - `effective_targets[i]` mirrors `targets[i]`'s order. For CPU
///   realizes it equals `targets[i]`; for GPU realizes it is the
///   NodeId of the spliced Op::Copy node, whose output the executor
///   produces as a fresh `BackendStorage::Cpu`.
///
/// Mutates the graph (takes a write lock); the executor takes its own
/// read lock after this returns.
fn prepare(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    device: &Device,
    initial: StorageCache,
) -> Result<(StorageCache, BackendId, Vec<NodeId>)> {
    prepare_split(graph, targets, targets.len(), device, initial)
}

/// [`prepare`] sibling that splices the D2H `Op::Copy { target: Cpu }`
/// only on the FIRST `n_host` targets. The remaining targets are
/// realized as themselves and their results stay wherever the picker
/// placed them — the realize-split capability (executor-unification
/// Session 5): the legacy `GraphExecutor::realize_split` kept new
/// parameter storage on-device while downloading only the loss
/// scalar, and the Trainer port needs the same shape on the
/// pipelined path.
///
/// `effective_targets[i]` is the spliced Copy NodeId for `i < n_host`
/// and `targets[i]` itself otherwise, preserving caller order.
fn prepare_split(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    n_host: usize,
    device: &Device,
    initial: StorageCache,
) -> Result<(StorageCache, BackendId, Vec<NodeId>)> {
    let backend_id = device_to_backend_id(device);

    // Phase 2 of bridge-retirement: splice an `Op::Copy { target:
    // Cpu }` at every realize root, regardless of source backend, so
    // D2H runs as a graph node the optimizer can see (architecture
    // identity check #1).
    //
    // Why always — even for CPU realizes:
    //   1. Strided / sliced / permuted realize roots are common; the
    //      executor's WorkItemKind::Copy arm runs `auto_contiguize`
    //      on the input before the kernel, so the output is the
    //      LOGICAL view's bytes, not the parent storage's full bytes.
    //      Without the splice on CPU, a `realize_f32` of a slice view
    //      returned the parent's full bytes (a long-standing bug
    //      inherited from the pre-9c `read_to_cpu_bytes`); routing
    //      through Op::Copy fixes it uniformly.
    //   2. The CPU→CPU Copy kernel is one memcpy that replaces the
    //      `.to_vec()` `read_to_cpu_bytes` used to do; no extra cost
    //      in the contiguous case.
    //   3. One code path through Op::Copy keeps the executor's
    //      semantics consistent across devices.
    //
    // The spliced node's shape + dtype match the source; the
    // executor's WorkItemKind::Copy arm allocates a fresh CPU storage
    // and runs the source-backend's registered Copy kernel.
    let effective_targets = {
        let mut g = graph
            .write()
            .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
        targets
            .iter()
            .enumerate()
            .map(|(i, &src_id)| {
                if i >= n_host {
                    // Device-resident target (realize-split tail):
                    // no D2H splice — the caller receives the
                    // storage Arc wherever the picker placed it.
                    return src_id;
                }
                let (shape, dtype) = {
                    let n = g.node(src_id);
                    (n.shape.clone(), n.dtype)
                };
                g.push(Node {
                    op: Op::Copy { target: DeviceLocation::Cpu },
                    inputs: vec![src_id],
                    shape,
                    dtype,
                })
            })
            .collect::<Vec<_>>()
    };

    let order = {
        let g = graph
            .read()
            .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
        topo_order_multi(&g, &effective_targets)
    };

    // Build the StorageCache on top of `initial` (which may carry
    // persistent storages from an InferenceContext). build_const_cache
    // adds any reachable Op::Const NodeId not already present.
    let cache = build_const_cache(graph, &order, device, initial)?;

    // Picker-arc step 4a: prepare() pins the realize DEVICE only.
    // Per-node `target_backend` is no longer stamped here — the
    // dispatch loops build an ExecutionPlan against the pinned
    // device (`PlanOptions::with_pinned_device`) and `optimize_graph`
    // commits each node's winner backend AFTER planning, then runs the
    // residency + layout passes over those final placements (cleanup
    // Step B), before the executor derives its ordering.
    //
    // The pre-4a monolithic loop's "always overwrite" doctrine
    // survives in `stamp_plan_backends`: graphs are shared
    // (`Arc<RwLock<Graph>>`) and a single graph may be realized on
    // multiple devices across a session, so every realize call
    // re-stamps from its own plan rather than trusting stale pins.
    Ok((cache, backend_id, effective_targets))
}

/// **Test-only helper** (cleanup A2). Production backend-stamping now
/// lives in `fuel_dispatch::optimize::optimize_graph` (the optimizer
/// writes its placement decision into the graph — "plan IS the graph");
/// the realize bridge no longer stamps. This replica is retained as a
/// unit-test fixture for the residency / layout / re-stamp tests below,
/// which build a plan by hand and need the graph stamped in isolation
/// without running the full optimizer.
///
/// Commits the plan's per-node winner to the graph's `target_backend`
/// side-table. Per computational node (skip `Op::Const` / `Op::Release` /
/// `Op::Contiguize` / view ops / `Op::Reshape`): stamp `winner.backend`
/// if the plan has an entry, else the pinned device's backend (structural
/// ops — `Op::Copy` / `Op::Move` / `Op::Alloc` / `Op::ZeroFill` — whose
/// `Op::Copy` / `Op::Move` stamps the optimizer's residency pass later
/// corrects to the source backend). Mirror of
/// `fuel_dispatch::optimize`'s `stamp_plan_backends`.
#[cfg(test)]
fn stamp_plan_backends(
    graph: &Arc<RwLock<Graph>>,
    roots: &[NodeId],
    plan: &ExecutionPlan,
    pinned_loc: DeviceLocation,
) -> Result<()> {
    let pinned_backend = location_to_backend_id(pinned_loc);
    let mut g = graph
        .write()
        .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
    let order = topo_order_multi(&g, roots);
    for &id in &order {
        let node = g.node(id);
        if matches!(node.op, Op::Const | Op::Release | Op::Contiguize)
            || node.op.is_view_op()
            || matches!(node.op, Op::Reshape(_))
        {
            continue;
        }
        let stamp = plan
            .alternatives(id)
            .and_then(|set| set.winner())
            .map(|c| c.backend)
            .unwrap_or(pinned_backend);
        g.set_target_backend(id, stamp);
    }
    Ok(())
}

/// For each reachable `Op::Const`, take its legacy storage slot,
/// extract bytes via the dyn host-buffer interface, and produce a
/// device-resident `fuel_memory::Storage` keyed in a StorageCache by
/// the Const's NodeId.
///
/// **CPU device** (target == `DeviceLocation::Cpu`): per-Const
/// CPU-storage construction — no transient graph, no executor
/// invocation. Just `CpuStorageBytes::from_bytes(host_bytes)`.
///
/// **Non-CPU device** (Phase 3b of bridge-retirement, post-9c):
/// builds a transient graph with one `Op::Const → Op::Copy { target }`
/// pair per user Const, seeds the transient StorageCache with CPU
/// storages of host bytes (+ a device-handle anchor), and realizes
/// the Op::Copy targets via `PipelinedExecutor::realize_many`. The
/// resulting device storages are inserted at the **original** user-
/// Const NodeIds. The transient graph isn't observable to the user
/// — only the user-Const NodeIds appear in the returned cache.
///
/// This replaces the deleted `upload_host_buffer`'s per-`DeviceLocation`
/// match. The per-target match now lives in the executor's
/// `WorkItemKind::Copy` arm (output allocation) and the
/// `copy_from_cpu_wrapper` (per-target H2D).
///
/// `pub(crate)`: [`crate::factories::LazyRealizer`] (the Judge's
/// realize seam) calls this directly to maintain a persistent
/// const cache across its warmup + timed iterations — the pipelined
/// replacement for the legacy executor's `const_pool` amortization.
pub(crate) fn build_const_cache(
    graph: &Arc<RwLock<Graph>>,
    order: &[NodeId],
    device: &Device,
    initial: StorageCache,
) -> Result<StorageCache> {
    let mut cache = initial;
    cache.reserve(order.len() / 4);

    // Pass 1: collect (user_const_id, host_bytes, dtype, need_bytes)
    // for every reachable Op::Const that isn't already in the cache
    // (persistent slots from InferenceContext take precedence).
    let consts_to_upload: Vec<(NodeId, Vec<u8>, fuel_ir::DType)> = {
        let g = graph
            .read()
            .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
        let mut out: Vec<(NodeId, Vec<u8>, fuel_ir::DType)> =
            Vec::with_capacity(order.len() / 4);
        for &id in order {
            if cache.contains_key(&id) {
                continue;
            }
            let node = g.node(id);
            if !matches!(node.op, Op::Const) {
                continue;
            }
            let slot_arc = g.storage_for(id).ok_or_else(|| {
                Error::Msg(format!(
                    "pipelined_bridge: Op::Const node {id:?} has no \
                     storage in graph.storage_map (constructor failed \
                     to seed the slot)",
                ))
                .bt()
            })?;
            let (host_buf, dtype) = {
                let slot = slot_arc
                    .read()
                    .map_err(|_| Error::Msg("slot lock poisoned".into()).bt())?;
                (slot.as_dyn().to_host_buffer_dyn()?, slot.dtype())
            };
            // Truncate to the node's declared shape. The slot's buffer
            // may hold more bytes than the node consumes (shared
            // storage across views, padding for alignment). Same
            // truncation contract the deleted `upload_host_buffer`'s
            // `truncate_to` parameter enforced.
            let need_bytes = node.shape.elem_count() * dtype.size_in_bytes();
            let mut bytes = host_buffer_to_bytes(&host_buf);
            if bytes.len() > need_bytes {
                bytes.truncate(need_bytes);
            }
            out.push((id, bytes, dtype));
        }
        out
    };

    if consts_to_upload.is_empty() {
        return Ok(cache);
    }

    let target_loc = device.location();
    if target_loc == DeviceLocation::Cpu {
        // CPU realize: short-circuit. CPU→CPU through the executor
        // would be one extra memcpy per Const for no architectural
        // benefit (the per-`DeviceLocation` match in the deleted
        // `upload_host_buffer` was about routing to the right
        // backend allocator; for CPU there's no routing decision).
        for (id, bytes, dtype) in consts_to_upload {
            let storage = Storage::new(
                BackendStorage::Cpu(fuel_cpu_backend::byte_storage::CpuStorageBytes::from_bytes(
                    &bytes,
                )),
                dtype,
            );
            cache.insert(id, Arc::new(RwLock::new(storage)));
        }
        return Ok(cache);
    }

    // Non-CPU realize: build a transient graph with `Op::Const →
    // Op::Copy { target: target_loc }` pairs and realize the Op::Copy
    // targets multi-target. The transient graph is internal — the
    // user's graph stays unmodified.
    let transient = Arc::new(RwLock::new(Graph::new()));
    let mut transient_cache = StorageCache::new();

    // Device-handle anchor: the executor's Op::Copy arm derives the
    // device handle by searching the cache for any storage on the
    // target backend. Without an anchor, the first Op::Copy can't
    // resolve a CUDA/Vulkan device handle. Push an Op::Const
    // placeholder first; its cache entry is the 4-byte device-seed
    // Storage.
    if let Some(seed) = device_seed_storage(device)? {
        let anchor_id = {
            let mut g = transient
                .write()
                .map_err(|_| Error::Msg("transient graph lock poisoned".into()).bt())?;
            g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: fuel_ir::Shape::from_dims(&[4]),
                dtype: fuel_ir::DType::U8,
            })
        };
        transient_cache.insert(anchor_id, Arc::new(RwLock::new(seed)));
    }

    // Push one Op::Const → Op::Copy pair per user Const. The
    // transient Const's cache entry is the CPU storage of the host
    // bytes; the Op::Copy reads it and produces a device-resident
    // output. Keep parallel vectors of (user_const_id, transient
    // copy_id) so we can write results into the user's cache.
    //
    // target_backend on the Op::Copy nodes = Cpu (the SOURCE
    // backend; the kernel that runs is `copy_from_cpu_wrapper`,
    // registered at `(OpKind::Copy, [dt, dt], Cpu)`). The
    // executor's WorkItemKind::Copy arm reads target_location from
    // the op's variant to know where to allocate the output.
    let mut user_to_copy: Vec<(NodeId, NodeId)> =
        Vec::with_capacity(consts_to_upload.len());
    {
        let mut g = transient
            .write()
            .map_err(|_| Error::Msg("transient graph lock poisoned".into()).bt())?;
        for (user_id, bytes, dtype) in consts_to_upload.into_iter() {
            let n_elem = if dtype.size_in_bytes() == 0 {
                0
            } else {
                bytes.len() / dtype.size_in_bytes()
            };
            let shape = fuel_ir::Shape::from_dims(&[n_elem]);
            let trans_const_id = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: shape.clone(),
                dtype,
            });
            let cpu_storage = Storage::new(
                BackendStorage::Cpu(fuel_cpu_backend::byte_storage::CpuStorageBytes::from_bytes(
                    &bytes,
                )),
                dtype,
            );
            transient_cache.insert(trans_const_id, Arc::new(RwLock::new(cpu_storage)));
            let copy_id = g.push(Node {
                op: Op::Copy { target: target_loc },
                inputs: vec![trans_const_id],
                shape,
                dtype,
            });
            g.set_target_backend(copy_id, BackendId::Cpu);
            user_to_copy.push((user_id, copy_id));
        }
    }

    let copy_targets: Vec<NodeId> = user_to_copy.iter().map(|(_, c)| *c).collect();
    let realized = PipelinedExecutor::realize_many(
        Arc::clone(&transient), &copy_targets, transient_cache,
    )?;
    if realized.len() != user_to_copy.len() {
        return Err(Error::Msg(format!(
            "build_const_cache: realize_many returned {} storages for {} \
             Op::Copy targets — internal bug",
            realized.len(), user_to_copy.len(),
        )).bt());
    }
    for ((user_id, _), (arc, _layout)) in user_to_copy.into_iter().zip(realized) {
        cache.insert(user_id, arc);
    }
    Ok(cache)
}

/// Extract the raw bytes from a `HostBuffer` via a per-variant match
/// (`bytemuck::cast_slice` for typed numeric vecs; identity for the
/// raw-byte sub-byte variants).
fn host_buffer_to_bytes(buf: &HostBuffer) -> Vec<u8> {
    match buf {
        HostBuffer::U8(v) => v.clone(),
        HostBuffer::I8(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::U32(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::I16(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::I32(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::I64(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::BF16(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::F16(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::F32(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::F64(v) => bytemuck::cast_slice(v).to_vec(),
        // F8E4M3 has no `Pod` impl in the float8 crate; reinterpret
        // via std::slice::from_raw_parts. `F8E4M3` is `Copy` + 1 byte
        // wide so this is a safe transmute over &[F8E4M3] → &[u8].
        HostBuffer::F8E4M3(v) => {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<float8::F8E4M3>(),
                )
            };
            bytes.to_vec()
        }
        HostBuffer::F6E2M3(v) => v.clone(),
        HostBuffer::F6E3M2(v) => v.clone(),
        HostBuffer::F4(v) => v.clone(),
        HostBuffer::F8E8M0(v) => v.clone(),
    }
}

/// Map a `Device` (the fuel-core wrapper around `DynBackendDevice`) to
/// the `BackendId` the kernel-binding-table keys on. Mirrors the
/// `DeviceLocation` variants 1:1.
fn device_to_backend_id(device: &Device) -> BackendId {
    location_to_backend_id(device.location())
}

/// Map a `DeviceLocation` to the backend that owns its storage
/// substrate. Total — `BackendId` mirrors `DeviceLocation` 1:1
/// (AOCL/MKL are `kernel_source` siblings under `Cpu`, not distinct
/// backends).
fn location_to_backend_id(loc: DeviceLocation) -> BackendId {
    match loc {
        DeviceLocation::Cpu => BackendId::Cpu,
        DeviceLocation::Cuda { .. } => BackendId::Cuda,
        DeviceLocation::Vulkan { .. } => BackendId::Vulkan,
        DeviceLocation::Metal { .. } => BackendId::Metal,
    }
}

// ---------------------------------------------------------------------------
// Runtime route picker (Picker 2) — Phase C PR-C1
// ---------------------------------------------------------------------------
//
// Re-introduces the selector + live-telemetry plumbing PR-A3b-2 removed
// (`production_selector_for` / `backend_runtime_lookup_for` /
// `DeviceRuntimeHandle`), now feeding the **branch route picker**
// (`fuel_dispatch::ranker::pick_route`) rather than a per-node selector.
// The bridge builds the production `ChainedSelector` (VRAM-pressure guard
// + Judge-aware rank) and the live per-tier free-memory lookup, then
// resolves one arm per `Op::Branch` into a `PickedRoute` the executor
// lowers via `lower_picked_route`. With no branches (CPU-only build) the
// route is empty and realize is unchanged from Phase B.

/// Opt-out knob for the runtime route picker. Matches the `FUEL_*`
/// env-var convention (`FUEL_FORCE_F32`, `FUEL_TRACE`, ...): set
/// `FUEL_DISABLE_RUNTIME_SELECTOR=1` to fall back to the static arm-0
/// lowering (no live-telemetry route picking). Default: picker ON.
fn runtime_selector_disabled() -> bool {
    std::env::var("FUEL_DISABLE_RUNTIME_SELECTOR").ok().as_deref() == Some("1")
}

/// Build the production Picker 2 for one realize call: a
/// [`ChainedSelector`] composing
///
/// - the **VramPressure guard** over live [`BackendRuntime`] handles
///   ([`backend_runtime_lookup_for`] — the realize device + the
///   always-present CPU backend), and
/// - the **JudgeAware rank** over the cached profile oracle
///   ([`crate::judge::cached_oracle`], the same Layer-2 source
///   `build_optimized_graph` feeds `PlanOptions::with_judge`).
///
/// Returns the selector **and** the live free-memory lookup (the route
/// picker's cache fingerprints telemetry through the same lookup).
/// `None` when [`runtime_selector_disabled`] — the dispatch path then
/// uses the static arm-0 lowering.
fn production_selector_for(
    device: &Device,
) -> Option<(Arc<dyn RuntimeSelector>, BackendRuntimeLookup)> {
    if runtime_selector_disabled() {
        return None;
    }
    let judge: Option<Arc<dyn JudgeOracle>> = crate::judge::cached_oracle()
        .map(|oracle| oracle as Arc<dyn JudgeOracle>);
    let lookup = backend_runtime_lookup_for(device);
    let selector: Arc<dyn RuntimeSelector> = Arc::new(
        ChainedSelector::with_default_estimator(judge, Some(lookup.clone())),
    );
    Some((selector, lookup))
}

/// [`BackendRuntime`] adapter over a live device handle. Owns the
/// `Arc<dyn DynBackendDevice>` so the boxed handle the selector borrows
/// stays valid for the duration of a `select` call; every query
/// delegates through [`DynBackendDevice::as_backend_runtime`]. Devices
/// without a runtime impl answer `None` / `Unknown` — honest no-signal,
/// never fabricated pressure.
struct DeviceRuntimeHandle(Arc<dyn DynBackendDevice>);

impl BackendRuntime for DeviceRuntimeHandle {
    fn available_bytes(&self) -> Option<u64> {
        self.0.as_backend_runtime().and_then(|r| r.available_bytes())
    }

    fn total_bytes(&self) -> Option<u64> {
        self.0.as_backend_runtime().and_then(|r| r.total_bytes())
    }

    // Delegate rather than re-derive: backends may override `would_fit`
    // with native predictive checks (e.g. Vulkan's VK_EXT_memory_budget
    // fragmentation awareness).
    fn would_fit(&self, size: u64) -> FitStatus {
        match self.0.as_backend_runtime() {
            Some(r) => r.would_fit(size),
            None => FitStatus::Unknown,
        }
    }
}

/// Live-handle lookup for the VramPressure guard / picker fingerprint.
/// Resolves:
///
/// - the realize device's `(backend, location)` → the realize `Device`'s
///   own handle (the live GPU handle the bridge holds — with today's
///   monolithic pinning every GPU arm in a branch targets exactly this
///   device);
/// - `(Cpu, DeviceLocation::Cpu)` → a fresh stateless
///   [`fuel_cpu_backend::dyn_impl::CpuBackendDevice`] (covers the
///   host-RAM arm + CPU candidates when realizing on a GPU);
/// - anything else → `None` (= `FitStatus::Unknown`, no signal).
///   Multi-GPU realizes will widen this when a live device registry
///   lands.
fn backend_runtime_lookup_for(device: &Device) -> BackendRuntimeLookup {
    let realize_loc = device.location();
    let realize_backend = location_to_backend_id(realize_loc);
    let inner: Arc<dyn DynBackendDevice> = device.inner.clone();
    Arc::new(move |backend, loc| {
        if backend == realize_backend && loc == realize_loc {
            return Some(Box::new(DeviceRuntimeHandle(Arc::clone(&inner)))
                as BackendRuntimeHandle);
        }
        if backend == BackendId::Cpu && loc == DeviceLocation::Cpu {
            return Some(Box::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice)
                as BackendRuntimeHandle);
        }
        None
    })
}

/// Resolve the runtime route — one arm per `Op::Branch` chosen by the
/// production Picker 2 (selector + live telemetry). `None` when the
/// picker is disabled; otherwise the [`PickedRoute`] the executor lowers
/// via `lower_picked_route`. A branchless graph yields an empty route
/// (the picker is a no-op), so realize is byte-identical to Phase B.
fn resolve_runtime_route(
    graph: &Arc<RwLock<Graph>>,
    roots: &[NodeId],
    plan: &ExecutionPlan,
    device: &Device,
) -> Result<Option<PickedRoute>> {
    let Some((selector, lookup)) = production_selector_for(device) else {
        return Ok(None);
    };
    let g = graph
        .read()
        .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
    // Cheap branchless fast-path: a graph with no `Op::Branch` (every
    // CPU-only build, every graph with no genuine fork) has no decision
    // for the picker. Skip the topo walk + selector entirely and take the
    // arm-0 lowering — realize is byte-identical to Phase B, with no
    // per-realize picker overhead on the common case.
    let has_branch = (0..g.len())
        .any(|i| matches!(g.node(NodeId(i)).op, Op::Branch { .. }));
    if !has_branch {
        return Ok(None);
    }
    let route = pick_route(&g, roots, plan, selector.as_ref(), Some(&lookup));
    Ok(Some(route))
}

// ---------------------------------------------------------------------------
// Cross-device copy insertion (picker arc Phase 2.1 wire-in)
// ---------------------------------------------------------------------------

/// Where do the bytes of a cache-resident [`Storage`] live?
///
/// Returns `None` when the variant can't self-report its device —
/// legacy Vulkan storages constructed without a backend handle
/// (`VulkanStorageBytes::from_device`), and Metal pending its byte
/// substrate. Callers treat unknown residency as "leave the edge
/// alone" — the status-quo executor behavior, never a wrong copy.
fn cached_storage_location(storage: &Storage) -> Option<DeviceLocation> {
    match &storage.inner {
        BackendStorage::Cpu(_) => Some(DeviceLocation::Cpu),
        #[cfg(feature = "cuda")]
        BackendStorage::Cuda(c) => Some(c.device().location()),
        #[cfg(feature = "vulkan")]
        BackendStorage::Vulkan(v) => v
            .backend()
            .map(|b| DeviceLocation::Vulkan { gpu_id: b.gpu_id }),
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        BackendStorage::Metal(_) => None,
    }
}

/// Allocate a small "device anchor" storage on `device` — enough bytes
/// to carry the device handle into the [`StorageCache`] so the
/// pipelined executor's [`WorkItemKind::Alloc`] arm can derive the
/// per-backend handle for `Op::Alloc` nodes.
///
/// Phase 3a of bridge-retirement (post-9c). This is the *residual*
/// of the deleted [`fuel-core::inference_context::alloc_zeroed_on`]:
/// it does only the per-backend "allocate-on-device" piece, not the
/// zero-fill (that moves to the executor's Alloc arm). Callers
/// (today: [`crate::inference_context::KvCache::with_capacity`])
/// insert the returned Storage into the StorageCache before realizing
/// Op::Alloc nodes; the executor finds the device handle by searching
/// the cache for any storage on the target backend.
///
/// For CPU targets returns `Ok(None)` — CPU's Op::Alloc arm doesn't
/// need a device-handle anchor (`alloc_cpu_zeroed` is allocator-free).
///
/// The 4-byte size is arbitrary: small enough to be ~free, large
/// enough that even Vulkan's strict `vkAllocateMemory` accepts it.
pub fn device_seed_storage(device: &Device) -> Result<Option<Storage>> {
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    const SEED_BYTES: usize = 4;
    match device.location() {
        DeviceLocation::Cpu => Ok(None),
        #[cfg(feature = "cuda")]
        DeviceLocation::Cuda { .. } => {
            let cuda_dev = crate::cuda_backend::as_device(device)?;
            let cuda_bytes =
                fuel_cuda_backend::CudaStorageBytes::alloc(cuda_dev, SEED_BYTES)?;
            Ok(Some(Storage::new(BackendStorage::Cuda(cuda_bytes), fuel_ir::DType::U8)))
        }
        #[cfg(not(feature = "cuda"))]
        DeviceLocation::Cuda { .. } => Err(Error::Msg(
            "device_seed_storage: CUDA device requested but fuel-core wasn't built \
             with --features cuda".into(),
        )
        .bt()),
        #[cfg(feature = "vulkan")]
        DeviceLocation::Vulkan { .. } => {
            let backend = crate::vulkan_backend::as_device(device)?;
            let zeros = vec![0_u8; SEED_BYTES];
            let vk_bytes = backend.upload_bytes_handle(&zeros)?;
            Ok(Some(Storage::new(BackendStorage::Vulkan(vk_bytes), fuel_ir::DType::U8)))
        }
        #[cfg(not(feature = "vulkan"))]
        DeviceLocation::Vulkan { .. } => Err(Error::Msg(
            "device_seed_storage: Vulkan device requested but fuel-core wasn't built \
             with --features vulkan".into(),
        )
        .bt()),
        other => Err(Error::Msg(format!(
            "device_seed_storage: device {other:?} not wired (CPU + CUDA + Vulkan \
             today; Metal pending its byte-storage substrate)",
        ))
        .bt()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::{DType, Shape};
    use fuel_dispatch::plan::compile_plan;

    fn push_node(g: &mut Graph, op: Op, inputs: Vec<NodeId>) -> NodeId {
        g.push(Node {
            op,
            inputs,
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        })
    }

    fn cpu_storage_f32(vals: &[f32]) -> Arc<RwLock<Storage>> {
        let bytes: &[u8] = bytemuck::cast_slice(vals);
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cpu(
                fuel_cpu_backend::byte_storage::CpuStorageBytes::from_bytes(bytes),
            ),
            DType::F32,
        )))
    }

    fn noop_kernel_for_tests(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[fuel_ir::Layout],
        _p: &fuel_dispatch::kernel::OpParams,
    ) -> Result<()> {
        Ok(())
    }

    /// Build an ExecutionPlan with one single-candidate winner per
    /// `(node, backend, device)` entry. The kernel ref is a no-op —
    /// these tests assert stamping/placement metadata, not dispatch.
    fn plan_with_winners(
        winners: &[(NodeId, BackendId, DeviceLocation)],
    ) -> ExecutionPlan {
        use fuel_dispatch::ranker::{AlternativeSet, Candidate};
        let mut alternatives = HashMap::new();
        for &(node, backend, device) in winners {
            let mut set = AlternativeSet::empty();
            set.push(Candidate {
                kernel: noop_kernel_for_tests,
                caps: fuel_dispatch::kernel::KernelCaps::empty(),
                backend,
                device,
                precision:
                    fuel_dispatch::fused::PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
                static_cost: Default::default(),
                inbound_transfer_ns: 0,
                op_params: fuel_dispatch::kernel::OpParams::None,
                coupling: Vec::new(),
                kernel_source: "",
            });
            alternatives.insert(node, set);
        }
        ExecutionPlan {
            order: Vec::new(),
            alternatives,
            generation: 0,
        }
    }

    /// Picker-arc step 4a: `stamp_plan_backends` commits the plan's
    /// winner backend per node; nodes without a plan entry (here the
    /// structural Op::Copy) get the pinned device's backend; Consts
    /// stay unstamped.
    #[test]
    fn stamp_plan_backends_winner_and_pinned_default() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, add, copy) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let add = push_node(&mut g, Op::Add, vec![c1, c1]);
            let copy = push_node(
                &mut g,
                Op::Copy { target: DeviceLocation::Cpu },
                vec![add],
            );
            (c1, add, copy)
        };
        let plan = plan_with_winners(&[(
            add,
            BackendId::Cuda,
            DeviceLocation::Cuda { gpu_id: 0 },
        )]);
        stamp_plan_backends(&graph, &[copy], &plan, DeviceLocation::Cpu).unwrap();
        let g = graph.read().unwrap();
        assert_eq!(
            g.target_backend(add),
            Some(BackendId::Cuda),
            "plan winner's backend stamped",
        );
        assert_eq!(
            g.target_backend(copy),
            Some(BackendId::Cpu),
            "no plan entry → pinned device's backend",
        );
        assert_eq!(g.target_backend(c1), None, "Const skipped");
    }

    /// Step 4a preserves the old monolithic loop's "always
    /// overwrite" doctrine: a stale stamp from a previous realize on
    /// another device is re-stamped from this call's plan, and the
    /// realize succeeds on CPU.
    #[test]
    fn stale_stamps_overwritten_per_realize() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, c2, add) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let c2 = push_node(&mut g, Op::Const, vec![]);
            let add = push_node(&mut g, Op::Add, vec![c1, c2]);
            // Stale pin from a hypothetical previous CUDA realize.
            g.set_target_backend(add, BackendId::Cuda);
            (c1, c2, add)
        };
        let mut initial = StorageCache::new();
        initial.insert(c1, cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]));
        initial.insert(c2, cpu_storage_f32(&[10.0, 20.0, 30.0, 40.0]));

        let device = crate::Device::cpu();
        let out =
            realize_one_as_with_initial::<f32>(&graph, add, &device, initial)
                .expect("CPU realize despite stale CUDA stamp");
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(
            graph.read().unwrap().target_backend(add),
            Some(BackendId::Cpu),
            "stale stamp overwritten by this call's plan winner",
        );
    }

    /// Planner Stage 2 fuel-core adapter, CPU-only integration: the
    /// `SystemTopology`-as-`TransferEstimator` impl + the cache-
    /// residency callback thread through `compile_plan` and change
    /// NOTHING on a single-device host — zero inbound-transfer
    /// terms, candidate-for-candidate identical plan. (Zero probed
    /// paths is pinned by `transfer_cost::tests::
    /// calibrate_cpu_only_is_empty`; this test never queries a
    /// cross-device pair, so the same-device short-circuit
    /// guarantees the lazy calibration probe can't fire even on
    /// GPU-featured builds — where the local CPU-only binding table
    /// keeps every candidate and residency on CPU anyway.)
    #[test]
    fn stage2_cpu_only_estimator_leaves_plan_unchanged() {
        use fuel_ir::dispatch::OpKind;
        use fuel_dispatch::kernel::{unknown_cost, KernelBindingTable, KernelCaps};

        let mut table = KernelBindingTable::new();
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel_for_tests,
            KernelCaps::empty(),
            fuel_dispatch::fused::PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );

        let mut g = Graph::new();
        let c1 = push_node(&mut g, Op::Const, vec![]);
        let add = push_node(&mut g, Op::Add, vec![c1, c1]);
        let order = topo_order_multi(&g, &[add]);

        // The const's bytes are CPU-resident — what build_const_cache
        // produces for a CPU realize.
        let mut cache = StorageCache::new();
        cache.insert(c1, cpu_storage_f32(&[1.0; 4]));

        let topology = SystemTopology::current();
        let placements_for = |dev: DeviceLocation| -> Vec<BackendId> {
            topology.backends_for(dev).to_vec()
        };
        let capabilities_for = |b: BackendId|
            -> Option<&fuel_ir::backend::BackendCapabilities>
        { topology.capabilities(b) };
        // Same closure shape build_optimized_graph wires.
        let input_residency = |id: NodeId| -> Option<DeviceLocation> {
            let slot = cache.get(&id)?;
            let guard = slot.read().ok()?;
            cached_storage_location(&guard)
        };

        let base_opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_for)
            .with_capabilities_for(&capabilities_for);
        let base = compile_plan(&g, &order, &table, &base_opts).expect("base plan");

        let wired_opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_for)
            .with_capabilities_for(&capabilities_for)
            .with_transfer_estimator(&*topology)
            .with_input_residency(&input_residency);
        let wired = compile_plan(&g, &order, &table, &wired_opts).expect("wired plan");

        let a = base.alternatives(add).expect("base set");
        let b = wired.alternatives(add).expect("wired set");
        assert_eq!(a.len(), b.len(), "same candidate count");
        for (ca, cb) in a.alternatives().iter().zip(b.alternatives()) {
            assert_eq!(ca.backend, cb.backend);
            assert_eq!(ca.device, cb.device);
            assert_eq!(ca.kernel as usize, cb.kernel as usize, "same kernel ref");
            assert_eq!(ca.kernel_source, cb.kernel_source);
            assert_eq!(
                cb.inbound_transfer_ns, 0,
                "CPU-only host: zero transfer terms",
            );
        }
    }

    /// CPU realize through `realize_one_as_with_initial` over the
    /// optimized ("plan IS the graph") dispatch path produces correct
    /// bytes — the single realize path after PR-A3b-2.
    #[test]
    fn optimized_cpu_realize_end_to_end() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, c2, add) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let c2 = push_node(&mut g, Op::Const, vec![]);
            let add = push_node(&mut g, Op::Add, vec![c1, c2]);
            (c1, c2, add)
        };
        let mut initial = StorageCache::new();
        initial.insert(c1, cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]));
        initial.insert(c2, cpu_storage_f32(&[10.0, 20.0, 30.0, 40.0]));

        let device = crate::Device::cpu();
        // PR-C1: the production runtime route picker (Picker 2) defaults
        // ON, so this CPU realize goes through `resolve_runtime_route` →
        // `pick_route`. A CPU-only graph has no `Op::Branch`, so the route
        // is empty ⇒ the executor uses the arm-0 lowering ⇒ realize is
        // unchanged. The correct bytes pin that no-branch-no-op contract
        // at the bridge level.
        assert!(
            production_selector_for(&device).is_some(),
            "the runtime route picker defaults ON (no opt-out env set)",
        );
        let out = realize_one_as_with_initial::<f32>(&graph, add, &device, initial)
            .expect("realize through the optimized + route-picker path");
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// PR-C1: the production selector builder returns both a selector and
    /// the live free-memory lookup (the picker's cache fingerprints
    /// telemetry through the same lookup). Defaults ON; the opt-out env
    /// var disables it.
    #[test]
    fn production_selector_for_returns_selector_and_lookup() {
        let device = crate::Device::cpu();
        let built = production_selector_for(&device);
        assert!(built.is_some(), "picker defaults ON without the opt-out");
        let (_selector, lookup) = built.unwrap();
        // The lookup resolves CPU (always-present backend).
        assert!(
            lookup(BackendId::Cpu, DeviceLocation::Cpu).is_some(),
            "the lookup resolves the always-present CPU backend",
        );
    }

    /// PR-C1 (re-introduced from A3b-2): the live-handle lookup resolves
    /// the realize device + CPU and answers `None` (no signal) for
    /// everything else. The CPU handle's `would_fit` must answer without
    /// panicking (the value is platform-dependent — only the wiring is
    /// asserted).
    #[test]
    fn backend_runtime_lookup_resolves_cpu_and_misses_others() {
        let device = crate::Device::cpu();
        let lookup = backend_runtime_lookup_for(&device);

        let cpu = lookup(BackendId::Cpu, DeviceLocation::Cpu)
            .expect("CPU handle always resolvable");
        let _ = cpu.would_fit(1); // platform-dependent; must not panic.

        assert!(
            lookup(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }).is_none(),
            "no live handle for a backend that isn't the realize device",
        );
    }

    /// Session 3 rider: the reporting realize returns the
    /// `kernel_source` of the picker's pick for the realize ROOT (the
    /// caller's node — not the spliced D2H `Op::Copy` on top of it).
    /// The CPU Add f32 cell always has at least one binding-table
    /// alternative, so the plan covers the root and the report is
    /// `Some`; its value must be one of the cell's registered tags
    /// (under default features that's the lone portable registration;
    /// with onemkl/aocl siblings it's whichever the picker ranks
    /// first — membership, not position, is the contract).
    #[test]
    fn reporting_realize_returns_root_kernel_source() {
        use fuel_ir::dispatch::OpKind;

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, c2, add) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let c2 = push_node(&mut g, Op::Const, vec![]);
            let add = push_node(&mut g, Op::Add, vec![c1, c2]);
            (c1, c2, add)
        };
        let mut initial = StorageCache::new();
        initial.insert(c1, cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]));
        initial.insert(c2, cpu_storage_f32(&[10.0, 20.0, 30.0, 40.0]));

        let device = crate::Device::cpu();
        let (out, root_kernel_source) =
            realize_one_as_with_initial_reporting::<f32>(&graph, add, &device, initial, &SymEnv::default())
                .expect("reporting realize");
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);

        let src = root_kernel_source
            .expect("plan covers the Add root → dispatched-sibling report present");
        let bindings = global_bindings();
        let alts = bindings.lookup_alternatives(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
        );
        assert!(
            alts.iter().any(|e| e.kernel_source == src),
            "reported kernel_source {src:?} must be a registered sibling at \
             the (AddElementwise, [F32;3], Cpu) cell",
        );
    }

    /// Real CPU negation kernel for the step-4b end-to-end test —
    /// reads `inputs[0]` as f32 and writes the negation into
    /// `outputs[0]`.
    fn neg_kernel_cpu_f32(
        inputs: &[Arc<RwLock<Storage>>],
        outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[fuel_ir::Layout],
        _params: &fuel_dispatch::kernel::OpParams,
    ) -> Result<()> {
        let negated: Vec<f32> = {
            let guard = inputs[0]
                .read()
                .map_err(|_| Error::Msg("input lock poisoned".into()).bt())?;
            let BackendStorage::Cpu(c) = &guard.inner else {
                return Err(Error::Msg("test kernel: input must be CPU".into()).bt());
            };
            let typed: &[f32] = c.as_slice().expect("f32 cast");
            typed.iter().map(|x| -x).collect()
        };
        let mut out = outputs[0]
            .write()
            .map_err(|_| Error::Msg("output lock poisoned".into()).bt())?;
        let BackendStorage::Cpu(c) = &mut out.inner else {
            return Err(Error::Msg("test kernel: output must be CPU".into()).bt());
        };
        c.as_slice_mut().expect("f32 cast").copy_from_slice(&negated);
        Ok(())
    }

    /// Step 4b end-to-end on CPU, no GPU needed: the realize device
    /// is pinned to CUDA but the (synthetic) binding table has Neg
    /// f32 ONLY on CPU. The picker's off-device fallback places the
    /// op on CPU, the stamping pass commits BackendId::Cpu, the
    /// residency pass proves no crossing (the const lives on CPU
    /// too), and the executor realizes the plan's winner kernel
    /// correctly on CPU.
    #[test]
    fn fallback_off_device_node_realizes_on_cpu_end_to_end() {
        use fuel_ir::dispatch::OpKind;
        use fuel_dispatch::kernel::{unknown_cost, KernelBindingTable, KernelCaps};

        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, neg) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let neg = push_node(&mut g, Op::Neg, vec![c1]);
            (c1, neg)
        };
        let mut cache = StorageCache::new();
        cache.insert(c1, cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]));

        // Synthetic table: Neg f32 registered ONLY under Cpu — the
        // pinned CUDA device has no implementation.
        let mut table = KernelBindingTable::new();
        table.register_full(
            OpKind::NegElementwise,
            &[DType::F32, DType::F32],
            BackendId::Cpu,
            neg_kernel_cpu_f32,
            KernelCaps::empty(),
            fuel_dispatch::fused::PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );

        let placements_fn = move |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == cuda0 { vec![BackendId::Cuda] } else { vec![BackendId::Cpu] }
        };
        let fallback_fn = |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            vec![(BackendId::Cpu, DeviceLocation::Cpu)]
        };
        let options = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(cuda0)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn);
        let plan = {
            let g = graph.read().unwrap();
            let order = fuel_graph::topo_order(&g, neg);
            compile_plan(&g, &order, &table, &options).expect("plan with fallback")
        };
        let winner = plan
            .alternatives(neg)
            .and_then(|s| s.winner())
            .expect("fallback winner");
        assert_eq!(winner.backend, BackendId::Cpu);
        assert_eq!(winner.device, DeviceLocation::Cpu, "placed off-device");

        stamp_plan_backends(&graph, &[neg], &plan, cuda0).unwrap();
        assert_eq!(
            graph.read().unwrap().target_backend(neg),
            Some(BackendId::Cpu),
            "off-device winner's backend stamped",
        );
        // (Residency stitching is now an optimize_graph pass — cleanup Step B;
        // this test realizes via `realize_with_plan` directly, and the const +
        // fallback node both live on CPU, so there is no crossing to stitch.)

        let (storage, _layout) = PipelinedExecutor::realize_with_plan(
            Arc::clone(&graph), neg, cache, Arc::new(plan),
        )
        .expect("realize the off-device fallback winner on CPU");
        let guard = storage.read().unwrap();
        let BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected CPU storage from the fallback node");
        };
        let typed: &[f32] = c.as_slice().expect("f32 cast");
        assert_eq!(typed, &[-1.0, -2.0, -3.0, -4.0]);
    }

    /// Realize-split (executor-unification Session 5): ONE executor
    /// pass; the first `n_host` targets come back as host bytes
    /// (through the spliced D2H Op::Copy), the rest as resident
    /// storage Arcs + layouts straight from the kernel output.
    #[test]
    fn realize_split_host_and_resident() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, c2, add, mul) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let c2 = push_node(&mut g, Op::Const, vec![]);
            let add = push_node(&mut g, Op::Add, vec![c1, c2]);
            let mul = push_node(&mut g, Op::Mul, vec![c1, c2]);
            (c1, c2, add, mul)
        };
        let mut cache = StorageCache::new();
        cache.insert(c1, cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]));
        cache.insert(c2, cpu_storage_f32(&[10.0, 20.0, 30.0, 40.0]));

        let (host, resident) = realize_split_as_with_initial::<f32>(
            &graph, &[add, mul], 1, &crate::Device::cpu(), cache,
        )
        .expect("realize_split");

        assert_eq!(host.len(), 1, "exactly n_host host results");
        assert_eq!(host[0], vec![11.0, 22.0, 33.0, 44.0]);

        assert_eq!(resident.len(), 1, "remaining targets stay resident");
        let (storage, layout) = &resident[0];
        assert_eq!(layout.shape().dims(), &[4]);
        let guard = storage.read().unwrap();
        match &guard.inner {
            BackendStorage::Cpu(c) => {
                let vals: &[f32] = bytemuck::cast_slice(c.bytes());
                assert_eq!(vals, &[10.0, 40.0, 90.0, 160.0]);
            }
            #[allow(unreachable_patterns)]
            other => panic!("expected CPU storage, got {other:?}"),
        }
    }

    /// The Trainer's step-to-step chaining pattern: a resident result
    /// from one realize seeds the next realize's StorageCache at a
    /// fresh placeholder NodeId — no D2H/H2D round-trip, no
    /// storage_map involvement for the carried state.
    #[test]
    fn realize_split_resident_feeds_next_step() {
        // Step 1: n_host = 0 — everything stays resident.
        let g1 = Arc::new(RwLock::new(Graph::new()));
        let (c1, dbl) = {
            let mut g = g1.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let dbl = push_node(&mut g, Op::Add, vec![c1, c1]);
            (c1, dbl)
        };
        let mut cache1 = StorageCache::new();
        cache1.insert(c1, cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]));
        let (host1, resident1) = realize_split_as_with_initial::<f32>(
            &g1, &[dbl], 0, &crate::Device::cpu(), cache1,
        )
        .expect("step 1");
        assert!(host1.is_empty());
        let (carried, _) = resident1.into_iter().next().expect("one resident");

        // Step 2: fresh graph; the carried Arc binds to a placeholder
        // Const (no storage_map slot — the cache provides it).
        let g2 = Arc::new(RwLock::new(Graph::new()));
        let (ph, sq) = {
            let mut g = g2.write().unwrap();
            let ph = push_node(&mut g, Op::Const, vec![]);
            let sq = push_node(&mut g, Op::Sqr, vec![ph]);
            (ph, sq)
        };
        let mut cache2 = StorageCache::new();
        cache2.insert(ph, carried);
        let (host2, resident2) = realize_split_as_with_initial::<f32>(
            &g2, &[sq], 1, &crate::Device::cpu(), cache2,
        )
        .expect("step 2");
        assert_eq!(host2[0], vec![4.0, 16.0, 36.0, 64.0]);
        assert!(resident2.is_empty());
    }

    /// PR-A3b-1: the fallback-flag parser. Unset / empty ⇒ the NEW
    /// `optimize_graph` default; any non-empty value ⇒ the LEGACY
    /// `compile_plan` path. (Tested via the pure `legacy_plan_from_env`
    /// seam so the process-global `OnceLock` cache doesn't pin us to
    /// whichever value the test binary observed first.)
    /// PR-A3b-2: the single optimized realize path runs the bridge's
    /// exact correctness sequence — `build_optimized_graph` (the ONE
    /// `compile_plan`, surfacing its `ExecutionPlan`) → stamp backends →
    /// residency stitch → layout fixups → `realize_with_optimized` (run
    /// / `lower_run` order + binding-table lookup) — and produces the
    /// expected values for `(a + b) * a`. This exercises the de-duped
    /// plan reuse directly: the SAME plan `optimize_graph` surfaces
    /// drives stamping/residency/layout, and the executor recomputes its
    /// order from the stamped graph.
    #[test]
    fn optimized_path_runs_full_correctness_sequence() {
        // Realize `(a + b) * a` with a = [1,2,3,4], b = [10,20,30,40].
        // add  = [11,22,33,44]
        // out  = add * a = [11, 44, 99, 176]
        let expected = vec![11.0_f32, 44.0, 99.0, 176.0];
        let a_vals = [1.0_f32, 2.0, 3.0, 4.0];
        let b_vals = [10.0_f32, 20.0, 30.0, 40.0];

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a, b, _add, out) = {
            let mut g = graph.write().unwrap();
            let a = push_node(&mut g, Op::Const, vec![]);
            let b = push_node(&mut g, Op::Const, vec![]);
            let add = push_node(&mut g, Op::Add, vec![a, b]);
            let out = push_node(&mut g, Op::Mul, vec![add, a]);
            (a, b, add, out)
        };

        let device = crate::Device::cpu();
        let pinned = device.location();

        let mut initial = StorageCache::new();
        initial.insert(a, cpu_storage_f32(&a_vals));
        initial.insert(b, cpu_storage_f32(&b_vals));

        // prepare(): const cache + realize-root Op::Copy{Cpu} splice.
        let (cache, _backend, mut eff) =
            prepare(&graph, &[out], &device, initial).expect("prepare");
        let cpu_target = eff.pop().expect("one effective target");

        // The bridge's correctness sequence is now entirely inside
        // `optimize_graph`: it stamps the winning backends AND runs the
        // residency + layout-fixup passes (cleanup A1 + Step B). The bridge
        // just dispatches the stamped, copy-stitched graph.
        let (optimized, _plan) =
            build_optimized_graph(&graph, &[cpu_target], pinned, &cache)
                .expect("optimize_graph");

        let (storage, _layout) = PipelinedExecutor::realize_with_optimized(
            graph.clone(), cpu_target, cache, &optimized,
        )
        .expect("optimized realize");
        let out_vals = extract_cpu_bytes_typed::<f32>(&storage).expect("host bytes");

        assert_eq!(out_vals, expected, "optimized path: (a+b)*a");
    }

    /// A default-path realize goes through `realize_one_as_with_initial`
    /// and produces correct values — confirming the optimized
    /// `optimize_graph` engine is the live path on the public realize
    /// entry point.
    #[test]
    fn default_realize_entry_point_produces_correct_values() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a, b, add, out) = {
            let mut g = graph.write().unwrap();
            let a = push_node(&mut g, Op::Const, vec![]);
            let b = push_node(&mut g, Op::Const, vec![]);
            let add = push_node(&mut g, Op::Add, vec![a, b]);
            let out = push_node(&mut g, Op::Mul, vec![add, a]);
            (a, b, add, out)
        };
        let _ = add;
        let mut initial = StorageCache::new();
        initial.insert(a, cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]));
        initial.insert(b, cpu_storage_f32(&[10.0, 20.0, 30.0, 40.0]));

        let out_vals = realize_one_as_with_initial::<f32>(
            &graph, out, &crate::Device::cpu(), initial,
        )
        .expect("default realize of (a+b)*a");
        assert_eq!(out_vals, vec![11.0, 44.0, 99.0, 176.0]);
    }

    /// `n_host` beyond the target count is a typed error, not a panic.
    #[test]
    fn realize_split_n_host_too_large_errors() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let c1 = {
            let mut g = graph.write().unwrap();
            push_node(&mut g, Op::Const, vec![])
        };
        let err = realize_split_as_with_initial::<f32>(
            &graph, &[c1], 2, &crate::Device::cpu(), StorageCache::new(),
        );
        assert!(err.is_err(), "n_host > targets.len() must be a typed Err");
    }
}
