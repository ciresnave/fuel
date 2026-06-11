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

use fuel_core_types::backend::{BackendRuntime, FitStatus};
use fuel_core_types::dyn_backend::DynBackendDevice;
use fuel_core_types::{
    probe::BackendId, DeviceLocation, Error, HostBuffer, Result,
};
use fuel_graph::{Graph, Node, NodeId, Op, topo_order_multi};
use fuel_graph::opt::{
    execution_plan, insert_cross_device_copies, insert_layout_fixups,
};
use fuel_dispatch::dispatch::global_bindings;
use fuel_dispatch::plan::{compile_plan, ExecutionPlan, PlanOptions};
use fuel_dispatch::pipelined::{PipelinedExecutor, StorageCache};
use fuel_dispatch::ranker::{
    BackendRuntimeHandle, BackendRuntimeLookup, ChainedSelector, JudgeOracle,
    RuntimeSelector,
};
use fuel_storage::{BackendStorage, Storage};

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
/// 2. `build_execution_plan` — picker enumeration + rank against the
///    pinned device; `stamp_plan_backends` commits each winner's
///    backend to the graph.
/// 3. `insert_resident_input_copies` + `apply_layout_fixups` —
///    residency + layout stitching against the final placements.
/// 4. `PipelinedExecutor::realize_with_plan[_and_selector]` — kick
///    the compile + execute pipeline; returns a
///    `BackendStorage::Cpu` for the spliced root.
/// 5. `bytemuck::cast_slice` — reinterpret the CPU bytes as `T`.
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
    realize_one_as_with_initial_reporting::<T>(graph, target, device, initial)
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
    // Phase 4.2: build an ExecutionPlan via SystemTopology-driven
    // placements so the optimizer ranker (Phases 1.1–1.5 + 3) gets
    // to pick among co-located backends per node. The plan ships
    // alongside the realize call; the executor's plan-aware dispatch
    // (Phase 4.1) reads `AlternativeSet::winner` per node.
    //
    // Phase 4.3: retry on `TopologyChanged`. If the live
    // SystemTopology generation advances between plan-build and
    // dispatch (a backend registered/unregistered mid-flight), the
    // executor surfaces `Error::TopologyChanged`; we rebuild the
    // plan against the fresh topology and try again. Retries back
    // off exponentially (`topology_retry_backoff`) so a burst of
    // generation bumps — device hotplug storms, or the test suite's
    // concurrent churn — settles instead of exhausting instant
    // rebuilds; capped at `MAX_PLAN_REBUILDS` to prevent infinite
    // spin under genuinely persistent churn.
    //
    // Picker-arc step 3: dispatch through the production runtime
    // selector (Picker 2) — VramPressure guard + JudgeAware rank.
    // See `production_selector_for`.
    let selector = production_selector_for(device);
    let (storage, root_kernel_source) = dispatch_with_plan_retry(
        graph, cpu_target, cache, selector, device.location(), target,
    )?;
    Ok((extract_cpu_bytes_typed::<T>(&storage)?, root_kernel_source))
}

/// Phase 4.3 retry-on-stale-plan loop for the single-target path.
/// Pulled out so the multi-target path can reuse the same retry
/// shape.
///
/// `selector` is the production runtime selector (Picker 2) from
/// [`production_selector_for`] — `Some` routes through
/// [`PipelinedExecutor::realize_with_plan_and_selector`]; `None`
/// (opt-out knob set) keeps the selector-less plan path, whose
/// dispatch takes `AlternativeSet::winner` per node.
///
/// Picker-arc step 4a: each attempt runs the full plan → stamp →
/// copy-insert → fixup sequence. The plan is built against the
/// pinned DEVICE (no per-node backends yet); the winners are then
/// stamped onto the graph (`stamp_plan_backends`), the
/// cross-device-copy pass stitches residency against those final
/// placements, and the layout-fixup pass runs last. Re-planning
/// after a `TopologyChanged` retry re-runs the whole sequence so
/// stamps stay consistent with the fresh plan.
/// Cap on plan rebuilds when the executor surfaces
/// `Error::TopologyChanged` mid-dispatch. Six attempts with the
/// exponential backoff below give a churn burst ~63ms to settle
/// before the error escapes to the caller — enough for hotplug
/// storms and the test suite's concurrent generation churn, while
/// genuinely persistent churn still fails loudly.
const MAX_PLAN_REBUILDS: usize = 6;

/// Exponential settle-wait between `TopologyChanged` plan rebuilds:
/// 1, 2, 4, 8, 16ms for attempts 0..=4 (the 6th attempt dispatches
/// immediately after the last wait). A topology generation bump
/// means registration state is actively moving; rebuilding the plan
/// in a tight loop just re-reads the same moving state — a brief
/// sleep is the correct primitive here, not a perf-path concern
/// (this only runs when a rebuild is already required).
fn topology_retry_backoff(attempt: usize) {
    let ms = 1u64 << attempt.min(4);
    std::thread::sleep(std::time::Duration::from_millis(ms));
}

fn dispatch_with_plan_retry(
    graph: &Arc<RwLock<Graph>>,
    cpu_target: NodeId,
    cache: StorageCache,
    selector: Option<Arc<dyn RuntimeSelector>>,
    pinned_loc: DeviceLocation,
    report_node: NodeId,
) -> Result<(Arc<RwLock<Storage>>, Option<&'static str>)> {
    let mut attempt = 0;
    // Hold a clone of `cache` outside the loop; the inner clone per-
    // attempt shares the Arcs (cheap) while keeping a fresh
    // structurally-empty layer for the executor to consume.
    loop {
        let plan = build_execution_plan(graph, &[cpu_target], pinned_loc, &cache)?;
        // Step 4a: commit the picker's winners to the graph BEFORE
        // residency stitching + ordering derivation read them.
        stamp_plan_backends(graph, &[cpu_target], &plan, pinned_loc)?;
        // Phase 2.1: materialize Op::Copy on cross-device edges now
        // that final placements are stamped. Idempotent on retry.
        insert_resident_input_copies(
            graph, &[cpu_target], &cache, pinned_loc, &plan,
        )?;
        // Phase 2.2: insert Op::Contiguize before any kernel whose
        // chosen winner doesn't advertise `KernelCaps::strided_input`
        // and whose live input layout is non-contiguous. CSE-deduped
        // across consumers; idempotent on retry.
        apply_layout_fixups(graph, &[cpu_target], &plan)?;
        let cache_for_attempt = cache.clone();
        let result = match selector.as_ref() {
            Some(sel) => PipelinedExecutor::realize_with_plan_and_selector(
                graph.clone(), cpu_target, cache_for_attempt,
                Arc::clone(&plan), Arc::clone(sel),
            ),
            None => PipelinedExecutor::realize_with_plan(
                graph.clone(), cpu_target, cache_for_attempt,
                Arc::clone(&plan),
            ),
        };
        match result {
            Ok((storage, _layout)) => {
                // Session 3 rider: report which sibling the picker
                // dispatched for `report_node`, from the SAME plan
                // (and selector) the successful attempt ran with.
                let dispatched = dispatched_kernel_source(
                    &plan, selector.as_deref(), report_node,
                );
                return Ok((storage, dispatched));
            }
            Err(e) if matches!(e, Error::TopologyChanged { .. })
                && attempt + 1 < MAX_PLAN_REBUILDS =>
            {
                topology_retry_backoff(attempt);
                attempt += 1;
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Replicate the executor's `resolve_compiled` pick for one node
/// against the plan that just dispatched: the runtime selector's
/// choice when one is active, the static `AlternativeSet::winner`
/// otherwise. `None` when the plan has no `AlternativeSet` for the
/// node — the executor then dispatched the first-registered binding
/// via its `compile_node` fallback.
///
/// Honesty note: a selector (e.g. JudgeAware) re-queries live state
/// per `select` call, so this post-realize re-query could in
/// principle diverge from the pick the executor's compile thread made
/// moments earlier — that requires a concurrent Judge-profile or
/// VRAM-pressure shift mid-realize. Acceptable for a diagnostic
/// attribution tag; if exactness ever matters, the stronger seam is
/// the executor recording its actual pick per node.
fn dispatched_kernel_source(
    plan: &ExecutionPlan,
    selector: Option<&dyn RuntimeSelector>,
    node: NodeId,
) -> Option<&'static str> {
    let set = plan.alternatives(node)?;
    let pick = match selector {
        Some(s) => s.select(set),
        None => set.winner(),
    }?;
    Some(pick.kernel_source)
}

/// Phase 2.2 wiring: insert `Op::Contiguize` before kernels whose
/// committed winner rejects strided inputs. The callback queries
/// the plan's per-node `AlternativeSet::winner()` for its caps.
/// Idempotent — safe to call on retry.
///
/// When a node has no plan entry (the picker skipped it because
/// `op_to_op_kind` returned `None`, typical for view ops + structural
/// ops), the callback returns `true` (= no fixup needed); the
/// executor's auto-Contiguize gate at execute time is the safety net
/// for these.
fn apply_layout_fixups(
    graph: &Arc<RwLock<Graph>>,
    roots: &[NodeId],
    plan: &Arc<ExecutionPlan>,
) -> Result<()> {
    let mut g = graph
        .write()
        .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
    insert_layout_fixups(&mut g, roots, |id| {
        plan.alternatives(id)
            .and_then(|set| set.winner())
            .map(|cand| cand.caps.strided_input)
            .unwrap_or(true)
    });
    Ok(())
}

/// Multi-target counterpart of [`realize_one_as_with_initial`].
pub fn realize_many_as_with_initial<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    device: &Device,
    initial: StorageCache,
) -> Result<Vec<Vec<T>>> {
    if targets.is_empty() {
        return Ok(Vec::new());
    }
    let (cache, _, effective_targets) = prepare(graph, targets, device, initial)?;
    // Phase 4.2 + 4.3 + 2.2 + picker-arc 4a: plan-aware dispatch
    // with topology-stale retry, post-plan winner stamping,
    // cross-device-copy stitching, and layout-fixup insertion. See
    // `dispatch_with_plan_retry` for the single-target version of
    // the loop; this multi-target sibling shares the same shape.
    // Picker-arc step 3: dispatch through the production runtime
    // selector (Picker 2).
    let selector = production_selector_for(device);
    let pinned_loc = device.location();
    let mut attempt = 0;
    let results = loop {
        let plan = build_execution_plan(graph, &effective_targets, pinned_loc, &cache)?;
        stamp_plan_backends(graph, &effective_targets, &plan, pinned_loc)?;
        insert_resident_input_copies(
            graph, &effective_targets, &cache, pinned_loc, &plan,
        )?;
        apply_layout_fixups(graph, &effective_targets, &plan)?;
        let cache_for_attempt = cache.clone();
        let result = match selector.as_ref() {
            Some(sel) => PipelinedExecutor::realize_many_with_plan_and_selector(
                graph.clone(), &effective_targets, cache_for_attempt, plan,
                Arc::clone(sel),
            ),
            None => PipelinedExecutor::realize_many_with_plan(
                graph.clone(), &effective_targets, cache_for_attempt, plan,
            ),
        };
        match result {
            Ok(r) => break r,
            Err(e) if matches!(e, Error::TopologyChanged { .. })
                && attempt + 1 < MAX_PLAN_REBUILDS =>
            {
                topology_retry_backoff(attempt);
                attempt += 1;
                continue;
            }
            Err(e) => return Err(e),
        }
    };
    let mut out = Vec::with_capacity(results.len());
    for (storage, _layout) in results {
        out.push(extract_cpu_bytes_typed::<T>(&storage)?);
    }
    Ok(out)
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
// Execution-plan build (Phase 4.2)
// ---------------------------------------------------------------------------

/// Build a topology-driven [`ExecutionPlan`] for the given realize
/// targets. The plan's per-node `AlternativeSet`s are populated
/// by [`fuel_dispatch::plan::compile_plan`] with placements derived
/// from [`SystemTopology::backends_for`] — every co-located backend
/// on each node's target device competes, with [`SystemTopology::capabilities`]
/// feeding the cost composer.
///
/// Phase 4.2 of the picker-work arc. The plan flows through to
/// `PipelinedExecutor::realize_with_plan` / `realize_many_with_plan`;
/// the executor's [`fuel_dispatch::pipelined`] dispatch reads the
/// per-node winner via [Phase 4.1's `resolve_compiled`].
///
/// Caller invariants:
///
/// - `prepare()` has already been called (const cache + realize-root
///   splices exist). Per-node `target_backend` is NOT required —
///   picker-arc step 4a: `compile_plan` resolves each node's
///   decision device from `pinned_device` (the realize call's
///   device) unless the node carries an explicit scheduler
///   placement.
/// - `targets` are the `effective_targets` returned by `prepare()`
///   — for non-CPU realizes that's the spliced `Op::Copy` nodes.
///
/// On every call the helper snapshots [`SystemTopology::current()`]
/// — generation-aware (cheap if nothing changed since last call,
/// rebuilds + atomically swaps on a probe / registration bump).
/// The snapshot is alive for the duration of `compile_plan`; closures
/// borrow it.
///
/// ## Planner Stage 2 (2026-06-11)
///
/// `cache` is `prepare()`'s StorageCache (const uploads + persistent
/// `initial` slots). It feeds `PlanOptions::with_input_residency` so
/// `compile_plan`'s residency walk prices already-resident inputs
/// from where their bytes actually live — the same
/// [`cached_storage_location`] derivation `effective_placements`
/// uses for the residency stitch. The topology snapshot doubles as
/// the `TransferEstimator` (`SystemTopology::estimate_transfer_ns`,
/// Stage-1 calibration behind it), making the inbound-transfer cost
/// term + the priced off-device admission live on every plan.
/// Single-device hosts see no change: zero transfer paths, zero
/// terms, and an empty `fallback_for` keeps every set on-device.
fn build_execution_plan(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    pinned_device: DeviceLocation,
    cache: &StorageCache,
) -> Result<Arc<ExecutionPlan>> {
    let topology = SystemTopology::current();

    // execution_plan respects both data-flow and ordering edges so
    // the plan's coverage matches the executor's walk exactly.
    let order = {
        let g = graph
            .read()
            .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
        execution_plan(&g, targets)
    };

    // Closures borrow the topology snapshot. `placements_for_device`
    // returns every backend co-located on the node's target device;
    // `capabilities_for` looks up that backend's BackendCapabilities
    // for the Layer-1 cost composer; `fallback_for` (picker-arc step
    // 4b, relaxed by planner Stage 2) enumerates every OTHER
    // device's backends. With the transfer estimator wired below,
    // those off-device candidates ALWAYS enumerate, priced by the
    // inbound-transfer term — locality emerges from the numbers
    // (explicitly placed nodes stay hard-pinned; destructive ops
    // never move). The cross-device-copy pass stitches residency
    // around any off-device winner.
    let placements_for = |dev: fuel_core_types::DeviceLocation|
        -> Vec<fuel_core_types::probe::BackendId>
    {
        topology.backends_for(dev).to_vec()
    };
    let capabilities_for = |b: fuel_core_types::probe::BackendId|
        -> Option<&fuel_core_types::backend::BackendCapabilities>
    {
        topology.capabilities(b)
    };
    let fallback_for = |dev: fuel_core_types::DeviceLocation|
        -> Vec<(fuel_core_types::probe::BackendId, fuel_core_types::DeviceLocation)>
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

    let g = graph
        .read()
        .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;

    // Layer-2 cost refinement (Phase 3 → production, 2026-06-10):
    // when the Judge has cached profile data — populated this
    // process or lazily loaded from a prior run's persisted report —
    // attach the oracle so `compile_plan`'s cost composer refines
    // Layer-1 static estimates with measured latencies per
    // `(op, dtype, size_class, backend, kernel_source)` cell. No
    // cached profile → `None` → pure Layer-1 ranking, identical to
    // the pre-oracle behavior. Cells the Judge never measured miss
    // inside the oracle and keep their Layer-1 estimate too.
    let judge_oracle = crate::judge::cached_oracle();

    // Planner Stage 2: residency for already-resident graph inputs
    // (consts uploaded by build_const_cache + InferenceContext
    // persistent slots) — where the bytes actually live, via the
    // same derivation `effective_placements` uses. Lock-poison /
    // unresolvable-location cases answer `None` (no pricing signal)
    // rather than erroring: planning degrades to unpriced edges,
    // and the catastrophic-poison case still surfaces on the
    // realize path proper.
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
        // Stage 2: the topology snapshot IS the production
        // TransferEstimator (see the trait impl in
        // `crate::topology`). Same-device queries cost zero without
        // touching the lazy calibration probe.
        .with_transfer_estimator(&*topology)
        .with_input_residency(&input_residency);
    if let Some(oracle) = judge_oracle.as_deref() {
        options = options.with_judge(oracle);
    }

    let bindings_guard = global_bindings();
    let plan = compile_plan(&g, &order, &bindings_guard, &options)?;
    Ok(Arc::new(plan))
}

// ---------------------------------------------------------------------------
// Production runtime selector (picker arc step 3)
// ---------------------------------------------------------------------------

/// Opt-out knob for the production runtime selector. Matches the
/// `FUEL_*` env-var convention (`FUEL_FORCE_F32`, `FUEL_TRACE`, ...):
/// set `FUEL_DISABLE_RUNTIME_SELECTOR=1` to fall back to the
/// selector-less plan path (per-node `AlternativeSet::winner`, the
/// pre-step-3 behavior). Default: selector ON.
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
///   `build_execution_plan` feeds `PlanOptions::with_judge`). The
///   selector re-queries at dispatch time, so measurements landing
///   after a plan freezes still influence the pick.
///
/// Both signals degrade honestly: no cached profile → static-cost
/// ranking; no runtime handle for a candidate → `FitStatus::Unknown`
/// (tier-equal with Comfortable). With neither signal the pick is
/// exactly `AlternativeSet::winner()` on every plan-produced set —
/// today's behavior.
///
/// Returns `None` when [`runtime_selector_disabled`] — the dispatch
/// loops then call the selector-less `realize_with_plan` APIs.
fn production_selector_for(device: &Device) -> Option<Arc<dyn RuntimeSelector>> {
    if runtime_selector_disabled() {
        return None;
    }
    let judge: Option<Arc<dyn JudgeOracle>> = crate::judge::cached_oracle()
        .map(|oracle| oracle as Arc<dyn JudgeOracle>);
    let lookup = backend_runtime_lookup_for(device);
    Some(Arc::new(ChainedSelector::with_default_estimator(
        judge,
        Some(lookup),
    )))
}

/// [`BackendRuntime`] adapter over a live device handle. Owns the
/// `Arc<dyn DynBackendDevice>` so the boxed handle the selector
/// borrows stays valid for the duration of a `select` call; every
/// query delegates through
/// [`DynBackendDevice::as_backend_runtime`]. Devices without a
/// runtime impl (Metal today) answer `None` / `Unknown` — honest
/// no-signal, never fabricated pressure.
struct DeviceRuntimeHandle(Arc<dyn DynBackendDevice>);

impl BackendRuntime for DeviceRuntimeHandle {
    fn available_bytes(&self) -> Option<u64> {
        self.0.as_backend_runtime().and_then(|r| r.available_bytes())
    }

    fn total_bytes(&self) -> Option<u64> {
        self.0.as_backend_runtime().and_then(|r| r.total_bytes())
    }

    // Delegate rather than re-derive: backends may override
    // `would_fit` with native predictive checks (e.g. Vulkan's
    // VK_EXT_memory_budget fragmentation awareness).
    fn would_fit(&self, size: u64) -> FitStatus {
        match self.0.as_backend_runtime() {
            Some(r) => r.would_fit(size),
            None => FitStatus::Unknown,
        }
    }
}

/// Live-handle lookup for the VramPressure guard. Resolves:
///
/// - the realize device's `(backend, location)` → the realize
///   `Device`'s own handle (the only live GPU handle the bridge
///   holds — with today's monolithic pinning every GPU candidate in
///   the plan targets exactly this device);
/// - `(Cpu, DeviceLocation::Cpu)` → a fresh stateless
///   [`fuel_cpu_backend::dyn_impl::CpuBackendDevice`] (covers the
///   realize-root D2H copies + CPU candidates when realizing on a
///   GPU);
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
/// of this function — the dispatch loops stamp each node's plan
/// winner via [`stamp_plan_backends`] after `compile_plan` runs.
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
            .map(|&src_id| {
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
    // device (`PlanOptions::with_pinned_device`) and
    // `stamp_plan_backends` commits each node's winner AFTER
    // planning, before `insert_resident_input_copies` (so the
    // residency stitch sees final placements) and before the
    // executor derives its ordering.
    //
    // The pre-4a monolithic loop's "always overwrite" doctrine
    // survives in `stamp_plan_backends`: graphs are shared
    // (`Arc<RwLock<Graph>>`) and a single graph may be realized on
    // multiple devices across a session, so every realize call
    // re-stamps from its own plan rather than trusting stale pins.
    Ok((cache, backend_id, effective_targets))
}

/// Picker-arc step 4a: commit the plan's per-node winners to the
/// graph's `target_backend` side-table. Runs AFTER `compile_plan`
/// (the winners exist) and BEFORE [`insert_resident_input_copies`]
/// + the executor's ordering derivation (both read the stamps).
///
/// Per computational node (the same skip-set the pre-4a monolithic
/// loop used — `Op::Const` / `Op::Release` / `Op::Contiguize` /
/// view ops / `Op::Reshape` inherit or don't need a stamp):
///
/// - **Plan winner present** → stamp `winner.backend`. Under the
///   locality policy every winner sits on the pinned device, so on
///   a single-backend system this is byte-identical to the old
///   monolithic pin; off-device fallback winners (step 4b) stamp
///   their owning backend so the executor allocates output on the
///   backend that actually runs the kernel.
/// - **No plan entry** (structural ops the planner skips —
///   `Op::Copy` / `Op::Move` / `Op::Alloc` / `Op::ZeroFill` — plus
///   ops without an OpKind mapping) → stamp the pinned device's
///   backend, exactly like the old loop. `Op::Copy` / `Op::Move`
///   stamps are subsequently corrected to the SOURCE backend by
///   [`insert_resident_input_copies`]'s re-stamp sweep where the
///   source's placement resolves.
///
/// Always overwrites (see `prepare()` doc) — re-planning after a
/// `TopologyChanged` retry re-stamps consistently from the fresh
/// plan.
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
/// device-resident `fuel_storage::Storage` keyed in a StorageCache by
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
    let consts_to_upload: Vec<(NodeId, Vec<u8>, fuel_core_types::DType)> = {
        let g = graph
            .read()
            .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
        let mut out: Vec<(NodeId, Vec<u8>, fuel_core_types::DType)> =
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
                shape: fuel_core_types::Shape::from_dims(&[4]),
                dtype: fuel_core_types::DType::U8,
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
            let shape = fuel_core_types::Shape::from_dims(&[n_elem]);
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

/// Compute every reachable node's *effective placement* for the
/// cross-device-copy pass. Resolution priority per node:
///
/// 1. **Residency-declaring ops** — `Op::Copy` / `Op::Move` /
///    `Op::Alloc` carry their output's location in the op variant;
///    that's definitional.
/// 2. **Explicit `Graph::placement`** — set by
///    [`insert_cross_device_copies`] on the copies it inserts (and,
///    later in the picker arc, by per-node picker placements).
/// 3. **Cache-resident storages** — consts (uploaded to the realize
///    device by [`build_const_cache`]) and persistent `initial`
///    slots (InferenceContext): effective placement is where the
///    bytes actually live. Authoritative even when residency can't
///    be determined (no fall-through to the pin — a cached slot is
///    never recomputed on the pinned backend).
/// 4. **Plan winner** (picker-arc step 4a) — a node the picker
///    planned runs on its winner's device. Under the locality
///    policy that's the pinned device; off-device fallback winners
///    (step 4b) place the node where the kernel actually runs so
///    this pass stitches the crossing.
/// 5. **Backend stamp** — a node with `target_backend` set but no
///    plan entry (structural ops, plus stamps left by
///    [`stamp_plan_backends`]'s no-entry arm) follows the realize
///    device.
/// 6. **Residency-inheriting pass-throughs** — view ops, `Reshape`,
///    and `Contiguize` carry no pin and produce no new residency;
///    they follow their data input (already resolved — `order` is
///    topological).
///
/// Nodes matching none of the above stay absent; the pass leaves
/// their edges alone.
fn effective_placements(
    g: &Graph,
    order: &[NodeId],
    cache: &StorageCache,
    pinned_loc: DeviceLocation,
    plan: &ExecutionPlan,
) -> Result<HashMap<NodeId, DeviceLocation>> {
    let mut map: HashMap<NodeId, DeviceLocation> =
        HashMap::with_capacity(order.len());
    for &id in order {
        let node = g.node(id);
        match node.op {
            Op::Copy { target } | Op::Move { target } | Op::Alloc { target } => {
                map.insert(id, target);
                continue;
            }
            _ => {}
        }
        if let Some(loc) = g.placement(id) {
            map.insert(id, loc);
            continue;
        }
        if let Some(slot) = cache.get(&id) {
            let guard = slot
                .read()
                .map_err(|_| Error::Msg("storage lock poisoned".into()).bt())?;
            if let Some(loc) = cached_storage_location(&guard) {
                map.insert(id, loc);
            }
            continue;
        }
        if let Some(winner) = plan.alternatives(id).and_then(|set| set.winner()) {
            map.insert(id, winner.device);
            continue;
        }
        if g.target_backend(id).is_some() {
            map.insert(id, pinned_loc);
            continue;
        }
        if node.op.is_view_op()
            || matches!(node.op, Op::Reshape(_) | Op::Contiguize)
        {
            if let Some(&loc) = node.inputs.first().and_then(|i| map.get(i)) {
                map.insert(id, loc);
            }
        }
    }
    Ok(map)
}

/// Phase 2.1 wire-in: insert `Op::Copy { target }` on every graph
/// edge whose producer's resident location doesn't share a storage
/// substrate with the consumer's placement, then stamp each inserted
/// copy's `target_backend` with the SOURCE backend (the pipelined
/// executor's Op::Copy convention — the transfer kernel runs on the
/// backend the bytes come FROM: `copy_from_cpu_wrapper` for H2D,
/// the source backend's download wrapper for D2H).
///
/// ## Ownership split vs. the bridge's other transfer mechanisms
///
/// Three owners coexist, each covering a disjoint class of edges:
///
/// - **Realize-root splice** (`prepare()` step 1): D2H for the FINAL
///   outputs. Those `Op::Copy` nodes are consumers this pass skips
///   (`Op::Copy`/`Op::Move` consumers are never re-wrapped), so no
///   double-insertion is possible.
/// - **[`build_const_cache`]** (Phase 3b): HOST→device
///   materialization of graph-owned consts (from
///   `graph.storage_map`), re-derived per realize call so the same
///   graph can be realized on different devices across a session.
///   This pass does NOT subsume it: const upload happens through a
///   transient graph precisely so the upload target can follow the
///   per-call pin; in-graph copies would be sticky (graph rewrites
///   survive the call) and would pin the upload target of the first
///   realize forever. Because `build_const_cache` runs first, those
///   consts are already co-located and this pass proves no-op on
///   their edges.
/// - **This pass**: cross-device edges among ALREADY-RESIDENT
///   storages — today exactly the `initial`-cache slots
///   (InferenceContext persistent storages) living on a different
///   device than the pinned backend. Previously nothing handled
///   these (the executor dispatched the consumer's kernel against a
///   foreign-device input).
///
/// ## Re-stamping on repeat realizes (and after `stamp_plan_backends`)
///
/// Graph rewrites are sticky and [`stamp_plan_backends`] overwrites
/// `target_backend` on every computational node — including copies
/// this pass inserted on a PREVIOUS realize call (whose correct
/// stamp is the source backend, not the realize backend). This
/// helper therefore runs after the stamping pass and re-stamps every
/// `Op::Copy` / `Op::Move` whose source location resolves with the
/// SOURCE's backend — the transfer kernel runs where the bytes come
/// from (Move dispatches the same `OpKind::Copy` kernel; only its
/// destructive-release bookkeeping differs).
/// That covers placement-carrying copies (this pass's own inserts)
/// AND placement-less copies (realize-root splices, safety copies):
/// under the locality policy a root splice's source resolves to the
/// pinned device, reproducing the old "follow the realize device"
/// stamp; when an off-device fallback winner (picker-arc step 4b)
/// produces the root's input, the source-backend rule stamps the
/// download kernel onto the backend that actually owns the bytes.
/// Copies whose source doesn't resolve keep the stamping pass's
/// pinned-backend stamp.
///
/// Returns the number of copies inserted (0 on any single-device
/// graph).
fn insert_resident_input_copies(
    graph: &Arc<RwLock<Graph>>,
    roots: &[NodeId],
    cache: &StorageCache,
    pinned_loc: DeviceLocation,
    plan: &ExecutionPlan,
) -> Result<usize> {
    let topology = SystemTopology::current();
    let mut g = graph
        .write()
        .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
    let order = topo_order_multi(&g, roots);
    let placements = effective_placements(&g, &order, cache, pinned_loc, plan)?;

    // Substrate query routed through SystemTopology. Identical
    // locations short-circuit to `true` BEFORE the topology lookup:
    // a location trivially shares bytes with itself, and
    // `SystemTopology::shares_storage` returns `false` for unknown
    // backends — without the short-circuit an unprobed topology
    // would wrap every same-device edge in a copy.
    let shares = |a: DeviceLocation, b: DeviceLocation| -> bool {
        if a == b {
            return true;
        }
        topology.shares_storage(
            (location_to_backend_id(a), a),
            (location_to_backend_id(b), b),
        )
    };

    let inserted = insert_cross_device_copies(
        &mut g,
        roots,
        |id| placements.get(&id).copied(),
        shares,
    );

    // Stamp the new copies: target_backend = SOURCE backend. The
    // pass only inserts a copy when the producer's placement
    // resolved to Some, so the lookup can't miss.
    for &copy_id in &inserted {
        let src = g.node(copy_id).inputs.first().copied().ok_or_else(|| {
            Error::Msg(format!(
                "insert_resident_input_copies: inserted Op::Copy {copy_id:?} \
                 has no input — fuel_graph::opt::insert_cross_device_copies \
                 broke its single-input invariant",
            ))
            .bt()
        })?;
        let src_loc = placements.get(&src).copied().ok_or_else(|| {
            Error::Msg(format!(
                "insert_resident_input_copies: inserted Op::Copy {copy_id:?} \
                 wraps producer {src:?} with no resolved placement — the \
                 pass should only fire on placed producers",
            ))
            .bt()
        })?;
        g.set_target_backend(copy_id, location_to_backend_id(src_loc));
    }

    // Re-stamp pre-existing copies/moves with their SOURCE backend
    // (see doc comment) — placement-carrying copies from previous
    // realize calls AND placement-less realize-root splices whose
    // source resolves. `order` predates the insertions above, so this
    // never touches the freshly stamped nodes.
    for &id in &order {
        if !matches!(g.node(id).op, Op::Copy { .. } | Op::Move { .. }) {
            continue;
        }
        let Some(&src) = g.node(id).inputs.first() else { continue };
        let Some(&src_loc) = placements.get(&src) else { continue };
        g.set_target_backend(id, location_to_backend_id(src_loc));
    }

    Ok(inserted.len())
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
            Ok(Some(Storage::new(BackendStorage::Cuda(cuda_bytes), fuel_core_types::DType::U8)))
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
            Ok(Some(Storage::new(BackendStorage::Vulkan(vk_bytes), fuel_core_types::DType::U8)))
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
    use fuel_core_types::{DType, Shape};

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

    /// Regression guard for the Phase 2.1 wire-in's no-op guarantee:
    /// when every resident input lives on the same device the nodes
    /// are pinned to, the pass mutates NOTHING — no new nodes, no
    /// rewired edges. This is the shape of every single-device
    /// realize under today's monolithic pinning (build_const_cache
    /// uploads consts to the realize device before the pass runs).
    #[test]
    fn resident_copies_noop_when_colocated() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, c2, add) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let c2 = push_node(&mut g, Op::Const, vec![]);
            let add = push_node(&mut g, Op::Add, vec![c1, c2]);
            // Mimic prepare()'s monolithic pinning loop.
            g.set_target_backend(add, BackendId::Cpu);
            (c1, c2, add)
        };
        let mut cache = StorageCache::new();
        cache.insert(c1, cpu_storage_f32(&[1.0; 4]));
        cache.insert(c2, cpu_storage_f32(&[2.0; 4]));

        let pre_len = graph.read().unwrap().len();
        let inserted = insert_resident_input_copies(
            &graph, &[add], &cache, DeviceLocation::Cpu, &ExecutionPlan::empty(),
        )
        .unwrap();

        assert_eq!(inserted, 0, "co-located graph must be a no-op");
        let g = graph.read().unwrap();
        assert_eq!(g.len(), pre_len, "no nodes appended");
        assert_eq!(g.node(add).inputs, vec![c1, c2], "edges untouched");
    }

    /// Positive case: a persistent input (initial-cache slot) is
    /// CPU-resident while the consumers are pinned to CUDA. Exactly
    /// ONE Op::Copy bridges the crossing, CSE-deduped across both
    /// consumers, stamped target_backend = SOURCE backend (Cpu — the
    /// H2D `copy_from_cpu_wrapper` registration) with its output
    /// placement on the consumer device. Placement metadata only —
    /// no GPU needed.
    #[test]
    fn resident_copies_one_copy_per_crossing_deduped() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, neg, sqr) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let neg = push_node(&mut g, Op::Neg, vec![c1]);
            let sqr = push_node(&mut g, Op::Sqr, vec![c1]);
            g.set_target_backend(neg, BackendId::Cuda);
            g.set_target_backend(sqr, BackendId::Cuda);
            (c1, neg, sqr)
        };
        // Persistent slot resident on CPU (the InferenceContext
        // `initial` pattern — build_const_cache skips slots already
        // in the cache, so they keep their residency).
        let mut cache = StorageCache::new();
        cache.insert(c1, cpu_storage_f32(&[1.0; 4]));

        let pre_len = graph.read().unwrap().len();
        let inserted = insert_resident_input_copies(
            &graph, &[neg, sqr], &cache, cuda0, &ExecutionPlan::empty(),
        )
        .unwrap();

        assert_eq!(inserted, 1, "one crossing → one copy, CSE-deduped");
        let g = graph.read().unwrap();
        assert_eq!(g.len(), pre_len + 1);
        let neg_in = g.node(neg).inputs[0];
        let sqr_in = g.node(sqr).inputs[0];
        assert_eq!(neg_in, sqr_in, "both consumers share the one copy");
        assert_ne!(neg_in, c1, "consumers were rewired off the raw input");
        let copy_node = g.node(neg_in);
        assert!(
            matches!(copy_node.op, Op::Copy { target } if target == cuda0),
            "bridge copy targets the consumer device; got {:?}",
            copy_node.op,
        );
        assert_eq!(copy_node.inputs, vec![c1], "copy reads the resident slot");
        assert_eq!(
            g.target_backend(neg_in),
            Some(BackendId::Cpu),
            "stamped with the SOURCE backend (H2D runs on the CPU wrapper)",
        );
        assert_eq!(
            g.placement(neg_in),
            Some(cuda0),
            "copy output placed on the consumer device",
        );
    }

    /// Sticky-graph idempotence: a second prepare()-shaped call on
    /// the already-rewritten graph (same device) inserts nothing and
    /// keeps the source-backend stamp intact — including across the
    /// monolithic pinning loop's overwrite, which the re-stamp sweep
    /// corrects.
    #[test]
    fn resident_copies_idempotent_and_restamped_on_second_call() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, neg) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let neg = push_node(&mut g, Op::Neg, vec![c1]);
            g.set_target_backend(neg, BackendId::Cuda);
            (c1, neg)
        };
        let mut cache = StorageCache::new();
        cache.insert(c1, cpu_storage_f32(&[1.0; 4]));

        let first = insert_resident_input_copies(
            &graph, &[neg], &cache, cuda0, &ExecutionPlan::empty(),
        )
        .unwrap();
        assert_eq!(first, 1);
        let copy_id = graph.read().unwrap().node(neg).inputs[0];

        // Second realize on the same graph: the monolithic loop
        // re-stamps every computational node — including the copy —
        // with the realize backend. Simulate that clobber, then
        // verify the pass both proves no-op AND restores the
        // source-backend stamp.
        {
            let mut g = graph.write().unwrap();
            g.set_target_backend(copy_id, BackendId::Cuda);
        }
        let pre_len = graph.read().unwrap().len();
        let second = insert_resident_input_copies(
            &graph, &[neg], &cache, cuda0, &ExecutionPlan::empty(),
        )
        .unwrap();
        assert_eq!(second, 0, "re-run inserts nothing");
        let g = graph.read().unwrap();
        assert_eq!(g.len(), pre_len, "no nodes appended on re-run");
        assert_eq!(
            g.target_backend(copy_id),
            Some(BackendId::Cpu),
            "re-stamp sweep restores the source-backend stamp after \
             the monolithic loop clobbered it",
        );
    }

    fn noop_kernel_for_tests(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[fuel_core_types::Layout],
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
        use fuel_core_types::dispatch::OpKind;
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
            -> Option<&fuel_core_types::backend::BackendCapabilities>
        { topology.capabilities(b) };
        // Same closure shape build_execution_plan wires.
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

    /// Picker-arc step 3 end-to-end: a CPU realize through
    /// `realize_one_as_with_initial` runs the production
    /// `ChainedSelector` path (`realize_with_plan_and_selector`,
    /// default ON) and produces correct bytes. With no pressure
    /// signal beyond Comfortable/Unknown and CPU-only candidates,
    /// the chained selector degenerates to the static winner —
    /// pinning the no-signal-parity guarantee at the bridge level.
    #[test]
    fn production_selector_cpu_realize_end_to_end() {
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
        assert!(
            production_selector_for(&device).is_some(),
            "production selector defaults ON (no opt-out env set)",
        );

        let out = realize_one_as_with_initial::<f32>(&graph, add, &device, initial)
            .expect("realize through the production selector path");
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
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
        use fuel_core_types::dispatch::OpKind;

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
            realize_one_as_with_initial_reporting::<f32>(&graph, add, &device, initial)
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
        _layouts: &[fuel_core_types::Layout],
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
        use fuel_core_types::dispatch::OpKind;
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
        let inserted = insert_resident_input_copies(
            &graph, &[neg], &cache, cuda0, &plan,
        )
        .unwrap();
        assert_eq!(
            inserted, 0,
            "const lives on CPU; the fallback node runs on CPU — no crossing",
        );

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

    /// Step 4b residency stitch: when an off-device fallback winner
    /// feeds a consumer planned on the pinned device, the
    /// cross-device-copy pass inserts exactly one Op::Copy on the
    /// crossing, targeting the consumer device and stamped with the
    /// SOURCE backend. Graph-rewrite assertions only — no GPU.
    #[test]
    fn fallback_winner_crossing_gets_copy_stitched() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, neg, sqr) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let neg = push_node(&mut g, Op::Neg, vec![c1]);
            let sqr = push_node(&mut g, Op::Sqr, vec![neg]);
            (c1, neg, sqr)
        };
        let mut cache = StorageCache::new();
        cache.insert(c1, cpu_storage_f32(&[1.0; 4]));

        // Neg fell back to CPU; Sqr stays on the pinned CUDA device.
        let plan = plan_with_winners(&[
            (neg, BackendId::Cpu, DeviceLocation::Cpu),
            (sqr, BackendId::Cuda, cuda0),
        ]);
        stamp_plan_backends(&graph, &[sqr], &plan, cuda0).unwrap();
        let inserted = insert_resident_input_copies(
            &graph, &[sqr], &cache, cuda0, &plan,
        )
        .unwrap();
        assert_eq!(inserted, 1, "one crossing (neg→sqr) → one copy");

        let g = graph.read().unwrap();
        let bridge_id = g.node(sqr).inputs[0];
        assert_ne!(bridge_id, neg, "sqr rewired off the raw fallback output");
        let bridge = g.node(bridge_id);
        assert!(
            matches!(bridge.op, Op::Copy { target } if target == cuda0),
            "bridge copy targets the consumer device; got {:?}",
            bridge.op,
        );
        assert_eq!(bridge.inputs, vec![neg], "copy reads the fallback output");
        assert_eq!(
            g.target_backend(bridge_id),
            Some(BackendId::Cpu),
            "copy stamped with the SOURCE backend (bytes live on CPU)",
        );
        assert_eq!(
            g.target_backend(neg),
            Some(BackendId::Cpu),
            "fallback node keeps its off-device stamp",
        );
        assert_eq!(
            g.target_backend(sqr),
            Some(BackendId::Cuda),
            "pinned-device consumer keeps its winner stamp",
        );
    }

    /// Step 4b realize-root coherence: a root Op::Copy (placement-
    /// less splice) whose input fell back off-device gets re-stamped
    /// with the SOURCE backend by the residency pass — the download
    /// kernel must run where the bytes actually live, not on the
    /// pinned device.
    #[test]
    fn realize_root_copy_restamped_to_fallback_source() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, neg, root_copy) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let neg = push_node(&mut g, Op::Neg, vec![c1]);
            // Mimic prepare()'s realize-root splice (no placement).
            let root_copy = push_node(
                &mut g,
                Op::Copy { target: DeviceLocation::Cpu },
                vec![neg],
            );
            (c1, neg, root_copy)
        };
        let mut cache = StorageCache::new();
        cache.insert(c1, cpu_storage_f32(&[1.0; 4]));

        let plan = plan_with_winners(&[(neg, BackendId::Cpu, DeviceLocation::Cpu)]);
        stamp_plan_backends(&graph, &[root_copy], &plan, cuda0).unwrap();
        assert_eq!(
            graph.read().unwrap().target_backend(root_copy),
            Some(BackendId::Cuda),
            "stamping pass defaults the planless root copy to the pinned backend",
        );

        let inserted = insert_resident_input_copies(
            &graph, &[root_copy], &cache, cuda0, &plan,
        )
        .unwrap();
        assert_eq!(inserted, 0, "Op::Copy consumers are never re-wrapped");
        assert_eq!(
            graph.read().unwrap().target_backend(root_copy),
            Some(BackendId::Cpu),
            "re-stamp sweep moves the root copy onto the SOURCE backend \
             (its input fell back to CPU)",
        );
    }

    /// The live-handle lookup resolves the realize device + CPU and
    /// answers `None` (no signal) for everything else. On this host
    /// the CPU handle reports real memory numbers on Windows/Linux;
    /// we only assert the wiring (a handle exists and `would_fit`
    /// answers without panicking), not the OS-specific values.
    #[test]
    fn backend_runtime_lookup_resolves_cpu_and_misses_others() {
        let device = crate::Device::cpu();
        let lookup = backend_runtime_lookup_for(&device);

        let cpu = lookup(BackendId::Cpu, DeviceLocation::Cpu)
            .expect("CPU handle always resolvable");
        // Any FitStatus is acceptable — platform-dependent signal —
        // the call itself must succeed.
        let _ = cpu.would_fit(1);

        assert!(
            lookup(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }).is_none(),
            "no live handle for a backend that isn't the realize device",
        );
    }
}
