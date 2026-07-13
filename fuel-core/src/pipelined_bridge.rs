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
use fuel_backend_contract::backend::{BackendRuntime, BackendStreams};
use fuel_ir::backend::FitStatus;
use fuel_backend_contract::dyn_backend::DynBackendDevice;
use fuel_graph::{Graph, Node, NodeId, Op, topo_order_multi};
use fuel_dispatch::dispatch::global_bindings;
use fuel_dispatch::dispatched_kernel_source;
use fuel_dispatch::optimize::{optimize_graph_with_runtime_fusion, OptimizedGraph};
use fuel_dispatch::plan::PlanOptions;
use fuel_dispatch::pipelined::{PipelinedExecutor, StorageCache};
use fuel_dispatch::ranker::{
    BackendRuntimeHandle, BackendRuntimeLookup, ChainedSelector, JudgeOracle,
    RuntimeSelector,
};
use fuel_memory::{BackendStorage, Storage};

use crate::Device;
use crate::topology::SystemTopology;

// ---------------------------------------------------------------------------
// Optimize-call telemetry (Phase D · D2a)
// ---------------------------------------------------------------------------

/// Process-global count of real `optimize_graph` invocations, bumped once
/// per [`build_optimized_graph`] call (the expensive placement DP + cost
/// composer + Judge + residency/layout mutation). Mirrors the B1 in-flight
/// counter idiom: a plain `AtomicUsize` with `Relaxed` ordering.
///
/// This is a **test/telemetry observation**, never a correctness gate. The
/// D2a optimize-skip born-red test reads it to prove the prebuilt realize
/// path (`realize_one_prebuilt_env`) does NOT re-run the optimizer on a
/// held, already-optimized graph. `TopologyChanged` retries legitimately
/// bump it (each retry re-optimizes against the fresh topology).
static OPTIMIZE_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

std::thread_local! {
    /// Per-thread mirror of [`OPTIMIZE_CALLS`], bumped in the SAME spot.
    /// `build_optimized_graph` always runs on the realize CALLER's thread
    /// (the optimize takes the graph write-lock synchronously, BEFORE the
    /// executor spawns its compiler thread), so a thread-local count
    /// isolates one realize sequence from every OTHER test thread's
    /// concurrent optimizes — which a single process-global counter cannot.
    /// The D2a born-red test reads the thread-local delta so it is robust
    /// under the full concurrent suite; the process-global
    /// [`optimize_calls`] stays the coarse telemetry surface.
    static OPTIMIZE_CALLS_TL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Read the process-global `optimize_graph` invocation count (D2a
/// telemetry — see [`OPTIMIZE_CALLS`]). Monotonically non-decreasing across
/// the process lifetime; process-wide, so it is polluted by concurrent test
/// threads — use [`optimize_calls_thread_local`] for a per-thread delta.
pub fn optimize_calls() -> usize {
    OPTIMIZE_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Read THIS thread's `optimize_graph` invocation count (see
/// [`OPTIMIZE_CALLS_TL`]). Robust for a single-threaded realize sequence
/// even while other threads optimize concurrently — the D2a optimize-skip
/// assertion measures a delta on this reader.
pub fn optimize_calls_thread_local() -> usize {
    OPTIMIZE_CALLS_TL.with(|c| c.get())
}

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
///    passes (cleanup Step B), consuming its internal `ExecutionPlan` and
///    returning only the `OptimizedGraph` view (cleanup Step D). The bridge
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

/// Multi-DEVICE realize: like [`realize_one_as`], but additionally seeds
/// the executor's input cache with a device-handle anchor for each device
/// in `extra_devices` — the backends that appear in the graph's per-node
/// placements OTHER than the primary realize `device`.
///
/// # Why this exists (the dual-device-seed gap)
///
/// A realize pins ONE `device` (the primary). [`build_const_cache`] uploads
/// the reachable `Op::Const`s to that device, so a primary-device handle
/// lands in the cache (carried by the uploaded const storages) and the
/// executor's H2D `Op::Copy`/`Op::Alloc` device-handle search
/// (`find_cuda_device_in_cache` / `find_vulkan_backend_in_cache`) succeeds
/// for the primary backend. But a genuinely multi-VENDOR graph (e.g. a
/// sub-DAG placed on CUDA reconverging with one placed on Vulkan) also has
/// H2D copies targeting the OTHER backend — and NO storage on that backend
/// is in the cache, so the handle search fails with "no storage in input
/// cache to derive the handle".
///
/// This entry closes that: for each `extra_devices` handle it pushes a tiny
/// `Op::Const` anchor node into the graph and inserts that backend's
/// 0-byte [`device_seed_storage`] at the anchor's NodeId. The anchor is
/// unreachable from `target`, so it is never dispatched (it does not appear
/// in the executor's run order) — it exists ONLY so the per-backend handle
/// search resolves. The anchor IS a cache key, so the executor's
/// `layout_cache` seeding (`g.layout(id)` over `inputs.keys()`) requires it
/// to be a real graph node — hence the push rather than a synthetic id.
///
/// Single-device realize is unaffected: pass an empty `extra_devices` and
/// this is byte-identical to [`realize_one_as`] (no anchors pushed, the
/// same `StorageCache::new()` initial).
///
/// CPU appearing among the extra devices is a no-op (CPU's allocation path
/// is handle-free; [`device_seed_storage`] returns `Ok(None)` for it).
pub fn realize_one_as_multi_device<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    target: NodeId,
    device: &Device,
    extra_devices: &[&Device],
) -> Result<Vec<T>> {
    let initial = seed_extra_device_handles(graph, device, extra_devices)?;
    realize_one_as_with_initial::<T>(graph, target, device, initial)
}

/// Build the `initial` [`StorageCache`] for a multi-device realize: one
/// device-handle anchor per `extra_devices` entry whose backend differs
/// from the primary `device`'s backend (and isn't CPU). See
/// [`realize_one_as_multi_device`] for the rationale.
///
/// Each anchor is a fresh `Op::Const` node pushed into `graph`, paired
/// with the backend's [`device_seed_storage`] in the returned cache. The
/// primary backend is skipped — the const upload already seeds it.
fn seed_extra_device_handles(
    graph: &Arc<RwLock<Graph>>,
    primary: &Device,
    extra_devices: &[&Device],
) -> Result<StorageCache> {
    let mut cache = StorageCache::new();
    if extra_devices.is_empty() {
        return Ok(cache);
    }
    let primary_backend = device_to_backend_id(primary);
    // De-dup: seed each distinct extra backend once.
    let mut seeded: Vec<BackendId> = vec![primary_backend];
    for dev in extra_devices {
        let backend = device_to_backend_id(dev);
        if seeded.contains(&backend) {
            continue;
        }
        // device_seed_storage returns Ok(None) for CPU (no handle anchor
        // needed) and a 0/4-byte device storage for GPU backends.
        let Some(seed) = device_seed_storage(dev)? else {
            seeded.push(backend);
            continue;
        };
        let anchor_id = {
            let mut g = graph
                .write()
                .map_err(|_| Error::Msg("graph lock poisoned during device-handle seed".into()).bt())?;
            g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: fuel_ir::Shape::from_dims(&[4]),
                dtype: fuel_ir::DType::U8,
            })
        };
        cache.insert(anchor_id, Arc::new(RwLock::new(seed)));
        seeded.push(backend);
    }
    Ok(cache)
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
    // Production realize: cost-based cross-device auto-placement enabled.
    realize_one_as_reporting_impl::<T>(graph, target, device, initial, sym_env, true)
}

/// Reference realize: single-device, cost-based cross-device placement
/// **suppressed** (`allow_cost_placement = false`). Every un-pinned node
/// stays on `device`, so a CPU-pinned call is a genuine all-CPU oracle —
/// never cost-relocated onto the very backend it's meant to validate. See
/// [`fuel_dispatch::plan::PlanOptions::allow_cost_placement`]. Backs
/// [`crate::lazy::LazyTensor::realize_f32_reference`] + the per-op parity
/// harness; the replacement for the retiring `fuel-reference-backend` oracle.
pub fn realize_one_reference_as<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    target: NodeId,
    device: &Device,
) -> Result<Vec<T>> {
    realize_one_as_reporting_impl::<T>(
        graph, target, device, StorageCache::new(), &SymEnv::default(), false,
    )
    .map(|(bytes, _root_kernel_source)| bytes)
}

fn realize_one_as_reporting_impl<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    target: NodeId,
    device: &Device,
    initial: StorageCache,
    sym_env: &SymEnv,
    allow_cost_placement: bool,
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
        graph, cpu_target, cache, device, target, sym_env, allow_cost_placement,
    )?;
    Ok((extract_cpu_bytes_typed::<T>(&storage)?, root_kernel_source))
}

// ---------------------------------------------------------------------------
// Phase D · D2a — the optimize-skip bridge seam (plan-once persistent decode)
// ---------------------------------------------------------------------------

/// First-realize sibling of [`realize_one_as_with_initial_env`] that ALSO
/// returns the reusable optimize artifacts so a caller (D2b's
/// `DecodeSession`) can re-realize the SAME graph on later tokens WITHOUT
/// re-paying `prepare`'s D2H-splice/const-cache or `build_optimized_graph`'s
/// placement DP.
///
/// Runs the normal path ONCE — `prepare` (splices the D2H `Op::Copy` at the
/// root + builds the const cache) then `optimize_graph` in place (stamps +
/// residency/layout fixups) then dispatch — and returns
/// `(effective_target, OptimizedGraph, result)`:
/// - `effective_target` — the D2H `Op::Copy` NodeId `prepare` spliced (the
///   node the executor was asked for; stable across tokens because the graph
///   structure is stable per D1).
/// - `OptimizedGraph` — the cached `{roots, generation}` view. It bakes NO
///   Const data / storage / `SymEnv`; the durable optimization output lives
///   in the (now-mutated) graph. Sound to reuse **iff the graph structure +
///   topology generation are unchanged** — exactly D1's guarantee.
/// - `result` — the first token's host bytes (so the caller can byte-compare
///   later prebuilt realizes against it).
///
/// Byte-identical to `realize_one_as_with_initial_env` for the value it
/// produces — it IS that path, just additionally surfacing the optimize view
/// and spliced root. `OPTIMIZE_CALLS` bumps exactly once here (plus once per
/// `TopologyChanged` retry, as on the normal path).
pub fn prebuild_optimized_env<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    target: NodeId,
    device: &Device,
    initial: StorageCache,
    sym_env: &SymEnv,
) -> Result<(NodeId, OptimizedGraph, Vec<T>)> {
    let (cache, _backend_id, mut effective_targets) =
        prepare(graph, &[target], device, initial)?;
    let Some(cpu_target) = effective_targets.pop() else {
        return Err(Error::Msg(
            "pipelined_bridge: prepare returned no effective target for a \
             single-target prebuild — internal bug"
                .into(),
        )
        .bt());
    };
    let (storage, optimized, _full_cache) = dispatch_with_plan_retry_capturing(
        graph, cpu_target, cache, device, sym_env,
    )?;
    let bytes = extract_cpu_bytes_typed::<T>(&storage)?;
    Ok((cpu_target, optimized, bytes))
}

/// Phase D · D2b — like [`prebuild_optimized_env`] but ALSO returns the
/// full realized [`StorageCache`] (all reachable `Op::Const` uploaded by
/// `build_const_cache` — the weights et al., merged over `initial`). The
/// held decode session keeps this cache so subsequent prebuilt realizes
/// (which SKIP the const-cache walk) still find every weight Const; only
/// the per-token data Consts (token-ids / RoPE / mask) are overwritten
/// in the held cache each token. Without this, the prebuilt path would
/// error on the first weight Const it can't resolve.
pub fn prebuild_optimized_env_capturing_cache<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    target: NodeId,
    device: &Device,
    initial: StorageCache,
    sym_env: &SymEnv,
) -> Result<(NodeId, OptimizedGraph, StorageCache, Vec<T>)> {
    let (cache, _backend_id, mut effective_targets) =
        prepare(graph, &[target], device, initial)?;
    let Some(cpu_target) = effective_targets.pop() else {
        return Err(Error::Msg(
            "pipelined_bridge: prepare returned no effective target for a \
             single-target prebuild — internal bug"
                .into(),
        )
        .bt());
    };
    let (storage, optimized, full_cache) = dispatch_with_plan_retry_capturing(
        graph, cpu_target, cache, device, sym_env,
    )?;
    let bytes = extract_cpu_bytes_typed::<T>(&storage)?;
    Ok((cpu_target, optimized, full_cache, bytes))
}

/// Plan-once realize: the `graph` has ALREADY been `prepare`d (the D2H
/// `Op::Copy` spliced at the root) AND `optimize_graph`'d (backend stamps +
/// residency `Op::Copy` + layout `Op::Contiguize` baked in place) on a prior
/// [`prebuild_optimized_env`] call; `optimized` is the cached view from that
/// call. This entry goes STRAIGHT to
/// [`PipelinedExecutor::realize_with_optimized_picking_env`], **skipping
/// BOTH** `prepare` (no re-splice, no const-cache walk) **AND**
/// `build_optimized_graph`/`optimize_graph` (no re-plan, no double-insert of
/// residency/layout nodes). `OPTIMIZE_CALLS` does NOT move.
///
/// `effective_target` is the D2H `Op::Copy` NodeId `prebuild_optimized_env`
/// returned; `cache` is the per-token [`StorageCache`] (the re-bound data
/// Consts, typically `InferenceContext::cloned_persistent()`); `sym_env`
/// carries the per-token `DynScalar` bindings (e.g. `cached_len`).
///
/// The `(selector, lookup)` are per-realize-stateless (Device + Judge
/// derived, no per-realize state — verified Q3), so we rebuild them cheaply
/// per call; for the branchless decode graph the selector is never consulted
/// (arm-0 lowering), so this is a no-op there.
///
/// `TopologyChanged` is surfaced as its typed error (NOT retried here): a
/// topology shift means the cached `optimized.generation` is stale and the
/// stamps may be wrong, so re-optimizing in place would be incorrect. The
/// caller (D2b) handles it by invalidating the held session and rebuilding
/// (§4/§5, Q5). Every other error propagates unchanged. Never panics.
pub fn realize_one_prebuilt_env<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    effective_target: NodeId,
    optimized: &OptimizedGraph,
    device: &Device,
    cache: StorageCache,
    sym_env: &SymEnv,
) -> Result<Vec<T>> {
    // NO prepare(), NO build_optimized_graph(). Straight to the executor.
    let (selector, lookup) = match production_selector_for(device) {
        Some((s, l)) => (Some(s), Some(l)),
        None => (None, None),
    };
    let (storage, _layout) = PipelinedExecutor::realize_with_optimized_picking_env(
        graph.clone(),
        effective_target,
        cache,
        optimized,
        selector,
        lookup,
        sym_env.clone(),
    )?;
    extract_cpu_bytes_typed::<T>(&storage)
}

/// Capturing sibling of [`dispatch_with_plan_retry`] used by
/// [`prebuild_optimized_env`]: identical retry-on-`TopologyChanged` loop, but
/// returns the FINAL successful [`OptimizedGraph`] view alongside the storage
/// so it can be cached for later prebuilt realizes. (The base
/// `dispatch_with_plan_retry` drops the view; the plan-once path needs to
/// keep it.) No `report_node` attribution here — the prebuild caller wants
/// the reusable view, not the root sibling tag.
fn dispatch_with_plan_retry_capturing(
    graph: &Arc<RwLock<Graph>>,
    cpu_target: NodeId,
    cache: StorageCache,
    device: &Device,
    sym_env: &SymEnv,
) -> Result<(Arc<RwLock<Storage>>, OptimizedGraph, StorageCache)> {
    let pinned_loc = device.location();
    let mut retry = TopologyRetryState::new();
    loop {
        let optimized =
            build_optimized_graph(graph, &[cpu_target], pinned_loc, &cache, true)?;
        let (selector, lookup) = match production_selector_for(device) {
            Some((s, l)) => (Some(s), Some(l)),
            None => (None, None),
        };
        let cache_for_attempt = cache.clone();
        let result = PipelinedExecutor::realize_with_optimized_picking_env(
            graph.clone(), cpu_target, cache_for_attempt, &optimized,
            selector, lookup, sym_env.clone(),
        );
        match result {
            Ok((storage, _layout)) => return Ok((storage, optimized, cache)),
            Err(e) if matches!(e, Error::TopologyChanged { .. })
                && retry.permit_retry() =>
            {
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Retry-on-stale-plan loop for the single-target path. Pulled out so
/// the multi-target path can reuse the same retry shape.
///
/// Each attempt runs the full optimize → stamp → copy-insert → fixup
/// sequence. `optimize_graph` (via [`build_optimized_graph`]) transforms
/// the graph in place against the pinned DEVICE: internally it stamps each
/// per-node winner's backend, stitches cross-device-copy residency against
/// those final placements, and runs the layout-fixup pass last — all
/// consuming its internal `ExecutionPlan`, which is not returned (cleanup
/// Step D). Re-optimizing after a `TopologyChanged` retry re-runs the whole
/// sequence so stamps stay consistent with the fresh placement.
///
/// Dispatch goes through [`PipelinedExecutor::realize_with_optimized`]
/// — the executor recomputes its run/`lower_run` dispatch order from the
/// (post-stamping) graph and resolves each node's kernel via the
/// binding-table lookup. No plan is threaded (Step D); the stamp/residency/
/// layout passes are internal to `optimize_graph`, and root attribution
/// reads the graph + registry.
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

/// Build the `OptimizedGraph` lowering view for the realize path.
///
/// `optimize_graph` transforms the graph **in place** into the "plan IS
/// the graph" form and returns only the [`OptimizedGraph`] view whose
/// `dispatch_order` (the runs' `lower_run` sequence) the executor walks.
/// Its internal `ExecutionPlan` accumulator (placement/cost/validation +
/// the stamp / residency / layout passes) is consumed inside
/// `optimize_graph` and never surfaced (Step D — the graph's
/// `target_backend` stamps + `Op::Branch` arms are the only output; the
/// executor re-derives any per-arm candidate from the binding registry).
/// Build-time diagnostics (missing binding, no device context) fire here
/// exactly as they did for the legacy `compile_plan` path.
fn build_optimized_graph(
    graph: &Arc<RwLock<Graph>>,
    roots: &[NodeId],
    pinned_device: DeviceLocation,
    cache: &StorageCache,
    allow_cost_placement: bool,
) -> Result<OptimizedGraph> {
    // D2a telemetry: one bump per real optimize (the expensive placement DP
    // + cost composer + Judge + in-place residency/layout mutation). The
    // prebuilt realize path skips this entirely; the born-red test asserts
    // the count does not move on a re-realize of a held, optimized graph.
    // Bump BOTH the process-global counter (coarse telemetry) and this
    // thread's mirror (the isolated per-realize delta the test reads).
    OPTIMIZE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    OPTIMIZE_CALLS_TL.with(|c| c.set(c.get() + 1));
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

    // Baracuda dispatch/miss telemetry (opt-in, behind the `telemetry`
    // feature). Snapshot the opt-in state + build the plan-time hooks BEFORE
    // `options` so the borrow outlives the plan; `hooks()` is `None` (⇒ no
    // hooks threaded, byte-identical plan) unless emission is enabled.
    #[cfg(feature = "telemetry")]
    let tele_install = crate::telemetry::TelemetryInstall::new(pinned_device);
    #[cfg(feature = "telemetry")]
    let tele_hooks = tele_install.hooks();

    let mut options = PlanOptions::new()
        .with_placements_for_device(&placements_for)
        .with_capabilities_for(&capabilities_for)
        .with_pinned_device(pinned_device)
        .with_fallback_placements_for(&fallback_for)
        .with_transfer_estimator(&*topology)
        .with_input_residency(&input_residency);
    // Reference / single-device realize: suppress cost-based cross-device
    // placement so every un-pinned node stays on the pinned device (the CPU
    // oracle must run on CPU, not be relocated to the backend it validates).
    if !allow_cost_placement {
        options = options.without_cost_placement();
    }
    if let Some(oracle) = judge_oracle.as_deref() {
        options = options.with_judge(oracle);
    }
    #[cfg(feature = "telemetry")]
    if let Some(ref hooks) = tele_hooks {
        options = options.with_telemetry(hooks);
    }

    let bindings_guard = global_bindings();
    let mut g = graph
        .write()
        .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
    // The PRODUCTION optimize entry includes runtime fusion: adopted (Tier-2 /
    // JIT-synthesized) fused ops get their gated `Op::Branch` arms emitted
    // before placement. With nothing adopted the sidecar scan is an empty-Vec
    // early return — byte-identical to the bare `optimize_graph` every test
    // that doesn't adopt observes (`runtime-fused-op-registration.md` §6).
    optimize_graph_with_runtime_fusion(&mut g, roots, &bindings_guard, &options)
}

fn dispatch_with_plan_retry(
    graph: &Arc<RwLock<Graph>>,
    cpu_target: NodeId,
    cache: StorageCache,
    device: &Device,
    report_node: NodeId,
    sym_env: &SymEnv,
    allow_cost_placement: bool,
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
        // on every kernel node, and that's an optimizer concern), and runs
        // the residency + layout passes — all consuming its internal
        // `ExecutionPlan`, which is not returned (Step D). Build-time
        // validation (missing binding / no device) fires inside `optimize_graph`.
        let optimized =
            build_optimized_graph(graph, &[cpu_target], pinned_loc, &cache, allow_cost_placement)?;
        // Residency (cross-device `Op::Copy`) and layout-fixup
        // (`Op::Contiguize`) are now optimizer passes inside `optimize_graph`
        // (cleanup Step B) — driven by the graph stamps + the `input_residency`
        // provider threaded through `build_optimized_graph`. The graph arrives
        // here already copy-stitched and fixed up; the bridge no longer runs
        // either pass.
        // Cleanup Step C: the EXECUTOR owns `Op::Branch` arm-selection now.
        // The bridge builds the runtime selector + live device lookup (Device-
        // and Judge-derived, so they stay here) and hands them to the executor,
        // which picks one arm per branch at dispatch — or arm-0 when the
        // selector is disabled / the graph is branchless (realize unchanged
        // from Phase B). Replaces the bridge's old `resolve_runtime_route`.
        // Step E Phase C / PR C1: hand the selector + lookup to the executor
        // as OWNED `Arc`s so the streaming compiler thread can `move` them in
        // and resolve each branch lazily at the frontier (the SAME VRAM-only
        // selector — byte-identical to the prior one-shot pick).
        let (selector, lookup) = match production_selector_for(device) {
            Some((s, l)) => (Some(s), Some(l)),
            None => (None, None),
        };
        let mut cache_for_attempt = cache.clone();
        // Option B — post-optimize device seed. If the plan offloaded any node
        // to a GPU the pinned-device const upload didn't seed (the CPU-pinned
        // mixed realize), give the executor a device-handle anchor per placed
        // backend so its H2D `Op::Copy`/`Op::Alloc` device-handle search
        // succeeds. No-op for a single-device realize (incl. the reference
        // oracle, which suppresses cross-device placement outright).
        seed_placed_device_handles(graph, cpu_target, pinned_loc, &mut cache_for_attempt)?;
        // Dispatch the "plan IS the graph" form: the executor recomputes its
        // run/`lower_run` dispatch order from the (now fully-stamped) graph,
        // resolves each branch's arm at the frontier (streaming), then
        // resolves each node's kernel via the binding-table lookup.
        let result = PipelinedExecutor::realize_with_optimized_picking_env(
            graph.clone(), cpu_target, cache_for_attempt, &optimized,
            selector, lookup, sym_env.clone(),
        );
        match result {
            Ok((storage, _layout)) => {
                // Session 3 rider: report which sibling dispatched for
                // `report_node` — derived from the SAME graph stamp + registry
                // the executor dispatched through (Step D: was the plan's
                // `AlternativeSet::winner`). Best-effort telemetry: a
                // lock/lookup miss yields `None`, not an error.
                let dispatched = graph
                    .read()
                    .ok()
                    .and_then(|g| dispatched_kernel_source(&g, report_node, &global_bindings()));
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
        let optimized =
            build_optimized_graph(graph, effective_targets, pinned_loc, &cache, true)?;
        // Cleanup Step C: the executor resolves each branch's arm at dispatch
        // (was the bridge's `resolve_runtime_route`); the bridge just builds +
        // hands over the Device/Judge-derived selector + live lookup. PR C1:
        // owned `Arc`s so the streaming compiler thread can `move` them in and
        // resolve branches lazily at the frontier.
        let (selector, lookup) = match production_selector_for(device) {
            Some((s, l)) => (Some(s), Some(l)),
            None => (None, None),
        };
        let cache_for_attempt = cache.clone();
        let result = PipelinedExecutor::realize_many_with_optimized_picking_env(
            graph.clone(), effective_targets, cache_for_attempt, &optimized,
            selector, lookup, sym_env.clone(),
        );
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

/// Phase D · D2b — build a single device-resident `fuel_memory::Storage`
/// Arc from a host buffer, the same upload path [`build_const_cache`]
/// uses per Const. The persistent-decode re-bind inserts the result into
/// the [`crate::inference_context::InferenceContext`]'s persistent map
/// under a STABLE data-Const NodeId each token (token-ids / RoPE / mask),
/// so the bytes stay out of the const-cache walk on the prebuilt realize.
///
/// **CPU device**: wraps the host bytes directly
/// (`CpuStorageBytes::from_bytes`) — no transient graph / executor. This
/// is the D2b born-red bed and the only path exercised at CPU parity.
///
/// **Non-CPU device**: builds a one-node transient `Op::Const → Op::Copy
/// { target }` graph (+ device-handle anchor) and realizes the copy —
/// the H2D upload. Mirrors [`build_const_cache`]'s non-CPU arm for a
/// single buffer.
///
/// The `dtype` tag comes from the `HostBuffer` variant. Never panics.
pub fn upload_host_buffer_to_device(
    device: &Device,
    buf: HostBuffer,
) -> Result<Arc<RwLock<Storage>>> {
    let dtype = host_buffer_dtype(&buf);
    let bytes = host_buffer_to_bytes(&buf);
    let target_loc = device.location();

    if target_loc == DeviceLocation::Cpu {
        let storage = Storage::new(
            BackendStorage::Cpu(
                fuel_cpu_backend::byte_storage::CpuStorageBytes::from_bytes(&bytes),
            ),
            dtype,
        );
        return Ok(Arc::new(RwLock::new(storage)));
    }

    // Non-CPU: transient Op::Const → Op::Copy { target } upload.
    let transient = Arc::new(RwLock::new(Graph::new()));
    let mut transient_cache = StorageCache::new();
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
    let n_elem = if dtype.size_in_bytes() == 0 {
        0
    } else {
        bytes.len() / dtype.size_in_bytes()
    };
    let shape = fuel_ir::Shape::from_dims(&[n_elem]);
    let copy_id = {
        let mut g = transient
            .write()
            .map_err(|_| Error::Msg("transient graph lock poisoned".into()).bt())?;
        let trans_const_id = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: shape.clone(),
            dtype,
        });
        let cpu_storage = Storage::new(
            BackendStorage::Cpu(
                fuel_cpu_backend::byte_storage::CpuStorageBytes::from_bytes(&bytes),
            ),
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
        copy_id
    };
    let realized = PipelinedExecutor::realize_many(
        Arc::clone(&transient), &[copy_id], transient_cache,
    )?;
    let (arc, _layout) = realized.into_iter().next().ok_or_else(|| {
        Error::Msg(
            "upload_host_buffer_to_device: realize_many returned no storage \
             for the Op::Copy target — internal bug"
                .into(),
        )
        .bt()
    })?;
    Ok(arc)
}

/// The `DType` a `HostBuffer` variant carries.
fn host_buffer_dtype(buf: &HostBuffer) -> fuel_ir::DType {
    use fuel_ir::DType;
    match buf {
        HostBuffer::U8(_) => DType::U8,
        HostBuffer::I8(_) => DType::I8,
        HostBuffer::U32(_) => DType::U32,
        HostBuffer::I16(_) => DType::I16,
        HostBuffer::I32(_) => DType::I32,
        HostBuffer::I64(_) => DType::I64,
        HostBuffer::BF16(_) => DType::BF16,
        HostBuffer::F16(_) => DType::F16,
        HostBuffer::F32(_) => DType::F32,
        HostBuffer::F64(_) => DType::F64,
        HostBuffer::F8E4M3(_) => DType::F8E4M3,
        HostBuffer::F6E2M3(_) => DType::F6E2M3,
        HostBuffer::F6E3M2(_) => DType::F6E3M2,
        HostBuffer::F4(_) => DType::F4,
        HostBuffer::F8E8M0(_) => DType::F8E8M0,
    }
}

/// Extract the raw bytes from a `HostBuffer` via a per-variant match
/// (`bytemuck::cast_slice` for typed numeric vecs; identity for the
/// raw-byte sub-byte variants).
///
/// `pub(crate)`: also reused by `LlamaModel::build_token_rope_mask_bytes`
/// (CapturedRun replay wiring) to extract raw per-token bytes from the
/// same `HostBuffer`s `build_token_rope_mask_arcs` uploads — one
/// dtype-dispatch table, not two.
pub(crate) fn host_buffer_to_bytes(buf: &HostBuffer) -> Vec<u8> {
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
// Runtime route picker (Picker 2) — selector construction (cleanup Step C)
// ---------------------------------------------------------------------------
//
// The selector + live-telemetry plumbing (`production_selector_for` /
// `backend_runtime_lookup_for` / `DeviceRuntimeHandle`) is built HERE because
// it needs the realize `Device` + the Judge oracle, both fuel-core. Cleanup
// Step C moved the *pick itself* into the executor: the bridge builds the
// production `ChainedSelector` (VRAM-pressure guard + Judge-aware rank) and
// the live per-tier free-memory lookup, then hands them to the executor's
// `realize_with_optimized_picking_env`, which runs `pick_route` (one arm per
// `Op::Branch`) at dispatch and lowers via `lower_picked_route`. With no
// branches (CPU-only build) the route is empty and realize is unchanged.

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
///   always-present CPU backend),
/// - the **live-load tier** (Step E Phase C / C2) read off the SAME
///   handles via their Tier-2 `BackendStreams` seam (B1's per-device
///   in-flight counter) — demoting arms on busy devices within a VRAM fit
///   tier, and
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

    /// Step E Phase C / B1: expose this handle's Tier-2 [`BackendStreams`]
    /// live-load surface so the load-aware selector (C2) reaches
    /// `pending_work_count` over the route picker's
    /// [`BackendRuntimeLookup`] — which only ever hands out a
    /// `&dyn BackendRuntime` — without naming `DeviceRuntimeHandle`. C2's
    /// `ChainedSelector` load leg downcasts the boxed handle this lookup
    /// returns via `as_backend_streams()`, so the SAME lookup that answers
    /// the VRAM guard also carries the load signal (no second lookup).
    ///
    /// A streaming device (CUDA / Vulkan) returns `Some(self)`; a
    /// synchronous device (CPU) returns `None` — the honesty contract:
    /// CPU has no queue, so it is NOT a `BackendStreams` (a selector reads
    /// "no load signal," never a fabricated "idle"). The CPU
    /// [`fuel_cpu_backend::dyn_impl::CpuBackendDevice`] handle the lookup
    /// builds for the host-RAM arm isn't a `DeviceRuntimeHandle` at all
    /// and keeps the base-trait default `None`, so CPU stays honest on
    /// both lookup paths.
    fn as_backend_streams(&self) -> Option<&dyn BackendStreams> {
        match self.0.location_dyn() {
            DeviceLocation::Cpu => None,
            _ => Some(self),
        }
    }
}

/// Step E Phase C / B1: the live-load read surface for the per-device
/// in-flight counter. `pending_work_count` reads the process-wide
/// `fuel-dispatch` counter for THIS handle's device — the executor's own
/// submitted-but-not-drained async-op count (CUDA events + eager Vulkan
/// batches), which is the "queue depth" signal `06-runtime` names. Only
/// streaming devices reach this impl (CPU is filtered out in
/// [`BackendRuntime::as_backend_streams`]); the `Cpu` arm here is a
/// belt-and-suspenders honesty guard, never hit in practice.
impl BackendStreams for DeviceRuntimeHandle {
    fn pending_work_count(&self) -> Option<u32> {
        match self.0.location_dyn() {
            DeviceLocation::Cpu => None,
            loc => Some(fuel_dispatch::dispatch::inflight_count(loc)),
        }
    }

    /// Advertised concurrent in-flight capacity. Reports a conservative
    /// constant: Fuel today drives one stream per CUDA device and one
    /// compute queue per Vulkan device (A3 / A4b), so the meaningful slot
    /// count is 1. C2's load tiering reads this as the denominator of
    /// `pending_work_count / slot_capacity`
    /// ([`fuel_dispatch::ranker::load_tier`]); with capacity 1 the tiering
    /// is binary (idle at 0 in flight, saturated at >=1), which is exactly
    /// right for a single-stream device. Per-device capacity wiring (a
    /// `BackendCapabilities` field) is the design §9 open-Q1 refinement.
    fn slot_capacity(&self) -> u32 {
        1
    }

    /// Barrier: block until all submitted work on this device's slots has
    /// retired. Delegates to the device's `synchronize` (CUDA
    /// `cuCtxSynchronize` / Vulkan device-wait) — the same realize-boundary
    /// drain the executor already performs.
    fn flush(&self) -> Result<()> {
        self.0.synchronize_dyn()
    }
}

/// Live-handle lookup for the VramPressure guard, the C2 load leg, and the
/// picker fingerprint — one lookup, both signals. Each handle it returns
/// (`DeviceRuntimeHandle`) answers `would_fit` (VRAM) AND, via its Tier-2
/// `BackendStreams` impl, `pending_work_count` (live load), so C2's
/// `ChainedSelector` reads VRAM + load off the same handle without a second
/// lookup (design §3.3). Resolves:
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

/// The distinct non-CPU device locations an optimized graph's cross-device
/// residency nodes (`Op::Copy`/`Op::Move`/`Op::Alloc { target }`) place work
/// on, EXCLUDING `pinned_loc` and any location already backed by a storage in
/// `cache`. Scans only the nodes reachable from `root`.
///
/// This is the detection half of Option B (the post-optimize device seed): a
/// CPU-pinned realize whose plan offloaded nodes to a GPU has H2D copies whose
/// destination device the executor must derive from the cache — but a
/// single-device realize's const upload only seeded the pinned device. The
/// returned locations are exactly those needing a [`device_seed_storage`]
/// anchor. Empty for a pure single-device realize (the reference oracle, or any
/// plan that didn't cross devices).
fn placed_device_locations_needing_seed(
    graph: &Arc<RwLock<Graph>>,
    root: NodeId,
    pinned_loc: DeviceLocation,
    cache: &StorageCache,
) -> Result<Vec<DeviceLocation>> {
    // Locations already represented by a cached storage — no seed needed.
    let already: Vec<DeviceLocation> = cache
        .values()
        .filter_map(|arc| {
            let guard = arc.read().ok()?;
            cached_storage_location(&guard)
        })
        .collect();
    let g = graph
        .read()
        .map_err(|_| Error::Msg("graph lock poisoned during device-seed scan".into()).bt())?;
    let mut needed: Vec<DeviceLocation> = Vec::new();
    for id in topo_order_multi(&g, &[root]) {
        let loc = match g.node(id).op {
            Op::Copy { target } | Op::Move { target } | Op::Alloc { target } => target,
            _ => continue,
        };
        if loc == DeviceLocation::Cpu || loc == pinned_loc {
            continue;
        }
        if already.contains(&loc) || needed.contains(&loc) {
            continue;
        }
        needed.push(loc);
    }
    Ok(needed)
}

/// Construct a `Device` handle for `loc`. CUDA maps ordinal → a fresh
/// [`fuel_cuda_backend::CudaDevice`]; CPU is the host device. Vulkan can't be
/// built from a bare ordinal (it needs an explicit `DeviceSelection`), so a
/// Vulkan target reached here surfaces an actionable error pointing at the
/// explicit multi-device seed entry rather than the executor's later cryptic
/// "no Vulkan storage in input cache" — callers wanting a mixed Vulkan realize
/// use [`realize_one_as_multi_device`], which seeds the Vulkan handle up front.
fn device_for_location(loc: DeviceLocation) -> Result<Device> {
    match loc {
        DeviceLocation::Cpu => Ok(Device::cpu()),
        DeviceLocation::Cuda { gpu_id } => crate::cuda_backend::new_device(gpu_id),
        DeviceLocation::Vulkan { gpu_id } => Err(Error::Msg(format!(
            "post-optimize device seed: the plan placed nodes on Vulkan \
             {{ gpu_id: {gpu_id} }} but this single-device realize has no \
             Vulkan handle to seed, and a Vulkan device can't be derived from \
             an ordinal alone (it needs an explicit DeviceSelection). Realize \
             the mixed graph via `realize_one_as_multi_device` with the Vulkan \
             device in `extra_devices`.",
        ))
        .bt()),
        other => Err(Error::Msg(format!(
            "post-optimize device seed: target {other:?} not wired (CPU + CUDA \
             auto-seed today; Vulkan via realize_one_as_multi_device).",
        ))
        .bt()),
    }
}

/// Option B — the post-optimize device seed. After `optimize_graph` stamps
/// placements and inserts cross-device residency `Op::Copy`/`Op::Alloc` nodes,
/// a CPU-pinned realize whose plan offloaded work to a GPU needs a device-
/// handle anchor for each such backend in the executor's cache: the executor
/// derives the per-backend device handle by searching the cache
/// (`find_cuda_device_in_cache`), and a single-device realize's const upload
/// only seeded the pinned device. Without this the first H2D
/// `Op::Copy { target: Cuda }` panics ("no CUDA storage in input cache … seed
/// via device_seed_storage").
///
/// For each location in [`placed_device_locations_needing_seed`] this pushes a
/// fresh unreachable `Op::Const` anchor node and inserts that backend's
/// [`device_seed_storage`] into `cache` at the anchor's NodeId — the same
/// mechanism as [`seed_extra_device_handles`], but driven by the *plan's actual
/// placements* rather than a caller-supplied device list. A strict no-op for a
/// single-device realize (the reference oracle, or any plan that stayed put),
/// so `realize_f32` / `realize_f32_reference` on a CPU-only host are unaffected.
///
/// Known minor: a repeated realize of the SAME graph that keeps crossing to the
/// same GPU accumulates one unreachable anchor `Op::Const` per call (graph
/// growth; harmless — never dispatched). Amortize via the persistent-cache
/// (InferenceContext) path when that matters.
fn seed_placed_device_handles(
    graph: &Arc<RwLock<Graph>>,
    root: NodeId,
    pinned_loc: DeviceLocation,
    cache: &mut StorageCache,
) -> Result<()> {
    let needed = placed_device_locations_needing_seed(graph, root, pinned_loc, cache)?;
    for loc in needed {
        let dev = device_for_location(loc)?;
        let Some(seed) = device_seed_storage(&dev)? else {
            continue; // CPU (unreachable given the filter) — no handle needed.
        };
        let anchor_id = {
            let mut g = graph.write().map_err(|_| {
                Error::Msg("graph lock poisoned during placed-device seed".into()).bt()
            })?;
            g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: fuel_ir::Shape::from_dims(&[4]),
                dtype: fuel_ir::DType::U8,
            })
        };
        cache.insert(anchor_id, Arc::new(RwLock::new(seed)));
    }
    Ok(())
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

    /// Option B detection: `placed_device_locations_needing_seed` reports the
    /// non-CPU device an `Op::Copy { target }` places on when the realize is
    /// pinned to CPU (the CPU-pinned mixed realize that panicked in phase6b),
    /// and reports nothing when the realize is pinned to that same device or
    /// the graph never crosses off CPU. This is the CPU-observable half of
    /// Option B — the actual CUDA seeding needs `--features cuda` + a GPU.
    #[test]
    fn option_b_detects_offloaded_device_only_when_cpu_pinned() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };

        // Graph: Const → Copy{ target: Cuda0 } (an H2D offload), root = Copy.
        let mut g = Graph::new();
        let c = push_node(&mut g, Op::Const, vec![]);
        let copy = push_node(&mut g, Op::Copy { target: cuda0 }, vec![c]);
        let graph = Arc::new(RwLock::new(g));
        let empty = StorageCache::new();

        // CPU-pinned realize: the Cuda offload needs a seed.
        assert_eq!(
            placed_device_locations_needing_seed(&graph, copy, DeviceLocation::Cpu, &empty)
                .unwrap(),
            vec![cuda0],
            "CPU-pinned realize with a Copy→Cuda must flag Cuda for seeding",
        );

        // Cuda-pinned realize (realize_f32_cuda): the target IS the pinned
        // device — const upload already seeded it, so nothing to add.
        assert!(
            placed_device_locations_needing_seed(&graph, copy, cuda0, &empty)
                .unwrap()
                .is_empty(),
            "a Copy to the pinned device needs no extra seed",
        );

        // Pure CPU graph: no cross-device node → no seed.
        let mut g2 = Graph::new();
        let a = push_node(&mut g2, Op::Const, vec![]);
        let cpu_copy = push_node(&mut g2, Op::Copy { target: DeviceLocation::Cpu }, vec![a]);
        let graph2 = Arc::new(RwLock::new(g2));
        assert!(
            placed_device_locations_needing_seed(&graph2, cpu_copy, DeviceLocation::Cpu, &empty)
                .unwrap()
                .is_empty(),
            "an all-CPU graph never needs a device seed",
        );
    }

    /// Step E Phase C / B1 honesty contract: a CPU `DeviceRuntimeHandle` is
    /// NOT a `BackendStreams` (CPU dispatches synchronously — no queue). Its
    /// `as_backend_streams` upcast returns `None`, so the load-aware selector
    /// (C2) reads "no load signal" for CPU, never a fabricated "idle." This is
    /// the CPU half of B1's seam; the streaming half (CUDA/Vulkan reading the
    /// in-flight counter) is covered by the live mid-realize gate.
    #[test]
    fn cpu_runtime_handle_is_not_backend_streams() {
        let handle = DeviceRuntimeHandle(
            Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice) as Arc<dyn DynBackendDevice>,
        );
        // The upcast a C2 selector performs over a `&dyn BackendRuntime`.
        let as_runtime: &dyn BackendRuntime = &handle;
        assert!(
            as_runtime.as_backend_streams().is_none(),
            "CPU has no queue concept — it must not expose BackendStreams",
        );
        // And the direct surface is honest too (None, not Some(0)).
        assert_eq!(
            handle.pending_work_count(),
            None,
            "CPU pending_work_count must be None (no queue), never Some(0)",
        );
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
        // Step C: the production runtime route picker (Picker 2) defaults
        // ON, so this CPU realize threads the bridge-built selector into the
        // executor's `realize_with_optimized_picking_env`, which runs the
        // pick. A CPU-only graph has no `Op::Branch`, so the route is empty
        // ⇒ the executor uses the arm-0 lowering ⇒ realize is unchanged. The
        // correct bytes pin that no-branch-no-op contract at the bridge level.
        assert!(
            production_selector_for(&device).is_some(),
            "the runtime route picker defaults ON (no opt-out env set)",
        );
        let out = realize_one_as_with_initial::<f32>(&graph, add, &device, initial)
            .expect("realize through the optimized + route-picker path");
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// Phase D · D2a born-red gate for the optimize-skip bridge seam.
    ///
    /// Realize a small CPU graph ONCE via [`prebuild_optimized_env`]
    /// (capturing the cached `OptimizedGraph` + the spliced D2H target), then
    /// re-realize the SAME graph via [`realize_one_prebuilt_env`] with a fresh
    /// per-call cache + empty `SymEnv`, and assert the three plan-once
    /// invariants:
    ///   (a) the optimizer count (this thread's isolated
    ///       [`optimize_calls_thread_local`]) did NOT increase on the 2nd
    ///       realize (the optimizer was skipped);
    ///   (b) the 2nd result is **exactly** `==` the 1st (same plan → same
    ///       kernels → bit-exact, NOT epsilon);
    ///   (c) the graph node `len()` did NOT grow between the two realizes (no
    ///       double-spliced D2H `Op::Copy`, no double-inserted
    ///       residency/`Op::Contiguize`).
    ///
    /// The test also demonstrates the HAZARD the seam avoids (the born-red
    /// shape): routing a 3rd realize of the same graph through the normal
    /// full path (`realize_one_as_with_initial`) DOES re-optimize (the count
    /// moves) and DOES re-splice a D2H `Op::Copy` (the node count grows) —
    /// which is exactly what the prebuilt seam skips.
    #[test]
    fn d2a_prebuilt_realize_skips_optimize_and_does_not_grow_graph() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, c2, add) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let c2 = push_node(&mut g, Op::Const, vec![]);
            let add = push_node(&mut g, Op::Add, vec![c1, c2]);
            (c1, c2, add)
        };
        // Stable Const storages (the "data" that would be re-bound per token).
        let c1_arc = cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]);
        let c2_arc = cpu_storage_f32(&[10.0, 20.0, 30.0, 40.0]);
        let device = crate::Device::cpu();

        // --- First realize: full path, capture the reusable optimize view. ---
        let mut initial = StorageCache::new();
        initial.insert(c1, Arc::clone(&c1_arc));
        initial.insert(c2, Arc::clone(&c2_arc));
        // Per-thread counter — isolated from other test threads' concurrent
        // optimizes (the process-global `optimize_calls()` is polluted under
        // the full suite; a global counter cannot give an isolated delta).
        let calls_before = optimize_calls_thread_local();
        let (effective_target, optimized, first) =
            prebuild_optimized_env::<f32>(&graph, add, &device, initial, &SymEnv::default())
                .expect("prebuild (first) realize");
        assert_eq!(first, vec![11.0, 22.0, 33.0, 44.0]);
        let calls_after_first = optimize_calls_thread_local();
        assert!(
            calls_after_first > calls_before,
            "the FIRST realize must run the optimizer (prepare + optimize_graph): \
             {calls_before} -> {calls_after_first}",
        );
        let len_after_first = graph.read().unwrap().len();

        // --- Second realize: PREBUILT seam — skip prepare + optimize. ---
        // Fresh per-call cache carrying the same const Arcs (the D2b caller
        // would re-bind per-token data here; on CPU it is a slice-identical
        // clone). No new prepare/const-cache walk runs on this path.
        let mut cache2 = StorageCache::new();
        cache2.insert(c1, Arc::clone(&c1_arc));
        cache2.insert(c2, Arc::clone(&c2_arc));
        let second = realize_one_prebuilt_env::<f32>(
            &graph,
            effective_target,
            &optimized,
            &device,
            cache2,
            &SymEnv::default(),
        )
        .expect("prebuilt (second) realize");
        let calls_after_second = optimize_calls_thread_local();
        let len_after_second = graph.read().unwrap().len();

        // (a) optimizer skipped on the prebuilt path.
        assert_eq!(
            calls_after_second, calls_after_first,
            "prebuilt realize must NOT re-run optimize_graph: \
             {calls_after_first} -> {calls_after_second}",
        );
        // (b) bit-exact (same plan → same kernels), NOT epsilon.
        assert_eq!(
            second, first,
            "prebuilt realize must reproduce the first result byte-for-byte",
        );
        // (c) no double-splice / double-insert.
        assert_eq!(
            len_after_second, len_after_first,
            "prebuilt realize must NOT grow the graph (no re-spliced D2H Copy \
             / re-inserted residency or Contiguize)",
        );

        // --- Control: the HAZARD the seam avoids (born-red shape). ---
        // Routing the SAME graph through the full path a THIRD time DOES
        // re-optimize and DOES splice a second D2H Op::Copy at the root —
        // exactly the double-work + node growth the prebuilt seam skips.
        let mut initial3 = StorageCache::new();
        initial3.insert(c1, Arc::clone(&c1_arc));
        initial3.insert(c2, Arc::clone(&c2_arc));
        let third =
            realize_one_as_with_initial::<f32>(&graph, add, &device, initial3)
                .expect("full-path (third) realize");
        assert_eq!(third, first, "full path still computes the same value");
        assert!(
            optimize_calls_thread_local() > calls_after_second,
            "the FULL path re-optimizes (this is what the prebuilt seam skips)",
        );
        assert!(
            graph.read().unwrap().len() > len_after_second,
            "the FULL path re-splices a D2H Op::Copy at the root, growing the \
             graph (this is the double-splice the prebuilt seam skips)",
        );
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
    /// `compile_plan`, internal: stamp backends → residency stitch →
    /// layout fixups) → `realize_with_optimized` (run / `lower_run` order +
    /// binding-table lookup) — and produces the expected values for
    /// `(a + b) * a`. `optimize_graph` returns only the `OptimizedGraph`
    /// (Step D); the executor recomputes its order from the stamped graph.
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
        let optimized =
            build_optimized_graph(&graph, &[cpu_target], pinned, &cache, true)
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
