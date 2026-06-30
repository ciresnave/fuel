//! Pipelined compile + execute. Phase 7.5 B4.
//!
//! [`PipelinedExecutor::realize`] runs compilation and execution on
//! separate threads connected by a channel: a compiler thread walks
//! the graph in topological order and emits work items; the
//! executor (this thread) consumes them, allocates output Storage,
//! calls the kernel, and stores the result in an internal cache.
//! Both threads run concurrently so execution can begin while
//! compilation is still resolving later nodes.
//!
//! Today's "compile" step is a single binding-table lookup —
//! roughly nanoseconds per node. The threading delivers no
//! measurable speedup in this regime. The pipelining infrastructure
//! exists *now* — built on the [`compile_node`] / [`execute_compiled`]
//! interface from B5 — so future work that grows the compile step
//! (residency-aware planning, transfer-cost minimization, kernel
//! auto-tuning) plugs in without revisiting call sites.
//!
//! ## Storage during the migration
//!
//! `fuel_graph::Graph::storage_map` uses the legacy
//! `fuel_backend_contract::Storage` (the `Box<dyn DynBackendStorage>`
//! newtype). The pipelined executor uses the new
//! `fuel_memory::Storage` (BackendStorage enum + dtype). During
//! the migration the two coexist — neither is converted on the fly.
//! The pipelined executor takes pre-realized inputs as a
//! `HashMap<NodeId, Arc<RwLock<fuel_memory::Storage>>>` rather
//! than reading from the graph's storage_map. Phase D unifies the
//! two paths once kernel migration completes.
//!
//! ## Op coverage
//!
//! B4 supports `Op::Const` (input-cache adoption — no kernel call)
//! and `Op::Add` on f32 (mapped to `OpKind::AddElementwise`).
//! Phase C adds the rest as more (op, dtype) bindings register.

use std::collections::HashMap;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, RwLock};
use std::thread;

use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, DeviceLocation, Error, Layout, Result, SymEnv};
use fuel_graph::opt::{execution_plan, insert_safety_copies};
use fuel_graph::{Graph, Node, NodeId, Op, PickedRoute};

use crate::compiled::{compile_node, execute_compiled, CompiledNode, CompletionHandle};
use crate::dispatch::global_bindings;
use crate::kernel::{KernelBindingTable, OpParams};
use crate::optimize::OptimizedGraph;
use crate::ranker::{pick_route, BackendRuntimeLookup, RuntimeSelector};
use fuel_memory::Storage;

/// How the executor derives its dispatch order + topology-generation
/// stamp for the `TopologyChanged` chunk-boundary check.
///
/// PR-A3b-1 of the "plan IS the graph" rebuild: the realize bridge can
/// now drive the executor from `optimize_graph`'s in-place form via
/// [`OptimizedGraph::dispatch_order`] (`extract_runs`/`lower_run`)
/// instead of the legacy `ExecutionPlan`-supplied order. Both arms
/// compute the *same* `NodeId` sequence on a branchless graph (the
/// A3a equivalence gate proved `dispatch_order == compile_plan(...).order`);
/// the only difference is the source of truth.
///
/// - [`OrderSource::Default`] — recompute via `execution_plan` inside
///   the executor (the pre-A3b-1 behavior; the legacy `ExecutionPlan`
///   path and the plain `realize`/`realize_many` entries use this).
/// - [`OrderSource::Optimized`] — derive from the `OptimizedGraph`'s
///   run lowering, computed AFTER `insert_safety_copies` so the order
///   covers any safety-copy nodes the executor just inserted. Carries
///   the optimize-time `generation` so the chunk-boundary
///   `TopologyChanged` check still fires (the new path dispatches with
///   `plan: None`, so the generation must come from here instead).
///
///   PR-C1: it also carries an optional **route** ([`PickedRoute`]) — the
///   runtime route picker (Picker 2) chose one arm per `Op::Branch`. When
///   `Some`, the order is the route-aware lowering
///   ([`fuel_graph::lower_picked_route`]); when `None` (or empty) it is
///   the arm-0 lowering (`lower_runs_arm0`), byte-identical to Phase B.
enum OrderSource<'a> {
    Default,
    Optimized {
        optimized: &'a OptimizedGraph,
        /// The route the picker resolved (one arm per branch). `None` ⇒
        /// arm-0 everywhere (no picker / no pressure ⇒ Phase B behavior).
        route: Option<&'a PickedRoute>,
    },
}

/// Map from NodeId to a realized Storage Arc. Used both as the
/// input cache (passed in by the caller for pre-realized leaves)
/// and as the output cache (built up during execution).
pub type StorageCache = HashMap<NodeId, Arc<RwLock<Storage>>>;

/// What flavor of work item the executor is processing.
/// Disambiguates the four cases:
enum WorkItemKind {
    /// `Op::Const` — its Storage Arc is already in the input cache.
    /// Executor verifies the entry exists and moves on.
    ConstAdopt,
    /// Metadata-only view op (`Op::Transpose`, `Op::Permute`,
    /// `Op::BroadcastTo`): the output's Storage Arc IS the
    /// input's Storage Arc (bytes shared); `output_layout`
    /// describes the strided view.
    ViewOf {
        input: NodeId,
    },
    /// Reshape-style adoption: the output is contiguous in
    /// `output_layout.shape()`. If the input is already contiguous
    /// + zero offset, the output Arc is the input Arc (zero copy).
    /// Otherwise, the executor auto-contiguizes the input into a
    /// fresh Arc and uses that.
    ContiguizeOf {
        input: NodeId,
    },
    /// Computational kernel: allocate output, run the compiled
    /// kernel, store the result. `compiled` is `Some(...)`.
    Kernel,
    /// `Op::Release` — metadata-only "this storage is no longer
    /// needed" directive. Per `Op::Release`'s contract: the output
    /// is a zero-element marker (NodeId placeholder for graph
    /// bookkeeping; never read by any consumer). The executor emits
    /// a zero-byte CPU `Storage` at the node's slot so downstream
    /// cache lookups don't surface "missing slot" errors; the actual
    /// deallocation of `inputs[0]` is driven by `destructive_input`
    /// in the realize loop (Phase B), which drops the Arc held by
    /// the cache. The held Arc's Drop chain frees device memory
    /// (sync for CPU, stream-deferred for async backends).
    ReleaseMarker,
    /// `Op::WriteSlice` — in-place scatter write. The output adopts
    /// the destination input's Storage Arc (zero-copy alias); the
    /// kernel mutates that Arc's bytes via a write lock. The source
    /// input's slab is copied into the destination's rectangular
    /// region defined by `OpParams::WriteSlice`'s `ranges`.
    ///
    /// Carries `dest`/`source` NodeIds so the executor can:
    /// 1. Adopt the dest's Arc as the kernel's output slot.
    /// 2. Pass only the source as a kernel input (the dest is not
    ///    a read-input to the kernel; it's the destination buffer
    ///    being mutated in place).
    ///
    /// The realize loop's `destructive_input` cleanup evicts the
    /// `dest` NodeId from the cache after this op runs (per
    /// `Op::WriteSlice`'s destructive_input == Some(0) contract).
    /// Downstream consumers read post-write bytes via this op's
    /// own NodeId, not the dest's.
    WriteSlice {
        dest: NodeId,
        source: NodeId,
    },
    /// `Op::WriteSliceRotating` — like `WriteSlice` but the rotating
    /// axis wraps modulo `modulus`. Carries `dest`/`source`/`position`
    /// NodeIds; the kernel reads the U32 position from `position`'s
    /// storage and splits the write across the ring boundary if
    /// `position % modulus + slab_len > modulus`.
    WriteSliceRotating {
        dest: NodeId,
        source: NodeId,
        position: NodeId,
    },
    /// `Op::Copy { target }` — produce a fresh Storage on
    /// `target_location`, copying bytes from `inputs[0]`'s residency.
    ///
    /// The kernel lookup goes through the standard binding-table path
    /// at `(OpKind::Copy, [dt, dt], source_backend)`, so D2H is a peer
    /// of every other op (architecture identity check #1). The
    /// dedicated WorkItemKind exists because Op::Copy is the one op
    /// whose output's device differs from `target_backend`:
    /// `target_backend` is the **source** backend (where the kernel
    /// runs — it owns the download path), while `target_location`
    /// drives output allocation. WorkItemKind::Kernel always allocates
    /// output on `target_backend`; Copy needs the override.
    ///
    /// Phase 2 of the bridge-retirement trajectory. Replaces the
    /// per-variant `match self` in `BackendStorage::read_to_cpu_bytes`
    /// (deleted alongside this commit) with a graph-level node the
    /// optimizer can see, cost, and eventually fuse.
    Copy {
        target_location: DeviceLocation,
    },
    /// `Op::Move { target }` — `Op::Copy`'s destructive sibling:
    /// produce a fresh Storage on `target_location` via the same
    /// data-movement kernel (binding-table lookup at
    /// `(OpKind::Copy, [dt, dt], source_backend)`; output allocation
    /// driven by `target_location`), then release the source. The
    /// release half is NOT handled in the executor's arm — it rides
    /// the realize loop's `destructive_input` cleanup
    /// (`Op::Move::destructive_input() == Some(0)`), exactly like
    /// `Op::Release`: the cache's Arc to the source drops after the
    /// move runs and the Drop chain frees device memory (sync for
    /// CPU, stream-deferred for async backends).
    ///
    /// Ordering safety is the graph's job, not this arm's:
    /// `execution_plan` integrates `derive_ordering`, which pins the
    /// Move AFTER every non-destructive reader of the source's alias
    /// set, and `insert_safety_copies` snapshots the source for any
    /// reader that data-flow-depends on the Move's output. A Move
    /// therefore never strands a sibling consumer of its source.
    ///
    /// Same-device moves are legal and match the legacy
    /// `GraphExecutor` contract: a plain copy producing fresh
    /// storage on the same device, with the source still evicted
    /// afterward.
    Move {
        target_location: DeviceLocation,
    },
    /// `Op::Alloc { target }` — produce a freshly-allocated, zero-
    /// initialized Storage on `target_location` with the node's shape
    /// + dtype. Zero inputs.
    ///
    /// Doesn't dispatch through the binding table — the executor's
    /// arm calls each backend's native allocator directly. For
    /// non-CPU targets it derives the device handle by searching the
    /// input cache for any storage on `target_location`'s backend
    /// (callers seed this via
    /// `fuel-core::pipelined_bridge::device_seed_storage`).
    ///
    /// Phase 3a of the bridge-retirement trajectory (post-9c).
    /// Replaces `fuel-core::inference_context::alloc_zeroed_on`'s
    /// per-`DeviceLocation` match with a graph node the optimizer
    /// can see; the per-backend match moves from fuel-core's bridge
    /// layer into fuel-storage's executor (the architectural dispatch
    /// layer).
    Alloc {
        target_location: DeviceLocation,
    },
    /// `Op::ZeroFill` — fill the input's storage bytes with zero,
    /// in place. Adopts the input's Storage Arc as the output (same
    /// Storage; bytes mutated). Destructive on `inputs[0]`.
    ///
    /// Direct executor dispatch (no binding-table lookup) per the
    /// same rationale as `WorkItemKind::Alloc`: structural op,
    /// per-backend dispatch. Per-backend behaviour:
    /// - CPU: `bytes_mut().fill(0)` via the CoW path.
    /// - CUDA: `CudaStorageBytes::zero_async` via baracuda alpha.30's
    ///   `DeviceBuffer::zero_async` (cuMemsetD8Async, in-place).
    /// - Vulkan: `VulkanBackend::fill_bytes_zero` via
    ///   `vkCmdFillBuffer` — device-side, ~2× the bandwidth of the
    ///   old host-staged zeros path that `alloc_zeroed_on` used.
    ///
    /// Phase 3a follow-up of bridge-retirement (post-9c). Pairs with
    /// the uninit-alloc `WorkItemKind::Alloc` to give the
    /// architecturally clean "Op::Alloc (uninit) → Op::ZeroFill
    /// (explicit fill)" pipeline.
    ZeroFill,
    /// In-place kernel — the output adopts the input at index
    /// `target_idx`'s Storage Arc and the kernel mutates that Arc's
    /// bytes through a single write lock. Phase 3 of the in-place ops
    /// infrastructure
    /// (`docs/session-prompts/in-place-ops-infrastructure.md`).
    ///
    /// Kernel lookup goes through the standard binding table at
    /// `(op_to_op_kind, [target_dtype], target_backend)`. The wrapper's
    /// `inputs` slice is empty; `outputs[0]` is the target Arc. The
    /// wrapper acquires `outputs[0]`'s write lock and calls the
    /// underlying single-pointer kernel (e.g. baracuda's
    /// `affine_inplace_*` on CUDA, or the chassis with `src == dst`
    /// on CPU).
    ///
    /// Output Arc adoption: identical to `WorkItemKind::WriteSlice`
    /// (the output's slot = the target's Arc; the realize loop's
    /// `destructive_input` cleanup evicts the target's NodeId from the
    /// cache afterward). Downstream consumers read post-mutation bytes
    /// through this op's NodeId, not the target's.
    InplaceKernel {
        /// Index into `inputs` whose Storage Arc the output adopts.
        /// Always equal to the source `Op::destructive_input().unwrap()`;
        /// stored explicitly so the executor doesn't re-derive it.
        target_idx: usize,
    },
    /// `Op::View { slot }` — multi-output projection (Option C,
    /// Session 4). The output's Storage Arc IS the producer's Arc;
    /// the WorkItem's `output_layout` was prepared by
    /// [`fuel_graph::Tensor::view`] with the slot's `byte_offset`
    /// baked into `Layout::start_offset` so downstream kernels reading
    /// the producer's bytes as `slot_dtype` elements land on the
    /// slot's first byte. Structurally identical to [`Self::ViewOf`]
    /// at execute time — kept distinct so error messages and
    /// telemetry have a clean dispatch point.
    SlotView {
        producer: NodeId,
    },
    /// `Op::ViewOwned { slot }` — multi-output projection with an
    /// independent destination buffer. At realize time, allocate a
    /// fresh contiguous Storage of the slot's `(shape, dtype)` on
    /// the producer's device, then memcpy
    /// `producer.bytes[slot.byte_offset .. slot.byte_offset + slot.len_bytes]`
    /// into the new storage. Session 4 ships the CPU path; non-CPU
    /// backends return a typed error until their copy-with-offset
    /// hooks land in the followup session.
    SlotOwn {
        producer: NodeId,
        slot:     u32,
    },
}

/// One unit of work emitted by the compiler thread to the executor
/// thread.
struct WorkItem {
    node_id: NodeId,
    inputs: Vec<NodeId>,
    /// Number of elements in the output (for output Storage
    /// allocation; multiplied by dtype size at allocation time).
    elem_count: usize,
    dtype: DType,
    target_backend: BackendId,
    /// What kind of work this represents (kernel vs adopt vs view).
    kind: WorkItemKind,
    /// `Some` for [`WorkItemKind::Kernel`]; `None` for the other
    /// two. Carries the resolved kernel ref + op_params.
    compiled: Option<CompiledNode>,
    /// The output's [`Layout`]. For kernels: always
    /// `Layout::contiguous(node.shape)`. For metadata-only view
    /// ops: a strided/broadcast Layout pointing at the input's
    /// Storage. Carried so the executor can publish the right
    /// Layout into its layout cache and ultimately return it from
    /// [`PipelinedExecutor::realize`].
    output_layout: Layout,
    /// Index into `inputs` whose storage gets destroyed by this op
    /// (`Op::Release` / `Op::Move` → `Some(0)`; non-destructive ops
    /// → `None`). The executor evicts the destroyed input from the
    /// cache after the op runs, unless it's also in the realize
    /// target set. Snapshot of `node.op.destructive_input()` at
    /// compile time.
    destructive_input: Option<usize>,
    /// Multi-output bundle metadata (Option C, item 3). `Some(_)`
    /// when the node was declared multi-output via
    /// `Graph::set_output_views`; the Kernel arm allocates one
    /// contiguous output Storage sized to fit every slot's bytes,
    /// then attaches the bundle via `Storage::with_bundle` so
    /// downstream `Op::View`/`Op::ViewOwned` nodes resolve
    /// correctly.
    output_bundle: Option<Arc<[fuel_ir::storage::OutputView]>>,
}

/// Merge the graph's side-effect roots into the caller's requested
/// roots. Preserves caller order (user roots come first), dedupes,
/// and appends side-effect-bearing nodes (`Op::Print`, `Op::Save`,
/// etc.) so their effects fire even when not reachable from the
/// caller's roots. Mirrors `fuel-graph-executor::extend_with_side_effect_roots`.
///
/// Short-circuits to a borrow-equivalent `Vec<NodeId>` when the graph
/// has no side-effect roots — the common case during inference.
fn extend_with_side_effect_roots(graph: &Graph, user_roots: &[NodeId]) -> Vec<NodeId> {
    let side = graph.side_effect_roots();
    if side.is_empty() {
        return user_roots.to_vec();
    }
    let mut out = Vec::with_capacity(user_roots.len() + side.len());
    out.extend_from_slice(user_roots);
    for &s in side {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

/// Pipelined executor: walks a graph, compiles each node in a
/// dedicated thread, executes the compiled stream on the calling
/// thread.
///
/// # Architectural commitments (Phase 7.6 step 9c)
///
/// ## Fail-fast dispatch — no runtime CPU fallback
///
/// The binding-table lookup is fail-fast. If
/// `global_bindings().lookup(op_kind, dtypes, backend)` returns
/// `None`, `realize` / `realize_many` surface a typed `Error::Msg`
/// at compile time. There is **no runtime fallback** to a CPU
/// sibling when a registered kernel returns `Err`; we commit to the
/// backend contract that registered kernels must succeed for shapes
/// matching their declared coverage.
///
/// Rationale: per-decision-point alternatives (architecture v1.0
/// §04) already let multiple backends register at the same key. If
/// a backend's coverage is partial, the right answer is to register
/// only the shapes it actually handles — not to silently fall back
/// at runtime, which masks bugs and slows the hot path. The
/// long-term home for "this op on this device" is graph-level
/// dispatch insertion (Op::Copy edges to a backend that does
/// support the shape), which makes the routing decision visible in
/// the IR rather than hidden in the executor. Picker-arc step 4b
/// landed exactly that: `compile_plan`'s off-device fallback
/// (`PlanOptions::fallback_placements_for`) admits a candidate on
/// another device when the pinned device lacks the implementation,
/// and the bridge's cross-device-copy pass stitches residency
/// around the off-device winner. There is exactly ONE fallback
/// owner — the plan-time picker; this executor still dispatches
/// fail-fast, and ops with no implementation anywhere surface the
/// plan-time `NoBackendForOp` error.
///
/// This is a deliberate departure from the legacy executor's
/// `cpu_fallback(op, inputs, shape, dtype, cache)` semantics.
///
/// ## Optimizer separation — callers compose
///
/// The pipelined executor does **not** run `fuel_graph::opt` or any
/// `RuleRegistry` pass itself. Callers that want graph-level
/// optimization compose it at the call site:
///
/// ```ignore
/// let optimized_targets = registry.optimize_to_fixpoint(&graph, &targets);
/// PipelinedExecutor::realize_many(graph, &optimized_targets, inputs)?;
/// ```
///
/// Rationale: the legacy `GraphExecutor<B>::with_optimization(bool)`
/// coupling reflected its per-typed-entry API surface
/// (`realize_f32`, etc.) where the convenience of a single
/// optimize+realize call was load-bearing. The pipelined path has
/// a uniform `realize_many` that composes trivially — no need to
/// re-couple them at the executor layer. This keeps the executor
/// focused on execution and lets the optimizer evolve
/// independently (different rule registries per call site, mid-
/// pipeline opt+realize+opt+realize patterns, etc.).
pub struct PipelinedExecutor;

impl PipelinedExecutor {
    /// Realize `target` and every transitive dependency. Compilation
    /// runs in a worker thread; execution runs on the calling
    /// thread.
    ///
    /// Pre-conditions:
    ///
    /// - Every reachable `Op::Const` node must have a corresponding
    ///   entry in `inputs` (pre-realized Storage Arc).
    /// - Every reachable non-`Const` node must have its
    ///   `target_backend` set in the graph
    ///   (`Graph::set_target_backend`).
    /// - The op + dtype must be registered in `global_bindings()`.
    ///
    /// Returns the realized `Storage` Arc for `target` plus its
    /// resolved [`Layout`]. The Layout is contiguous for kernel
    /// outputs; for graphs whose target is a metadata-only view
    /// op (`Op::Transpose`, `Op::Permute`, `Op::BroadcastTo`),
    /// the returned Storage shares its bytes with an upstream
    /// node and the Layout encodes the view's strides + offset.
    ///
    /// Production-correct: errors on any unmet precondition rather
    /// than panicking.
    pub fn realize(
        graph: Arc<RwLock<Graph>>,
        target: NodeId,
        inputs: StorageCache,
    ) -> Result<(Arc<RwLock<Storage>>, Layout)> {
        Self::realize_inner(graph, target, inputs, OrderSource::Default, SymEnv::default())
    }

    /// Env-carrying sibling of [`realize`]: realize `target` with a
    /// per-pass [`SymEnv`] supplying the runtime bindings for any
    /// `DynScalar` op params (today: `Op::WriteSlice`'s dynamic start
    /// offset — Phase D symbolic extents). An **empty** env is
    /// byte-identical to [`realize`]; the env is consulted only by ops
    /// that carry a `DynScalar`, so a graph with none ignores it. Uses
    /// the default (`execution_plan`) dispatch order.
    ///
    /// This is the input-determined path for persistent decode (the
    /// per-token `cached_len` write offset). The `OptimizedGraph`/route
    /// entry points keep an empty env until the realize bridge threads
    /// the session env through them (Phase D step 1, fuel-core).
    pub fn realize_with_env(
        graph: Arc<RwLock<Graph>>,
        target: NodeId,
        inputs: StorageCache,
        sym_env: SymEnv,
    ) -> Result<(Arc<RwLock<Storage>>, Layout)> {
        Self::realize_inner(graph, target, inputs, OrderSource::Default, sym_env)
    }

    /// PR-A3b-1 entry: realize `target` driven from `optimized`'s
    /// in-place "plan IS the graph" form. The dispatch order is the
    /// `OptimizedGraph`'s `extract_runs`/`lower_run` lowering (computed
    /// after safety-copy insertion); per-node kernel resolution uses
    /// the binding-table-lookup path (no `ExecutionPlan`, so no
    /// route-picking — A3b-1 is branchless). The optimize-time
    /// `generation` drives the `TopologyChanged` chunk-boundary check.
    ///
    /// Pre-conditions match [`realize`]: every reachable kernel-bearing
    /// node must have `target_backend` set (the bridge's
    /// `stamp_plan_backends` does this) and a registered binding.
    pub fn realize_with_optimized(
        graph: Arc<RwLock<Graph>>,
        target: NodeId,
        inputs: StorageCache,
        optimized: &OptimizedGraph,
    ) -> Result<(Arc<RwLock<Storage>>, Layout)> {
        Self::realize_inner(
            graph,
            target,
            inputs,
            OrderSource::Optimized { optimized, route: None },
            SymEnv::default(),
        )
    }

    /// Env-carrying sibling of [`realize_with_optimized`]: same
    /// optimized-graph dispatch, but with a per-pass [`SymEnv`] supplying
    /// the runtime bindings for `DynScalar` op params (Phase D symbolic
    /// extents). An empty env is byte-identical to
    /// [`realize_with_optimized`]. The realize bridge threads its session
    /// env through here for persistent decode.
    pub fn realize_with_optimized_env(
        graph: Arc<RwLock<Graph>>,
        target: NodeId,
        inputs: StorageCache,
        optimized: &OptimizedGraph,
        sym_env: SymEnv,
    ) -> Result<(Arc<RwLock<Storage>>, Layout)> {
        Self::realize_inner(
            graph,
            target,
            inputs,
            OrderSource::Optimized { optimized, route: None },
            sym_env,
        )
    }

    /// PR-C1 entry: realize `target` driven from `optimized`'s in-place
    /// form **following the runtime route picker's chosen arms**. The
    /// dispatch order is the route-aware lowering
    /// ([`fuel_graph::lower_picked_route`]) over `route` — the per-branch
    /// arm the picker (Picker 2) selected by live telemetry. A branch
    /// absent from `route` defaults to arm 0, so an **empty** route is
    /// byte-identical to [`realize_with_optimized`] (the no-pressure /
    /// no-telemetry contract). A branchless graph has no branches ⇒ the
    /// route is empty ⇒ this is exactly the arm-0 path.
    pub fn realize_with_optimized_route(
        graph: Arc<RwLock<Graph>>,
        target: NodeId,
        inputs: StorageCache,
        optimized: &OptimizedGraph,
        route: &PickedRoute,
    ) -> Result<(Arc<RwLock<Storage>>, Layout)> {
        Self::realize_inner(
            graph,
            target,
            inputs,
            OrderSource::Optimized { optimized, route: Some(route) },
            SymEnv::default(),
        )
    }

    /// Env-carrying sibling of [`realize_with_optimized_route`] — the
    /// route-aware lowering with a per-pass [`SymEnv`]. An empty env is
    /// byte-identical to [`realize_with_optimized_route`].
    pub fn realize_with_optimized_route_env(
        graph: Arc<RwLock<Graph>>,
        target: NodeId,
        inputs: StorageCache,
        optimized: &OptimizedGraph,
        route: &PickedRoute,
        sym_env: SymEnv,
    ) -> Result<(Arc<RwLock<Storage>>, Layout)> {
        Self::realize_inner(
            graph,
            target,
            inputs,
            OrderSource::Optimized { optimized, route: Some(route) },
            sym_env,
        )
    }

    /// Cleanup Step C/D entry: the executor OWNS `Op::Branch` arm-selection.
    /// Given the runtime `selector` + the live device `lookup` (built
    /// bridge-side from the realize `Device` + Judge oracle and passed in,
    /// exactly like the residency `input_residency` provider), pick one arm per
    /// branch HERE — at dispatch — then lower the chosen route. The per-arm
    /// candidates are re-enumerated from the runtime binding registry + the
    /// graph (Step D: no threaded `ExecutionPlan`).
    ///
    /// No `selector` (the `runtime_selector_disabled()` opt-out) or a
    /// branchless graph yields `None` ⇒ arm-0 lowering, byte-identical to Phase
    /// B. Step E later makes this re-pick per decision-point by live device
    /// queue depth.
    pub fn realize_with_optimized_picking_env(
        graph: Arc<RwLock<Graph>>,
        target: NodeId,
        inputs: StorageCache,
        optimized: &OptimizedGraph,
        selector: Option<&dyn RuntimeSelector>,
        lookup: Option<&BackendRuntimeLookup>,
        sym_env: SymEnv,
    ) -> Result<(Arc<RwLock<Storage>>, Layout)> {
        match Self::pick_route_for(&graph, &[target], selector, lookup)? {
            Some(route) => Self::realize_with_optimized_route_env(
                graph, target, inputs, optimized, &route, sym_env,
            ),
            None => Self::realize_with_optimized_env(
                graph, target, inputs, optimized, sym_env,
            ),
        }
    }

    /// Compute the runtime route for the executor's picking entry points — the
    /// relocated body of the bridge's old `resolve_runtime_route`. Returns
    /// `None` (⇒ arm-0 lowering) when there is no `selector` (the disabled
    /// opt-out) or the graph has no `Op::Branch` (the branchless fast-path:
    /// skip the topo walk + selector entirely so the common case is unchanged).
    /// The per-arm candidates are re-enumerated from the runtime binding
    /// registry (Step D — was the threaded `ExecutionPlan`).
    fn pick_route_for(
        graph: &Arc<RwLock<Graph>>,
        roots: &[NodeId],
        selector: Option<&dyn RuntimeSelector>,
        lookup: Option<&BackendRuntimeLookup>,
    ) -> Result<Option<PickedRoute>> {
        let Some(sel) = selector else {
            return Ok(None);
        };
        let g = graph
            .read()
            .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
        let has_branch =
            (0..g.len()).any(|i| matches!(g.node(NodeId(i)).op, Op::Branch { .. }));
        if !has_branch {
            return Ok(None);
        }
        let bindings = global_bindings();
        Ok(Some(pick_route(&g, roots, &bindings, sel, lookup)))
    }

    fn realize_inner(
        graph: Arc<RwLock<Graph>>,
        target: NodeId,
        inputs: StorageCache,
        order_source: OrderSource<'_>,
        sym_env: SymEnv,
    ) -> Result<(Arc<RwLock<Storage>>, Layout)> {
        // Auto-insert safety copies for in-place ops whose target
        // has additional readers in this realize set (residual-
        // connection cycle break). No-op when no destructive ops
        // are present.
        {
            let mut g = graph.write().map_err(|_| poisoned("graph lock"))?;
            let effective_roots = extend_with_side_effect_roots(&g, &[target]);
            insert_safety_copies(&mut g, &effective_roots);
        }

        // Execution plan + initial layouts for the input cache
        // entries, computed on the calling thread to keep the
        // compiler thread free of graph-locking responsibilities.
        // Side-effect roots (Op::Print, Op::Save, etc.) are merged
        // into the walk so their effects fire even when not
        // reachable from the user's `target`. `execution_plan`
        // integrates `derive_ordering`'s view-aware pinning so
        // destructive ops run AFTER non-destructive readers of
        // their targets.
        let (order, mut layout_cache): (Vec<NodeId>, HashMap<NodeId, Layout>) = {
            let g = graph.read().map_err(|_| poisoned("graph lock"))?;
            let effective_roots = extend_with_side_effect_roots(&g, &[target]);
            // PR-A3b-1: the OptimizedGraph path lowers via
            // `extract_runs`/`lower_run` (computed here, AFTER
            // `insert_safety_copies`, so the order covers freshly
            // inserted safety-copy nodes). The default path keeps the
            // pre-A3b-1 `execution_plan` walk. On a branchless graph the
            // two sequences are identical (A3a equivalence gate).
            let order = order_for(&g, &effective_roots, &order_source);
            let mut layouts = HashMap::with_capacity(inputs.len());
            for &id in inputs.keys() {
                layouts.insert(id, g.layout(id));
            }
            (order, layouts)
        };

        let (tx, rx) = channel::<Result<WorkItem>>();
        let graph_for_compiler = Arc::clone(&graph);
        let order_for_compiler = order.clone();

        // Compiler thread: read graph nodes, resolve kernels,
        // push WorkItems. On error, push the error and bail.
        let compiler = thread::spawn(move || {
            compiler_thread_body(
                graph_for_compiler, order_for_compiler, sym_env, tx,
            );
        });

        // Executor on this thread: consume WorkItems, gather
        // inputs from the cache, allocate outputs, call kernels,
        // populate the cache. After each destructive op
        // (Op::Release / Op::Move) evict the destroyed input from
        // the cache, unless it's the realize target.
        //
        // Phase 4.3: at each dispatch-chunk boundary (target_backend
        // change), check the live SystemTopology generation against
        // the plan's stamped generation. Mismatch surfaces
        // `Error::TopologyChanged`; the realize layer
        // (`pipelined_bridge`) catches it, rebuilds the plan against
        // the fresh topology, and retries.
        let plan_generation: Option<u64> = generation_for(&order_source);
        let mut current_chunk_backend: Option<BackendId> = None;
        let mut cache: StorageCache = inputs;
        // Step E A4b-1: per-node async-completion handles. Each producing node's
        // `execute_compiled` returns a `CompletionHandle` (CUDA → a recorded
        // `Event`; CPU/Vulkan → `Ready`) instead of being `wait`ed inline. We
        // store them here, wait the SOURCE producer's handle before a
        // cross-device copy (A4b-3, below), and drain whatever remains at
        // realize-end.
        let mut handles: HashMap<NodeId, CompletionHandle> = HashMap::new();
        // Step E A4b-4: eagerly-submitted-but-not-yet-waited Vulkan batches. The
        // OVERLAP enabler — when we LEAVE a Vulkan chunk (or are about to block on
        // a CUDA copy) we `submit_pending` the open Vulkan batch so the iGPU runs
        // it while the executor records/dispatches the next (CUDA) chunk, instead
        // of letting it sit merely-recorded until realize-end (the A4b-2 behavior,
        // which never overlapped). Each submitted batch is tracked here and waited
        // at the in-flight-lifetime guard (before a Vulkan read / Vulkan-referenced
        // eviction / realize-end). `multi_backend` gates ALL eager submits so a
        // single-device realize never touches this path (byte-identical to A4b-2).
        let mut inflight_vulkan: Vec<InflightVulkan> = Vec::new();
        // Step E A4b-4: set true the first time the dispatch order transitions
        // between two DIFFERENT non-CPU... actually any two distinct backends — a
        // genuine backend switch. On a pure-CUDA or pure-Vulkan graph there is no
        // switch, so this stays false and the eager-submit / in-flight machinery is
        // unreachable (single-device byte-identical + throughput-neutral, §5).
        let mut multi_backend = false;
        for item in rx {
            let item = item?;
            // Chunk-boundary hook (target_backend change). Existing duty: the
            // `TopologyChanged` generation check. Step E A4b-4 adds two duties:
            // (1) detect a genuine backend switch → arm `multi_backend`; (2) when
            // LEAVING a Vulkan chunk on a multi-backend graph, eager-submit the
            // open Vulkan batch so the iGPU starts it concurrently with the chunk
            // we are about to dispatch (the overlap win).
            if current_chunk_backend != Some(item.target_backend) {
                if let Some(plan_gen) = plan_generation {
                    let live = crate::dispatch::topology_generation();
                    if live != plan_gen {
                        return Err(Error::TopologyChanged {
                            plan_generation: plan_gen,
                            current_generation: live,
                        }
                        .bt());
                    }
                }
                let leaving = current_chunk_backend;
                if let Some(prev) = leaving {
                    if prev != item.target_backend {
                        // A real backend switch occurred — this realize is
                        // genuinely multi-backend.
                        multi_backend = true;
                        // Leaving a Vulkan chunk → submit it NOW so it runs on the
                        // iGPU while we record/dispatch the next chunk. Whole-chunk
                        // granularity (the open batch IS the just-finished Vulkan
                        // chunk with all its intra-CB barriers) — UAF/race-safe per
                        // `eager_submit_all_vulkan`'s contract.
                        if prev == BackendId::Vulkan {
                            eager_submit_all_vulkan(&cache, &mut inflight_vulkan)?;
                        }
                    }
                }
                current_chunk_backend = Some(item.target_backend);
            }
            // Step E A4b-4 (Invariant E — same-device cross-chunk RAW safety):
            // before RECORDING a new Vulkan op while in-flight Vulkan batches
            // exist, wait them. Vulkan only guarantees batches BEGIN in submission
            // order (they may overlap / complete out of order), so a later Vulkan
            // op that reads a buffer an in-flight batch wrote would race without
            // this. The in-flight batch was submitted at the prior chunk boundary
            // and an intervening (CUDA) chunk has since run, so this fence is
            // typically already signalled — the iGPU work overlapped, and we only
            // re-sync as Vulkan work resumes. No-op when single-device
            // (`multi_backend` false) or no batches are in flight.
            if multi_backend
                && item.target_backend == BackendId::Vulkan
                && !inflight_vulkan.is_empty()
            {
                drain_inflight_vulkan(&mut inflight_vulkan)?;
            }
            // Step E A4b-3 + A4b-4: the cross-device residency boundary.
            if matches!(item.kind, WorkItemKind::Copy { .. } | WorkItemKind::Move { .. }) {
                // A4b-4: before we (possibly) block the host on the CUDA producer,
                // eager-submit any open Vulkan chunk so the iGPU runs it WHILE we
                // wait on CUDA (the §5.1 ordering — submit Vulkan *before* draining
                // CUDA). Cheap + safe; gated to multi-backend (no-op single-device).
                if multi_backend {
                    eager_submit_all_vulkan(&cache, &mut inflight_vulkan)?;
                }
                if let Some(&producer) = item.inputs.first() {
                    // A4b-4: a VULKAN-source D2H must read COMPLETED data, and the
                    // Vulkan download path only flushes the OPEN batch (not our
                    // in-flight ones) — so wait the in-flight Vulkan batches here.
                    // For a CUDA/CPU source we do NOT wait Vulkan (the copy reads no
                    // Vulkan buffer); waiting it would needlessly serialize the
                    // independent Vulkan sub-DAG and kill the overlap. The Vulkan
                    // work that was eager-submitted at the chunk boundary keeps
                    // running concurrently until ITS result is the one being copied.
                    if multi_backend && copy_source_is_vulkan(&cache, producer)? {
                        drain_inflight_vulkan(&mut inflight_vulkan)?;
                    }
                    // Step E A4b-3: FINER CUDA source-drain — wait ONLY the source
                    // producer's recorded event (not the whole source device), so
                    // the OTHER sub-DAG's independent in-flight CUDA work keeps
                    // running. No-op for a Vulkan/CPU producer (no CUDA handle).
                    wait_producer_handle(&mut handles, producer)?;
                }
            }
            let handle = execute_work_item(&item, &mut cache, &mut layout_cache)?;
            store_handle(&mut handles, item.node_id, handle);
            if let Some(d_idx) = item.destructive_input {
                if let Some(&destroyed) = item.inputs.get(d_idx) {
                    if destroyed != target {
                        // Step E A4b-4 (in-flight-batch DATA-BUFFER lifetime — the
                        // UAF guard). A `SubmittedBatch` owns its CB/descriptors/
                        // transients but NOT the DATA buffers it reads (those live
                        // here in the cache). Freeing a data buffer an in-flight CB
                        // still reads is a use-after-free. Before this destructive
                        // eviction, wait ALL in-flight Vulkan batches (conservative
                        // — we can't cheaply map buffer→batch, so wait every one
                        // that could reference `destroyed`). This is design §3
                        // row 4's "wait H(evicted)", now coherent because under
                        // A4b-4 the batch IS submitted. No-op single-device / no
                        // in-flight batches.
                        if multi_backend && !inflight_vulkan.is_empty() {
                            drain_inflight_vulkan(&mut inflight_vulkan)?;
                        }
                        // Step E A2: ALSO drain the still-OPEN (recorded-but-
                        // unsubmitted) Vulkan batch — `force_flush` submits+waits it
                        // so a recorded command never reads freed memory either.
                        // (Together the two drains cover every Vulkan reference to
                        // `destroyed`; CUDA eviction is stream-ordered-safe via A3.)
                        if let Some(d_arc) = cache.get(&destroyed) {
                            force_flush_vulkan(d_arc)?;
                        }
                        cache.remove(&destroyed);
                        layout_cache.remove(&destroyed);
                        // Drop the evicted node's CUDA handle (if still present)
                        // so the realize-end drain's empty-map assert stays
                        // meaningful. The free itself is stream-ordered-safe (A3).
                        handles.remove(&destroyed);
                    }
                }
            }
        }

        compiler
            .join()
            .map_err(|_| Error::Msg("compiler thread panicked".to_string()).bt())?;

        // Step E A4b-1: drain every outstanding async handle before the result is
        // read / the cache drops (freeing intermediates). For CUDA this waits the
        // recorded events (one stream/device ⇒ waiting the latest drains all prior
        // stream work too; see `drain_handles`).
        drain_handles(&mut handles)?;
        // Step E A4b-4: wait every eagerly-submitted (in-flight) Vulkan batch
        // before the cache drops, freeing the data buffers their CBs read. Empty
        // on a single-device realize (nothing was eager-submitted).
        drain_inflight_vulkan(&mut inflight_vulkan)?;
        // Step E A4b-2: drain all deferred Vulkan work before reading/returning
        // the result (and before the cache drops, freeing buffers). The A2
        // submit+wait is now SPLIT — `drain_vulkan_pending` submits the open
        // batch then waits it via a `VulkanCompletion` handle. Byte-identical to
        // A2 for pure-Vulkan (one submission at realize-end, just split).
        drain_vulkan_pending(&cache)?;
        debug_assert!(
            inflight_vulkan.is_empty(),
            "A4b-4: in-flight Vulkan batch list must be empty after realize-end drain",
        );

        let storage = cache.remove(&target).ok_or_else(|| {
            Error::Msg(format!(
                "PipelinedExecutor::realize: target slot {:?} not populated after execution",
                target
            ))
            .bt()
        })?;
        let layout = layout_cache.remove(&target).ok_or_else(|| {
            Error::Msg(format!(
                "PipelinedExecutor::realize: target layout {:?} not populated after execution",
                target
            ))
            .bt()
        })?;
        Ok((storage, layout))
    }

    /// Realize multiple targets in one walk. Each target's transitive
    /// dependency set is collapsed into a single execution plan
    /// (via [`execution_plan`]); shared subgraphs are evaluated
    /// once. Returns a `Vec<(Storage, Layout)>` whose order matches
    /// `targets`.
    ///
    /// This is Phase A of Phase 7.6 step 9c — feature-parity with the
    /// legacy `GraphExecutor::realize_many_f32`. Multi-session
    /// migration target ([memory: project_phase_7_6_step_9c_parity_audit.md]).
    ///
    /// Pre-conditions: every reachable `Op::Const` must be in
    /// `inputs`; every reachable non-`Const` must have its
    /// `target_backend` set; the op + dtype must be registered in
    /// `global_bindings()`. Same as single-target [`realize`].
    pub fn realize_many(
        graph: Arc<RwLock<Graph>>,
        targets: &[NodeId],
        inputs: StorageCache,
    ) -> Result<Vec<(Arc<RwLock<Storage>>, Layout)>> {
        Self::realize_many_inner(graph, targets, inputs, OrderSource::Default, SymEnv::default())
    }

    /// Multi-target PR-A3b-1 entry — the `realize_many` sibling of
    /// [`realize_with_optimized`]. Drives the executor from the
    /// `OptimizedGraph`'s run lowering via the binding-table-lookup
    /// path, with the optimize-time `generation` keying the
    /// `TopologyChanged` chunk-boundary check.
    pub fn realize_many_with_optimized(
        graph: Arc<RwLock<Graph>>,
        targets: &[NodeId],
        inputs: StorageCache,
        optimized: &OptimizedGraph,
    ) -> Result<Vec<(Arc<RwLock<Storage>>, Layout)>> {
        Self::realize_many_inner(
            graph,
            targets,
            inputs,
            OrderSource::Optimized { optimized, route: None },
            SymEnv::default(),
        )
    }

    /// Env-carrying sibling of [`realize_many_with_optimized`] (Phase D
    /// symbolic extents). An empty env is byte-identical to it.
    pub fn realize_many_with_optimized_env(
        graph: Arc<RwLock<Graph>>,
        targets: &[NodeId],
        inputs: StorageCache,
        optimized: &OptimizedGraph,
        sym_env: SymEnv,
    ) -> Result<Vec<(Arc<RwLock<Storage>>, Layout)>> {
        Self::realize_many_inner(
            graph,
            targets,
            inputs,
            OrderSource::Optimized { optimized, route: None },
            sym_env,
        )
    }

    /// Multi-target PR-C1 entry — the `realize_many` sibling of
    /// [`realize_with_optimized_route`]. Lowers each target's runs
    /// following the picker's chosen arms via
    /// [`fuel_graph::lower_picked_route`].
    pub fn realize_many_with_optimized_route(
        graph: Arc<RwLock<Graph>>,
        targets: &[NodeId],
        inputs: StorageCache,
        optimized: &OptimizedGraph,
        route: &PickedRoute,
    ) -> Result<Vec<(Arc<RwLock<Storage>>, Layout)>> {
        Self::realize_many_inner(
            graph,
            targets,
            inputs,
            OrderSource::Optimized { optimized, route: Some(route) },
            SymEnv::default(),
        )
    }

    /// Env-carrying sibling of [`realize_many_with_optimized_route`]
    /// (Phase D symbolic extents). An empty env is byte-identical to it.
    pub fn realize_many_with_optimized_route_env(
        graph: Arc<RwLock<Graph>>,
        targets: &[NodeId],
        inputs: StorageCache,
        optimized: &OptimizedGraph,
        route: &PickedRoute,
        sym_env: SymEnv,
    ) -> Result<Vec<(Arc<RwLock<Storage>>, Layout)>> {
        Self::realize_many_inner(
            graph,
            targets,
            inputs,
            OrderSource::Optimized { optimized, route: Some(route) },
            sym_env,
        )
    }

    /// Multi-target sibling of [`realize_with_optimized_picking_env`] —
    /// the executor picks one arm per `Op::Branch` (cleanup Step C/D) over the
    /// effective targets, then lowers the chosen route.
    pub fn realize_many_with_optimized_picking_env(
        graph: Arc<RwLock<Graph>>,
        targets: &[NodeId],
        inputs: StorageCache,
        optimized: &OptimizedGraph,
        selector: Option<&dyn RuntimeSelector>,
        lookup: Option<&BackendRuntimeLookup>,
        sym_env: SymEnv,
    ) -> Result<Vec<(Arc<RwLock<Storage>>, Layout)>> {
        match Self::pick_route_for(&graph, targets, selector, lookup)? {
            Some(route) => Self::realize_many_with_optimized_route_env(
                graph, targets, inputs, optimized, &route, sym_env,
            ),
            None => Self::realize_many_with_optimized_env(
                graph, targets, inputs, optimized, sym_env,
            ),
        }
    }

    fn realize_many_inner(
        graph: Arc<RwLock<Graph>>,
        targets: &[NodeId],
        inputs: StorageCache,
        order_source: OrderSource<'_>,
        sym_env: SymEnv,
    ) -> Result<Vec<(Arc<RwLock<Storage>>, Layout)>> {
        if targets.is_empty() {
            return Ok(Vec::new());
        }

        // Auto-insert safety copies for in-place ops whose target
        // has additional readers in this realize set (residual-
        // connection cycle break). No-op when no destructive ops
        // are present.
        {
            let mut g = graph.write().map_err(|_| poisoned("graph lock"))?;
            let effective_roots = extend_with_side_effect_roots(&g, targets);
            insert_safety_copies(&mut g, &effective_roots);
        }

        // Execution plan covering every target's dependency set +
        // initial layouts for the input cache entries. Side-effect
        // roots merge in so their effects fire even when not
        // reachable from any target. `execution_plan` integrates
        // `derive_ordering`'s view-aware pinning so destructive ops
        // run AFTER non-destructive readers of their targets.
        let (order, mut layout_cache): (Vec<NodeId>, HashMap<NodeId, Layout>) = {
            let g = graph.read().map_err(|_| poisoned("graph lock"))?;
            let effective_roots = extend_with_side_effect_roots(&g, targets);
            // PR-A3b-1: derive from the OptimizedGraph's run lowering
            // when present (post-safety-copies); else the legacy
            // `execution_plan` walk. Identical on branchless graphs.
            let order = order_for(&g, &effective_roots, &order_source);
            let mut layouts = HashMap::with_capacity(inputs.len());
            for &id in inputs.keys() {
                layouts.insert(id, g.layout(id));
            }
            (order, layouts)
        };

        // Caller's target set — used to gate destructive-input
        // eviction so we don't drop a tensor the caller asked for.
        let target_set: std::collections::HashSet<NodeId> = targets.iter().copied().collect();

        let (tx, rx) = channel::<Result<WorkItem>>();
        let graph_for_compiler = Arc::clone(&graph);
        let order_for_compiler = order.clone();

        let compiler = thread::spawn(move || {
            compiler_thread_body(
                graph_for_compiler, order_for_compiler, sym_env, tx,
            );
        });

        // Phase 4.3: per-chunk SystemTopology generation check (see
        // realize_inner for the rationale). PR-A3b-1: the OptimizedGraph
        // path supplies its own optimize-time generation.
        let plan_generation: Option<u64> = generation_for(&order_source);
        let mut current_chunk_backend: Option<BackendId> = None;
        let mut cache: StorageCache = inputs;
        // Step E A4b-1/A4b-3: per-node async-completion handles (see realize_inner
        // for the full rationale). Map drained at realize-end; the cross-device
        // copy waits the SOURCE producer's handle (A4b-3).
        let mut handles: HashMap<NodeId, CompletionHandle> = HashMap::new();
        // Step E A4b-4: eagerly-submitted in-flight Vulkan batches + the
        // multi-backend gate. See realize_inner for the full rationale — this is
        // the identical eager-submit / in-flight-lifetime machinery applied to the
        // multi-target walk. `multi_backend` arms on a genuine backend switch and
        // gates every eager submit, so a single-device realize_many is
        // byte-identical to A4b-2.
        let mut inflight_vulkan: Vec<InflightVulkan> = Vec::new();
        let mut multi_backend = false;
        for item in rx {
            let item = item?;
            // Chunk-boundary hook: TopologyChanged check + (A4b-4) backend-switch
            // detection + eager-submit-on-leaving-a-Vulkan-chunk.
            if current_chunk_backend != Some(item.target_backend) {
                if let Some(plan_gen) = plan_generation {
                    let live = crate::dispatch::topology_generation();
                    if live != plan_gen {
                        return Err(Error::TopologyChanged {
                            plan_generation: plan_gen,
                            current_generation: live,
                        }
                        .bt());
                    }
                }
                let leaving = current_chunk_backend;
                if let Some(prev) = leaving {
                    if prev != item.target_backend {
                        multi_backend = true;
                        if prev == BackendId::Vulkan {
                            eager_submit_all_vulkan(&cache, &mut inflight_vulkan)?;
                        }
                    }
                }
                current_chunk_backend = Some(item.target_backend);
            }
            // Step E A4b-4 (Invariant E): wait in-flight Vulkan before recording a
            // new Vulkan op (same-device cross-chunk RAW safety). See realize_inner.
            if multi_backend
                && item.target_backend == BackendId::Vulkan
                && !inflight_vulkan.is_empty()
            {
                drain_inflight_vulkan(&mut inflight_vulkan)?;
            }
            // Step E A4b-3 + A4b-4: cross-device residency boundary. See
            // realize_inner for the full rationale.
            if matches!(item.kind, WorkItemKind::Copy { .. } | WorkItemKind::Move { .. }) {
                // A4b-4: submit any open Vulkan chunk so the iGPU runs while the
                // host (possibly) waits on the CUDA producer.
                if multi_backend {
                    eager_submit_all_vulkan(&cache, &mut inflight_vulkan)?;
                }
                if let Some(&producer) = item.inputs.first() {
                    // A4b-4: wait in-flight Vulkan ONLY for a Vulkan-source D2H
                    // (reads completed data; download flushes only the OPEN batch).
                    // A CUDA/CPU source does NOT wait Vulkan (preserves overlap).
                    if multi_backend && copy_source_is_vulkan(&cache, producer)? {
                        drain_inflight_vulkan(&mut inflight_vulkan)?;
                    }
                    // Step E A4b-3: finer CUDA source-drain — wait ONLY the producer.
                    wait_producer_handle(&mut handles, producer)?;
                }
            }
            let handle = execute_work_item(&item, &mut cache, &mut layout_cache)?;
            store_handle(&mut handles, item.node_id, handle);
            if let Some(d_idx) = item.destructive_input {
                if let Some(&destroyed) = item.inputs.get(d_idx) {
                    if !target_set.contains(&destroyed) {
                        // Step E A4b-4 (in-flight DATA-BUFFER lifetime — UAF guard):
                        // wait ALL in-flight Vulkan batches before freeing a buffer
                        // their CBs may read (conservative buffer→batch mapping).
                        // See realize_inner for the UAF argument. No-op
                        // single-device / no in-flight batches.
                        if multi_backend && !inflight_vulkan.is_empty() {
                            drain_inflight_vulkan(&mut inflight_vulkan)?;
                        }
                        // Step E A2: ALSO drain the still-OPEN recorded Vulkan batch
                        // (force_flush submit+wait) so a recorded command never
                        // reads freed memory. CUDA eviction is A3 stream-safe.
                        if let Some(d_arc) = cache.get(&destroyed) {
                            force_flush_vulkan(d_arc)?;
                        }
                        cache.remove(&destroyed);
                        layout_cache.remove(&destroyed);
                        // CUDA eviction is stream-ordered-safe (A3); drop the
                        // handle so the realize-end empty-map assert stays valid.
                        handles.remove(&destroyed);
                    }
                }
            }
        }

        compiler
            .join()
            .map_err(|_| Error::Msg("compiler thread panicked".to_string()).bt())?;

        // Step E A4b-1: drain every outstanding async handle (CUDA events) before
        // results are read / the cache drops.
        drain_handles(&mut handles)?;
        // Step E A4b-4: wait every eagerly-submitted in-flight Vulkan batch before
        // the cache drops. Empty on a single-device realize.
        drain_inflight_vulkan(&mut inflight_vulkan)?;
        // Step E A4b-2: drain all deferred Vulkan work — submit the open batch
        // then wait it via a `VulkanCompletion` handle (the A2 submit+wait, split;
        // byte-identical for pure-Vulkan — one submission at realize-end).
        drain_vulkan_pending(&cache)?;
        debug_assert!(
            inflight_vulkan.is_empty(),
            "A4b-4: in-flight Vulkan batch list must be empty after realize-end drain",
        );

        // Verify every target was realized + collect outputs in
        // target order. We don't `remove` from the cache because the
        // same NodeId can appear twice in `targets` (caller wants the
        // same storage twice) — clone the Arc instead.
        let mut out = Vec::with_capacity(targets.len());
        for &target in targets {
            let storage = cache.get(&target).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor::realize_many: target {:?} not populated after execution",
                    target
                ))
                .bt()
            })?;
            let layout = layout_cache.get(&target).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor::realize_many: target layout {:?} not populated after execution",
                    target
                ))
                .bt()
            })?;
            out.push((storage, layout));
        }
        Ok(out)
    }
}

/// Resolve the executor's dispatch order from the configured
/// [`OrderSource`]. PR-A3b-1: `Optimized` lowers the in-place
/// "plan IS the graph" form via `extract_runs`/`lower_run`; `Default`
/// keeps the pre-A3b-1 `execution_plan` walk. Both produce the same
/// `NodeId` sequence on a branchless graph (A3a equivalence gate),
/// computed here AFTER safety-copy insertion so either path covers
/// any freshly inserted copies.
fn order_for(
    graph: &Graph,
    effective_roots: &[NodeId],
    order_source: &OrderSource<'_>,
) -> Vec<NodeId> {
    match order_source {
        OrderSource::Default => execution_plan(graph, effective_roots),
        OrderSource::Optimized { optimized, route } => {
            // Lower the runs over the *effective* roots (user targets +
            // any spliced/side-effect roots) so the order matches the
            // walk's reachable set rather than only the optimize-time
            // roots. On a branchless graph this is the same concatenated
            // `lower_run` sequence the OptimizedGraph reports.
            //
            // PR-C1: when the picker supplied a route, follow the chosen
            // arms (`lower_picked_route`); otherwise the arm-0 lowering
            // (`OptimizedGraph::dispatch_order`). An empty route is
            // byte-identical to arm-0, so the two paths agree whenever
            // the picker chose arm-0 everywhere (the no-pressure case).
            match route {
                Some(route) => {
                    fuel_graph::lower_picked_route(graph, effective_roots, route)
                }
                None => {
                    let view = OptimizedGraph {
                        roots: effective_roots.to_vec(),
                        generation: optimized.generation,
                    };
                    view.dispatch_order(graph)
                }
            }
        }
    }
}

/// The topology generation that keys the executor's `TopologyChanged`
/// chunk-boundary check. Uses the `OptimizedGraph`'s optimize-time
/// generation (PR-A3b-1 path). `None` ⇒ no check (the plain
/// `realize`/`realize_many` entries).
fn generation_for(
    order_source: &OrderSource<'_>,
) -> Option<u64> {
    match order_source {
        OrderSource::Optimized { optimized, .. } => Some(optimized.generation),
        OrderSource::Default => None,
    }
}

/// Compiler thread body. Reads each node in topo order, resolves
/// its kernel via `global_bindings()`, and pushes a `WorkItem` on
/// the channel. Sends `Err(...)` and stops on the first failure.
fn compiler_thread_body(
    graph: Arc<RwLock<Graph>>,
    order: Vec<NodeId>,
    sym_env: SymEnv,
    tx: Sender<Result<WorkItem>>,
) {
    let bindings = global_bindings();

    let g = match graph.read() {
        Ok(g) => g,
        Err(_) => {
            let _ = tx.send(Err(poisoned("graph lock in compiler")));
            return;
        }
    };

    // Compiler-thread-local layout cache. Populated as compile_one
    // walks topologically; downstream nodes look up their inputs'
    // layouts here (rather than from the graph side-table) to honor
    // the strided layouts emitted by metadata-only view ops earlier
    // in the same realize call.
    let mut layout_cache: HashMap<NodeId, Layout> = HashMap::new();

    for id in order {
        let item = compile_one(&g, id, &mut layout_cache, &bindings, &sym_env);
        let stop_on_err = item.is_err();
        if tx.send(item).is_err() {
            return;
        }
        if stop_on_err {
            return;
        }
    }
}

/// Resolve the [`CompiledNode`] for a kernel-bearing graph node via
/// the first-registered [`compile_node`] binding-table lookup.
///
/// `op_params` always comes from the executor's
/// `op_to_op_params(graph, node, layout_cache, sym_env)` — the live
/// OpParams shape (reduce dims, conv geometry, the per-pass
/// `SymEnv`-resolved `WriteSlice` offset, etc.) derives from the graph
/// node + its resolved input layouts at execute time.
fn resolve_compiled(
    op_kind: OpKind,
    dtypes: &[DType],
    target_backend: BackendId,
    op_params: OpParams,
    bindings: &KernelBindingTable,
) -> Result<CompiledNode> {
    compile_node(op_kind, dtypes, target_backend, op_params, bindings)
}

/// Resolve one node into a `WorkItem` and update `layout_cache`
/// with the node's output layout. Three op shapes:
///
/// - `Op::Const` — adopts the entry from the input cache; layout is
///   read from the graph's side-table (or its contiguous fallback).
///
/// - Metadata-only view op — output layout is read from the graph's
///   side-table (populated by `Graph::push` at construction time);
///   the executor adopts the input's Storage Arc (no allocation, no
///   kernel).
///
/// - Computational op — output layout is `Layout::contiguous(node.shape)`
///   because today's kernels write contiguous output. The compiler
///   resolves the kernel ref and emits a Kernel work item.
fn compile_one(
    graph: &Graph,
    id: NodeId,
    layout_cache: &mut HashMap<NodeId, Layout>,
    bindings: &KernelBindingTable,
    sym_env: &SymEnv,
) -> Result<WorkItem> {
    let node = graph.node(id);
    let elem_count = node.shape.elem_count();
    let inputs = node.inputs.clone();
    // Snapshot destructive-input metadata from the graph at compile
    // time so the executor can evict the destroyed input from its
    // cache after the op runs (Op::Release / Op::Move semantics).
    let destructive_input = node.op.destructive_input();

    if matches!(node.op, Op::Const) {
        let output_layout = graph.layout(id);
        layout_cache.insert(id, output_layout.clone());
        return Ok(WorkItem {
            node_id: id,
            inputs,
            elem_count,
            dtype: node.dtype,
            target_backend: BackendId::Cpu,
            kind: WorkItemKind::ConstAdopt,
            compiled: None,
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    if node.op.is_view_op() {
        if inputs.len() != 1 {
            return Err(Error::Msg(format!(
                "view op {:?} expects 1 input, got {}",
                node.op,
                inputs.len(),
            ))
            .bt());
        }
        // Layout is read from the graph's side-table — populated by
        // `Graph::push` at construction time for view ops, and by
        // graph-rewriting opt passes that emit view nodes. The
        // compiler does not re-derive: graph.layout(id) is the single
        // source of truth.
        let output_layout = graph.layout(id);
        layout_cache.insert(id, output_layout.clone());
        // Inherit the upstream's target_backend (or default CPU) —
        // metadata-only adoption doesn't actually run on a backend,
        // but downstream consumers look at target_backend so it
        // needs to be set sensibly. Any device works.
        let target_backend = graph
            .target_backend(id)
            .or_else(|| graph.target_backend(inputs[0]))
            .unwrap_or(BackendId::Cpu);
        return Ok(WorkItem {
            node_id: id,
            inputs,
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::ViewOf { input: node.inputs[0] },
            compiled: None,
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    // Op::View — multi-output projection (Option C, Session 4).
    // Structurally identical to a metadata-only view op at realize
    // time: clone the producer's Storage Arc into the View's cache
    // slot. The graph's layout side-table, populated by
    // `Tensor::view`, already carries the slot's effective layout
    // (slot.byte_offset baked into start_offset in slot-dtype-element
    // units).
    if let Op::View { .. } = node.op {
        if inputs.len() != 1 {
            return Err(Error::Msg(format!(
                "Op::View expects 1 input (the producer), got {}",
                inputs.len(),
            )).bt());
        }
        let output_layout = graph.layout(id);
        layout_cache.insert(id, output_layout.clone());
        let target_backend = graph
            .target_backend(id)
            .or_else(|| graph.target_backend(inputs[0]))
            .unwrap_or(BackendId::Cpu);
        return Ok(WorkItem {
            node_id: id,
            inputs: inputs.clone(),
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::SlotView { producer: inputs[0] },
            compiled: None,
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    // Op::ViewOwned — copies the slot's bytes into a fresh
    // contiguous Storage at realize time. Output layout is
    // contiguous starting at offset 0 (the memcpy produced an
    // independent buffer).
    if let Op::ViewOwned { slot } = node.op {
        if inputs.len() != 1 {
            return Err(Error::Msg(format!(
                "Op::ViewOwned expects 1 input (the producer), got {}",
                inputs.len(),
            )).bt());
        }
        let output_layout = Layout::contiguous(node.shape.clone());
        layout_cache.insert(id, output_layout.clone());
        let target_backend = graph
            .target_backend(id)
            .or_else(|| graph.target_backend(inputs[0]))
            .unwrap_or(BackendId::Cpu);
        return Ok(WorkItem {
            node_id: id,
            inputs: inputs.clone(),
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::SlotOwn { producer: inputs[0], slot },
            compiled: None,
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    if matches!(node.op, Op::Release) {
        // Op::Release produces a zero-element marker output. The
        // marker storage is CPU + 0 bytes — nothing reads it; it
        // exists so cache lookups on the Release NodeId resolve.
        // The actual deallocation of `inputs[0]` happens in the
        // realize loop via `destructive_input` cleanup.
        if inputs.len() != 1 {
            return Err(Error::Msg(format!(
                "Op::Release expects 1 input, got {}",
                inputs.len(),
            ))
            .bt());
        }
        let output_layout = Layout::contiguous(node.shape.clone());
        layout_cache.insert(id, output_layout.clone());
        // Inherit the source's target_backend so any downstream
        // backend-aware logic (telemetry, scheduling) sees a sensible
        // value. The marker doesn't actually run on any backend.
        let target_backend = graph
            .target_backend(id)
            .or_else(|| graph.target_backend(inputs[0]))
            .unwrap_or(BackendId::Cpu);
        return Ok(WorkItem {
            node_id: id,
            inputs,
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::ReleaseMarker,
            compiled: None,
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    // In-place ops (Phase 3 of in-place ops infrastructure): kernel
    // lookup goes through the standard binding-table path, but the
    // executor adopts the target's Arc as the output instead of
    // allocating fresh bytes. The compile-time predicate is exactly
    // `op.destructive_input().is_some()` AND NOT one of the
    // structural ops (WriteSlice / ZeroFill / Release / Move) which
    // have their own dedicated WorkItemKind arms in this function.
    if !matches!(node.op,
        Op::WriteSlice { .. } | Op::WriteSliceRotating { .. } | Op::ZeroFill | Op::Release | Op::Move { .. },
    ) {
        if let Some(target_idx) = node.op.destructive_input() {
            if inputs.len() <= target_idx {
                return Err(Error::Msg(format!(
                    "in-place op {:?} declares destructive_input={target_idx} \
                     but has only {} input(s)",
                    node.op, inputs.len(),
                ))
                .bt());
            }
            let target_backend = graph.target_backend(id).ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: in-place op node {id:?} has no target_backend set",
                ))
                .bt()
            })?;
            let op_kind = op_to_op_kind(&node.op).ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: in-place op {:?} has no op_to_op_kind mapping",
                    node.op,
                ))
                .bt()
            })?;
            let op_params = op_to_op_params(graph, node, layout_cache, sym_env)?;
            let dtypes = build_lookup_dtypes(graph, node);
            let compiled = resolve_compiled(
                op_kind, &dtypes, target_backend, op_params, bindings,
            )?;
            // Output adopts target's Layout (same Storage Arc, same shape).
            let output_layout = graph.layout(inputs[target_idx]);
            layout_cache.insert(id, output_layout.clone());
            return Ok(WorkItem {
                node_id: id,
                inputs: inputs.clone(),
                elem_count,
                dtype: node.dtype,
                target_backend,
                kind: WorkItemKind::InplaceKernel { target_idx },
                compiled: Some(compiled),
                output_layout,
                destructive_input,
                output_bundle: None,
            });
        }
    }

    if matches!(node.op, Op::WriteSlice { .. }) {
        // WriteSlice: kernel lookup happens like a normal kernel, but
        // the work item carries a dedicated kind so the executor
        // knows to adopt the destination's Arc as the output rather
        // than allocating a fresh buffer. The destination's NodeId is
        // captured here so the executor doesn't have to re-derive it.
        if inputs.len() != 2 {
            return Err(Error::Msg(format!(
                "Op::WriteSlice expects 2 inputs (destination, source), got {}",
                inputs.len(),
            ))
            .bt());
        }
        let target_backend = graph.target_backend(id).ok_or_else(|| {
            Error::Msg(format!(
                "PipelinedExecutor: WriteSlice node {:?} has no target_backend set",
                id
            ))
            .bt()
        })?;
        let op_params = op_to_op_params(graph, node, layout_cache, sym_env)?;
        let dtypes = build_lookup_dtypes(graph, node);
        let compiled = resolve_compiled(
            OpKind::WriteSlice, &dtypes, target_backend, op_params, bindings,
        )?;
        // Output adopts the destination's Layout — same Storage Arc,
        // same shape. Downstream consumers that want a post-write
        // sub-extent compose an explicit Op::Slice (WriteSlice does
        // not encode partial extents).
        let output_layout = graph.layout(inputs[0]);
        layout_cache.insert(id, output_layout.clone());
        return Ok(WorkItem {
            node_id: id,
            inputs: inputs.clone(),
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::WriteSlice { dest: inputs[0], source: inputs[1] },
            compiled: Some(compiled),
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    if matches!(node.op, Op::WriteSliceRotating { .. }) {
        // WriteSliceRotating: same shape as WriteSlice but with a
        // third input (position scalar). Dispatch through the
        // binding table at `OpKind::WriteSliceRotating`; the
        // executor's arm reads the position from `position`'s
        // storage and the kernel splits across the ring boundary.
        if inputs.len() != 3 {
            return Err(Error::Msg(format!(
                "Op::WriteSliceRotating expects 3 inputs (destination, source, position), got {}",
                inputs.len(),
            ))
            .bt());
        }
        let target_backend = graph.target_backend(id).ok_or_else(|| {
            Error::Msg(format!(
                "PipelinedExecutor: WriteSliceRotating node {:?} has no target_backend set",
                id
            ))
            .bt()
        })?;
        let op_params = op_to_op_params(graph, node, layout_cache, sym_env)?;
        let dtypes = build_lookup_dtypes(graph, node);
        let compiled = resolve_compiled(
            OpKind::WriteSliceRotating, &dtypes, target_backend, op_params, bindings,
        )?;
        let output_layout = graph.layout(inputs[0]);
        layout_cache.insert(id, output_layout.clone());
        return Ok(WorkItem {
            node_id: id,
            inputs: inputs.clone(),
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::WriteSliceRotating {
                dest: inputs[0], source: inputs[1], position: inputs[2],
            },
            compiled: Some(compiled),
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    if let Op::Alloc { target } = node.op {
        // Op::Alloc { target }: produce a fresh zero-init Storage on
        // `target` with node.shape * node.dtype. Zero inputs.
        // Direct executor dispatch via WorkItemKind::Alloc — no
        // binding-table lookup (the device handle threading isn't
        // expressible through the binding table's current key shape).
        if !inputs.is_empty() {
            return Err(Error::Msg(format!(
                "Op::Alloc expects 0 inputs, got {}",
                inputs.len(),
            ))
            .bt());
        }
        let output_layout = Layout::contiguous(node.shape.clone());
        layout_cache.insert(id, output_layout.clone());
        // target_backend is informational here (the executor's Alloc
        // arm consults target_location for allocation routing); set
        // it to target's backend for consistency with how downstream
        // nodes inherit/read it.
        let target_backend = match target {
            DeviceLocation::Cpu => BackendId::Cpu,
            DeviceLocation::Cuda { .. } => BackendId::Cuda,
            DeviceLocation::Vulkan { .. } => BackendId::Vulkan,
            DeviceLocation::Metal { .. } => BackendId::Metal,
        };
        return Ok(WorkItem {
            node_id: id,
            inputs,
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::Alloc { target_location: target },
            compiled: None,
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    if matches!(node.op, Op::ZeroFill) {
        // Op::ZeroFill: 1 input (the buffer to zero); output aliases
        // input's Storage Arc (destructive in-place). No binding-
        // table lookup — per-backend dispatch in execute_work_item.
        if inputs.len() != 1 {
            return Err(Error::Msg(format!(
                "Op::ZeroFill expects 1 input, got {}",
                inputs.len(),
            ))
            .bt());
        }
        // Output layout: contiguous in the node's shape. We don't
        // require the input to be contiguous (auto_contiguize in the
        // executor handles strided inputs), but the destructive
        // semantic means callers should only ZeroFill an Op::Alloc
        // output today.
        let output_layout = Layout::contiguous(node.shape.clone());
        layout_cache.insert(id, output_layout.clone());
        let target_backend = graph
            .target_backend(id)
            .or_else(|| graph.target_backend(inputs[0]))
            .unwrap_or(BackendId::Cpu);
        return Ok(WorkItem {
            node_id: id,
            inputs,
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::ZeroFill,
            compiled: None,
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    if let Op::Copy { target } = node.op {
        // Op::Copy { target }: download bytes from the source residency
        // to a fresh Storage allocated on `target`. Kernel lookup uses
        // `target_backend` = source backend (set by `prepare()` on the
        // realize device the source is on); the wrapper does the actual
        // transfer (e.g. Vulkan::download_bytes → memcpy into a CPU
        // CpuStorageBytes). The output Storage's variant is determined
        // by `target_location` and allocated in `execute_work_item`'s
        // WorkItemKind::Copy arm.
        if inputs.len() != 1 {
            return Err(Error::Msg(format!(
                "Op::Copy expects 1 input, got {}",
                inputs.len(),
            ))
            .bt());
        }
        let target_backend = graph.target_backend(id).ok_or_else(|| {
            Error::Msg(format!(
                "PipelinedExecutor: Op::Copy node {:?} has no target_backend \
                 set (= source backend, the one whose kernel runs the \
                 download)",
                id
            ))
            .bt()
        })?;
        let op_params = OpParams::None;
        let dtypes = build_lookup_dtypes(graph, node);
        let compiled = resolve_compiled(
            OpKind::Copy, &dtypes, target_backend, op_params, bindings,
        )?;
        // Output layout: contiguous in the node's shape (mirrors the
        // source's logical shape). Auto-contiguize on the input side
        // (in execute_work_item) handles strided source views by
        // materializing them into a contiguous source buffer before
        // the kernel runs.
        let output_layout = Layout::contiguous(node.shape.clone());
        layout_cache.insert(id, output_layout.clone());
        return Ok(WorkItem {
            node_id: id,
            inputs,
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::Copy { target_location: target },
            compiled: Some(compiled),
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    if let Op::Move { target } = node.op {
        // Op::Move { target }: identical data-movement half to
        // Op::Copy — kernel lookup at (OpKind::Copy, [dt, dt],
        // source_backend); output allocated on `target` by the
        // executor's Copy/Move arm. The destructive half (release
        // the source) rides the realize loop's `destructive_input`
        // cleanup via the snapshot taken at the top of this function
        // (`Op::Move::destructive_input() == Some(0)`). Ordering
        // safety (a Move must not strand a sibling consumer of its
        // source) is `execution_plan`/`derive_ordering`'s job — the
        // Move is pinned after every non-destructive reader of the
        // source's alias set before any work item is emitted.
        if inputs.len() != 1 {
            return Err(Error::Msg(format!(
                "Op::Move expects 1 input, got {}",
                inputs.len(),
            ))
            .bt());
        }
        let target_backend = graph.target_backend(id).ok_or_else(|| {
            Error::Msg(format!(
                "PipelinedExecutor: Op::Move node {:?} has no target_backend \
                 set (= source backend, the one whose kernel runs the \
                 transfer)",
                id
            ))
            .bt()
        })?;
        let op_params = OpParams::None;
        let dtypes = build_lookup_dtypes(graph, node);
        let compiled = resolve_compiled(
            OpKind::Copy, &dtypes, target_backend, op_params, bindings,
        )?;
        // Output layout: contiguous in the node's shape, same as
        // Op::Copy (the transfer materializes strided sources via
        // auto-contiguize on the input side).
        let output_layout = Layout::contiguous(node.shape.clone());
        layout_cache.insert(id, output_layout.clone());
        return Ok(WorkItem {
            node_id: id,
            inputs,
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::Move { target_location: target },
            compiled: Some(compiled),
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    if matches!(node.op, Op::Reshape(_)) {
        if inputs.len() != 1 {
            return Err(Error::Msg(format!(
                "Op::Reshape expects 1 input, got {}",
                inputs.len(),
            ))
            .bt());
        }
        let input_layout = graph.layout(inputs[0]);
        // Output is contiguous in the new shape — bytes per element
        // are unchanged, so a contiguous input flows through with
        // zero copy. A non-contiguous input is auto-contiguized at
        // execute time and the result is naturally contiguous.
        let output_layout = Layout::contiguous(node.shape.clone());
        layout_cache.insert(id, output_layout.clone());
        let target_backend = graph
            .target_backend(id)
            .or_else(|| graph.target_backend(inputs[0]))
            .unwrap_or(BackendId::Cpu);
        // Sanity: same total element count.
        let in_elem_count = input_layout.shape().elem_count();
        if in_elem_count != elem_count {
            return Err(Error::Msg(format!(
                "Op::Reshape changes element count: input {} → output {}",
                in_elem_count, elem_count,
            ))
            .bt());
        }
        return Ok(WorkItem {
            node_id: id,
            inputs,
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::ContiguizeOf { input: node.inputs[0] },
            compiled: None,
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    if matches!(node.op, Op::Contiguize) {
        // Op::Contiguize: same execute-time path as Reshape, but
        // output shape == input shape so there's no element-count
        // check. Zero-copy when input is already contiguous +
        // zero-offset (the ContiguizeOf arm in execute_work_item
        // adopts the input Arc); otherwise auto_contiguize allocates
        // a fresh contiguous Storage on the input's backend.
        if inputs.len() != 1 {
            return Err(Error::Msg(format!(
                "Op::Contiguize expects 1 input, got {}",
                inputs.len(),
            ))
            .bt());
        }
        let output_layout = Layout::contiguous(node.shape.clone());
        layout_cache.insert(id, output_layout.clone());
        let target_backend = graph
            .target_backend(id)
            .or_else(|| graph.target_backend(inputs[0]))
            .unwrap_or(BackendId::Cpu);
        return Ok(WorkItem {
            node_id: id,
            inputs,
            elem_count,
            dtype: node.dtype,
            target_backend,
            kind: WorkItemKind::ContiguizeOf { input: node.inputs[0] },
            compiled: None,
            output_layout,
            destructive_input,
            output_bundle: None,
        });
    }

    let target_backend = graph.target_backend(id).ok_or_else(|| {
        Error::Msg(format!(
            "PipelinedExecutor: node {:?} ({:?}) has no target_backend set",
            id, node.op
        ))
        .bt()
    })?;

    let op_kind = op_to_op_kind(&node.op).ok_or_else(|| {
        Error::Msg(format!(
            "PipelinedExecutor: op {:?} not yet mapped to an OpKind \
             (Phase C migrates more ops as they're registered)",
            node.op,
        ))
        .bt()
    })?;

    let op_params = op_to_op_params(graph, node, layout_cache, sym_env)?;
    // Build the per-operand dtype list for the binding-table lookup —
    // inputs in order, then outputs. Variadic uniform-dtype ops
    // (Concat) collapse to the canonical `[T_in, T_out]` shorthand to
    // match how registrations index them.
    let dtypes = build_lookup_dtypes(graph, node);
    let compiled = resolve_compiled(op_kind, &dtypes, target_backend, op_params, bindings)?;
    let output_layout = Layout::contiguous(node.shape.clone());
    layout_cache.insert(id, output_layout.clone());
    // Multi-output: snapshot the producer's bundle metadata. When
    // `Some(_)`, the Kernel arm allocates a single Storage sized to
    // fit every slot and attaches the bundle via
    // `Storage::with_bundle` so downstream Op::View/Op::ViewOwned
    // resolve correctly.
    let output_bundle = graph.output_views_arc(id);
    Ok(WorkItem {
        node_id: id,
        inputs,
        elem_count,
        dtype: node.dtype,
        target_backend,
        kind: WorkItemKind::Kernel,
        compiled: Some(compiled),
        output_layout,
        destructive_input,
        output_bundle,
    })
}

/// Build the per-operand dtype list used as the binding-table lookup
/// key. Inputs in order, then the output. Variadic-uniform ops
/// (Concat) collapse to the canonical `[T_in, T_out]` shorthand to
/// match how those wrappers are registered (otherwise an N-way concat
/// would need N+1 distinct registrations per dtype).
///
/// `pub(crate)` — `crate::plan::compile_plan` (step 9b) reuses this
/// to key the route picker's binding-table lookup the same way the
/// pipelined path does.
pub(crate) fn build_lookup_dtypes(graph: &Graph, node: &Node) -> Vec<DType> {
    if matches!(node.op, Op::Concat { .. }) {
        // Concat: all inputs share node.dtype by construction.
        let in_dt = node
            .inputs
            .first()
            .map(|&id| graph.node(id).dtype)
            .unwrap_or(node.dtype);
        return vec![in_dt, node.dtype];
    }
    if matches!(node.op, Op::WriteSlice { .. }) {
        // WriteSlice: the destination (inputs[0]) is exposed to the
        // executor as the OUTPUT slot (in-place adoption), not as an
        // input the kernel reads. Canonicalize the binding-table key
        // to `[T_source, T_out]` so registrations match the kernel's
        // actual surface (one input slab + one output buffer).
        let src_dt = node
            .inputs
            .get(1)
            .map(|&id| graph.node(id).dtype)
            .unwrap_or(node.dtype);
        return vec![src_dt, node.dtype];
    }
    if matches!(node.op, Op::WriteSliceRotating { .. }) {
        // Same canonicalization as WriteSlice: dest = output slot;
        // position (inputs[2]) is read by the kernel via the
        // position-input parameter, not the binding-table key.
        let src_dt = node
            .inputs
            .get(1)
            .map(|&id| graph.node(id).dtype)
            .unwrap_or(node.dtype);
        return vec![src_dt, node.dtype];
    }
    let mut dts: Vec<DType> = node
        .inputs
        .iter()
        .map(|&id| graph.node(id).dtype)
        .collect();
    dts.push(node.dtype);
    dts
}

/// SoftmaxLastDim is reachable only through `Op::Fused(FusedOps::
/// SOFTMAX_LAST_DIM, _)` post-step-5 (the legacy `Op::SoftmaxLastDim`
/// primitive variant was retired). This helper collapses the
/// op-to-OpKind/OpParams shapes for callers that want a single-arm
/// match.
fn op_is_softmax_last_dim(op: &Op) -> bool {
    matches!(op, Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::SOFTMAX_LAST_DIM)
}

/// FusedLinear is reachable only through `Op::Fused(FusedOps::
/// FUSED_LINEAR, _)` post-step-5.
fn op_is_fused_linear(op: &Op) -> bool {
    matches!(op, Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::FUSED_LINEAR)
}

/// Map a `fuel_graph::Op` to a `fuel_ir::dispatch::OpKind`.
/// Returns `None` for ops that haven't been wired into the new
/// dispatch path yet — Phase C extends this as op families migrate.
///
/// `pub(crate)` — `crate::plan::compile_plan` (step 9b) reuses this
/// to skip nodes the binding-table doesn't index (view ops,
/// `Op::Const`, not-yet-migrated ops) when building bindings.
pub(crate) fn op_to_op_kind(op: &Op) -> Option<OpKind> {
    match op {
        Op::Add           => Some(OpKind::AddElementwise),
        Op::Sub           => Some(OpKind::SubElementwise),
        Op::Mul           => Some(OpKind::MulElementwise),
        Op::Div           => Some(OpKind::DivElementwise),
        Op::Relu          => Some(OpKind::ReluElementwise),
        Op::Neg           => Some(OpKind::NegElementwise),
        Op::Sqr           => Some(OpKind::SqrElementwise),
        Op::Sqrt          => Some(OpKind::SqrtElementwise),
        Op::Tanh          => Some(OpKind::TanhElementwise),
        Op::Exp           => Some(OpKind::ExpElementwise),
        Op::Log           => Some(OpKind::LogElementwise),
        Op::Sin           => Some(OpKind::SinElementwise),
        Op::Cos           => Some(OpKind::CosElementwise),
        Op::Sigmoid       => Some(OpKind::SigmoidElementwise),
        Op::Silu          => Some(OpKind::SiluElementwise),
        Op::Gelu          => Some(OpKind::GeluElementwise),
        Op::Step          => Some(OpKind::StepElementwise),
        Op::Recip         => Some(OpKind::RecipElementwise),
        Op::Abs           => Some(OpKind::AbsElementwise),
        Op::Equal         => Some(OpKind::EqualElementwise),
        Op::Ne            => Some(OpKind::NotEqualElementwise),
        Op::Lt            => Some(OpKind::LessElementwise),
        Op::Le            => Some(OpKind::LessEqualElementwise),
        Op::Gt            => Some(OpKind::GreaterElementwise),
        Op::Ge            => Some(OpKind::GreaterEqualElementwise),
        Op::Where         => Some(OpKind::Where),
        Op::Floor         => Some(OpKind::FloorElementwise),
        Op::Ceil          => Some(OpKind::CeilElementwise),
        Op::Round         => Some(OpKind::RoundElementwise),
        Op::Sign          => Some(OpKind::SignElementwise),
        Op::Erf           => Some(OpKind::ErfElementwise),
        Op::GeluErf       => Some(OpKind::GeluErfElementwise),
        Op::Pow           => Some(OpKind::PowElementwise),
        Op::Rsqrt         => Some(OpKind::RsqrtElementwise),
        Op::Rem           => Some(OpKind::RemElementwise),
        Op::Flip { .. }   => Some(OpKind::Flip),
        Op::Roll { .. }   => Some(OpKind::Roll),
        Op::CumSum { .. } => Some(OpKind::CumSum),
        Op::Triu { .. }   => Some(OpKind::Triu),
        Op::Tril { .. }   => Some(OpKind::Tril),
        Op::LogSoftmaxLastDim => Some(OpKind::LogSoftmaxLastDim),
        Op::LogSoftmaxLastDimBackward => Some(OpKind::LogSoftmaxLastDimBackward),
        Op::MaskedFill { .. } => Some(OpKind::MaskedFill),
        Op::Pad { .. }    => Some(OpKind::Pad),
        Op::PadBackward { .. } => Some(OpKind::PadBackward),
        Op::SumDim(_)     => Some(OpKind::SumReduce),
        Op::MaxDim(_)     => Some(OpKind::MaxReduce),
        Op::MinDim(_)     => Some(OpKind::MinReduce),
        Op::MeanDim(_)    => Some(OpKind::MeanReduce),
        Op::SumAll        => Some(OpKind::SumReduce),
        Op::MaxAll        => Some(OpKind::MaxReduce),
        Op::MinAll        => Some(OpKind::MinReduce),
        Op::MeanAll       => Some(OpKind::MeanReduce),
        Op::MatMul        => Some(OpKind::MatMul),
        Op::Cast(_)       => Some(OpKind::Cast),
        // Phase 7.6 step 5: fused-op dispatch routes through
        // `Op::Fused(fid, _)`; the legacy `Op::Conv2D` /
        // `Op::FusedLinear` / `Op::SoftmaxLastDim` /
        // `Op::RmsNormLastDim` / `Op::LayerNormLastDim` / `Op::Rope`
        // arms were dropped together with the variants.
        Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::CONV2D => {
            Some(OpKind::Conv2D)
        }
        // Phase 7.6 step 5 (final): legacy `Op::ConvTranspose2D` /
        // `Op::FlashAttn` / `Op::PagedAttn` arms dropped with the
        // variants; dispatch flows only through `Op::Fused`.
        Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::CONV_TRANSPOSE2D => {
            Some(OpKind::ConvTranspose2D)
        }
        Op::ReduceSumTo(_) => Some(OpKind::ReduceSumTo),
        Op::ReduceMaxTo(_) => Some(OpKind::ReduceMaxTo),
        Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::FLASH_ATTN => {
            Some(OpKind::FlashAttn)
        }
        Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::PAGED_ATTN => {
            Some(OpKind::PagedAttn)
        }
        Op::AddScalar(_)  => Some(OpKind::Affine),
        Op::MulScalar(_)  => Some(OpKind::Affine),
        Op::Clamp { .. }  => Some(OpKind::ClampElementwise),
        Op::PowI(_)       => Some(OpKind::PowIElementwise),
        Op::Maximum       => Some(OpKind::MaximumElementwise),
        Op::Minimum       => Some(OpKind::MinimumElementwise),
        Op::Concat { .. } => Some(OpKind::Concat),
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::SOFTMAX_LAST_DIM =>
        {
            Some(OpKind::SoftmaxLastDim)
        }
        // Phase 7.6 step 6 follow-up: backward helpers now route
        // through the byte-level binding table too.
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::SOFTMAX_LAST_DIM_BACKWARD =>
        {
            Some(OpKind::SoftmaxLastDimBackward)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::LAYER_NORM_LAST_DIM_BACKWARD =>
        {
            Some(OpKind::LayerNormLastDimBackward)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::RMS_NORM_LAST_DIM_BACKWARD =>
        {
            Some(OpKind::RmsNormLastDimBackward)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::REDUCE_MAX_TO_BACKWARD =>
        {
            Some(OpKind::ReduceMaxToBackward)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::POWI_BACKWARD =>
        {
            Some(OpKind::PowIElementwiseBackward)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::FUSED_LINEAR =>
        {
            Some(OpKind::FusedLinear)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::RMS_NORM_LAST_DIM =>
        {
            Some(OpKind::RmsNormLastDim)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::LAYER_NORM_LAST_DIM =>
        {
            Some(OpKind::LayerNormLastDim)
        }
        Op::IndexSelect { .. } => Some(OpKind::IndexSelect),
        Op::Gather { .. } => Some(OpKind::Gather),
        Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::ROPE => {
            Some(OpKind::Rope)
        }
        Op::IndexAdd { .. } => Some(OpKind::IndexAdd),
        Op::ScatterAdd { .. } => Some(OpKind::ScatterAdd),
        Op::ArgMaxDim(_) => Some(OpKind::ArgMaxDim),
        Op::ArgMinDim(_) => Some(OpKind::ArgMinDim),
        // Phase 7.6 step 5 (final): legacy `Op::QMatMul` arm dropped.
        Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::QMATMUL => {
            Some(OpKind::QMatMul)
        }
        Op::WriteSlice { .. } => Some(OpKind::WriteSlice),
        Op::WriteSliceRotating { .. } => Some(OpKind::WriteSliceRotating),
        // Phase 2 of bridge-retirement (post-9c): Op::Copy dispatches
        // through the binding table at OpKind::Copy. The BackendId
        // axis encodes the source backend (the kernel runs there —
        // download from local memory). The executor handles output
        // allocation on the target via a dedicated WorkItemKind::Copy
        // arm, since target_location differs from target_backend for
        // this op (the only one with that property today).
        Op::Copy { .. } => Some(OpKind::Copy),
        // Op::Move dispatches the same data-movement kernel as
        // Op::Copy — the destructive release of the source is
        // realize-loop bookkeeping (`destructive_input` cleanup), not
        // a kernel. Same OpKind, same source-backend key convention.
        // `compile_plan` skips both (residency-determined, not a
        // picker decision).
        Op::Move { .. } => Some(OpKind::Copy),
        // Phase 3 of the in-place ops infrastructure: each in-place Op
        // variant maps to its OpKind so the binding-table dispatch can
        // resolve a kernel. The executor's dedicated WorkItemKind
        // arms (added alongside) handle the storage-Arc adoption.
        Op::ReluInplace        => Some(OpKind::ReluInplace),
        Op::SiluInplace        => Some(OpKind::SiluInplace),
        Op::GeluInplace        => Some(OpKind::GeluInplace),
        Op::TanhInplace        => Some(OpKind::TanhInplace),
        Op::SigmoidInplace     => Some(OpKind::SigmoidInplace),
        Op::NegInplace         => Some(OpKind::NegInplace),
        Op::AbsInplace         => Some(OpKind::AbsInplace),
        Op::SqrInplace         => Some(OpKind::SqrInplace),
        Op::SqrtInplace        => Some(OpKind::SqrtInplace),
        Op::RsqrtInplace       => Some(OpKind::RsqrtInplace),
        Op::RecipInplace       => Some(OpKind::RecipInplace),
        Op::ExpInplace         => Some(OpKind::ExpInplace),
        Op::LogInplace         => Some(OpKind::LogInplace),
        Op::SinInplace         => Some(OpKind::SinInplace),
        Op::CosInplace         => Some(OpKind::CosInplace),
        Op::SignInplace        => Some(OpKind::SignInplace),
        Op::FloorInplace       => Some(OpKind::FloorInplace),
        Op::CeilInplace        => Some(OpKind::CeilInplace),
        Op::RoundInplace       => Some(OpKind::RoundInplace),
        Op::ErfInplace         => Some(OpKind::ErfInplace),
        Op::GeluErfInplace     => Some(OpKind::GeluErfInplace),
        Op::ClampInplace { .. } => Some(OpKind::ClampInplace),
        Op::PowIInplace(_)      => Some(OpKind::PowIInplace),
        // In-place binary variants (Add/Sub/Mul/Div/MaskedFill) and
        // their OpKind siblings are tracked by a parallel session and
        // toggle in and out of the Op enum as that work progresses.
        // No multi-output arm references them; nothing to map here in
        // the current HEAD.
        Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::INPLACE_AFFINE => {
            Some(OpKind::InplaceAffine)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY =>
        {
            Some(OpKind::FusedSoftmaxCrossEntropy)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::CAUSAL_CONV1D =>
        {
            Some(OpKind::CausalConv1d)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::SELECTIVE_SCAN =>
        {
            Some(OpKind::SelectiveScan)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::SSD_CHUNK_SCAN =>
        {
            Some(OpKind::SsdChunkScan)
        }
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::NF4_MATMUL =>
        {
            Some(OpKind::Nf4Matmul)
        }
        // Phase 3a (post-9c): Op::Alloc + Op::ZeroFill are structural
        // ops dispatched directly via `WorkItemKind::Alloc` /
        // `WorkItemKind::ZeroFill` (no binding-table lookup).
        // `compile_plan` skips these.
        Op::Alloc { .. } | Op::ZeroFill => None,
        _ => None,
    }
}

/// Build the [`OpParams`] for `node`'s op. Most ops use
/// `OpParams::None`; reductions / matmul / conv / slice carry their
/// op-specific extras here. The graph is consulted to read input
/// shapes (e.g. reductions need the input shape to walk the
/// multi-index — Storage only carries bytes + dtype).
///
/// Phase C — extends as op families migrate. Returns Err if a
/// graph-shape lookup fails (currently can't, but the signature is
/// `Result` so future cases needing validation slot in cleanly).
/// Encode an `f64` value into the byte pattern of `dtype`. Used by
/// `Op::Pad` (Constant mode) to pre-convert the fill value once at
/// op_params time, so the kernel itself stays dtype-agnostic.
fn encode_value_to_bytes(dtype: DType, value: f64) -> Result<Vec<u8>> {
    match dtype {
        DType::F32 => Ok((value as f32).to_le_bytes().to_vec()),
        DType::F64 => Ok(value.to_le_bytes().to_vec()),
        DType::BF16 => Ok(half::bf16::from_f32(value as f32).to_le_bytes().to_vec()),
        DType::F16 => Ok(half::f16::from_f32(value as f32).to_le_bytes().to_vec()),
        DType::U8 => Ok(vec![value as u8]),
        DType::U32 => Ok((value as u32).to_le_bytes().to_vec()),
        other => Err(Error::Msg(format!(
            "encode_value_to_bytes: dtype {other:?} not yet supported for Pad fill",
        )).bt()),
    }
}

/// Encode a typed `Scalar` to its little-endian byte representation.
/// Used by Op::MaskedFill so the kernel only sees bytes — never has
/// to know the value's dtype.
fn scalar_to_bytes(s: fuel_ir::Scalar) -> Vec<u8> {
    use fuel_ir::Scalar;
    match s {
        Scalar::U8(v)  => vec![v],
        Scalar::I8(v)  => vec![v as u8],
        Scalar::U32(v) => v.to_le_bytes().to_vec(),
        Scalar::I16(v) => v.to_le_bytes().to_vec(),
        Scalar::I32(v) => v.to_le_bytes().to_vec(),
        Scalar::I64(v) => v.to_le_bytes().to_vec(),
        Scalar::BF16(v) => v.to_le_bytes().to_vec(),
        Scalar::F16(v) => v.to_le_bytes().to_vec(),
        Scalar::F32(v) => v.to_le_bytes().to_vec(),
        Scalar::F64(v) => v.to_le_bytes().to_vec(),
        Scalar::F8E4M3(v) => vec![v.to_bits()],
    }
}

fn op_to_op_params(
    graph: &Graph,
    node: &Node,
    layout_cache: &HashMap<NodeId, Layout>,
    sym_env: &SymEnv,
) -> Result<OpParams> {
    // Helper: read an input's layout from the compiler-thread-local
    // cache (which is current within the realize call), falling back
    // to the graph's side-table if the input wasn't visited (which
    // shouldn't happen in topo order, but the fallback keeps the
    // path safe).
    let input_layout = |input_id: NodeId| -> Layout {
        layout_cache
            .get(&input_id)
            .cloned()
            .unwrap_or_else(|| graph.layout(input_id))
    };
    Ok(match &node.op {
        // Phase 7.6 step 5 (final): legacy `Op::QMatMul` arm dropped.
        Op::Fused(_, fuel_graph::registry::FusedOpParams::QMatMul { quant_type, k, n }) => {
            // Inputs: (activations f32 [..., m, k], weight_bytes
            // u32-typed). Output shape (this Node's shape) is
            // [..., m, n]. Flatten leading dims into batch_count.
            if node.inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "Op::QMatMul expects 2 inputs (activations, weight_bytes), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let act_layout = input_layout(node.inputs[0]);
            let act_dims = act_layout.shape().dims();
            if act_dims.len() < 2 {
                return Err(Error::Msg(format!(
                    "Op::QMatMul: activations must be rank ≥ 2, got {act_dims:?}",
                ))
                .bt());
            }
            let m = act_dims[act_dims.len() - 2];
            let k_act = act_dims[act_dims.len() - 1];
            if k_act != *k {
                return Err(Error::Msg(format!(
                    "Op::QMatMul: activation last dim ({k_act}) must equal Op's k ({k})",
                ))
                .bt());
            }
            let batch_count: usize = act_dims[..act_dims.len() - 2].iter().product();
            OpParams::QMatMul {
                quant_type: *quant_type,
                batch_count,
                m,
                n: *n,
                k: *k,
            }
        }
        Op::ArgMaxDim(d) | Op::ArgMinDim(d) => {
            // Reuse OpParams::Reduce — same shape contract; the
            // single reduce dim is the argmax/argmin axis. Input
            // layout flows through KernelRef's `layouts[0]`.
            OpParams::Reduce {
                dims: vec![*d],
                keepdim: false,
            }
        }
        Op::SumDim(d) | Op::MaxDim(d) | Op::MinDim(d) | Op::MeanDim(d) => {
            OpParams::Reduce {
                dims: vec![*d],
                keepdim: false,
            }
        }
        Op::SumAll | Op::MaxAll | Op::MinAll | Op::MeanAll => {
            let il = input_layout(node.inputs[0]);
            let rank = il.shape().rank();
            OpParams::Reduce {
                dims: (0..rank).collect(),
                keepdim: false,
            }
        }
        Op::MatMul => {
            // Batched matmul: lhs `[..lhs_batch.., m, k]` @
            // rhs `[..rhs_batch.., k, n]` → out `[..lhs_batch.., m, n]`.
            // Per-axis the batch dims either match or follow GQA-style
            // divisibility (lhs_dim > rhs_dim && lhs_dim % rhs_dim == 0);
            // the kernel honors the latter via `rhs_idx = lhs_idx / n_rep`.
            if node.inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "Op::MatMul expects 2 inputs, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let lhs = input_layout(node.inputs[0]);
            let rhs = input_layout(node.inputs[1]);
            let lhs_dims = lhs.shape().dims();
            let rhs_dims = rhs.shape().dims();
            if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
                return Err(Error::Msg(format!(
                    "Op::MatMul requires both inputs rank ≥ 2; got lhs={:?} rhs={:?}",
                    lhs_dims, rhs_dims,
                ))
                .bt());
            }
            if lhs_dims.len() != rhs_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::MatMul: ranks must match (auto-broadcast happens at \
                     graph construction time); got lhs rank {} vs rhs rank {}",
                    lhs_dims.len(),
                    rhs_dims.len(),
                ))
                .bt());
            }
            let rank = lhs_dims.len();
            let batch_rank = rank - 2;
            // Per-axis validation: equal or GQA-divisible.
            for i in 0..batch_rank {
                let la = lhs_dims[i];
                let ra = rhs_dims[i];
                let ok = la == ra || (ra > 0 && la > ra && la % ra == 0);
                if !ok {
                    return Err(Error::Msg(format!(
                        "Op::MatMul: batch dim {i} disallowed combination \
                         (lhs={la}, rhs={ra}); must be equal or \
                         GQA-divisible (lhs > rhs && lhs % rhs == 0)",
                    ))
                    .bt());
                }
            }
            let lhs_batch_dims: Vec<usize> = lhs_dims[..batch_rank].to_vec();
            let rhs_batch_dims: Vec<usize> = rhs_dims[..batch_rank].to_vec();
            let (m, k_lhs) = (lhs_dims[rank - 2], lhs_dims[rank - 1]);
            let (k_rhs, n) = (rhs_dims[rank - 2], rhs_dims[rank - 1]);
            if k_lhs != k_rhs {
                return Err(Error::Msg(format!(
                    "Op::MatMul: contracting dims disagree — lhs trailing is \
                     [{m}, {k_lhs}], rhs trailing is [{k_rhs}, {n}]",
                ))
                .bt());
            }
            OpParams::Matmul {
                lhs_batch_dims,
                rhs_batch_dims,
                m,
                n,
                k: k_lhs,
            }
        }
        // Phase 7.6 step 4: legacy `Op::FusedLinear` and registry-extended
        // `Op::Fused(FUSED_LINEAR, _)` share the same shape contract and
        // params encoding (reuses OpParams::Matmul). One body covers both.
        op if op_is_fused_linear(op) => {
            // Inputs: [a, b, bias]. Same shape semantics as MatMul on
            // a/b; bias is rank-1 [N] and broadcasts along all leading
            // dims. We reuse OpParams::Matmul (kernel reads bias from
            // inputs[2] directly).
            if node.inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "Op::FusedLinear expects 3 inputs (a, b, bias), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let lhs = input_layout(node.inputs[0]);
            let rhs = input_layout(node.inputs[1]);
            let bias = input_layout(node.inputs[2]);
            let lhs_dims = lhs.shape().dims();
            let rhs_dims = rhs.shape().dims();
            let bias_dims = bias.shape().dims();
            if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
                return Err(Error::Msg(format!(
                    "Op::FusedLinear requires both a, b rank ≥ 2; got a={:?} b={:?}",
                    lhs_dims, rhs_dims,
                ))
                .bt());
            }
            if lhs_dims.len() != rhs_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::FusedLinear: ranks must match (auto-broadcast happens at \
                     graph construction time); got a rank {} vs b rank {}",
                    lhs_dims.len(),
                    rhs_dims.len(),
                ))
                .bt());
            }
            let rank = lhs_dims.len();
            let batch_rank = rank - 2;
            for i in 0..batch_rank {
                let la = lhs_dims[i];
                let ra = rhs_dims[i];
                let ok = la == ra || (ra > 0 && la > ra && la % ra == 0);
                if !ok {
                    return Err(Error::Msg(format!(
                        "Op::FusedLinear: batch dim {i} disallowed (a={la}, b={ra})",
                    ))
                    .bt());
                }
            }
            let lhs_batch_dims: Vec<usize> = lhs_dims[..batch_rank].to_vec();
            let rhs_batch_dims: Vec<usize> = rhs_dims[..batch_rank].to_vec();
            let (m, k_lhs) = (lhs_dims[rank - 2], lhs_dims[rank - 1]);
            let (k_rhs, n) = (rhs_dims[rank - 2], rhs_dims[rank - 1]);
            if k_lhs != k_rhs {
                return Err(Error::Msg(format!(
                    "Op::FusedLinear: contracting dims disagree — a trailing is \
                     [{m}, {k_lhs}], b trailing is [{k_rhs}, {n}]",
                ))
                .bt());
            }
            if bias_dims.len() != 1 || bias_dims[0] != n {
                return Err(Error::Msg(format!(
                    "Op::FusedLinear: bias must be rank-1 [{n}], got {bias_dims:?}",
                ))
                .bt());
            }
            OpParams::Matmul {
                lhs_batch_dims,
                rhs_batch_dims,
                m,
                n,
                k: k_lhs,
            }
        }
        // FusedSoftmaxCrossEntropy: flatten logits `[..., V]` to
        // `[n_rows, V]` for the kernel. n_rows is the product of all
        // leading dims (≥1 even for rank-1 logits, where targets are
        // a scalar). The kernel walks rows internally.
        Op::Fused(
            _,
            fuel_graph::registry::FusedOpParams::FusedSoftmaxCrossEntropy {
                reduction,
                ignore_index,
            },
        ) => {
            if node.inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "FusedSoftmaxCrossEntropy expects 2 inputs (logits, targets), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let logits_layout = input_layout(node.inputs[0]);
            let logits_dims = logits_layout.shape().dims();
            if logits_dims.is_empty() {
                return Err(Error::Msg(
                    "FusedSoftmaxCrossEntropy: logits must have rank ≥ 1".to_string(),
                )
                .bt());
            }
            let vocab = *logits_dims.last().unwrap();
            let n_rows: usize = logits_dims[..logits_dims.len() - 1]
                .iter()
                .product::<usize>()
                .max(1);
            let targets_layout = input_layout(node.inputs[1]);
            let target_count: usize = targets_layout.shape().dims().iter().product::<usize>().max(1);
            if target_count != n_rows {
                return Err(Error::Msg(format!(
                    "FusedSoftmaxCrossEntropy: targets element count {target_count} must \
                     equal logits row count {n_rows} (logits {logits_dims:?}, targets {:?})",
                    targets_layout.shape().dims(),
                ))
                .bt());
            }
            OpParams::FusedSoftmaxCrossEntropy {
                n_rows,
                vocab,
                reduction: *reduction,
                ignore_index: *ignore_index,
            }
        }
        // CausalConv1d: derive (batch, channels, seq_in, seq_out, kernel)
        // from the input layouts. x is `[batch, channels, seq_in]`
        // (caller pre-pads with kernel-1 zeros), weight is
        // `[channels, 1, kernel]`. seq_out = seq_in - (kernel - 1).
        Op::Fused(
            _,
            fuel_graph::registry::FusedOpParams::CausalConv1d { use_silu },
        ) => {
            if node.inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "CausalConv1d expects 3 inputs (x, weight, bias), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let x_layout = input_layout(node.inputs[0]);
            let w_layout = input_layout(node.inputs[1]);
            let x_dims = x_layout.shape().dims();
            let w_dims = w_layout.shape().dims();
            if x_dims.len() != 3 {
                return Err(Error::Msg(format!(
                    "CausalConv1d: x must be rank 3 [batch, channels, seq+pad], got {x_dims:?}",
                ))
                .bt());
            }
            if w_dims.len() != 3 {
                return Err(Error::Msg(format!(
                    "CausalConv1d: weight must be rank 3 [channels, 1, kernel], got {w_dims:?}",
                ))
                .bt());
            }
            let batch = x_dims[0];
            let channels = x_dims[1];
            let seq_in = x_dims[2];
            let kernel = w_dims[2];
            if seq_in < kernel - 1 {
                return Err(Error::Msg(format!(
                    "CausalConv1d: x time dim {seq_in} must be ≥ kernel-1 = {} \
                     (caller must pre-pad with kernel-1 zeros)",
                    kernel - 1,
                ))
                .bt());
            }
            let seq_out = seq_in - (kernel - 1);
            OpParams::CausalConv1d {
                batch,
                channels,
                seq_in,
                seq_out,
                kernel,
                use_silu: *use_silu,
            }
        }
        // SelectiveScan: derive (batch, seqlen, dim, dstate) from the
        // input layouts. u: [batch, seqlen, dim]; a: [dim, dstate].
        Op::Fused(
            _,
            fuel_graph::registry::FusedOpParams::SelectiveScan { delta_softplus },
        ) => {
            if node.inputs.len() != 5 {
                return Err(Error::Msg(format!(
                    "SelectiveScan expects 5 inputs (u, delta, a, b, c), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let u_layout = input_layout(node.inputs[0]);
            let a_layout = input_layout(node.inputs[2]);
            let u_dims = u_layout.shape().dims();
            let a_dims = a_layout.shape().dims();
            if u_dims.len() != 3 {
                return Err(Error::Msg(format!(
                    "SelectiveScan: u must be rank 3 [batch, seqlen, dim], got {u_dims:?}",
                ))
                .bt());
            }
            if a_dims.len() != 2 {
                return Err(Error::Msg(format!(
                    "SelectiveScan: a must be rank 2 [dim, dstate], got {a_dims:?}",
                ))
                .bt());
            }
            let batch = u_dims[0];
            let seqlen = u_dims[1];
            let dim = u_dims[2];
            let dstate = a_dims[1];
            if a_dims[0] != dim {
                return Err(Error::Msg(format!(
                    "SelectiveScan: a's first dim {} must equal dim {dim}", a_dims[0],
                ))
                .bt());
            }
            OpParams::SelectiveScan {
                batch,
                seqlen,
                dim,
                dstate,
                delta_softplus: *delta_softplus,
            }
        }
        // SsdChunkScan: derive (batch, seqlen, heads, head_dim,
        // state_dim) from x and b layouts. x: [batch, seqlen, heads,
        // head_dim]; b: [batch, seqlen, heads, state_dim].
        Op::Fused(
            _,
            fuel_graph::registry::FusedOpParams::SsdChunkScan { chunk_size },
        ) => {
            if node.inputs.len() != 5 {
                return Err(Error::Msg(format!(
                    "SsdChunkScan expects 5 inputs (x, dt, a, b, c), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let x_layout = input_layout(node.inputs[0]);
            let b_layout = input_layout(node.inputs[3]);
            let x_dims = x_layout.shape().dims();
            let b_dims = b_layout.shape().dims();
            if x_dims.len() != 4 {
                return Err(Error::Msg(format!(
                    "SsdChunkScan: x must be rank 4 [batch, seqlen, heads, head_dim], got {x_dims:?}",
                ))
                .bt());
            }
            if b_dims.len() != 4 {
                return Err(Error::Msg(format!(
                    "SsdChunkScan: b must be rank 4 [batch, seqlen, heads, state_dim], got {b_dims:?}",
                ))
                .bt());
            }
            let batch = x_dims[0];
            let seqlen = x_dims[1];
            let heads = x_dims[2];
            let head_dim = x_dims[3];
            let state_dim = b_dims[3];
            OpParams::SsdChunkScan {
                batch,
                seqlen,
                heads,
                head_dim,
                state_dim,
                chunk_size: *chunk_size,
            }
        }
        // Nf4Matmul: derive (batch, m, n, k) from activations + w_packed
        // layouts. activations: [..., m, k] (leading dims flatten into
        // batch); w_packed: [n, k/2] U8.
        Op::Fused(
            _,
            fuel_graph::registry::FusedOpParams::Nf4Matmul { block_size },
        ) => {
            if node.inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "Nf4Matmul expects 3 inputs (activations, w_packed, absmax), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let a_layout = input_layout(node.inputs[0]);
            let w_layout = input_layout(node.inputs[1]);
            let a_dims = a_layout.shape().dims();
            let w_dims = w_layout.shape().dims();
            if a_dims.len() < 2 {
                return Err(Error::Msg(format!(
                    "Nf4Matmul: activations must be rank ≥ 2, got {a_dims:?}",
                ))
                .bt());
            }
            if w_dims.len() != 2 {
                return Err(Error::Msg(format!(
                    "Nf4Matmul: w_packed must be rank 2 [n, k/2], got {w_dims:?}",
                ))
                .bt());
            }
            let m = a_dims[a_dims.len() - 2];
            let k = a_dims[a_dims.len() - 1];
            let n = w_dims[0];
            let batch: usize = a_dims[..a_dims.len() - 2].iter().product::<usize>().max(1);
            if k % 2 != 0 {
                return Err(Error::Msg(format!(
                    "Nf4Matmul: k={k} must be even (w_packed holds 2 nibbles per byte along k)",
                ))
                .bt());
            }
            if k % *block_size != 0 {
                return Err(Error::Msg(format!(
                    "Nf4Matmul: k={k} must be a multiple of block_size={block_size}",
                ))
                .bt());
            }
            if w_dims[1] != k / 2 {
                return Err(Error::Msg(format!(
                    "Nf4Matmul: w_packed second dim {} must equal k/2 = {}", w_dims[1], k / 2,
                ))
                .bt());
            }
            OpParams::Nf4Matmul {
                batch,
                m,
                n,
                k,
                block_size: *block_size,
            }
        }
        // Phase 7.6 step 3: SoftmaxLastDim flows through
        // `Op::Fused(FusedOps::SOFTMAX_LAST_DIM, _)`. The
        // SOFTMAX_LAST_DIM_BACKWARD variant added in step 6
        // follow-up shares the same `outer × last_dim` shape
        // contract (input 0 is `y` for forward / forward-output
        // for backward; both rank-≥1 with last-dim as the
        // reduction axis).
        op if op_is_softmax_last_dim(op)
            || matches!(op, Op::Fused(fid, _) if *fid == fuel_graph::registry::FusedOps::SOFTMAX_LAST_DIM_BACKWARD) =>
        {
            let il = input_layout(node.inputs[0]);
            let dims = il.shape().dims();
            if dims.is_empty() {
                return Err(Error::Msg(
                    "Op::SoftmaxLastDim requires rank ≥ 1".to_string(),
                )
                .bt());
            }
            let last_dim = *dims.last().unwrap();
            let outer_count: usize = dims[..dims.len() - 1].iter().product();
            OpParams::SoftmaxLastDim { outer_count, last_dim }
        }
        Op::Flip { dim } => {
            // Single input. Precompute the flat-3-axis split
            // (outer × dim × inner) from the input shape.
            if node.inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "Op::Flip expects 1 input, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let in_layout = input_layout(node.inputs[0]);
            let in_dims = in_layout.shape().dims();
            if *dim >= in_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::Flip: dim {dim} out of range for rank {}",
                    in_dims.len(),
                ))
                .bt());
            }
            let outer_count: usize = in_dims[..*dim].iter().product();
            let dim_size = in_dims[*dim];
            let inner_count: usize = in_dims[*dim + 1..].iter().product();
            OpParams::Flip { outer_count, dim_size, inner_count, axis: *dim }
        }
        Op::Roll { dim, shift } => {
            if node.inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "Op::Roll expects 1 input, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let in_layout = input_layout(node.inputs[0]);
            let in_dims = in_layout.shape().dims();
            if *dim >= in_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::Roll: dim {dim} out of range for rank {}",
                    in_dims.len(),
                ))
                .bt());
            }
            let outer_count: usize = in_dims[..*dim].iter().product();
            let dim_size = in_dims[*dim];
            let inner_count: usize = in_dims[*dim + 1..].iter().product();
            OpParams::Roll {
                outer_count, dim_size, inner_count, shift: *shift, axis: *dim,
            }
        }
        Op::CumSum { dim } => {
            if node.inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "Op::CumSum expects 1 input, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let in_layout = input_layout(node.inputs[0]);
            let in_dims = in_layout.shape().dims();
            if *dim >= in_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::CumSum: dim {dim} out of range for rank {}",
                    in_dims.len(),
                ))
                .bt());
            }
            let outer_count: usize = in_dims[..*dim].iter().product();
            let dim_size = in_dims[*dim];
            let inner_count: usize = in_dims[*dim + 1..].iter().product();
            OpParams::CumSum { outer_count, dim_size, inner_count, axis: *dim }
        }
        Op::Triu { diagonal } | Op::Tril { diagonal } => {
            if node.inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "Op::Triu/Tril expects 1 input, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let in_layout = input_layout(node.inputs[0]);
            let in_dims = in_layout.shape().dims();
            if in_dims.len() < 2 {
                return Err(Error::Msg(format!(
                    "Op::Triu/Tril requires rank >= 2, got {}",
                    in_dims.len(),
                ))
                .bt());
            }
            let rows = in_dims[in_dims.len() - 2];
            let cols = in_dims[in_dims.len() - 1];
            let batch_count: usize = in_dims[..in_dims.len() - 2]
                .iter().product::<usize>().max(1);
            OpParams::Triangular { batch_count, rows, cols, diagonal: *diagonal }
        }
        Op::LogSoftmaxLastDim => {
            if node.inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "Op::LogSoftmaxLastDim expects 1 input, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let in_layout = input_layout(node.inputs[0]);
            let in_dims = in_layout.shape().dims();
            if in_dims.is_empty() {
                return Err(Error::Msg(
                    "Op::LogSoftmaxLastDim requires rank ≥ 1".to_string(),
                )
                .bt());
            }
            let last_dim = *in_dims.last().unwrap();
            let outer_count: usize = in_dims[..in_dims.len() - 1].iter().product();
            OpParams::LogSoftmaxLastDim { outer_count, last_dim }
        }
        Op::LogSoftmaxLastDimBackward => {
            // Inputs: (forward_output_y, upstream_grad). Same shape contract
            // as SoftmaxLastDimBackward — last-dim row-wise dispatch.
            if node.inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "Op::LogSoftmaxLastDimBackward expects 2 inputs, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let in_layout = input_layout(node.inputs[0]);
            let in_dims = in_layout.shape().dims();
            if in_dims.is_empty() {
                return Err(Error::Msg(
                    "Op::LogSoftmaxLastDimBackward requires rank ≥ 1".to_string(),
                )
                .bt());
            }
            let last_dim = *in_dims.last().unwrap();
            let outer_count: usize = in_dims[..in_dims.len() - 1].iter().product();
            OpParams::LogSoftmaxLastDim { outer_count, last_dim }
        }
        Op::MaskedFill { value } => {
            if node.inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "Op::MaskedFill expects 2 inputs (x, mask), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let fill_bytes = scalar_to_bytes(*value);
            OpParams::MaskedFill { fill_bytes }
        }
        // Op::MaskedFillInplace tracked by the parallel in-place
        // binary family session — not in HEAD's Op enum right now.
        Op::PadBackward { in_shape, padding, mode } => {
            // Single input is the upstream gradient; output is the
            // input gradient (shape == `in_shape`).
            if node.inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "Op::PadBackward expects 1 input, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let in_dims: Vec<usize> = in_shape.dims().to_vec();
            // Output of the original forward Pad is the input shape
            // of the backward (== upstream's shape).
            let out_dims: Vec<usize> = in_dims.iter().zip(padding.iter())
                .map(|(&d, &(b, a))| d + b + a)
                .collect();
            let mode_tag: u8 = match mode {
                fuel_graph::PadMode::Constant => 0,
                fuel_graph::PadMode::Reflect => 1,
                fuel_graph::PadMode::Replicate => 2,
            };
            OpParams::PadBackward {
                in_shape: in_dims,
                out_shape: out_dims,
                padding: padding.clone(),
                mode_tag,
            }
        }
        Op::Pad { padding, mode, value } => {
            if node.inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "Op::Pad expects 1 input, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let in_layout = input_layout(node.inputs[0]);
            let in_dims: Vec<usize> = in_layout.shape().dims().to_vec();
            if padding.len() != in_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::Pad: padding.len() ({}) != input rank ({})",
                    padding.len(), in_dims.len(),
                ))
                .bt());
            }
            let out_dims: Vec<usize> = in_dims.iter().zip(padding.iter())
                .map(|(&d, &(b, a))| d + b + a)
                .collect();
            let mode_tag: u8 = match mode {
                fuel_graph::PadMode::Constant => 0,
                fuel_graph::PadMode::Reflect => 1,
                fuel_graph::PadMode::Replicate => 2,
            };
            // Encode fill value as bytes for the output dtype. The
            // kernel is dtype-agnostic — it just memcopies the
            // pattern. Conversion happens once here per node, not
            // per element in the kernel.
            let fill_bytes = encode_value_to_bytes(node.dtype, *value)?;
            OpParams::Pad {
                in_shape: in_dims,
                out_shape: out_dims,
                padding: padding.clone(),
                mode_tag,
                fill_bytes,
            }
        }
        Op::IndexAdd { dim } => {
            // Inputs: (base, indices, src). All same dtype except
            // indices is U32. Output shape == base shape.
            if node.inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "Op::IndexAdd expects 3 inputs (base, indices, src), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let base_layout = input_layout(node.inputs[0]);
            let idx_layout = input_layout(node.inputs[1]);
            let src_layout = input_layout(node.inputs[2]);
            let base_dims = base_layout.shape().dims();
            let idx_dims = idx_layout.shape().dims();
            let src_dims = src_layout.shape().dims();
            if *dim >= base_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::IndexAdd: dim {dim} out of range for base rank {}",
                    base_dims.len(),
                ))
                .bt());
            }
            if idx_dims.len() != 1 {
                return Err(Error::Msg(format!(
                    "Op::IndexAdd: indices must be rank 1, got {idx_dims:?}",
                ))
                .bt());
            }
            if base_dims.len() != src_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::IndexAdd: base rank ({}) != src rank ({})",
                    base_dims.len(), src_dims.len(),
                ))
                .bt());
            }
            let outer_count: usize = base_dims[..*dim].iter().product();
            let base_dim_size = base_dims[*dim];
            let inner_count: usize = base_dims[*dim + 1..].iter().product();
            let n_indices = idx_dims[0];
            OpParams::IndexAdd {
                outer_count,
                base_dim_size,
                n_indices,
                inner_count,
            }
        }
        Op::ScatterAdd { dim } => {
            if node.inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "Op::ScatterAdd expects 3 inputs (base, indices, src), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let base_layout = input_layout(node.inputs[0]);
            let src_layout = input_layout(node.inputs[2]);
            let base_shape: Vec<usize> = base_layout.shape().dims().to_vec();
            let src_shape: Vec<usize> = src_layout.shape().dims().to_vec();
            if base_shape.len() != src_shape.len() {
                return Err(Error::Msg(format!(
                    "Op::ScatterAdd: base rank ({}) != src rank ({})",
                    base_shape.len(), src_shape.len(),
                ))
                .bt());
            }
            if *dim >= base_shape.len() {
                return Err(Error::Msg(format!(
                    "Op::ScatterAdd: dim {dim} out of range for rank {}",
                    base_shape.len(),
                ))
                .bt());
            }
            OpParams::ScatterAdd {
                base_shape,
                src_shape,
                dim: *dim,
            }
        }
        // Phase 7.6 step 5: Rope routes through the registry; the
        // legacy `Op::Rope` arm was retired with the variant.
        Op::Fused(_, fuel_graph::registry::FusedOpParams::Rope) => {
            // Inputs: (x, cos, sin). x is [..., seq, head_dim];
            // cos/sin are [seq, head_dim] (validated at graph build).
            if node.inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "Op::Rope expects 3 inputs (x, cos, sin), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let x_layout = input_layout(node.inputs[0]);
            let x_dims = x_layout.shape().dims();
            if x_dims.len() < 2 {
                return Err(Error::Msg(format!(
                    "Op::Rope: x must have rank ≥ 2, got {x_dims:?}",
                ))
                .bt());
            }
            let head_dim = *x_dims.last().unwrap();
            let seq = x_dims[x_dims.len() - 2];
            let outer_count: usize = x_dims[..x_dims.len() - 2].iter().product();
            OpParams::Rope { outer_count, seq, head_dim }
        }
        Op::Gather { dim } => {
            // inputs[0] = source, inputs[1] = U32 indices (same rank).
            if node.inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "Op::Gather expects 2 inputs (source, indices), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let src_layout = input_layout(node.inputs[0]);
            let idx_layout = input_layout(node.inputs[1]);
            let source_shape: Vec<usize> = src_layout.shape().dims().to_vec();
            let output_shape: Vec<usize> = idx_layout.shape().dims().to_vec();
            if source_shape.len() != output_shape.len() {
                return Err(Error::Msg(format!(
                    "Op::Gather: source rank ({}) != indices rank ({})",
                    source_shape.len(),
                    output_shape.len(),
                ))
                .bt());
            }
            if *dim >= source_shape.len() {
                return Err(Error::Msg(format!(
                    "Op::Gather: dim {dim} out of range for rank {}",
                    source_shape.len(),
                ))
                .bt());
            }
            OpParams::Gather {
                source_shape,
                output_shape,
                dim: *dim,
            }
        }
        Op::IndexSelect { dim } => {
            // inputs[0] = source, inputs[1] = U32 indices (rank 1).
            if node.inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "Op::IndexSelect expects 2 inputs (source, indices), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let src_layout = input_layout(node.inputs[0]);
            let idx_layout = input_layout(node.inputs[1]);
            let src_dims = src_layout.shape().dims();
            let idx_dims = idx_layout.shape().dims();
            if *dim >= src_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::IndexSelect: dim {dim} out of range for source rank {}",
                    src_dims.len(),
                ))
                .bt());
            }
            if idx_dims.len() != 1 {
                return Err(Error::Msg(format!(
                    "Op::IndexSelect: indices must be rank 1, got shape {idx_dims:?}",
                ))
                .bt());
            }
            let outer_count: usize = src_dims[..*dim].iter().product();
            let source_dim_size = src_dims[*dim];
            let inner_count: usize = src_dims[*dim + 1..].iter().product();
            let n_indices = idx_dims[0];
            OpParams::IndexSelect {
                outer_count,
                source_dim_size,
                n_indices,
                inner_count,
            }
        }
        // Phase 7.6 step 5: legacy `Op::RmsNormLastDim` /
        // `Op::LayerNormLastDim` arms retired with the variants;
        // both now route through `Op::Fused(NORM_LAST_DIM, _)`.
        // Step 6 follow-up extends this arm to also cover the
        // backward variants (same outer × last_dim + eps geometry).
        Op::Fused(fid, params)
            if *fid == fuel_graph::registry::FusedOps::RMS_NORM_LAST_DIM
                || *fid == fuel_graph::registry::FusedOps::LAYER_NORM_LAST_DIM
                || *fid == fuel_graph::registry::FusedOps::RMS_NORM_LAST_DIM_BACKWARD
                || *fid == fuel_graph::registry::FusedOps::LAYER_NORM_LAST_DIM_BACKWARD =>
        {
            let eps = match params {
                fuel_graph::registry::FusedOpParams::RmsNormLastDim { eps } => *eps,
                fuel_graph::registry::FusedOpParams::LayerNormLastDim { eps } => *eps,
                fuel_graph::registry::FusedOpParams::RmsNormLastDimBackward { eps } => *eps,
                fuel_graph::registry::FusedOpParams::LayerNormLastDimBackward { eps } => *eps,
                _ => return Err(Error::Msg(format!(
                    "Op::Fused(NORM_LAST_DIM[_BACKWARD], _) expected \
                     RmsNormLastDim[Backward] or LayerNormLastDim[Backward] params, got {params:?}",
                )).bt()),
            };
            let il = input_layout(node.inputs[0]);
            let dims = il.shape().dims();
            if dims.is_empty() {
                return Err(Error::Msg(
                    "Op::Fused(NORM_LAST_DIM[_BACKWARD], _) requires rank ≥ 1".to_string(),
                )
                .bt());
            }
            let last_dim = *dims.last().unwrap();
            let outer_count: usize = dims[..dims.len() - 1].iter().product();
            OpParams::NormLastDim { outer_count, last_dim, eps }
        }
        // ReduceMaxToBackward: shape pair via the new
        // OpParams::ReduceMaxToBackward variant. x's shape is
        // `input_shape`; upstream's shape is `output_shape`.
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::REDUCE_MAX_TO_BACKWARD =>
        {
            if node.inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "Op::Fused(REDUCE_MAX_TO_BACKWARD) expects 2 inputs, got {}",
                    node.inputs.len(),
                )).bt());
            }
            let x_layout = input_layout(node.inputs[0]);
            let up_layout = input_layout(node.inputs[1]);
            let input_shape: Vec<usize> = x_layout.shape().dims().to_vec();
            let output_shape: Vec<usize> = up_layout.shape().dims().to_vec();
            OpParams::ReduceMaxToBackward { input_shape, output_shape }
        }
        Op::Concat { dim } => {
            // Output's shape: [..., total_dim, ...]. Compute outer
            // and inner counts from output_shape[..dim] and
            // [dim+1..]; per-input dim sizes from each input's
            // layout shape at index `dim`.
            if node.inputs.is_empty() {
                return Err(Error::Msg(
                    "Op::Concat requires at least 1 input".to_string(),
                )
                .bt());
            }
            let out_dims = node.shape.dims();
            if *dim >= out_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::Concat: dim {dim} out of range for output rank {}",
                    out_dims.len(),
                ))
                .bt());
            }
            let outer_count: usize = out_dims[..*dim].iter().product();
            let inner_count: usize = out_dims[*dim + 1..].iter().product();
            let mut input_dim_sizes: Vec<usize> = Vec::with_capacity(node.inputs.len());
            for in_id in &node.inputs {
                let il = input_layout(*in_id);
                let il_dims = il.shape().dims();
                if *dim >= il_dims.len() {
                    return Err(Error::Msg(format!(
                        "Op::Concat: input {in_id:?} has rank {} but concat dim is {dim}",
                        il_dims.len(),
                    ))
                    .bt());
                }
                input_dim_sizes.push(il_dims[*dim]);
            }
            OpParams::Concat {
                outer_count,
                input_dim_sizes,
                inner_count,
                axis: *dim,
            }
        }
        Op::AddScalar(c) => OpParams::Affine { mul: 1.0, add: *c },
        Op::MulScalar(c) => OpParams::Affine { mul: *c, add: 0.0 },
        Op::Clamp { min, max } => OpParams::Clamp { min: *min, max: *max },
        Op::PowI(exp) => OpParams::PowI { exp: *exp },
        Op::ClampInplace { min, max } => OpParams::Clamp { min: *min, max: *max },
        Op::PowIInplace(exp) => OpParams::PowI { exp: *exp },
        // PowI backward — `(x, upstream) → grad_x = exp · x^(exp-1) ·
        // upstream`. Pulls the same `exp` as the forward through
        // FusedOpParams::PowIBackward (autograd carries it across).
        Op::Fused(_, fuel_graph::registry::FusedOpParams::PowIBackward { exp }) => {
            OpParams::PowI { exp: *exp }
        }
        // In-place affine (Phase 3 of the in-place ops infrastructure):
        // reuse the existing `OpParams::Affine` payload — the kernel
        // side doesn't care whether the destination is fresh or aliases
        // the input; the structural decision lives in the executor's
        // dedicated `WorkItemKind::InplaceKernel` arm.
        Op::Fused(_, fuel_graph::registry::FusedOpParams::InplaceAffine { mul, add }) => {
            OpParams::Affine { mul: *mul, add: *add }
        }
        // Phase 7.6 step 5: legacy `Op::Conv2D` arm retired with the
        // variant; Conv2D routes through `Op::Fused(CONV2D, _)`.
        Op::Fused(_, fuel_graph::registry::FusedOpParams::Conv2D { stride, padding, groups }) => {
            // Inputs[0] = x [N, Cin, Hin, Win]; inputs[1] = weight
            // [Cout, Cin/groups, Kh, Kw]; inputs[2] (optional) = bias [Cout].
            // Output (this Node's shape) = [N, Cout, Hout, Wout].
            if node.inputs.len() != 2 && node.inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "Op::Conv2D expects 2 or 3 inputs, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let x_layout = input_layout(node.inputs[0]);
            let w_layout = input_layout(node.inputs[1]);
            let x_dims = x_layout.shape().dims();
            let w_dims = w_layout.shape().dims();
            if x_dims.len() != 4 || w_dims.len() != 4 {
                return Err(Error::Msg(format!(
                    "Op::Conv2D requires rank-4 x and weight; got x={x_dims:?} w={w_dims:?}",
                ))
                .bt());
            }
            let out_dims = node.shape.dims();
            if out_dims.len() != 4 {
                return Err(Error::Msg(format!(
                    "Op::Conv2D output must be rank 4, got {out_dims:?}",
                ))
                .bt());
            }
            let x_shape = [x_dims[0], x_dims[1], x_dims[2], x_dims[3]];
            let w_shape = [w_dims[0], w_dims[1], w_dims[2], w_dims[3]];
            let out_shape = [out_dims[0], out_dims[1], out_dims[2], out_dims[3]];
            OpParams::Conv2D {
                x_shape,
                w_shape,
                out_shape,
                stride: *stride,
                padding: *padding,
                dilation: (1, 1),
                groups: *groups,
            }
        }
        // Phase 7.6 step 5 (final): legacy `Op::PagedAttn` arm dropped.
        Op::Fused(_, fuel_graph::registry::FusedOpParams::PagedAttn {
            softmax_scale, block_size, softcap,
        }) => {
            // Inputs[0]=q [B,Hq,Sq,D], inputs[1]=k_cache [num_blocks,
            // block_size, Hkv, D], inputs[2]=v_cache same shape,
            // inputs[3]=block_table [B, max_blocks_per_seq] U32,
            // inputs[4]=context_lens [B] U32, inputs[5]=alibi [Hq] (optional).
            if node.inputs.len() != 5 && node.inputs.len() != 6 {
                return Err(Error::Msg(format!(
                    "Op::PagedAttn expects 5 or 6 inputs, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let q_layout = input_layout(node.inputs[0]);
            let kc_layout = input_layout(node.inputs[1]);
            let vc_layout = input_layout(node.inputs[2]);
            let bt_layout = input_layout(node.inputs[3]);
            let q_dims = q_layout.shape().dims();
            let kc_dims = kc_layout.shape().dims();
            let vc_dims = vc_layout.shape().dims();
            let bt_dims = bt_layout.shape().dims();
            if q_dims.len() != 4 || kc_dims.len() != 4 || vc_dims.len() != 4 {
                return Err(Error::Msg(format!(
                    "Op::PagedAttn requires rank-4 q/k_cache/v_cache; \
                     got q={q_dims:?} k_cache={kc_dims:?} v_cache={vc_dims:?}",
                ))
                .bt());
            }
            if kc_dims != vc_dims {
                return Err(Error::Msg(format!(
                    "Op::PagedAttn: k_cache {kc_dims:?} and v_cache {vc_dims:?} must match",
                ))
                .bt());
            }
            if bt_dims.len() != 2 {
                return Err(Error::Msg(format!(
                    "Op::PagedAttn: block_table must be rank 2 [B, max_blocks_per_seq], got {bt_dims:?}",
                ))
                .bt());
            }
            if kc_dims[1] != *block_size {
                return Err(Error::Msg(format!(
                    "Op::PagedAttn: k_cache block_size dim ({}) must equal Op's block_size ({})",
                    kc_dims[1], block_size,
                ))
                .bt());
            }
            OpParams::PagedAttn {
                b: q_dims[0],
                hq: q_dims[1],
                hkv: kc_dims[2],
                sq: q_dims[2],
                d: q_dims[3],
                block_size: *block_size,
                max_blocks_per_seq: bt_dims[1],
                num_blocks: kc_dims[0],
                softmax_scale: *softmax_scale,
                softcap: *softcap,
            }
        }
        // Phase 7.6 step 5 (final): legacy `Op::FlashAttn` arm dropped.
        Op::Fused(_, fuel_graph::registry::FusedOpParams::FlashAttn {
            softmax_scale, causal, window_size_left, window_size_right, softcap, k_len,
        }) => {
            // Inputs[0]=q [B,Hq,Sq,D], inputs[1]=k [B,Hkv,Sk,D],
            // inputs[2]=v [B,Hkv,Sk,D], inputs[3]=alibi_slopes [Hq] (optional).
            // `sk` is the PHYSICAL K extent (capacity, for strides/bytes);
            // `k_len` (resolved below) is the LOGICAL attended length.
            if node.inputs.len() != 3 && node.inputs.len() != 4 {
                return Err(Error::Msg(format!(
                    "Op::FlashAttn expects 3 or 4 inputs, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let q_layout = input_layout(node.inputs[0]);
            let k_layout = input_layout(node.inputs[1]);
            let v_layout = input_layout(node.inputs[2]);
            let q_dims = q_layout.shape().dims();
            let k_dims = k_layout.shape().dims();
            let v_dims = v_layout.shape().dims();
            if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
                return Err(Error::Msg(format!(
                    "Op::FlashAttn requires rank-4 q/k/v; got q={q_dims:?} k={k_dims:?} v={v_dims:?}",
                ))
                .bt());
            }
            if k_dims != v_dims {
                return Err(Error::Msg(format!(
                    "Op::FlashAttn: k {k_dims:?} and v {v_dims:?} must share shape",
                ))
                .bt());
            }
            if q_dims[0] != k_dims[0] || q_dims[3] != k_dims[3] {
                return Err(Error::Msg(format!(
                    "Op::FlashAttn: q {q_dims:?} and k {k_dims:?} must share B and D",
                ))
                .bt());
            }
            let sk = k_dims[2];
            // Resolve the logical attended length. `None` ⇒ attend the
            // full K extent (today's behavior). `Some(dyn)` ⇒ resolve
            // against the per-pass SymEnv; the resolved length must fit
            // the capacity and cover Sq (a valid causal prefix).
            let attended_k_len = match k_len {
                None => sk,
                Some(dyn_k) => {
                    let kl = dyn_k.resolve(sym_env).ok_or_else(|| {
                        Error::Msg(format!(
                            "Op::FlashAttn: dynamic k_len {dyn_k:?} is unbound in the SymEnv at realize",
                        ))
                        .bt()
                    })?;
                    if kl > sk {
                        return Err(Error::Msg(format!(
                            "Op::FlashAttn: resolved k_len ({kl}) exceeds K capacity ({sk})",
                        ))
                        .bt());
                    }
                    if kl < q_dims[2] {
                        return Err(Error::Msg(format!(
                            "Op::FlashAttn: resolved k_len ({kl}) < Sq ({}) — not a valid causal prefix",
                            q_dims[2],
                        ))
                        .bt());
                    }
                    kl
                }
            };
            OpParams::FlashAttn {
                b: q_dims[0],
                hq: q_dims[1],
                hkv: k_dims[1],
                sq: q_dims[2],
                sk,
                d: q_dims[3],
                k_len: attended_k_len,
                softmax_scale: *softmax_scale,
                causal: *causal,
                window_size_left: *window_size_left,
                window_size_right: *window_size_right,
                softcap: *softcap,
            }
        }
        Op::ReduceSumTo(target_shape) => {
            if node.inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "Op::ReduceSumTo expects 1 input, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let in_layout = input_layout(node.inputs[0]);
            let input_shape: Vec<usize> = in_layout.shape().dims().to_vec();
            let output_shape: Vec<usize> = target_shape.dims().to_vec();
            OpParams::ReduceSumTo { input_shape, output_shape }
        }
        Op::ReduceMaxTo(target_shape) => {
            if node.inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "Op::ReduceMaxTo expects 1 input, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let in_layout = input_layout(node.inputs[0]);
            let input_shape: Vec<usize> = in_layout.shape().dims().to_vec();
            let output_shape: Vec<usize> = target_shape.dims().to_vec();
            OpParams::ReduceMaxTo { input_shape, output_shape }
        }
        // Phase 7.6 step 5 (final): legacy `Op::ConvTranspose2D` arm dropped.
        Op::Fused(_, fuel_graph::registry::FusedOpParams::ConvTranspose2D {
            stride, padding, output_padding, dilation, groups,
        }) => {
            // Inputs[0] = x [N, Cin, Hin, Win]; inputs[1] = weight
            // [Cin, Cout/groups, Kh, Kw]; inputs[2] (optional) = bias [Cout].
            // Output (this Node's shape) = [N, Cout, Hout, Wout].
            if node.inputs.len() != 2 && node.inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "Op::ConvTranspose2D expects 2 or 3 inputs, got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let x_layout = input_layout(node.inputs[0]);
            let w_layout = input_layout(node.inputs[1]);
            let x_dims = x_layout.shape().dims();
            let w_dims = w_layout.shape().dims();
            if x_dims.len() != 4 || w_dims.len() != 4 {
                return Err(Error::Msg(format!(
                    "Op::ConvTranspose2D requires rank-4 x and weight; got x={x_dims:?} w={w_dims:?}",
                ))
                .bt());
            }
            let out_dims = node.shape.dims();
            if out_dims.len() != 4 {
                return Err(Error::Msg(format!(
                    "Op::ConvTranspose2D output must be rank 4, got {out_dims:?}",
                ))
                .bt());
            }
            let x_shape = [x_dims[0], x_dims[1], x_dims[2], x_dims[3]];
            let w_shape = [w_dims[0], w_dims[1], w_dims[2], w_dims[3]];
            let out_shape = [out_dims[0], out_dims[1], out_dims[2], out_dims[3]];
            OpParams::ConvTranspose2D {
                x_shape,
                w_shape,
                out_shape,
                stride: *stride,
                padding: *padding,
                output_padding: *output_padding,
                dilation: *dilation,
                groups: *groups,
            }
        }
        Op::WriteSlice { ranges, dyn_offset } => {
            // inputs[0] = destination (its shape == this node's shape
            // since WriteSlice's output adopts the destination's
            // Layout). inputs[1] = source slab; its shape is implied
            // by `ranges`. The kernel needs the destination shape to
            // compute strides for the slab walk.
            if node.inputs.len() != 2 {
                return Err(Error::Msg(format!(
                    "Op::WriteSlice expects 2 inputs (destination, source), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let dest_dims = node.shape.dims().to_vec();
            if ranges.len() != dest_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::WriteSlice: ranges.len() ({}) must equal destination rank ({})",
                    ranges.len(), dest_dims.len(),
                ))
                .bt());
            }
            // Phase D symbolic extents: when present, resolve the runtime
            // start offset against the per-pass SymEnv, overriding
            // `ranges[axis].0`. The slab WIDTH on that axis stays
            // `ranges[axis].1 - ranges[axis].0`, so the effective range
            // becomes `(base, base + width)`. `None` ⇒ fully static, and
            // the SymEnv is never consulted (empty-env realize unchanged).
            // The bounds loop below runs on the RESOLVED ranges, so an
            // out-of-capacity runtime offset surfaces as a typed error.
            let mut ranges = ranges.clone();
            if let Some((axis, off)) = dyn_offset {
                if *axis >= ranges.len() {
                    return Err(Error::Msg(format!(
                        "Op::WriteSlice: dyn_offset axis ({axis}) out of bounds for rank {}",
                        ranges.len(),
                    ))
                    .bt());
                }
                let base = off.resolve(sym_env).ok_or_else(|| {
                    Error::Msg(format!(
                        "Op::WriteSlice: dynamic offset {off:?} on axis {axis} is unbound \
                         in the SymEnv at realize",
                    ))
                    .bt()
                })?;
                let (start, end) = ranges[*axis];
                let width = end - start;
                ranges[*axis] = (base, base + width);
            }
            for (i, &(start, end)) in ranges.iter().enumerate() {
                if end < start || end > dest_dims[i] {
                    return Err(Error::Msg(format!(
                        "Op::WriteSlice: ranges[{i}] = ({start}, {end}) invalid \
                         for destination dim {i} = {}",
                        dest_dims[i],
                    ))
                    .bt());
                }
            }
            OpParams::WriteSlice {
                dest_shape: dest_dims,
                ranges,
            }
        }
        Op::WriteSliceRotating { axis, modulus, ranges } => {
            // inputs[0] = destination, inputs[1] = source slab,
            // inputs[2] = dynamic position scalar (U32). The builder
            // already validated rank / dtype / shape parity; the
            // exec-time check below mirrors WriteSlice for the
            // structural invariants the executor depends on.
            if node.inputs.len() != 3 {
                return Err(Error::Msg(format!(
                    "Op::WriteSliceRotating expects 3 inputs (destination, source, position), got {}",
                    node.inputs.len(),
                ))
                .bt());
            }
            let dest_dims = node.shape.dims().to_vec();
            if ranges.len() != dest_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::WriteSliceRotating: ranges.len() ({}) must equal destination rank ({})",
                    ranges.len(), dest_dims.len(),
                ))
                .bt());
            }
            if *axis >= dest_dims.len() {
                return Err(Error::Msg(format!(
                    "Op::WriteSliceRotating: axis {axis} out of bounds for rank {}",
                    dest_dims.len(),
                ))
                .bt());
            }
            if *modulus == 0 || *modulus > dest_dims[*axis] {
                return Err(Error::Msg(format!(
                    "Op::WriteSliceRotating: modulus {modulus} invalid for dest_shape[{axis}] = {}",
                    dest_dims[*axis],
                ))
                .bt());
            }
            OpParams::WriteSliceRotating {
                dest_shape: dest_dims,
                axis: *axis,
                modulus: *modulus,
                ranges: ranges.clone(),
            }
        }
        _ => OpParams::None,
    })
}

/// Execute one work item. Three branches by `WorkItemKind`:
///
/// - `ConstAdopt` — verify cache has an entry pre-seeded by the
///   caller; record the layout from the WorkItem.
/// - `ViewOf { input }` — clone the input's Storage Arc into the
///   output slot (bytes are shared); record the strided layout
///   from the WorkItem.
/// - `Kernel` — gather input Arcs, allocate the output, run the
///   compiled kernel, store the result; record the contiguous
///   layout from the WorkItem.
/// Execute one [`WorkItem`] against the cache, returning the node's
/// [`CompletionHandle`].
///
/// Step E A4b-1: the producing arms (Kernel / WriteSlice / WriteSliceRotating /
/// in-place / cross-device Copy alloc) return the handle from `execute_compiled`
/// *without* waiting it inline — the realize loop stores it in its per-node
/// handle map and drains at realize-end. Non-producing arms (const adopt, view,
/// slot projection, release marker, contiguize) are pure Arc-clones / host
/// allocs whose work is already complete → [`CompletionHandle::Ready`].
fn execute_work_item(
    item: &WorkItem,
    cache: &mut StorageCache,
    layout_cache: &mut HashMap<NodeId, Layout>,
) -> Result<CompletionHandle> {
    match &item.kind {
        WorkItemKind::ConstAdopt => {
            if !cache.contains_key(&item.node_id) {
                return Err(Error::Msg(format!(
                    "PipelinedExecutor: Const node {:?} not in input cache",
                    item.node_id
                ))
                .bt());
            }
            // Layout for input cache entries was seeded at realize
            // start; refresh from the WorkItem in case the side-table
            // was set after seeding.
            layout_cache.insert(item.node_id, item.output_layout.clone());
            Ok(CompletionHandle::Ready)
        }
        WorkItemKind::ViewOf { input } => {
            let input_arc = cache.get(input).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: view-op input {:?} of {:?} not realized",
                    input, item.node_id,
                ))
                .bt()
            })?;
            cache.insert(item.node_id, input_arc);
            layout_cache.insert(item.node_id, item.output_layout.clone());
            Ok(CompletionHandle::Ready)
        }
        WorkItemKind::SlotView { producer } => {
            // Op::View — multi-output projection. Same realization
            // shape as ViewOf: clone the producer's Storage Arc into
            // the View's cache slot. The WorkItem's output_layout was
            // computed at graph-build time by `Tensor::view`, with
            // the slot's byte_offset baked into start_offset (in
            // slot-dtype-element units). Downstream kernels reading
            // the producer's bytes as the slot's dtype starting at
            // that offset see the slot's typed window.
            let producer_arc = cache.get(producer).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: Op::View producer {:?} of {:?} not realized",
                    producer, item.node_id,
                ))
                .bt()
            })?;
            cache.insert(item.node_id, producer_arc);
            layout_cache.insert(item.node_id, item.output_layout.clone());
            Ok(CompletionHandle::Ready)
        }
        WorkItemKind::SlotOwn { producer, slot } => {
            // Op::ViewOwned — allocate a fresh Storage of the slot's
            // (shape, dtype) on the producer's device and copy the
            // slot's bytes in. Per-backend fast paths:
            //   - CPU:    direct &[u8]→&mut[u8] memcpy via CpuStorageBytes
            //   - CUDA:   D2D via CudaStorageBytes::slot_copy_to_new
            //             (cuMemcpyDtoDAsync with offset src pointer)
            //   - Vulkan: vkCmdCopyBuffer via
            //             VulkanBackend::slot_copy_to_new_handle
            //             (srcOffset/dstOffset/size)
            // For backends without a copy-with-offset hook we fall
            // back to a host round-trip (`to_host_buffer_dyn` →
            // slice → `storage_from_host_buffer_owned_dyn`), which
            // works but is wasteful.
            let producer_arc = cache.get(producer).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: Op::ViewOwned producer {:?} of {:?} not realized",
                    producer, item.node_id,
                ))
                .bt()
            })?;
            let new_storage = {
                let guard = producer_arc.read()
                    .map_err(|_| poisoned("ViewOwned producer storage"))?;
                let bundle = guard.bundle().ok_or_else(|| {
                    Error::Msg(format!(
                        "Op::ViewOwned: producer {:?} has no bundle side-table \
                         — the multi-output authoring contract requires \
                         Storage::with_bundle / new_bundled at allocation \
                         time. Node {:?} slot {} cannot resolve its source.",
                        producer, item.node_id, slot,
                    )).bt()
                })?;
                let sv = bundle.get(*slot as usize).cloned().ok_or_else(|| {
                    Error::Msg(format!(
                        "Op::ViewOwned: slot {} out of range for producer {:?} \
                         (bundle has {} slots)",
                        slot, producer, bundle.len(),
                    )).bt()
                })?;
                use fuel_memory::BackendStorage;
                match &guard.inner {
                    BackendStorage::Cpu(cpu_bytes) => {
                        let src_bytes = cpu_bytes.bytes();
                        let end = sv.byte_offset + sv.len_bytes();
                        if end > src_bytes.len() {
                            return Err(Error::Msg(format!(
                                "Op::ViewOwned: slot {} byte range [{}..{}) \
                                 exceeds producer's CPU byte length {}",
                                slot, sv.byte_offset, end, src_bytes.len(),
                            )).bt());
                        }
                        let owned = fuel_cpu_backend::CpuStorageBytes::from_bytes(
                            &src_bytes[sv.byte_offset..end],
                        );
                        fuel_memory::Storage::new(
                            BackendStorage::Cpu(owned),
                            sv.dtype,
                        )
                    }
                    #[cfg(feature = "cuda")]
                    BackendStorage::Cuda(cuda_bytes) => {
                        let dst = cuda_bytes.slot_copy_to_new(
                            sv.byte_offset, sv.len_bytes(),
                        )?;
                        fuel_memory::Storage::new(
                            BackendStorage::Cuda(dst),
                            sv.dtype,
                        )
                    }
                    #[cfg(feature = "vulkan")]
                    BackendStorage::Vulkan(vk_bytes) => {
                        let backend = vk_bytes.backend().ok_or_else(|| {
                            Error::Msg(format!(
                                "Op::ViewOwned: producer {:?} is on Vulkan but \
                                 its storage has no VulkanBackend handle. \
                                 Storages flowing through the pipelined executor \
                                 must be constructed via alloc_bytes_handle / \
                                 upload_bytes_handle so the bundle's slot extraction \
                                 path can reach the backend.",
                                producer,
                            )).bt()
                        })?;
                        let dst = backend.slot_copy_to_new_handle(
                            vk_bytes, sv.byte_offset, sv.len_bytes(),
                        )?;
                        fuel_memory::Storage::new(
                            BackendStorage::Vulkan(dst),
                            sv.dtype,
                        )
                    }
                    #[allow(unreachable_patterns)]
                    _ => {
                        return Err(Error::Msg(format!(
                            "Op::ViewOwned: producer {:?} is on a backend \
                             without a slot-copy-with-offset hook \
                             (CPU + CUDA + Vulkan wired). Add a per-backend \
                             `slot_copy_to_new`-equivalent and extend the \
                             match arm.",
                            producer,
                        )).bt());
                    }
                }
            };
            cache.insert(item.node_id, Arc::new(RwLock::new(new_storage)));
            layout_cache.insert(item.node_id, item.output_layout.clone());
            // A4b-1: Op::ViewOwned's slot copy (CUDA D2D) is same-stream; its
            // completion is carried by the realize-end full-stream sync in
            // `to_cpu_bytes` (unchanged) — `Ready` here is behavior-preserving.
            Ok(CompletionHandle::Ready)
        }
        WorkItemKind::WriteSlice { dest, source } => {
            let compiled = item.compiled.as_ref().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: WriteSlice work item {:?} has no compiled node",
                    item.node_id,
                ))
                .bt()
            })?;
            let dest_arc = cache.get(dest).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: WriteSlice destination {:?} of {:?} not realized",
                    dest, item.node_id,
                ))
                .bt()
            })?;
            let source_arc = cache.get(source).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: WriteSlice source {:?} of {:?} not realized",
                    source, item.node_id,
                ))
                .bt()
            })?;
            let dest_layout = layout_cache.get(dest).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: WriteSlice destination {:?} of {:?} has no cached layout",
                    dest, item.node_id,
                ))
                .bt()
            })?;
            let source_layout = layout_cache.get(source).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: WriteSlice source {:?} of {:?} has no cached layout",
                    source, item.node_id,
                ))
                .bt()
            })?;
            // v1 contract: destination must be contiguous + zero
            // offset. Strided dests would force the kernel to walk
            // both source and dest strides; defer until KV cache
            // exposes a non-contig dest. Source CAN be non-contig —
            // we auto-contiguize it here so the kernel sees a flat
            // slab.
            if !dest_layout.is_contiguous() || dest_layout.start_offset() != 0 {
                return Err(Error::Msg(format!(
                    "Op::WriteSlice (Phase E.3.2 v1): destination {:?} must be \
                     contiguous + zero-offset; got Layout {:?}",
                    dest, dest_layout,
                ))
                .bt());
            }
            let source_arc_contig = {
                let s_dtype = source_arc
                    .read()
                    .map_err(|_| poisoned("WriteSlice source storage"))?
                    .dtype;
                let s_len_bytes = source_arc
                    .read()
                    .map_err(|_| poisoned("WriteSlice source storage"))?
                    .inner
                    .len_bytes();
                let layout_bytes =
                    source_layout.shape().elem_count() * s_dtype.size_in_bytes();
                let bytes_match_shape = s_len_bytes == layout_bytes;
                let already_contig = source_layout.is_contiguous()
                    && source_layout.start_offset() == 0
                    && bytes_match_shape;
                if already_contig {
                    source_arc
                } else {
                    auto_contiguize(&source_arc, &source_layout)?
                }
            };
            let source_layout_kernel =
                fuel_ir::Layout::contiguous(source_layout.shape().clone());
            // The kernel sees inputs=[source] and outputs=[dest_arc].
            // The dest input slot is intentionally absent — the kernel
            // doesn't read from dest; it only writes to its bytes
            // through the output slot's write lock.
            let input_arcs = vec![source_arc_contig];
            let mut output_arcs = vec![dest_arc.clone()];
            let kernel_layouts = vec![source_layout_kernel, dest_layout.clone()];
            // A4b-1: defer the wait — return the handle to the realize loop.
            let handle =
                execute_compiled(compiled, &input_arcs, &mut output_arcs, &kernel_layouts)?;
            // Adopt the dest Arc at this WriteSlice node's slot. The
            // realize loop's destructive_input cleanup evicts the
            // dest's own NodeId from the cache afterward — downstream
            // readers go through this node's NodeId.
            cache.insert(item.node_id, dest_arc);
            layout_cache.insert(item.node_id, item.output_layout.clone());
            Ok(handle)
        }
        WorkItemKind::WriteSliceRotating { dest, source, position } => {
            // Same shape as WriteSlice but with a position input.
            // Kernel sees `[source, position]` as inputs; `dest` is the
            // pre-allocated output buffer (in-place adoption).
            let compiled = item.compiled.as_ref().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: WriteSliceRotating work item {:?} has no compiled node",
                    item.node_id,
                ))
                .bt()
            })?;
            let dest_arc = cache.get(dest).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: WriteSliceRotating destination {:?} of {:?} not realized",
                    dest, item.node_id,
                ))
                .bt()
            })?;
            let source_arc = cache.get(source).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: WriteSliceRotating source {:?} of {:?} not realized",
                    source, item.node_id,
                ))
                .bt()
            })?;
            let position_arc = cache.get(position).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: WriteSliceRotating position {:?} of {:?} not realized",
                    position, item.node_id,
                ))
                .bt()
            })?;
            let dest_layout = layout_cache.get(dest).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: WriteSliceRotating destination {:?} of {:?} has no cached layout",
                    dest, item.node_id,
                ))
                .bt()
            })?;
            let source_layout = layout_cache.get(source).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: WriteSliceRotating source {:?} of {:?} has no cached layout",
                    source, item.node_id,
                ))
                .bt()
            })?;
            // v1: destination contiguous + zero offset (same as WriteSlice).
            if !dest_layout.is_contiguous() || dest_layout.start_offset() != 0 {
                return Err(Error::Msg(format!(
                    "Op::WriteSliceRotating (v1): destination {:?} must be \
                     contiguous + zero-offset; got Layout {:?}",
                    dest, dest_layout,
                ))
                .bt());
            }
            let source_arc_contig = {
                let s_dtype = source_arc
                    .read()
                    .map_err(|_| poisoned("WriteSliceRotating source storage"))?
                    .dtype;
                let s_len_bytes = source_arc
                    .read()
                    .map_err(|_| poisoned("WriteSliceRotating source storage"))?
                    .inner
                    .len_bytes();
                let layout_bytes =
                    source_layout.shape().elem_count() * s_dtype.size_in_bytes();
                let bytes_match_shape = s_len_bytes == layout_bytes;
                let already_contig = source_layout.is_contiguous()
                    && source_layout.start_offset() == 0
                    && bytes_match_shape;
                if already_contig {
                    source_arc
                } else {
                    auto_contiguize(&source_arc, &source_layout)?
                }
            };
            let source_layout_kernel =
                fuel_ir::Layout::contiguous(source_layout.shape().clone());
            let position_layout = fuel_ir::Layout::contiguous(
                fuel_ir::Shape::from_dims(&[]),
            );
            let input_arcs = vec![source_arc_contig, position_arc];
            let mut output_arcs = vec![dest_arc.clone()];
            let kernel_layouts = vec![
                source_layout_kernel, position_layout, dest_layout.clone(),
            ];
            // A4b-1: defer the wait — return the handle to the realize loop.
            let handle =
                execute_compiled(compiled, &input_arcs, &mut output_arcs, &kernel_layouts)?;
            cache.insert(item.node_id, dest_arc);
            layout_cache.insert(item.node_id, item.output_layout.clone());
            Ok(handle)
        }
        WorkItemKind::Alloc { target_location } => {
            // Op::Alloc { target }: allocate a fresh, zero-init storage
            // on target_location. No inputs, no kernel call — direct
            // executor dispatch. For non-CPU targets we derive the
            // per-backend device handle by searching the input cache
            // for any storage on the target backend (callers seed this
            // via `pipelined_bridge::device_seed_storage`).
            let n_bytes = item.elem_count * item.dtype.size_in_bytes();
            let alloced: Storage = match target_location {
                // CPU has no separate uninit alloc primitive in safe
                // Rust (`vec![0; n]` is the canonical path). Op::Alloc
                // on CPU returns zero-init storage; the following
                // Op::ZeroFill is a redundant memset the optimizer
                // can elide and LLVM folds away regardless.
                DeviceLocation::Cpu => fuel_memory::alloc_cpu_zeroed(item.dtype, item.elem_count)?,
                #[cfg(feature = "cuda")]
                DeviceLocation::Cuda { gpu_id } => {
                    let cuda_dev = find_cuda_device_in_cache(cache, *gpu_id)
                        .ok_or_else(|| Error::Msg(format!(
                            "Op::Alloc on Cuda {{ gpu_id: {} }}: no CUDA \
                             storage in input cache to derive the device \
                             handle from. The caller must seed the cache \
                             (e.g. via `fuel-core::pipelined_bridge::\
                             device_seed_storage`) before realizing.",
                            gpu_id,
                        )).bt())?;
                    // Phase 3a follow-up: true uninit alloc via
                    // baracuda alpha.30's unsafe alloc. The bytes are
                    // uninit until a subsequent Op::ZeroFill or full-
                    // overwrite op runs. The byte-storage Arc has no
                    // typed-slice accessor that would dereference
                    // uninit bytes.
                    let cuda_bytes =
                        fuel_cuda_backend::CudaStorageBytes::alloc_uninit(&cuda_dev, n_bytes)?;
                    Storage::new(fuel_memory::BackendStorage::Cuda(cuda_bytes), item.dtype)
                }
                #[cfg(not(feature = "cuda"))]
                DeviceLocation::Cuda { .. } => {
                    return Err(Error::Msg(
                        "Op::Alloc on Cuda but fuel-storage wasn't built \
                         with --features cuda".to_string(),
                    ).bt());
                }
                #[cfg(feature = "vulkan")]
                DeviceLocation::Vulkan { gpu_id } => {
                    let backend = find_vulkan_backend_in_cache(cache, *gpu_id)
                        .ok_or_else(|| Error::Msg(format!(
                            "Op::Alloc on Vulkan {{ gpu_id: {} }}: no Vulkan \
                             storage in input cache to derive the backend \
                             handle from. The caller must seed the cache \
                             (e.g. via `fuel-core::pipelined_bridge::\
                             device_seed_storage`) before realizing.",
                            gpu_id,
                        )).bt())?;
                    // Phase 3a follow-up: true uninit alloc. The old
                    // `upload_bytes_handle(vec![0u8; n])` host-staged
                    // zeros (~2× the bandwidth of a device-side fill);
                    // we now allocate uninit and rely on a paired
                    // Op::ZeroFill (vkCmdFillBuffer) for the
                    // initialization. Faster KV-cache init on Vulkan.
                    let vk_bytes = backend.alloc_bytes_handle(n_bytes)?;
                    Storage::new(fuel_memory::BackendStorage::Vulkan(vk_bytes), item.dtype)
                }
                #[cfg(not(feature = "vulkan"))]
                DeviceLocation::Vulkan { .. } => {
                    return Err(Error::Msg(
                        "Op::Alloc on Vulkan but fuel-storage wasn't built \
                         with --features vulkan".to_string(),
                    ).bt());
                }
                other => {
                    return Err(Error::Msg(format!(
                        "Op::Alloc on {:?}: target not yet wired (CPU + \
                         CUDA + Vulkan land in Phase 3a of bridge-retirement; \
                         Metal extends when its byte-storage substrate is \
                         ready).", other,
                    )).bt());
                }
            };
            cache.insert(item.node_id, Arc::new(RwLock::new(alloced)));
            layout_cache.insert(item.node_id, item.output_layout.clone());
            Ok(CompletionHandle::Ready)
        }
        WorkItemKind::ZeroFill => {
            // Op::ZeroFill: in-place zero the input's bytes. Output
            // adopts the input's Storage Arc (same Storage; bytes
            // are mutated). Direct per-backend dispatch — no kernel
            // call, no binding-table lookup.
            if item.inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "PipelinedExecutor: ZeroFill work item {:?} expects \
                     1 input, got {}",
                    item.node_id, item.inputs.len(),
                )).bt());
            }
            let src_id = item.inputs[0];
            let src_arc = cache.get(&src_id).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: ZeroFill input {:?} of {:?} \
                     not realized",
                    src_id, item.node_id,
                ))
                .bt()
            })?;
            // Per-backend in-place fill. Acquired the write lock once
            // for the duration of the fill — kernels on async backends
            // (CUDA / Vulkan) issue the memset and return; the fence
            // happens when downstream readers acquire the read lock
            // through the executor's standard synchronization path.
            {
                let mut guard = src_arc
                    .write()
                    .map_err(|_| poisoned("ZeroFill destination storage"))?;
                match &mut guard.inner {
                    fuel_memory::BackendStorage::Cpu(c) => {
                        let bytes = c.bytes_mut();
                        bytes.fill(0);
                    }
                    #[cfg(feature = "cuda")]
                    fuel_memory::BackendStorage::Cuda(c) => {
                        c.zero_async()?;
                    }
                    #[cfg(feature = "vulkan")]
                    fuel_memory::BackendStorage::Vulkan(v) => {
                        let backend = v.backend().ok_or_else(|| {
                            Error::Msg(
                                "Op::ZeroFill on Vulkan: input has no \
                                 attached backend handle. Storage must \
                                 come from VulkanBackend::alloc_bytes_handle \
                                 / upload_bytes_handle.".to_string()
                            ).bt()
                        })?.clone();
                        backend.fill_bytes_zero(v)?;
                    }
                    #[allow(unreachable_patterns)]
                    other => {
                        return Err(Error::Msg(format!(
                            "Op::ZeroFill: backend not wired ({other:?}); \
                             CPU + CUDA + Vulkan covered, Metal extends \
                             when its byte-storage substrate is ready.",
                        )).bt());
                    }
                }
            }
            // Adopt the (now-zero) Storage Arc at this node's slot.
            // Downstream readers go through `node_id`; the original
            // input's NodeId is evicted from the cache by the
            // realize loop's destructive_input cleanup.
            cache.insert(item.node_id, src_arc);
            layout_cache.insert(item.node_id, item.output_layout.clone());
            // A4b-1: ZeroFill's CUDA memset is same-stream (and Op::Alloc
            // typically zero-inits, so this is usually elided). Its completion
            // is carried by the realize-end full-stream sync in `to_cpu_bytes`
            // (unchanged) and same-stream ordering — `Ready` is behavior-preserving.
            Ok(CompletionHandle::Ready)
        }
        WorkItemKind::Copy { target_location }
        | WorkItemKind::Move { target_location } => {
            // Op::Copy / Op::Move { target }: kernel lookup at
            // (OpKind::Copy, [dt, dt], source_backend) — the wrapper
            // downloads from its own residency into a freshly-
            // allocated output on `target_location`. We auto-
            // contiguize the input first so the kernel always sees
            // the logical view's bytes (a transpose-view source
            // materializes into a contiguous buffer before download).
            //
            // Move shares this entire data-movement half; its
            // destructive half (release the source) is driven by the
            // realize loop's `destructive_input` cleanup after this
            // arm returns — see `WorkItemKind::Move`.
            let op_label = if matches!(item.kind, WorkItemKind::Move { .. }) {
                "Move"
            } else {
                "Copy"
            };
            let compiled = item.compiled.as_ref().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: {op_label} work item {:?} has no compiled node",
                    item.node_id,
                ))
                .bt()
            })?;
            if item.inputs.len() != 1 {
                return Err(Error::Msg(format!(
                    "PipelinedExecutor: {op_label} work item {:?} expects 1 input, got {}",
                    item.node_id, item.inputs.len(),
                )).bt());
            }
            let src_id = item.inputs[0];
            let src_arc = cache.get(&src_id).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: {op_label} input {:?} of {:?} not realized",
                    src_id, item.node_id,
                ))
                .bt()
            })?;
            // Multi-output Option C guard: Op::Copy on a bundled
            // producer would silently drop the bundle metadata on
            // the destination, and the auto-contiguize step below
            // would treat the bundle as a flat slot-0-shaped buffer
            // and corrupt non-primary slots. Until the per-backend
            // whole-bundle-Copy hook lands, point callers at the
            // supported per-slot path:
            // Op::View → Op::Copy → Op::ViewOwned.
            //
            // Exception: when the immediate input is an Op::View
            // projection (its layout's byte extent is strictly less
            // than the bundle's total bytes), the View has already
            // narrowed the slab. auto-contiguize walks the View's
            // layout and produces a flat slot-shaped buffer that's
            // correctly sized — no data loss, no slot corruption.
            // The guard only fires when the Copy targets the bundle
            // root directly.
            {
                let guard = src_arc.read().map_err(|_| poisoned("Copy source storage"))?;
                if guard.is_bundled() {
                    let layout_bytes_for_guard = layout_cache
                        .get(&src_id)
                        .map(|l| l.shape().elem_count() * guard.dtype.size_in_bytes())
                        .unwrap_or(usize::MAX);
                    let bundle_bytes = guard.inner.len_bytes();
                    let is_whole_bundle = layout_bytes_for_guard >= bundle_bytes;
                    if is_whole_bundle {
                        return Err(Error::Msg(format!(
                            "PipelinedExecutor: Op::{op_label} on a bundled producer \
                             ({:?}) is not yet supported (Option C followup). \
                             Use Op::View → Op::Copy → Op::ViewOwned for \
                             per-slot cross-device moves, or wait for the \
                             per-backend whole-bundle-Copy hook.",
                            src_id,
                        )).bt());
                    }
                }
            }
            let src_layout = layout_cache.get(&src_id).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: {op_label} input {:?} of {:?} has no cached layout",
                    src_id, item.node_id,
                ))
                .bt()
            })?;
            let src_dtype = src_arc
                .read()
                .map_err(|_| poisoned("Copy source storage"))?
                .dtype;
            let src_len_bytes = src_arc
                .read()
                .map_err(|_| poisoned("Copy source storage"))?
                .inner
                .len_bytes();
            let layout_bytes =
                src_layout.shape().elem_count() * src_dtype.size_in_bytes();
            let bytes_match_shape = src_len_bytes == layout_bytes;
            let already_contig = src_layout.is_contiguous()
                && src_layout.start_offset() == 0
                && bytes_match_shape;
            let src_input = if already_contig {
                src_arc
            } else {
                auto_contiguize(&src_arc, &src_layout)?
            };
            let kernel_input_layout =
                fuel_ir::Layout::contiguous(src_layout.shape().clone());
            // Allocate the output on `target_location`. Phase 2
            // covered D2H (target=Cpu); Phase 3b extends to H2D
            // (target=Cuda / Vulkan from a CPU source). For non-CPU
            // targets the executor's Op::Alloc-style device-handle
            // search applies (find_cuda_device_in_cache /
            // find_vulkan_backend_in_cache); callers seed the cache
            // (e.g. `pipelined_bridge::device_seed_storage`) before
            // realizing.
            let n_bytes = item.elem_count * item.dtype.size_in_bytes();
            let output = match target_location {
                DeviceLocation::Cpu => {
                    fuel_memory::alloc_cpu_zeroed(item.dtype, item.elem_count)?
                }
                #[cfg(feature = "cuda")]
                DeviceLocation::Cuda { gpu_id } => {
                    let cuda_dev = find_cuda_device_in_cache(cache, *gpu_id)
                        .ok_or_else(|| Error::Msg(format!(
                            "Op::{op_label} on Cuda {{ gpu_id: {} }}: no CUDA \
                             storage in input cache to derive the device \
                             handle from. The caller must seed the cache \
                             (e.g. via `fuel-core::pipelined_bridge::\
                             device_seed_storage`) before realizing an \
                             H2D Op::{op_label}.",
                            gpu_id,
                        )).bt())?;
                    let cuda_bytes =
                        fuel_cuda_backend::CudaStorageBytes::alloc_uninit(&cuda_dev, n_bytes)?;
                    Storage::new(fuel_memory::BackendStorage::Cuda(cuda_bytes), item.dtype)
                }
                #[cfg(not(feature = "cuda"))]
                DeviceLocation::Cuda { .. } => {
                    return Err(Error::Msg(format!(
                        "Op::{op_label} target Cuda but fuel-storage wasn't built \
                         with --features cuda",
                    )).bt());
                }
                #[cfg(feature = "vulkan")]
                DeviceLocation::Vulkan { gpu_id } => {
                    let backend = find_vulkan_backend_in_cache(cache, *gpu_id)
                        .ok_or_else(|| Error::Msg(format!(
                            "Op::{op_label} on Vulkan {{ gpu_id: {} }}: no Vulkan \
                             storage in input cache to derive the backend \
                             handle from. The caller must seed the cache \
                             (e.g. via `fuel-core::pipelined_bridge::\
                             device_seed_storage`) before realizing an \
                             H2D Op::{op_label}.",
                            gpu_id,
                        )).bt())?;
                    let vk_bytes = backend.alloc_bytes_handle(n_bytes)?;
                    Storage::new(fuel_memory::BackendStorage::Vulkan(vk_bytes), item.dtype)
                }
                #[cfg(not(feature = "vulkan"))]
                DeviceLocation::Vulkan { .. } => {
                    return Err(Error::Msg(format!(
                        "Op::{op_label} target Vulkan but fuel-storage wasn't built \
                         with --features vulkan",
                    )).bt());
                }
                other => {
                    return Err(Error::Msg(format!(
                        "PipelinedExecutor: Op::{op_label} target_location {:?} not yet \
                         wired (CPU + CUDA + Vulkan covered; Metal extends when \
                         its byte-storage substrate is ready).",
                        other
                    )).bt());
                }
            };
            let input_arcs = vec![src_input];
            let mut output_arcs = vec![Arc::new(RwLock::new(output))];
            let kernel_layouts =
                vec![kernel_input_layout, item.output_layout.clone()];
            // A4b-1: defer the wait — return the handle to the realize loop.
            let handle =
                execute_compiled(compiled, &input_arcs, &mut output_arcs, &kernel_layouts)?;
            let arc = output_arcs.into_iter().next().expect("one output");
            cache.insert(item.node_id, arc);
            layout_cache.insert(item.node_id, item.output_layout.clone());
            Ok(handle)
        }
        WorkItemKind::ReleaseMarker => {
            // Emit a zero-byte CPU storage at the Release node's
            // slot. Per Op::Release's contract the marker is never
            // read — this exists so any downstream cache lookup of
            // `release_id` resolves rather than failing. The actual
            // deallocation of `inputs[0]` is driven by the realize
            // loop's `destructive_input` cleanup (Phase B).
            let marker = fuel_memory::alloc_cpu_zeroed(item.dtype, 0)?;
            cache.insert(item.node_id, Arc::new(RwLock::new(marker)));
            layout_cache.insert(item.node_id, item.output_layout.clone());
            Ok(CompletionHandle::Ready)
        }
        WorkItemKind::ContiguizeOf { input } => {
            let input_arc = cache.get(input).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: reshape input {:?} of {:?} not realized",
                    input, item.node_id,
                ))
                .bt()
            })?;
            let input_layout = layout_cache.get(input).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: reshape input {:?} of {:?} has no cached layout",
                    input, item.node_id,
                ))
                .bt()
            })?;
            // Zero-copy when the input is already contiguous + zero
            // offset; allocate + copy via the contiguize kernel
            // otherwise.
            let out_arc =
                if input_layout.is_contiguous() && input_layout.start_offset() == 0 {
                    input_arc
                } else {
                    auto_contiguize(&input_arc, &input_layout)?
                };
            cache.insert(item.node_id, out_arc);
            layout_cache.insert(item.node_id, item.output_layout.clone());
            // A4b-1: a materializing contiguize (CUDA) is same-stream; its
            // completion is carried by the realize-end full-stream sync in
            // `to_cpu_bytes` (unchanged) — `Ready` here is behavior-preserving.
            Ok(CompletionHandle::Ready)
        }
        WorkItemKind::Kernel => {
            let compiled = item.compiled.as_ref().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: Kernel work item {:?} has no compiled node",
                    item.node_id,
                ))
                .bt()
            })?;
            // Gather input Arcs from the cache, auto-contiguizing
            // any input whose layout is non-contiguous (typically
            // produced by an upstream metadata-only view op).
            // Today's kernels assume contiguous; this pass keeps
            // that invariant true at every kernel call site.
            //
            // We assemble a parallel `kernel_layouts` vec — the layout
            // of the bytes the kernel actually receives. After
            // auto-contiguize, that's `Layout::contiguous(shape)` for
            // the input's shape; for inputs already contiguous we use
            // the cached layout directly. Output layout comes last.
            let mut input_arcs: Vec<Arc<RwLock<Storage>>> = Vec::with_capacity(item.inputs.len());
            let mut kernel_layouts: Vec<fuel_ir::Layout> =
                Vec::with_capacity(item.inputs.len() + 1);
            // The kernel's `strided_input` cap lets non-contiguous
            // inputs (broadcast, transpose, etc.) flow through without
            // materialization — the kernel walks strides itself. Inputs
            // with non-zero `start_offset` still go through auto-
            // Contiguize today; offset honoring on the byte buffer is
            // a separate concern from stride support.
            let kernel_handles_strided = compiled.caps.strided_input;
            for in_id in &item.inputs {
                let in_arc = cache.get(in_id).cloned().ok_or_else(|| {
                    Error::Msg(format!(
                        "PipelinedExecutor: input {:?} of {:?} not realized",
                        in_id, item.node_id,
                    ))
                    .bt()
                })?;
                let in_layout = layout_cache.get(in_id).cloned().ok_or_else(|| {
                    Error::Msg(format!(
                        "PipelinedExecutor: input {:?} of {:?} has no cached layout",
                        in_id, item.node_id,
                    ))
                    .bt()
                })?;
                // "Already contig" means more than `is_contiguous()`:
                // the LAYOUT's element count must also match the
                // STORAGE's byte count for shape-validating kernels
                // (concat_cpu, etc.) that read `storage.len_bytes()`.
                // A first-N slice view has `is_contiguous() == true`
                // (strides match canonical [1] for shape [N]) but its
                // storage is the parent's full bytes — too big. In
                // that case we auto-contiguize to materialize only
                // the view's portion as a fresh storage.
                //
                // Same constraint applies to strided_ok: kernels that
                // declare strided support can walk strided inputs, but
                // the byte count check at the kernel surface still
                // assumes storage.len_bytes() == layout.elem_count() *
                // dtype.size_bytes(). Until kernels universally trust
                // layout over storage, we conservatively materialize.
                let in_dtype = in_arc
                    .read()
                    .map_err(|_| poisoned("input storage"))?
                    .dtype;
                let in_len_bytes = in_arc
                    .read()
                    .map_err(|_| poisoned("input storage"))?
                    .inner
                    .len_bytes();
                let layout_bytes =
                    in_layout.shape().elem_count() * in_dtype.size_in_bytes();
                let bytes_match_shape = in_len_bytes == layout_bytes;
                let already_contig = in_layout.is_contiguous()
                    && in_layout.start_offset() == 0
                    && bytes_match_shape;
                let strided_ok = kernel_handles_strided
                    && in_layout.start_offset() == 0
                    && bytes_match_shape;
                if already_contig || strided_ok {
                    input_arcs.push(in_arc);
                    kernel_layouts.push(in_layout);
                } else {
                    let contig_arc = auto_contiguize(&in_arc, &in_layout)?;
                    input_arcs.push(contig_arc);
                    kernel_layouts.push(fuel_ir::Layout::contiguous(
                        in_layout.shape().clone(),
                    ));
                }
            }
            kernel_layouts.push(item.output_layout.clone());

            // Allocate output on the target backend. For GPU backends
            // we derive the device handle from the first input —
            // every kernel has ≥1 input (Op::Const is handled via
            // ConstAdopt, never via Kernel), and an input on the
            // target backend carries its own device. This avoids
            // threading a device handle through `realize`.
            //
            // Multi-output (Option C, item 3): when the WorkItem
            // carries an `output_bundle`, size the allocation to
            // span every slot's bytes (primary-dtype-element count
            // rounded up from the bundle's total byte span) and
            // attach the bundle metadata after allocation.
            let (alloc_elem_count, bundle_to_attach) = match &item.output_bundle {
                Some(bundle) => {
                    let dtype_bytes = item.dtype.size_in_bytes().max(1);
                    let total_bytes = bundle.iter().fold(0usize, |acc, s| {
                        let end = s.byte_offset.saturating_add(s.len_bytes());
                        acc.max(end)
                    });
                    let elems = total_bytes.div_ceil(dtype_bytes);
                    (elems, Some(Arc::clone(bundle)))
                }
                None => (item.elem_count, None),
            };
            let output = match item.target_backend {
                BackendId::Cpu => fuel_memory::alloc_cpu_zeroed(item.dtype, alloc_elem_count)?,
                #[cfg(feature = "cuda")]
                BackendId::Cuda => {
                    let first_in = input_arcs.first().ok_or_else(|| {
                        Error::Msg(format!(
                            "PipelinedExecutor: kernel {:?} on Cuda has no inputs; \
                             cannot derive device for output allocation",
                            item.node_id,
                        ))
                        .bt()
                    })?;
                    let guard = first_in.read().map_err(|_| poisoned("input storage"))?;
                    let cuda_in = match &guard.inner {
                        fuel_memory::BackendStorage::Cuda(c) => c,
                        other => {
                            return Err(Error::Msg(format!(
                                "PipelinedExecutor: kernel {:?} target_backend=Cuda but \
                                 input has BackendStorage::{:?}; mixed-backend kernels \
                                 require an explicit Op::Copy first",
                                item.node_id,
                                std::mem::discriminant(other),
                            ))
                            .bt());
                        }
                    };
                    let n_bytes = alloc_elem_count * item.dtype.size_in_bytes();
                    let cuda_bytes =
                        fuel_cuda_backend::CudaStorageBytes::alloc(cuda_in.device(), n_bytes)?;
                    fuel_memory::Storage::new(fuel_memory::BackendStorage::Cuda(cuda_bytes), item.dtype)
                }
                #[cfg(feature = "vulkan")]
                BackendId::Vulkan => {
                    // Mirror the CUDA path: derive the backend handle
                    // from the first input's `VulkanStorageBytes::backend()`.
                    // Vulkan catch-up V.1.B (2026-05-21) — requires
                    // inputs constructed via `alloc_bytes_handle` /
                    // `upload_bytes_handle` so they carry the
                    // `Arc<VulkanBackend>` (mirroring CUDA's
                    // `Arc<CudaDevice>`).
                    let first_in = input_arcs.first().ok_or_else(|| {
                        Error::Msg(format!(
                            "PipelinedExecutor: kernel {:?} on Vulkan has no inputs; \
                             cannot derive backend handle for output allocation",
                            item.node_id,
                        ))
                        .bt()
                    })?;
                    let guard = first_in.read().map_err(|_| poisoned("input storage"))?;
                    let vk_in = match &guard.inner {
                        fuel_memory::BackendStorage::Vulkan(v) => v,
                        other => {
                            return Err(Error::Msg(format!(
                                "PipelinedExecutor: kernel {:?} target_backend=Vulkan but \
                                 input has BackendStorage::{:?}; mixed-backend kernels \
                                 require an explicit Op::Copy first",
                                item.node_id,
                                std::mem::discriminant(other),
                            ))
                            .bt());
                        }
                    };
                    let backend = vk_in.backend().ok_or_else(|| {
                        Error::Msg(format!(
                            "PipelinedExecutor: Vulkan kernel {:?} input has no backend \
                             handle. Storages flowing through the pipelined executor's \
                             Vulkan path must be constructed via \
                             VulkanBackend::alloc_bytes_handle / upload_bytes_handle \
                             (not the legacy alloc_bytes / upload_bytes which leave \
                             VulkanStorageBytes::backend = None).",
                            item.node_id,
                        ))
                        .bt()
                    })?;
                    let n_bytes = alloc_elem_count * item.dtype.size_in_bytes();
                    let vk_bytes = backend.alloc_bytes_handle(n_bytes)?;
                    fuel_memory::Storage::new(fuel_memory::BackendStorage::Vulkan(vk_bytes), item.dtype)
                }
                other => {
                    return Err(Error::Msg(format!(
                        "PipelinedExecutor: target_backend {:?} output allocation \
                         not yet implemented (CPU + CUDA + Vulkan wired; Metal extends later)",
                        other
                    ))
                    .bt());
                }
            };
            // Attach bundle metadata when this is a multi-output op.
            // The kernel sees the allocated bytes; the bundle lives
            // at the Storage wrapper level so downstream Op::View /
            // Op::ViewOwned can resolve their slot windows.
            let output = match bundle_to_attach {
                Some(bundle) => output.with_bundle(bundle)?,
                None => output,
            };
            let mut output_arcs = vec![Arc::new(RwLock::new(output))];

            // A4b-1: defer the wait — return the handle to the realize loop.
            let handle =
                execute_compiled(compiled, &input_arcs, &mut output_arcs, &kernel_layouts)?;

            let arc = output_arcs.into_iter().next().expect("one output");
            cache.insert(item.node_id, arc);
            layout_cache.insert(item.node_id, item.output_layout.clone());
            Ok(handle)
        }
        WorkItemKind::InplaceKernel { target_idx } => {
            let compiled = item.compiled.as_ref().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: InplaceKernel work item {:?} has no compiled node",
                    item.node_id,
                ))
                .bt()
            })?;
            let target_id = *item.inputs.get(*target_idx).ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: InplaceKernel work item {:?} target_idx={} \
                     out of bounds (inputs.len()={})",
                    item.node_id, target_idx, item.inputs.len(),
                ))
                .bt()
            })?;
            let target_arc = cache.get(&target_id).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: InplaceKernel target {:?} of {:?} not realized",
                    target_id, item.node_id,
                ))
                .bt()
            })?;
            let target_layout = layout_cache.get(&target_id).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: InplaceKernel target {:?} of {:?} has no cached layout",
                    target_id, item.node_id,
                ))
                .bt()
            })?;
            // v1 contract: target must be contiguous + zero offset.
            // Strided in-place targets would force the kernel to walk
            // strides for writes too; defer until a concrete consumer
            // needs it (mirrors `Op::WriteSlice`'s v1 dest contract).
            if !target_layout.is_contiguous() || target_layout.start_offset() != 0 {
                return Err(Error::Msg(format!(
                    "InplaceKernel (Phase 3 v1): target {:?} must be contiguous + \
                     zero-offset; got Layout {:?}",
                    target_id, target_layout,
                ))
                .bt());
            }
            // Kernel sees inputs=[non-target IR inputs in order] and
            // outputs=[target_arc]. For unary in-place (1 IR input at
            // target_idx=0) the input vec is empty, matching the
            // original Phase 3e contract. For binary in-place
            // (target_idx=0, 2 IR inputs) the wrapper sees the
            // non-destructive RHS as `inputs[0]`. The wrapper acquires
            // outputs[0]'s write lock and mutates through its
            // bytes_mut(). Layout vector contains the kernel's
            // inputs+outputs layouts in that order.
            let mut input_arcs: Vec<Arc<RwLock<Storage>>> = Vec::new();
            let mut kernel_layouts: Vec<Layout> = Vec::new();
            for (i, &input_id) in item.inputs.iter().enumerate() {
                if i == *target_idx {
                    continue;
                }
                let arc = cache.get(&input_id).cloned().ok_or_else(|| {
                    Error::Msg(format!(
                        "PipelinedExecutor: InplaceKernel non-target input {:?} of {:?} \
                         not realized",
                        input_id, item.node_id,
                    ))
                    .bt()
                })?;
                let lay = layout_cache.get(&input_id).cloned().ok_or_else(|| {
                    Error::Msg(format!(
                        "PipelinedExecutor: InplaceKernel non-target input {:?} of {:?} \
                         has no cached layout",
                        input_id, item.node_id,
                    ))
                    .bt()
                })?;
                input_arcs.push(arc);
                kernel_layouts.push(lay);
            }
            let mut output_arcs = vec![target_arc.clone()];
            kernel_layouts.push(target_layout.clone());
            // A4b-1: defer the wait — return the handle to the realize loop.
            let handle =
                execute_compiled(compiled, &input_arcs, &mut output_arcs, &kernel_layouts)?;
            // Adopt the target Arc at this node's slot. The realize
            // loop's destructive_input cleanup evicts the target's own
            // NodeId from the cache afterward.
            cache.insert(item.node_id, target_arc);
            layout_cache.insert(item.node_id, item.output_layout.clone());
            Ok(handle)
        }
    }
}

fn poisoned(what: &'static str) -> Error {
    Error::Msg(format!("PipelinedExecutor: {} poisoned", what)).bt()
}

/// Step E A4b-1: record a node's [`CompletionHandle`] in the executor's
/// per-node handle map.
///
/// `Ready` handles (CPU / Vulkan / view-and-alloc arms) carry no async work, so
/// we DON'T store them — keeping the map to genuinely-pending entries makes the
/// realize-end empty-after-drain assert (open question #6) meaningful and keeps
/// the drain O(pending) rather than O(nodes). A re-used NodeId (in-place adopt
/// re-keys an existing slot) waits the prior handle first so nothing leaks.
#[inline]
fn store_handle(
    handles: &mut HashMap<NodeId, CompletionHandle>,
    node_id: NodeId,
    handle: CompletionHandle,
) {
    if matches!(handle, CompletionHandle::Ready) {
        // A prior pending handle at this slot must still be waited (don't drop a
        // Pending silently). In practice NodeIds are unique per realize, so this
        // is just defensive.
        if let Some(prev) = handles.remove(&node_id) {
            let _ = prev.wait();
        }
        return;
    }
    if let Some(prev) = handles.insert(node_id, handle) {
        let _ = prev.wait();
    }
}

/// Step E A4b-3: the FINER cross-device producer wait.
///
/// Before dispatching a cross-device `Op::Copy` / `Op::Move` WorkItem, the
/// executor calls this on the copy's single source input. It removes the
/// source producer's [`CompletionHandle`] from the map and waits it — so the
/// specific producer P is complete before the D2H reads P's device buffer,
/// WITHOUT draining the whole source device (the conservative pre-A4b-3
/// behavior, where `to_cpu_bytes`'s `device.synchronize()` blocked on ALL
/// in-flight work on that device — including the OTHER sub-DAG's independent
/// work A4b-4 wants to overlap).
///
/// - **CUDA producer** → waits P's recorded `Event` (`cuEventSynchronize`):
///   fires after P + its already-ordered stream predecessors, nothing else.
/// - **Absent / `Ready` producer** (CPU producer, or a Vulkan producer — which
///   under A4b-2 is `Ready` mid-walk because nothing is submitted until
///   realize-end) → no-op. A Vulkan SOURCE's own ordering is handled inside
///   `download_bytes` (it `force_flush`es the open batch then reads), so there
///   is no mid-walk Vulkan handle to wait here; the cross-device Vulkan→CPU hop
///   stays correct without one.
///
/// Removing the handle here (rather than waiting it again at realize-end) keeps
/// `drain_handles`' post-drain empty-map assert meaningful and avoids a
/// double-wait. Waiting it twice would be harmless (`cuEventSynchronize` on an
/// already-signalled event is a fast no-op), but the consume-once contract is
/// cleaner. The producing node's *storage* stays in the cache for downstream
/// readers; only its completion handle is consumed.
fn wait_producer_handle(
    handles: &mut HashMap<NodeId, CompletionHandle>,
    producer: NodeId,
) -> Result<()> {
    if let Some(handle) = handles.remove(&producer) {
        handle.wait()?;
    }
    Ok(())
}

/// Step E A4b-1: realize-end drain — wait every outstanding async handle, then
/// clear the map.
///
/// **Event-recording / wait strategy (A4b-1 §1.2, open question #4).** CUDA has
/// one stream per device, and an `Event` recorded after a node's launch signals
/// when that node *and all prior stream work* have completed. So once the
/// latest-recorded event on a device has signalled, every earlier event on that
/// device is already signalled too: waiting them is a non-blocking
/// `cuEventQuery` fast-path, not N host stalls. We therefore wait every stored
/// handle (correct + simple); the only per-node cost is the `cuEventRecord` at
/// production time (a lightweight stream marker — measured negligible on the
/// long_chain 32-op stress, see the PR notes). We keep a per-node handle (rather
/// than a single per-device event) because A4b-3's finer cross-device wait needs
/// to wait a *specific producer's* completion — the per-node handle is that seam.
fn drain_handles(handles: &mut HashMap<NodeId, CompletionHandle>) -> Result<()> {
    for (_node, handle) in handles.drain() {
        handle.wait()?;
    }
    // Open question #6: once handles are stored instead of `wait`ed inline, the
    // `#[must_use]` on `CompletionHandle` can't catch a leaked (never-waited)
    // handle at the call site. The drain consumes the whole map, so it is empty
    // here by construction; assert it to catch a future leak (e.g. a new
    // dispatch path that forgets to thread its handle through `store_handle`).
    debug_assert!(handles.is_empty(), "A4b-1: handle map must be empty after drain");
    Ok(())
}

/// Search the input cache for any `BackendStorage::Cuda(s)` whose
/// device matches `gpu_id`. Returns a clone of the `CudaDevice` Arc
/// the storage carries — the caller uses it to call
/// `CudaStorageBytes::alloc` for `Op::Alloc` on Cuda. Returns `None`
/// if no matching storage exists.
///
/// Phase 3a of bridge-retirement: callers seed the cache with a 0-byte
/// CUDA storage (`fuel-core::pipelined_bridge::device_seed_storage`)
/// so this lookup succeeds for the first Op::Alloc; subsequent
/// Op::Allocs see prior Op::Alloc outputs in the cache.
#[cfg(feature = "cuda")]
fn find_cuda_device_in_cache(
    cache: &StorageCache,
    gpu_id: usize,
) -> Option<fuel_cuda_backend::CudaDevice> {
    for arc in cache.values() {
        let guard = arc.read().ok()?;
        if let fuel_memory::BackendStorage::Cuda(c) = &guard.inner {
            // CudaDevice carries its DeviceLocation (gpu_id ordinal)
            // through `location()`; match against the target gpu_id.
            // Multi-GPU future work can refine the match; today's
            // single-GPU setups always match.
            if matches!(c.device().location(), DeviceLocation::Cuda { gpu_id: g } if g == gpu_id) {
                return Some(c.device().clone());
            }
        }
    }
    None
}

/// Step E A4b-2: a Vulkan [`Completion`](crate::compiled::Completion) over an
/// in-flight (submitted, not-yet-waited) batch. `wait` blocks on the batch's
/// fence (`vkWaitForFences`) — which fires when the whole submitted command
/// buffer has retired on the single compute queue — then releases the batch
/// (frees the CB / descriptor sets / transient buffers / retired pool, all now
/// idle on the GPU).
///
/// The fence is per-BATCH, not per-node: one `submit_pending` flushes every op
/// recorded since the last submit, so this single handle covers a contiguous run
/// of Vulkan nodes (the A2 batching win, preserved — fewer fences, larger
/// submissions). `backend` keeps the [`VulkanBackend`] alive so the
/// post-fence pool retire + the `DeviceInner` the batch's resources reference
/// outlive the wait.
///
/// `Send` (required by `Completion`): every owned field is `Send` — `Fence` /
/// `CommandBuffer` / `DescriptorSet` / `Buffer` / `Allocation` / `CommandPool`
/// are all `Send` in vulkane (Arc-of-`DeviceInner` + explicit `unsafe impl`s),
/// and `Arc<VulkanBackend>` is `Send + Sync`.
#[cfg(feature = "vulkan")]
struct VulkanCompletion {
    backend: Arc<fuel_vulkan_backend::VulkanBackend>,
    batch: fuel_vulkan_backend::SubmittedBatch,
}

/// Step E A4b-4: the element type of the executor's in-flight-Vulkan-batch list.
/// A [`VulkanCompletion`] on a Vulkan build; a zero-sized placeholder otherwise
/// so the realize loops carry one `Vec<InflightVulkan>` local without `cfg` on
/// every use site (the helpers `eager_submit_all_vulkan` / `drain_inflight_vulkan`
/// are the only code that ever pushes/drains it, and they are `cfg`-gated).
#[cfg(feature = "vulkan")]
type InflightVulkan = VulkanCompletion;
#[cfg(not(feature = "vulkan"))]
type InflightVulkan = ();

/// Step E A4b-4: is the cross-device copy's SOURCE buffer Vulkan-resident?
///
/// Used at the cross-device `Op::Copy`/`Op::Move` boundary to decide whether the
/// executor must wait the in-flight Vulkan batches before the copy. A Vulkan→CPU
/// D2H reads the source on the host, and the Vulkan `download` path only
/// `force_flush`es the OPEN batch (never the executor's eagerly-submitted
/// in-flight ones), so for a VULKAN source we must `drain_inflight_vulkan` first
/// to read completed data. For a CUDA / CPU source we must NOT wait Vulkan — the
/// copy doesn't read any Vulkan buffer, and waiting Vulkan there would needlessly
/// serialize the independent Vulkan sub-DAG (killing the overlap A4b-4 exists to
/// win). Returns `false` when the producer's storage isn't in the cache (treated
/// as not-Vulkan → no Vulkan wait; the CUDA/CPU producer handle still covers it).
#[cfg(feature = "vulkan")]
fn copy_source_is_vulkan(cache: &StorageCache, producer: NodeId) -> Result<bool> {
    if let Some(arc) = cache.get(&producer) {
        let guard = arc
            .read()
            .map_err(|_| poisoned("storage lock probing copy-source backend"))?;
        return Ok(matches!(guard.inner, fuel_memory::BackendStorage::Vulkan(_)));
    }
    Ok(false)
}

#[cfg(not(feature = "vulkan"))]
#[inline]
fn copy_source_is_vulkan(_cache: &StorageCache, _producer: NodeId) -> Result<bool> {
    Ok(false)
}

#[cfg(feature = "vulkan")]
impl crate::compiled::Completion for VulkanCompletion {
    fn wait(self: Box<Self>) -> Result<()> {
        self.into_wait()
    }
}

#[cfg(feature = "vulkan")]
impl VulkanCompletion {
    /// Block on this batch's fence, retire pools, then free the batch. Consumes
    /// `self` by value (the by-value sibling of the boxed `Completion::wait`),
    /// used by the executor's `inflight_vulkan` drain (A4b-4) where we hold the
    /// `VulkanCompletion` unboxed in a `Vec`.
    fn into_wait(self) -> Result<()> {
        // `wait_submitted` consumes the batch by value: waits the fence, retires
        // pools, then drops the batch (UAF-safe — nothing the in-flight CB
        // references frees before the fence signals).
        let VulkanCompletion { backend, batch } = self;
        backend.wait_submitted(batch)
    }
}

/// Step E A4b-4 — eager Vulkan submission during the walk (the OVERLAP enabler).
///
/// Submits each DISTINCT Vulkan backend's currently-open batch WITHOUT waiting,
/// pushing the resulting in-flight [`VulkanCompletion`]s onto `inflight` so the
/// executor waits them later (at the in-flight-lifetime guard / realize-end). An
/// empty open batch yields `None` and is skipped (idempotent).
///
/// This is what makes a Vulkan sub-DAG actually START on the iGPU while the
/// executor records/dispatches the next (CUDA) chunk — without it the Vulkan
/// batch sits merely *recorded* until realize-end and never overlaps the CUDA
/// stream (design §5). Called ONLY on a genuinely multi-backend realize (gated by
/// the executor's `multi_backend` flag); on a single-device graph it is never
/// reached, so pure-Vulkan stays byte-identical to A4b-2 (one realize-end submit).
///
/// **Whole-chunk granularity (UAF/race-safety, design §4 + the Vulkan
/// single-queue OOO-completion semantics).** We submit only at chunk boundaries
/// and before a cross-device wait — never mid-chunk — so the batch handed to the
/// GPU is a COMPLETE contiguous run of Vulkan nodes with all its intra-CB
/// dependency barriers (`recorder.rs` `vkCmdPipelineBarrier`) already in place.
/// Vulkan only guarantees that batches on one queue BEGIN in submission order;
/// they may overlap / complete out of order, so a split mid-chunk would lose the
/// write-before-read ordering for ops that read an earlier op's buffer in the
/// SAME chunk. By submitting whole chunks and re-syncing (`drain_inflight_vulkan`)
/// before any later Vulkan op records, before any cross-device copy, and before
/// any Vulkan-referenced eviction, no in-flight batch is ever read or freed while
/// still running.
#[cfg(feature = "vulkan")]
fn eager_submit_all_vulkan(
    cache: &StorageCache,
    inflight: &mut Vec<VulkanCompletion>,
) -> Result<()> {
    // Collect distinct Vulkan backends referenced by the cache (dedup by gpu_id).
    let mut seen: Vec<usize> = Vec::new();
    let mut backends: Vec<Arc<fuel_vulkan_backend::VulkanBackend>> = Vec::new();
    for arc in cache.values() {
        let guard = arc
            .read()
            .map_err(|_| poisoned("storage lock during eager vulkan submit"))?;
        if let fuel_memory::BackendStorage::Vulkan(v) = &guard.inner {
            if let Some(backend) = v.backend() {
                if !seen.contains(&backend.gpu_id) {
                    seen.push(backend.gpu_id);
                    backends.push(backend.clone());
                }
            }
        }
    }
    for backend in backends {
        if let Some(batch) = backend.submit_pending()? {
            inflight.push(VulkanCompletion { backend, batch });
        }
    }
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
#[inline]
fn eager_submit_all_vulkan(_cache: &StorageCache, _inflight: &mut Vec<InflightVulkan>) -> Result<()> {
    Ok(())
}

/// Step E A4b-4 — wait every in-flight (eagerly-submitted) Vulkan batch, then
/// clear the list. The re-sync half of [`eager_submit_all_vulkan`].
///
/// Each `into_wait` blocks on the batch fence (`vkWaitForFences`), retires the
/// backend's descriptor pools, and frees the batch (CB / descriptor sets /
/// transient buffers / retired command pool — all now idle on the GPU). After
/// this returns, NO submitted Vulkan command buffer is still executing, so it is
/// safe to (a) record a new Vulkan op that may read a just-produced buffer,
/// (b) D2H-copy a Vulkan buffer on the host, or (c) free a buffer a batch read.
///
/// Because the eager submit happened at the PRIOR chunk boundary and an
/// intervening (e.g. CUDA) chunk has since been recorded/dispatched, the fence is
/// typically already signalled here — the wait is a fast no-op, and the iGPU work
/// genuinely overlapped the other device's stream. Waiting the Vulkan fence does
/// NOT touch the CUDA stream, so the other device keeps running (concurrency
/// preserved — design §5 "the per-device guards don't cross-stall").
#[cfg(feature = "vulkan")]
fn drain_inflight_vulkan(inflight: &mut Vec<VulkanCompletion>) -> Result<()> {
    for vc in inflight.drain(..) {
        vc.into_wait()?;
    }
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
#[inline]
fn drain_inflight_vulkan(_inflight: &mut Vec<InflightVulkan>) -> Result<()> {
    Ok(())
}

/// Step E A4b-2: the realize-end Vulkan drain, SPLIT into submit-then-wait.
///
/// Replaces A2's `force_flush_all_vulkan` (which submitted + waited atomically
/// inside `force_flush`). For each DISTINCT Vulkan backend referenced by the
/// cache, `submit_pending` submits the still-open batch WITHOUT waiting (→ a
/// `SubmittedBatch`), wraps it in the executor's [`VulkanCompletion`] handle, and
/// `wait`s it. Net for a pure-Vulkan realize: the batch still accumulates across
/// the whole walk (the per-op path keeps returning `Ready`; no eager submit —
/// that is A4b-4) and is submitted ONCE here, then waited — **byte-identical to
/// A2**, same single submission at realize-end, just split through the handle.
///
/// Idempotent across repeated cache arcs: the first `submit_pending` on a backend
/// returns `Some(batch)`; subsequent calls on the same backend see an empty batch
/// and return `None` (skipped) — exactly the `force_flush` idempotence A2 relied
/// on. No-op for a non-Vulkan build.
#[cfg(feature = "vulkan")]
fn drain_vulkan_pending(cache: &StorageCache) -> Result<()> {
    use crate::compiled::CompletionHandle;
    for arc in cache.values() {
        let backend = {
            let guard = arc
                .read()
                .map_err(|_| poisoned("storage lock during vulkan submit_pending"))?;
            match &guard.inner {
                fuel_memory::BackendStorage::Vulkan(v) => v.backend().cloned(),
                _ => None,
            }
        };
        let Some(backend) = backend else { continue };
        // Submit the open batch (if any) without waiting, then wait via the
        // completion handle. Empty batch → None → nothing to wait.
        if let Some(batch) = backend.submit_pending()? {
            let handle: CompletionHandle =
                CompletionHandle::Pending(Box::new(VulkanCompletion { backend, batch }));
            handle.wait()?;
        }
    }
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn drain_vulkan_pending(_cache: &StorageCache) -> Result<()> {
    Ok(())
}

/// Step E A2: force the Vulkan backend behind `arc` (if any) to submit + wait
/// its deferred batch — called before freeing a buffer the batch may reference
/// (a destructive eviction) so a recorded-but-unsubmitted command never reads
/// freed memory. No-op for non-Vulkan storage, an empty batch, or a non-Vulkan
/// build.
///
/// A4b-2/A4b-3 KEEP this as the blocking submit+wait. Design §3 row 4 wants
/// `wait H(evicted)`, but that is not coherent until A4b-4's eager submit
/// exists: under A4b-2 the Vulkan per-op path only RECORDS into the batch and
/// returns `Ready` during the walk (no eager submit), so mid-walk a Vulkan
/// buffer has NO completion handle to wait — its in-flight command is still
/// unsubmitted. `force_flush` (submit + wait) is the only coherent drain before
/// freeing such a buffer. The conversion to a handle wait rides A4b-4. Only the
/// realize-end drain ([`drain_vulkan_pending`]) is split (A4b-2).
#[cfg(feature = "vulkan")]
fn force_flush_vulkan(arc: &Arc<RwLock<Storage>>) -> Result<()> {
    let guard = arc
        .read()
        .map_err(|_| poisoned("storage lock during vulkan force_flush"))?;
    if let fuel_memory::BackendStorage::Vulkan(v) = &guard.inner {
        if let Some(backend) = v.backend() {
            backend.force_flush()?;
        }
    }
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn force_flush_vulkan(_arc: &Arc<RwLock<Storage>>) -> Result<()> {
    Ok(())
}

/// Vulkan counterpart of [`find_cuda_device_in_cache`]. Returns an
/// `Arc<VulkanBackend>` clone derived from any cached Vulkan storage
/// whose backend's `gpu_id` matches.
#[cfg(feature = "vulkan")]
fn find_vulkan_backend_in_cache(
    cache: &StorageCache,
    gpu_id: usize,
) -> Option<std::sync::Arc<fuel_vulkan_backend::VulkanBackend>> {
    for arc in cache.values() {
        let guard = arc.read().ok()?;
        if let fuel_memory::BackendStorage::Vulkan(v) = &guard.inner {
            if let Some(backend) = v.backend() {
                if backend.gpu_id == gpu_id {
                    return Some(backend.clone());
                }
            }
        }
    }
    None
}

/// Materialize a contiguous Storage Arc from a non-contiguous one.
/// Allocates a fresh buffer on the input's backend and copies the
/// strided / offset / broadcast input into it via the backend's
/// contiguize kernel. The returned Arc is a brand-new buffer; the
/// caller is responsible for replacing the cache entry only for the
/// duration of one kernel call (the upstream view op's output stays
/// in the cache so other consumers still see the strided view).
///
/// Stage 4 of Layout-on-Node — auto-Contiguize.
fn auto_contiguize(
    arc: &Arc<RwLock<Storage>>,
    layout: &Layout,
) -> Result<Arc<RwLock<Storage>>> {
    let in_guard = arc
        .read()
        .map_err(|_| poisoned("input storage lock during auto_contiguize"))?;
    let dtype = in_guard.dtype;
    let dtype_size = dtype.size_in_bytes();
    let new_storage = match &in_guard.inner {
        fuel_memory::BackendStorage::Cpu(c) => {
            let new_bytes = fuel_cpu_backend::byte_kernels::contiguize_cpu(c, layout, dtype_size)?;
            Storage::new(fuel_memory::BackendStorage::Cpu(new_bytes), dtype)
        }
        #[cfg(feature = "cuda")]
        fuel_memory::BackendStorage::Cuda(c) => {
            // Native baracuda contiguize (alpha.29). Byte-width-
            // dispatched: 1/2/4/8/16 byte elements all route to the
            // appropriate kernel. Handles signed strides (Flip),
            // zero strides (BroadcastTo), and non-zero element offset.
            //
            // Three host-side fast paths bake into the baracuda
            // launchers: already-contiguous → single cuMemcpyDtoDAsync;
            // innermost-stride-1 → per-outer-coord run copy; generic
            // → one thread per output element. Retires the prior
            // D2H → CPU contiguize_cpu → H2D fallback (two device
            // round-trips per non-contig input).
            let contig =
                fuel_cuda_backend::baracuda::contiguize::contiguize_to_fresh(
                    c, layout, dtype_size,
                )?;
            Storage::new(fuel_memory::BackendStorage::Cuda(contig), dtype)
        }
        #[cfg(feature = "vulkan")]
        fuel_memory::BackendStorage::Vulkan(v) => {
            // V.1.B stopgap: D2H → CPU contiguize_cpu → H2D. Mirrors
            // the pre-alpha.29 CUDA fallback. V.3.2 of the Vulkan
            // catch-up writes a native Slang contiguize kernel (with
            // signed-stride support that strided_copy.slang currently
            // lacks) and replaces this arm with a direct
            // VulkanStorageBytes → VulkanStorageBytes path.
            let backend = v.backend().ok_or_else(|| {
                Error::Msg(
                    "auto_contiguize: Vulkan input has no backend handle. \
                     Storages flowing through the pipelined executor's Vulkan \
                     path must be constructed via VulkanBackend::alloc_bytes_handle \
                     / upload_bytes_handle (not the legacy alloc_bytes / \
                     upload_bytes which leave VulkanStorageBytes::backend = None)."
                        .to_string(),
                )
                .bt()
            })?;
            let host_bytes = backend.download_bytes(v)?;
            let host_cpu =
                fuel_cpu_backend::byte_storage::CpuStorageBytes::from_bytes(&host_bytes);
            let contig_cpu = fuel_cpu_backend::byte_kernels::contiguize_cpu(
                &host_cpu, layout, dtype_size,
            )?;
            let host_contig_bytes = contig_cpu.bytes().to_vec();
            let vk_bytes = backend.upload_bytes_handle(&host_contig_bytes)?;
            Storage::new(fuel_memory::BackendStorage::Vulkan(vk_bytes), dtype)
        }
        #[allow(unreachable_patterns)]
        _ => {
            return Err(Error::Msg(
                "auto_contiguize: backend not wired (CPU + CUDA + Vulkan are \
                 wired today; Metal extends this match when its byte-storage \
                 substrate lands)"
                    .to_string(),
            )
            .bt());
        }
    };
    Ok(Arc::new(RwLock::new(new_storage)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::Shape;
    use fuel_graph::Node;

    /// A4b-1 test double: a `Completion` whose `wait` flips a shared flag, so a
    /// test can observe whether the drain actually waited it.
    struct FlagCompletion(std::sync::Arc<std::sync::atomic::AtomicUsize>);
    impl crate::compiled::Completion for FlagCompletion {
        fn wait(self: Box<Self>) -> Result<()> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    /// A4b-1: `store_handle` keeps only `Pending` handles (so the post-drain
    /// empty-map assert is meaningful and the drain is O(pending)), and
    /// `drain_handles` waits each pending handle exactly once then empties the map.
    #[test]
    fn handle_map_stores_pending_and_drains_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        let mut handles: HashMap<NodeId, CompletionHandle> = HashMap::new();

        // Ready handles are not stored.
        store_handle(&mut handles, NodeId(1), CompletionHandle::Ready);
        assert!(handles.is_empty(), "Ready handles must not be stored");

        // Two pending handles are stored and each waited once on drain.
        let waited = StdArc::new(AtomicUsize::new(0));
        store_handle(
            &mut handles,
            NodeId(2),
            CompletionHandle::Pending(Box::new(FlagCompletion(waited.clone()))),
        );
        store_handle(
            &mut handles,
            NodeId(3),
            CompletionHandle::Pending(Box::new(FlagCompletion(waited.clone()))),
        );
        assert_eq!(handles.len(), 2);

        drain_handles(&mut handles).expect("drain");
        assert_eq!(waited.load(Ordering::SeqCst), 2, "each pending handle waited once");
        assert!(handles.is_empty(), "map empty after drain");
    }

    /// A4b-1: re-keying a NodeId with a new Pending handle waits the prior one
    /// (no leaked async work), and storing `Ready` over a prior Pending also
    /// waits it. Defensive — NodeIds are unique per realize in practice.
    #[test]
    fn handle_map_rekey_waits_previous() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        let mut handles: HashMap<NodeId, CompletionHandle> = HashMap::new();
        let waited = StdArc::new(AtomicUsize::new(0));

        store_handle(
            &mut handles,
            NodeId(7),
            CompletionHandle::Pending(Box::new(FlagCompletion(waited.clone()))),
        );
        // Re-key with another Pending → prior is waited.
        store_handle(
            &mut handles,
            NodeId(7),
            CompletionHandle::Pending(Box::new(FlagCompletion(waited.clone()))),
        );
        assert_eq!(waited.load(Ordering::SeqCst), 1, "rekey waited the prior pending");

        // Store Ready over the live Pending → it is waited too.
        store_handle(&mut handles, NodeId(7), CompletionHandle::Ready);
        assert_eq!(waited.load(Ordering::SeqCst), 2, "Ready-over-Pending waited it");
        assert!(handles.is_empty());

        drain_handles(&mut handles).expect("drain empty");
    }

    /// A4b-3: `wait_producer_handle` waits the named producer's handle EXACTLY
    /// once and removes it from the map (the finer cross-device source-drain
    /// consumes the producer's completion before the Copy reads its buffer), and
    /// is a no-op for an absent producer (CPU / Vulkan-`Ready` source, or an
    /// already-consumed handle). The realize-end drain then waits only what
    /// remains — no double-wait.
    #[test]
    fn wait_producer_handle_consumes_named_handle_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        let mut handles: HashMap<NodeId, CompletionHandle> = HashMap::new();
        let waited = StdArc::new(AtomicUsize::new(0));

        // Two producers in flight; the Copy's source is NodeId(2).
        store_handle(
            &mut handles,
            NodeId(2),
            CompletionHandle::Pending(Box::new(FlagCompletion(waited.clone()))),
        );
        store_handle(
            &mut handles,
            NodeId(5),
            CompletionHandle::Pending(Box::new(FlagCompletion(waited.clone()))),
        );

        // Absent producer → no-op (a CPU/Vulkan-Ready source has no handle).
        wait_producer_handle(&mut handles, NodeId(99)).expect("absent is no-op");
        assert_eq!(waited.load(Ordering::SeqCst), 0, "absent producer waited nothing");
        assert_eq!(handles.len(), 2);

        // The cross-device copy's source: waited once, removed from the map.
        wait_producer_handle(&mut handles, NodeId(2)).expect("wait source");
        assert_eq!(waited.load(Ordering::SeqCst), 1, "source producer waited once");
        assert!(!handles.contains_key(&NodeId(2)), "source handle consumed");
        assert!(handles.contains_key(&NodeId(5)), "unrelated handle untouched");

        // Re-waiting the (now absent) source is a no-op — no double-wait.
        wait_producer_handle(&mut handles, NodeId(2)).expect("re-wait is no-op");
        assert_eq!(waited.load(Ordering::SeqCst), 1, "no double-wait");

        // Realize-end drain waits only the remaining unrelated producer.
        drain_handles(&mut handles).expect("drain");
        assert_eq!(waited.load(Ordering::SeqCst), 2, "drain waited the remaining one");
        assert!(handles.is_empty());
    }

    /// Op::Contiguize on a contiguous-already input is zero-copy:
    /// the executor adopts the input Storage Arc unchanged.
    #[test]
    fn op_contiguize_zero_copy_on_contiguous_input() {
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (const_id, contig_id) = {
            let mut g = graph.write().unwrap();
            let const_id = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2, 2]),
                dtype: DType::F32,
            });
            let contig_id = g.push(Node {
                op: Op::Contiguize,
                inputs: vec![const_id],
                shape: Shape::from_dims(&[2, 2]),
                dtype: DType::F32,
            });
            // No need to set target_backend on Contiguize — the
            // executor inherits or defaults to CPU.
            (const_id, contig_id)
        };

        let mut inputs = StorageCache::new();
        let input_arc = Arc::new(RwLock::new(storage));
        let input_arc_for_compare = Arc::clone(&input_arc);
        inputs.insert(const_id, input_arc);

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, contig_id, inputs).expect("realize");

        // Same Arc → zero-copy adoption. Compare by raw pointer.
        assert!(
            Arc::ptr_eq(&result_arc, &input_arc_for_compare),
            "Op::Contiguize on contiguous input must adopt the input Arc (zero copy)",
        );
    }

    /// Op::Contiguize on a strided input (here: a transposed view)
    /// materializes a fresh contiguous Storage. The output bytes
    /// reflect the strided view's logical content, not the input
    /// storage's underlying memory order.
    #[test]
    fn op_contiguize_materializes_on_strided_input() {
        // 2x3 row-major: [[1,2,3],[4,5,6]] → transpose → 3x2:
        // [[1,4],[2,5],[3,6]]. The transposed view shares the
        // original buffer; Op::Contiguize on the transpose
        // materializes contiguous bytes matching the transposed
        // logical order.
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (const_id, _transpose_id, contig_id) = {
            let mut g = graph.write().unwrap();
            let const_id = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2, 3]),
                dtype: DType::F32,
            });
            let transpose_id = g.push(Node {
                op: Op::Transpose,
                inputs: vec![const_id],
                shape: Shape::from_dims(&[3, 2]),
                dtype: DType::F32,
            });
            let contig_id = g.push(Node {
                op: Op::Contiguize,
                inputs: vec![transpose_id],
                shape: Shape::from_dims(&[3, 2]),
                dtype: DType::F32,
            });
            (const_id, transpose_id, contig_id)
        };

        let mut inputs = StorageCache::new();
        inputs.insert(const_id, Arc::new(RwLock::new(storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, contig_id, inputs).expect("realize");

        let guard = result_arc.read().unwrap();
        let cpu = if let fuel_memory::BackendStorage::Cpu(c) = &guard.inner {
            c
        } else {
            panic!("expected Cpu storage");
        };
        let typed: &[f32] = cpu.as_slice().expect("f32 cast");
        // Transposed logical order: [1,4,2,5,3,6].
        assert_eq!(
            typed,
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            "contiguized transpose must reflect logical (transposed) byte order",
        );
    }

    /// E2E: 3-node graph (Const + Const + Add), pre-seeded inputs,
    /// realized through the compiler+executor thread pair, returns
    /// expected sum bytes.
    #[test]
    fn pipelined_realize_const_const_add() {
        let lhs_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
        let rhs_storage = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (lhs_id, rhs_id, add_id) = {
            let mut g = graph.write().unwrap();
            let lhs_id = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3]),
                dtype: DType::F32,
            });
            let rhs_id = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3]),
                dtype: DType::F32,
            });
            let add_id = g.push(Node {
                op: Op::Add,
                inputs: vec![lhs_id, rhs_id],
                shape: Shape::from_dims(&[3]),
                dtype: DType::F32,
            });
            g.set_target_backend(add_id, BackendId::Cpu);
            (lhs_id, rhs_id, add_id)
        };

        let mut inputs = StorageCache::new();
        inputs.insert(lhs_id, Arc::new(RwLock::new(lhs_storage)));
        inputs.insert(rhs_id, Arc::new(RwLock::new(rhs_storage)));

        let (result_arc, _result_layout) =
            PipelinedExecutor::realize(graph, add_id, inputs).expect("realize");

        let guard = result_arc.read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().expect("f32 cast");
            assert_eq!(typed, &[11.0, 22.0, 33.0]);
        } else {
            panic!("expected CPU output");
        }
    }

    /// PR-C1 behavior contract: `realize_with_optimized_route` with an
    /// **empty** route (the no-pressure / no-telemetry case) is
    /// byte-identical to `realize_with_optimized` (arm-0). On a branchless
    /// graph the route has no branches to resolve, so both must produce
    /// the same bytes — realize is unchanged from Phase B.
    #[test]
    fn realize_with_optimized_route_empty_equals_arm0() {
        use crate::optimize::OptimizedGraph;

        let build = || {
            let graph = Arc::new(RwLock::new(Graph::new()));
            let (lhs_id, rhs_id, add_id) = {
                let mut g = graph.write().unwrap();
                let lhs_id = g.push(Node {
                    op: Op::Const,
                    inputs: vec![],
                    shape: Shape::from_dims(&[3]),
                    dtype: DType::F32,
                });
                let rhs_id = g.push(Node {
                    op: Op::Const,
                    inputs: vec![],
                    shape: Shape::from_dims(&[3]),
                    dtype: DType::F32,
                });
                let add_id = g.push(Node {
                    op: Op::Add,
                    inputs: vec![lhs_id, rhs_id],
                    shape: Shape::from_dims(&[3]),
                    dtype: DType::F32,
                });
                g.set_target_backend(add_id, BackendId::Cpu);
                (lhs_id, rhs_id, add_id)
            };
            let mut inputs = StorageCache::new();
            inputs.insert(
                lhs_id,
                Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0]))),
            );
            inputs.insert(
                rhs_id,
                Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0]))),
            );
            (graph, add_id, inputs)
        };

        let read_bytes = |arc: &Arc<RwLock<Storage>>| -> Vec<f32> {
            let guard = arc.read().unwrap();
            match &guard.inner {
                fuel_memory::BackendStorage::Cpu(c) => {
                    c.as_slice::<f32>().expect("f32 cast").to_vec()
                }
                _ => panic!("expected CPU output"),
            }
        };

        // Arm-0 path.
        let (g_a, add_a, in_a) = build();
        let opt_a = OptimizedGraph { roots: vec![add_a], generation: 0 };
        let (arm0_arc, _) =
            PipelinedExecutor::realize_with_optimized(g_a, add_a, in_a, &opt_a)
                .expect("arm-0 realize");
        let arm0 = read_bytes(&arm0_arc);

        // Empty-route path — must match arm-0 exactly.
        let (g_r, add_r, in_r) = build();
        let opt_r = OptimizedGraph { roots: vec![add_r], generation: 0 };
        let empty_route = PickedRoute::new();
        let (route_arc, _) = PipelinedExecutor::realize_with_optimized_route(
            g_r, add_r, in_r, &opt_r, &empty_route,
        )
        .expect("empty-route realize");
        let routed = read_bytes(&route_arc);

        assert_eq!(arm0, vec![11.0, 22.0, 33.0]);
        assert_eq!(
            routed, arm0,
            "an empty route realizes byte-identically to arm-0 (Phase B contract)",
        );
    }

    /// Cleanup Step C: the executor owns the arm-pick. `pick_route_for` (the
    /// relocated `resolve_runtime_route` body) must (1) forward to `pick_route`
    /// when a selector + a branch are present, (2) return `None` with no
    /// selector (the `runtime_selector_disabled` opt-out), and (3) short-circuit
    /// to `None` on a branchless graph (the fast-path) — each matching the old
    /// bridge behavior. `pick_route` itself is covered by the route_picker tests.
    #[test]
    fn pick_route_for_gates_and_forwards() {
        use crate::ranker::{AlternativeSet, Candidate};

        // A selector that always takes the LAST arm (arm 1), so a non-empty
        // route is observable (WinnerSelector picks arm-0 ⇒ empty route).
        #[derive(Debug)]
        struct PickLast;
        impl RuntimeSelector for PickLast {
            fn select<'a>(&self, set: &'a AlternativeSet) -> Option<&'a Candidate> {
                set.alternatives().last()
            }
        }

        let node = |g: &mut Graph, op: Op, inputs: Vec<NodeId>| {
            g.push(Node { op, inputs, shape: Shape::from_dims(&[2]), dtype: DType::F32 })
        };

        // 2-arm diamond (mirrors ranker::route_picker::tests::diamond): the
        // arm exits carry their `target_backend` so the picker reads each
        // arm's placement. Candidates are re-enumerated from the global
        // registry (Step D); PickLast selects the last arm regardless.
        let mut g = Graph::new();
        let pre = node(&mut g, Op::Const, vec![]);
        let diverge = node(&mut g, Op::Relu, vec![pre]);
        let arm0 = node(&mut g, Op::Silu, vec![diverge]);
        let arm1 = node(&mut g, Op::Gelu, vec![diverge]);
        g.set_target_backend(arm0, BackendId::Cuda);
        g.set_target_backend(arm1, BackendId::Cpu);
        let reconverge = node(&mut g, Op::Relu, vec![arm0]);
        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        let branch = b
            .finalize_branches(&mut g, reconverge)
            .expect("well-formed 2-arm branch")
            .expect("2 arms survive");
        let post = node(&mut g, Op::Tanh, vec![reconverge]);
        let graph = Arc::new(RwLock::new(g));

        // (1) selector + branch ⇒ Some(route); PickLast ⇒ arm 1 at the branch,
        //     and the route is byte-for-byte what `pick_route` produces directly
        //     (both re-enumerate from the same global registry).
        let route = PipelinedExecutor::pick_route_for(
            &graph, &[post], Some(&PickLast), None,
        )
        .expect("pick_route_for ok")
        .expect("a branched graph + selector yields a route");
        assert_eq!(
            route.get(&branch).copied(),
            Some(1),
            "PickLast selects arm 1 at the branch; route={route:?}",
        );
        let direct = {
            let g = graph.read().unwrap();
            pick_route(&g, &[post], &global_bindings(), &PickLast, None)
        };
        assert_eq!(route, direct, "the executor's pick forwards pick_route verbatim");

        // (2) no selector (the runtime-selector-disabled opt-out) ⇒ None.
        assert!(
            PipelinedExecutor::pick_route_for(&graph, &[post], None, None)
                .expect("ok")
                .is_none(),
            "no selector ⇒ no route (arm-0 lowering)",
        );

        // (3) branchless graph + selector ⇒ None (the fast-path).
        let (bl_graph, bl_root) = {
            let mut g = Graph::new();
            let c1 = node(&mut g, Op::Const, vec![]);
            let c2 = node(&mut g, Op::Const, vec![]);
            let add = node(&mut g, Op::Add, vec![c1, c2]);
            g.set_target_backend(add, BackendId::Cpu);
            (Arc::new(RwLock::new(g)), add)
        };
        assert!(
            PipelinedExecutor::pick_route_for(
                &bl_graph, &[bl_root], Some(&PickLast), None,
            )
            .expect("ok")
            .is_none(),
            "branchless graph ⇒ no branches ⇒ no route (fast-path)",
        );
    }

    /// Realizing a node whose target_backend isn't set surfaces a
    /// typed error (no panic).
    #[test]
    fn pipelined_errors_on_unset_target_backend() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (lhs_id, rhs_id, add_id) = {
            let mut g = graph.write().unwrap();
            let lhs = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2]),
                dtype: DType::F32,
            });
            let rhs = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2]),
                dtype: DType::F32,
            });
            // Deliberately do NOT call set_target_backend on the Add.
            let add = g.push(Node {
                op: Op::Add,
                inputs: vec![lhs, rhs],
                shape: Shape::from_dims(&[2]),
                dtype: DType::F32,
            });
            (lhs, rhs, add)
        };

        let mut inputs = StorageCache::new();
        inputs.insert(
            lhs_id,
            Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[1.0_f32, 2.0]))),
        );
        inputs.insert(
            rhs_id,
            Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[3.0_f32, 4.0]))),
        );

        let result = PipelinedExecutor::realize(graph, add_id, inputs);
        assert!(result.is_err(), "missing target_backend must error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("target_backend"),
            "error names the unmet precondition: {msg}"
        );
    }

    /// Realizing a Const-only graph adopts the pre-seeded input
    /// without calling any kernel.
    #[test]
    fn pipelined_realize_const_only() {
        let storage = fuel_memory::from_slice_cpu(&[5.0_f32, 6.0, 7.0]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let const_id = {
            let mut g = graph.write().unwrap();
            g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3]),
                dtype: DType::F32,
            })
        };

        let mut inputs = StorageCache::new();
        inputs.insert(const_id, Arc::new(RwLock::new(storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, const_id, inputs).expect("realize const");
        let guard = result_arc.read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            assert_eq!(typed, &[5.0, 6.0, 7.0]);
        }
    }

    /// Missing input-cache entry for a Const surfaces a typed error.
    #[test]
    fn pipelined_errors_on_missing_const_input() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let const_id = {
            let mut g = graph.write().unwrap();
            g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2]),
                dtype: DType::F32,
            })
        };
        // Deliberately empty input cache.
        let inputs = StorageCache::new();
        let result = PipelinedExecutor::realize(graph, const_id, inputs);
        assert!(result.is_err());
    }

    /// E2E: 2-node graph (Const + Relu) — exercises the unary
    /// dispatch wrapper + kernel through the pipelined executor.
    #[test]
    fn pipelined_realize_const_relu() {
        let storage = fuel_memory::from_slice_cpu(&[-1.0_f32, 0.0, 0.5, -3.5, 7.25]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, relu_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[5]),
                dtype: DType::F32,
            });
            let relu_id = g.push(Node {
                op: Op::Relu,
                inputs: vec![in_id],
                shape: Shape::from_dims(&[5]),
                dtype: DType::F32,
            });
            g.set_target_backend(relu_id, BackendId::Cpu);
            (in_id, relu_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, relu_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().expect("f32 cast");
            assert_eq!(typed, &[0.0, 0.0, 0.5, 0.0, 7.25]);
        }
    }

    /// E2E: Const + Const + Sub + Mul + Div — exercises three more
    /// of the freshly-migrated binary kernels in one graph. Verifies
    /// that intermediates flow through the cache as expected.
    #[test]
    fn pipelined_realize_chained_binary_ops() {
        let a_storage = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0]);
        let b_storage = fuel_memory::from_slice_cpu(&[3.0_f32, 5.0]);
        let c_storage = fuel_memory::from_slice_cpu(&[2.0_f32, 4.0]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, b_id, c_id, sub_id, mul_id, div_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let c = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            // (a - b)         = [7, 15]
            let sub = g.push(Node {
                op: Op::Sub, inputs: vec![a, b],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            // (a - b) * c     = [14, 60]
            let mul = g.push(Node {
                op: Op::Mul, inputs: vec![sub, c],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            // ((a-b)*c) / b   = [14/3, 12]
            let div = g.push(Node {
                op: Op::Div, inputs: vec![mul, b],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            g.set_target_backend(sub, BackendId::Cpu);
            g.set_target_backend(mul, BackendId::Cpu);
            g.set_target_backend(div, BackendId::Cpu);
            (a, b, c, sub, mul, div)
        };

        let _ = (sub_id, mul_id);
        let mut inputs = StorageCache::new();
        inputs.insert(a_id, Arc::new(RwLock::new(a_storage)));
        inputs.insert(b_id, Arc::new(RwLock::new(b_storage)));
        inputs.insert(c_id, Arc::new(RwLock::new(c_storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, div_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            assert!((typed[0] - (14.0_f32 / 3.0)).abs() < 1e-6);
            assert!((typed[1] - 12.0).abs() < 1e-6);
        }
    }

    /// E2E: Const + Transpose — verifies metadata-only view ops
    /// share the input's Storage Arc and produce a strided Layout.
    /// Stage 3 of Layout-on-Node.
    #[test]
    fn pipelined_realize_transpose_is_metadata_only() {
        // shape [2, 3]; transpose → shape [3, 2], strided
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, t_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let t_id = g.push(Node {
                op: Op::Transpose, inputs: vec![in_id],
                shape: Shape::from_dims(&[3, 2]), dtype: DType::F32,
            });
            // Note: NO set_target_backend — transpose is metadata-only
            // and doesn't run on a backend.
            (in_id, t_id)
        };
        let mut inputs = StorageCache::new();
        let in_arc = Arc::new(RwLock::new(storage));
        inputs.insert(in_id, Arc::clone(&in_arc));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, t_id, inputs).expect("realize");

        // The output Storage Arc is the SAME Arc as the input —
        // metadata-only adoption shares bytes.
        assert!(Arc::ptr_eq(&result_arc, &in_arc), "transpose must share input bytes");

        // The output Layout is the transposed view.
        assert_eq!(result_layout.shape().dims(), &[3, 2]);
        assert_eq!(result_layout.stride(), &[1, 3]);
        assert!(!result_layout.is_contiguous());
    }

    /// E2E: Const + Flip(dim=0) — verifies Op::Flip is metadata-only.
    /// The output Storage Arc must be the SAME Arc (no copy);
    /// the output Layout has a negated stride at dim 0 and a
    /// shifted start_offset.
    #[test]
    fn pipelined_realize_flip_is_metadata_only() {
        // shape [3, 4]; flip dim 0 → shape [3, 4], stride [-4, 1], offset 8.
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let storage = fuel_memory::from_slice_cpu(&data);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, f_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3, 4]), dtype: DType::F32,
            });
            let f_id = g.push(Node {
                op: Op::Flip { dim: 0 }, inputs: vec![in_id],
                shape: Shape::from_dims(&[3, 4]), dtype: DType::F32,
            });
            (in_id, f_id)
        };
        let mut inputs = StorageCache::new();
        let in_arc = Arc::new(RwLock::new(storage));
        inputs.insert(in_id, Arc::clone(&in_arc));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, f_id, inputs).expect("realize");

        assert!(Arc::ptr_eq(&result_arc, &in_arc), "flip must share input bytes");
        assert_eq!(result_layout.shape().dims(), &[3, 4]);
        assert_eq!(result_layout.stride(), &[-4_isize, 1]);
        assert_eq!(result_layout.start_offset(), 8);
        assert!(!result_layout.is_contiguous());
    }

    /// E2E: Const + Permute(rank-3 axes [2, 0, 1]) — verifies the
    /// general permute path through metadata-only adoption.
    #[test]
    fn pipelined_realize_permute_is_metadata_only() {
        // shape [2, 3, 4]; permute axes [2, 0, 1] → shape [4, 2, 3]
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let storage = fuel_memory::from_slice_cpu(&data);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, p_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3, 4]), dtype: DType::F32,
            });
            let p_id = g.push(Node {
                op: Op::Permute(vec![2, 0, 1]), inputs: vec![in_id],
                shape: Shape::from_dims(&[4, 2, 3]), dtype: DType::F32,
            });
            (in_id, p_id)
        };
        let mut inputs = StorageCache::new();
        let in_arc = Arc::new(RwLock::new(storage));
        inputs.insert(in_id, Arc::clone(&in_arc));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, p_id, inputs).expect("realize");

        assert!(Arc::ptr_eq(&result_arc, &in_arc));
        assert_eq!(result_layout.shape().dims(), &[4, 2, 3]);
        // Original strides for shape [2, 3, 4] are [12, 4, 1].
        // After permute axes [2, 0, 1]: [strides[2], strides[0], strides[1]] = [1, 12, 4].
        assert_eq!(result_layout.stride(), &[1, 12, 4]);
    }

    /// E2E: Const + Cast(f32→f64) — verifies cast through the
    /// pipelined executor. Output Storage has the target dtype;
    /// bytes encode the widened values.
    #[test]
    fn pipelined_realize_cast_f32_to_f64() {
        let storage = fuel_memory::from_slice_cpu(&[1.5_f32, -2.25, 100.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, c_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let c_id = g.push(Node {
                op: Op::Cast(DType::F64), inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F64,
            });
            g.set_target_backend(c_id, BackendId::Cpu);
            (in_id, c_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, c_id, inputs).expect("realize");
        assert_eq!(result_layout.shape().dims(), &[3]);

        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::F64);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f64>().unwrap(), &[1.5_f64, -2.25, 100.0]);
    }

    /// E2E: Const + Cast(f32→bf16) + Cast(bf16→f32) — round trip
    /// through bf16; verifies the Cast wrapper's source-dtype
    /// dispatch (different sources hit different match arms).
    /// Inputs chosen to round-trip exactly through bf16.
    #[test]
    fn pipelined_realize_cast_round_trip_via_bf16() {
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, -3.0, 0.5]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, c1_id, c2_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let c1_id = g.push(Node {
                op: Op::Cast(DType::BF16), inputs: vec![in_id],
                shape: Shape::from_dims(&[4]), dtype: DType::BF16,
            });
            let c2_id = g.push(Node {
                op: Op::Cast(DType::F32), inputs: vec![c1_id],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            g.set_target_backend(c1_id, BackendId::Cpu);
            g.set_target_backend(c2_id, BackendId::Cpu);
            (in_id, c1_id, c2_id)
        };
        let _ = c1_id;
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, c2_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::F32);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[1.0_f32, 2.0, -3.0, 0.5]);
    }

    /// E2E: Const + Slice — slice is metadata-only; the output Arc
    /// shares bytes with the input, and the Layout's start_offset
    /// + narrowed shape reflect the slice. Stage 3 of Layout-on-Node
    /// extended to cover Op::Slice via Layout::narrow.
    #[test]
    fn pipelined_realize_slice_is_metadata_only() {
        // shape [5]; slice dim 0 from index 1 with len 3 → shape [3]
        // Source: [10, 20, 30, 40, 50]; slice → [20, 30, 40]
        let storage = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0, 40.0, 50.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, s_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let s_id = g.push(Node {
                op: Op::Slice { dim: 0, start: 1, len: 3 },
                inputs: vec![in_id],
                shape: Shape::from_dims(&[3]),
                dtype: DType::F32,
            });
            (in_id, s_id)
        };
        let mut inputs = StorageCache::new();
        let in_arc = Arc::new(RwLock::new(storage));
        inputs.insert(in_id, Arc::clone(&in_arc));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, s_id, inputs).expect("realize");

        // Bytes shared with the input.
        assert!(Arc::ptr_eq(&result_arc, &in_arc));
        assert_eq!(result_layout.shape().dims(), &[3]);
        // Slice into a contiguous source: the resulting layout has
        // start_offset = original_stride[0] * start = 1 * 1 = 1,
        // and stride [1] (still contiguous within the narrowed dim).
        assert_eq!(result_layout.start_offset(), 1);
        assert_eq!(result_layout.stride(), &[1]);
    }

    /// E2E: Const + Slice + SumAll — slice is metadata-only, but
    /// sum needs contiguous bytes, so auto-Contiguize materializes
    /// the slice before reduce. Tests the stage 3+4 integration
    /// through Op::Slice. Sum of `[20, 30, 40]` is 90.
    #[test]
    fn pipelined_realize_slice_then_sum_all() {
        let storage = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0, 40.0, 50.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, s_id, sum_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let s_id = g.push(Node {
                op: Op::Slice { dim: 0, start: 1, len: 3 },
                inputs: vec![in_id],
                shape: Shape::from_dims(&[3]),
                dtype: DType::F32,
            });
            let sum_id = g.push(Node {
                op: Op::SumAll, inputs: vec![s_id],
                shape: Shape::from_dims(&[]), dtype: DType::F32,
            });
            g.set_target_backend(sum_id, BackendId::Cpu);
            (in_id, s_id, sum_id)
        };
        let _ = s_id;
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, sum_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[90.0]);
    }

    /// E2E: Const + Flip + ReluElementwise — Flip is metadata-only,
    /// then a kernel that doesn't yet handle negative strides forces
    /// auto-Contiguize to materialize through StridedIndex (which
    /// handles signed strides natively). Verifies the post-flip
    /// data ordering ([4, 3, 2, 1]) is correctly seen by the kernel.
    #[test]
    fn pipelined_realize_flip_then_relu_materializes_in_reverse() {
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, _f_id, r_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let f_id = g.push(Node {
                op: Op::Flip { dim: 0 }, inputs: vec![in_id],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let r_id = g.push(Node {
                op: Op::Relu, inputs: vec![f_id],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            g.set_target_backend(r_id, BackendId::Cpu);
            (in_id, f_id, r_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, r_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[4.0, 3.0, 2.0, 1.0]);
        assert_eq!(result_layout.shape().dims(), &[4]);
        assert!(result_layout.is_contiguous(), "relu output is contiguous");
    }

    /// E2E: Const + BroadcastTo — verifies that broadcast layouts
    /// have stride 0 on the broadcast dim while sharing the input's
    /// bytes. Stage 3 of Layout-on-Node.
    #[test]
    fn pipelined_realize_broadcast_is_metadata_only() {
        // shape [3]; broadcast to [4, 3] — leading dim is 0-stride
        let storage = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, b_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let b_id = g.push(Node {
                op: Op::BroadcastTo(Shape::from_dims(&[4, 3])),
                inputs: vec![in_id],
                shape: Shape::from_dims(&[4, 3]),
                dtype: DType::F32,
            });
            (in_id, b_id)
        };
        let mut inputs = StorageCache::new();
        let in_arc = Arc::new(RwLock::new(storage));
        inputs.insert(in_id, Arc::clone(&in_arc));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, b_id, inputs).expect("realize");

        assert!(Arc::ptr_eq(&result_arc, &in_arc));
        assert_eq!(result_layout.shape().dims(), &[4, 3]);
        // Broadcasted leading dim has stride 0.
        assert_eq!(result_layout.stride(), &[0, 1]);
    }

    /// E2E: Const + Const + MatMul — exercises rank-2 matmul
    /// through the pipelined executor. Inputs are contiguous (no
    /// auto-Contiguize needed); the kernel walks them via the
    /// (m, n, k) carried in OpParams::Matmul.
    #[test]
    fn pipelined_realize_matmul_2x3_times_3x2() {
        // [[1, 2, 3], [4, 5, 6]] @ [[7, 8], [9, 10], [11, 12]]
        //   = [[58, 64], [139, 154]]
        let lhs_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs_storage = fuel_memory::from_slice_cpu(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (lhs_id, rhs_id, mm_id) = {
            let mut g = graph.write().unwrap();
            let lhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let rhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3, 2]), dtype: DType::F32,
            });
            let mm = g.push(Node {
                op: Op::MatMul, inputs: vec![lhs, rhs],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::F32,
            });
            g.set_target_backend(mm, BackendId::Cpu);
            (lhs, rhs, mm)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(lhs_id, Arc::new(RwLock::new(lhs_storage)));
        inputs.insert(rhs_id, Arc::new(RwLock::new(rhs_storage)));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, mm_id, inputs).expect("realize");
        assert_eq!(result_layout.shape().dims(), &[2, 2]);
        assert!(result_layout.is_contiguous());

        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    /// E2E: batched matmul through the pipelined executor. Two
    /// batches of [2, 2] @ [2, 2]; the kernel iterates over
    /// `batch_count` and produces concatenated outputs.
    #[test]
    fn pipelined_realize_matmul_batched_2x_2x2_times_2x2() {
        let lhs_storage = fuel_memory::from_slice_cpu(&[
            1.0_f32, 2.0, 3.0, 4.0, // batch 0
            1.0, 0.0, 0.0, 1.0,     // batch 1 (identity)
        ]);
        let rhs_storage = fuel_memory::from_slice_cpu(&[
            5.0_f32, 6.0, 7.0, 8.0, // batch 0
            10.0, 20.0, 30.0, 40.0, // batch 1
        ]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (lhs_id, rhs_id, mm_id) = {
            let mut g = graph.write().unwrap();
            let lhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 2, 2]), dtype: DType::F32,
            });
            let rhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 2, 2]), dtype: DType::F32,
            });
            let mm = g.push(Node {
                op: Op::MatMul, inputs: vec![lhs, rhs],
                shape: Shape::from_dims(&[2, 2, 2]), dtype: DType::F32,
            });
            g.set_target_backend(mm, BackendId::Cpu);
            (lhs, rhs, mm)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(lhs_id, Arc::new(RwLock::new(lhs_storage)));
        inputs.insert(rhs_id, Arc::new(RwLock::new(rhs_storage)));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, mm_id, inputs).expect("realize");
        assert_eq!(result_layout.shape().dims(), &[2, 2, 2]);
        assert!(result_layout.is_contiguous());

        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        // batch 0: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        // batch 1: identity @ [[10,20],[30,40]]   = [[10,20],[30,40]]
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[19.0, 22.0, 43.0, 50.0, 10.0, 20.0, 30.0, 40.0]
        );
    }

    /// E2E: F64 elementwise add through the pipelined executor.
    /// Verifies that capability-driven dispatch correctly routes
    /// (AddElementwise, F64) to the f64 wrapper/kernel.
    #[test]
    fn pipelined_realize_add_f64() {
        let lhs = fuel_memory::from_slice_cpu(&[1.0_f64, 2.0, 3.0]);
        let rhs = fuel_memory::from_slice_cpu(&[10.0_f64, 20.0, 30.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (l_id, r_id, op_id) = {
            let mut g = graph.write().unwrap();
            let l = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F64,
            });
            let r = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F64,
            });
            let op = g.push(Node {
                op: Op::Add, inputs: vec![l, r],
                shape: Shape::from_dims(&[3]), dtype: DType::F64,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (l, r, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(l_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(r_id, Arc::new(RwLock::new(rhs)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::F64);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f64>().unwrap(), &[11.0_f64, 22.0, 33.0]);
    }

    /// E2E: Op::Equal F32 → U8 mask through the pipelined executor.
    /// Verifies (a) the binding-table key `(EqualElementwise, [F32, F32, U8],
    /// Cpu)` resolves, (b) the executor allocates a U8-sized output
    /// buffer (1 byte per element, not 4), (c) the kernel writes the
    /// expected mask bits including IEEE-754 NaN handling
    /// (`NaN == NaN` is false).
    #[test]
    fn pipelined_realize_eq_f32_to_u8_mask() {
        let lhs = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, f32::NAN, 0.0]);
        let rhs = fuel_memory::from_slice_cpu(&[1.0_f32, 5.0, 3.0, f32::NAN, -0.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (l_id, r_id, eq_id) = {
            let mut g = graph.write().unwrap();
            let l = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let r = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let eq = g.push(Node {
                op: Op::Equal, inputs: vec![l, r],
                shape: Shape::from_dims(&[5]), dtype: DType::U8,
            });
            g.set_target_backend(eq, BackendId::Cpu);
            (l, r, eq)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(l_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(r_id, Arc::new(RwLock::new(rhs)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, eq_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::U8);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let mask: &[u8] = c.as_slice().expect("u8 view");
        // Index 0: 1.0 == 1.0 → 1.
        // Index 1: 2.0 != 5.0 → 0.
        // Index 2: 3.0 == 3.0 → 1.
        // Index 3: NaN == NaN → 0 (IEEE-754).
        // Index 4: 0.0 == -0.0 → 1 (IEEE-754 zero equality).
        assert_eq!(mask, &[1, 0, 1, 0, 1]);
    }

    /// E2E: Op::Ne F32 → U8 mask. Mirrors the Eq F32 test with
    /// inverted predicate; NaN-vs-NaN slot now yields `1` (since
    /// `NaN != NaN` per IEEE-754).
    #[test]
    fn pipelined_realize_ne_f32_to_u8_mask() {
        let lhs = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, f32::NAN, 0.0]);
        let rhs = fuel_memory::from_slice_cpu(&[1.0_f32, 5.0, 3.0, f32::NAN, -0.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (l_id, r_id, ne_id) = {
            let mut g = graph.write().unwrap();
            let l = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let r = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let ne = g.push(Node {
                op: Op::Ne, inputs: vec![l, r],
                shape: Shape::from_dims(&[5]), dtype: DType::U8,
            });
            g.set_target_backend(ne, BackendId::Cpu);
            (l, r, ne)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(l_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(r_id, Arc::new(RwLock::new(rhs)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, ne_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::U8);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let mask: &[u8] = c.as_slice().expect("u8 view");
        // Inverse of the Eq test:
        // 1.0 != 1.0 → 0;  2.0 != 5.0 → 1;  3.0 != 3.0 → 0;
        // NaN != NaN → 1 (IEEE-754);  0.0 != -0.0 → 0 (IEEE-754).
        assert_eq!(mask, &[0, 1, 0, 1, 0]);
    }

    /// E2E: Op::Lt F32 → U8 mask. Confirms strict-less-than semantics
    /// + IEEE-754 NaN handling (any comparison with NaN is unordered →
    /// `0`).
    #[test]
    fn pipelined_realize_lt_f32_to_u8_mask() {
        let lhs = fuel_memory::from_slice_cpu(&[1.0_f32, 5.0, 3.0, f32::NAN, -1.0]);
        let rhs = fuel_memory::from_slice_cpu(&[2.0_f32, 5.0, 3.0, 0.0,      0.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (l_id, r_id, lt_id) = {
            let mut g = graph.write().unwrap();
            let l = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let r = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let lt = g.push(Node {
                op: Op::Lt, inputs: vec![l, r],
                shape: Shape::from_dims(&[5]), dtype: DType::U8,
            });
            g.set_target_backend(lt, BackendId::Cpu);
            (l, r, lt)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(l_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(r_id, Arc::new(RwLock::new(rhs)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, lt_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::U8);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let mask: &[u8] = c.as_slice().expect("u8 view");
        // 1.0 < 2.0 → 1;  5.0 < 5.0 → 0 (strict);  3.0 < 3.0 → 0;
        // NaN < 0.0 → 0 (unordered);  -1.0 < 0.0 → 1.
        assert_eq!(mask, &[1, 0, 0, 0, 1]);
    }

    /// E2E: Op::Le F32 → U8 mask. Distinct from Lt at the equal slot
    /// (`5.0 <= 5.0` = 1, vs Lt's `5.0 < 5.0` = 0). NaN unordered → 0.
    #[test]
    fn pipelined_realize_le_f32_to_u8_mask() {
        let lhs = fuel_memory::from_slice_cpu(&[1.0_f32, 5.0, 3.0, f32::NAN, -1.0]);
        let rhs = fuel_memory::from_slice_cpu(&[2.0_f32, 5.0, 2.0, 0.0,      0.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (l_id, r_id, le_id) = {
            let mut g = graph.write().unwrap();
            let l = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let r = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let le = g.push(Node {
                op: Op::Le, inputs: vec![l, r],
                shape: Shape::from_dims(&[5]), dtype: DType::U8,
            });
            g.set_target_backend(le, BackendId::Cpu);
            (l, r, le)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(l_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(r_id, Arc::new(RwLock::new(rhs)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, le_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::U8);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let mask: &[u8] = c.as_slice().expect("u8 view");
        // 1.0 <= 2.0 → 1;  5.0 <= 5.0 → 1 (key Lt difference);
        // 3.0 <= 2.0 → 0;  NaN <= 0.0 → 0 (unordered);  -1.0 <= 0.0 → 1.
        assert_eq!(mask, &[1, 1, 0, 0, 1]);
    }

    /// E2E: Op::Gt F32 → U8 mask. Strict-greater: equality slot is
    /// `0`. NaN unordered → `0`.
    #[test]
    fn pipelined_realize_gt_f32_to_u8_mask() {
        let lhs = fuel_memory::from_slice_cpu(&[3.0_f32, 5.0, 2.0, f32::NAN, 1.0]);
        let rhs = fuel_memory::from_slice_cpu(&[2.0_f32, 5.0, 3.0, 0.0,      0.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (l_id, r_id, gt_id) = {
            let mut g = graph.write().unwrap();
            let l = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let r = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let gt = g.push(Node {
                op: Op::Gt, inputs: vec![l, r],
                shape: Shape::from_dims(&[5]), dtype: DType::U8,
            });
            g.set_target_backend(gt, BackendId::Cpu);
            (l, r, gt)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(l_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(r_id, Arc::new(RwLock::new(rhs)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, gt_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::U8);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let mask: &[u8] = c.as_slice().expect("u8 view");
        // 3.0 > 2.0 → 1;  5.0 > 5.0 → 0 (strict);  2.0 > 3.0 → 0;
        // NaN > 0.0 → 0 (unordered);  1.0 > 0.0 → 1.
        assert_eq!(mask, &[1, 0, 0, 0, 1]);
    }

    /// E2E: Op::Ge F32 → U8 mask. Greater-or-equal: equality slot
    /// is `1` (distinguishes from Gt). NaN unordered → `0`. Closes
    /// the comparison family with full `[Eq, Ne, Lt, Le, Gt, Ge]`
    /// coverage.
    #[test]
    fn pipelined_realize_ge_f32_to_u8_mask() {
        let lhs = fuel_memory::from_slice_cpu(&[3.0_f32, 5.0, 2.0, f32::NAN, 0.0]);
        let rhs = fuel_memory::from_slice_cpu(&[2.0_f32, 5.0, 3.0, 0.0,      0.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (l_id, r_id, ge_id) = {
            let mut g = graph.write().unwrap();
            let l = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let r = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let ge = g.push(Node {
                op: Op::Ge, inputs: vec![l, r],
                shape: Shape::from_dims(&[5]), dtype: DType::U8,
            });
            g.set_target_backend(ge, BackendId::Cpu);
            (l, r, ge)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(l_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(r_id, Arc::new(RwLock::new(rhs)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, ge_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::U8);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let mask: &[u8] = c.as_slice().expect("u8 view");
        // 3.0 >= 2.0 → 1;  5.0 >= 5.0 → 1 (key Gt difference);
        // 2.0 >= 3.0 → 0;  NaN >= 0.0 → 0 (unordered);  0.0 >= 0.0 → 1.
        assert_eq!(mask, &[1, 1, 0, 0, 1]);
    }

    /// E2E: Op::Equal F64 → U8 mask. Confirms the F64 wrapper is
    /// independently registered and routed (binding-table key
    /// `(EqualElementwise, [F64, F64, U8], Cpu)`).
    #[test]
    fn pipelined_realize_eq_f64_to_u8_mask() {
        let lhs = fuel_memory::from_slice_cpu(&[1.0_f64, 2.0, 3.0]);
        let rhs = fuel_memory::from_slice_cpu(&[1.0_f64, 2.0, 4.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (l_id, r_id, eq_id) = {
            let mut g = graph.write().unwrap();
            let l = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F64,
            });
            let r = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F64,
            });
            let eq = g.push(Node {
                op: Op::Equal, inputs: vec![l, r],
                shape: Shape::from_dims(&[3]), dtype: DType::U8,
            });
            g.set_target_backend(eq, BackendId::Cpu);
            (l, r, eq)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(l_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(r_id, Arc::new(RwLock::new(rhs)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, eq_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::U8);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let mask: &[u8] = c.as_slice().expect("u8 view");
        assert_eq!(mask, &[1, 1, 0]);
    }

    /// E2E: Op::Where ternary select — `out[i] = if cond[i] != 0 { a[i] } else { b[i] }`.
    /// Validates (a) the binding-table key `(Where, [U8, F32, F32, F32], Cpu)`
    /// resolves to the where_f32 wrapper, (b) the U8 cond input drives
    /// the per-slot pick, (c) outputs preserve the input dtype.
    #[test]
    fn pipelined_realize_where_f32_picks_per_slot_from_u8_mask() {
        // cond = [1, 0, 1, 0, 1]; a = [1, 2, 3, 4, 5]; b = [10, 20, 30, 40, 50]
        // expected = [1, 20, 3, 40, 5]
        let cond_storage = fuel_memory::from_slice_cpu(&[1u8, 0, 1, 0, 1]);
        let a_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]);
        let b_storage = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0, 40.0, 50.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (cond_id, a_id, b_id, where_id) = {
            let mut g = graph.write().unwrap();
            let cond = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::U8,
            });
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            let w = g.push(Node {
                op: Op::Where, inputs: vec![cond, a, b],
                shape: Shape::from_dims(&[5]), dtype: DType::F32,
            });
            g.set_target_backend(w, BackendId::Cpu);
            (cond, a, b, w)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(cond_id, Arc::new(RwLock::new(cond_storage)));
        inputs.insert(a_id, Arc::new(RwLock::new(a_storage)));
        inputs.insert(b_id, Arc::new(RwLock::new(b_storage)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, where_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::F32);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let out: &[f32] = c.as_slice().expect("f32 view");
        assert_eq!(out, &[1.0, 20.0, 3.0, 40.0, 5.0]);
    }

    /// E2E: full chain `eq → where`. Compares two f32 vectors, then
    /// uses the resulting U8 mask to pick from a third tensor (or a
    /// fallback). Validates the comparison-family + Where ops compose
    /// end-to-end.
    #[test]
    fn pipelined_realize_eq_then_where_full_chain() {
        // a = [1, 2, 3]; b = [1, 5, 3] → eq = [1, 0, 1]
        // pick = [10, 20, 30]; fallback = [99, 99, 99]
        // result = [10, 99, 30]
        let a_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
        let b_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 5.0, 3.0]);
        let pick_storage = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);
        let fb_storage = fuel_memory::from_slice_cpu(&[99.0_f32, 99.0, 99.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, b_id, pick_id, fb_id, where_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let pick = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let fb = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let eq = g.push(Node {
                op: Op::Equal, inputs: vec![a, b],
                shape: Shape::from_dims(&[3]), dtype: DType::U8,
            });
            let w = g.push(Node {
                op: Op::Where, inputs: vec![eq, pick, fb],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(eq, BackendId::Cpu);
            g.set_target_backend(w, BackendId::Cpu);
            (a, b, pick, fb, w)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(a_id, Arc::new(RwLock::new(a_storage)));
        inputs.insert(b_id, Arc::new(RwLock::new(b_storage)));
        inputs.insert(pick_id, Arc::new(RwLock::new(pick_storage)));
        inputs.insert(fb_id, Arc::new(RwLock::new(fb_storage)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, where_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let out: &[f32] = c.as_slice().expect("f32 view");
        assert_eq!(out, &[10.0, 99.0, 30.0]);
    }

    /// E2E: Q4_0 QMatMul through the pipelined executor — proves
    /// quantized weights can flow into the unified path. Activations
    /// are F32, weights are U32-typed (raw block bytes).
    /// Construct a Q4_0 weight tensor where every weight = 1.0
    /// (d=1.0, every nibble=9 → 1*(9-8)=1), so A @ W^T computes
    /// the per-row sum of activations.
    #[test]
    fn pipelined_realize_qmatmul_q4_0_unit_weight_sums_activations() {
        use fuel_graph::QuantType;
        use half::f16;
        let block_size = std::mem::size_of::<fuel_quantized::BlockQ4_0>();
        let mut w_bytes = vec![0u8; 2 * block_size];
        for block_idx in 0..2 {
            let off = block_idx * block_size;
            let d_bytes = f16::from_f32(1.0).to_le_bytes();
            w_bytes[off..off + 2].copy_from_slice(&d_bytes);
            for i in 0..16 {
                w_bytes[off + 2 + i] = 0x99;
            }
        }
        // Weight tensor is U32-typed (rank-1, length = bytes/4)
        let w_storage = fuel_memory::Storage::new(
            fuel_memory::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_bytes(&w_bytes),
            ),
            DType::U32,
        );

        let act_vec: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let act_storage = fuel_memory::from_slice_cpu(&act_vec);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (act_id, w_id, mm_id) = {
            let mut g = graph.write().unwrap();
            let act = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 32]), dtype: DType::F32,
            });
            let w = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[w_bytes.len() / 4]),
                dtype: DType::U32,
            });
            let mm = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::QMATMUL,
                    fuel_graph::registry::FusedOpParams::QMatMul {
                        quant_type: QuantType::Q4_0, k: 32, n: 2,
                    },
                ),
                inputs: vec![act, w],
                shape: Shape::from_dims(&[1, 2]),
                dtype: DType::F32,
            });
            g.set_target_backend(mm, BackendId::Cpu);
            (act, w, mm)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(act_id, Arc::new(RwLock::new(act_storage)));
        inputs.insert(w_id, Arc::new(RwLock::new(w_storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, mm_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::F32);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r: &[f32] = c.as_slice().unwrap();
        // Both rows = sum(1..=32) = 528, within Q8_1 round-trip
        // tolerance.
        assert!((r[0] - 528.0).abs() < 0.5, "got {}, want 528", r[0]);
        assert!((r[1] - 528.0).abs() < 0.5, "got {}, want 528", r[1]);
    }

    /// E2E: QMatMul with Q5_0 weights — verifies the new quant
    /// dispatch arm picks `qmatmul_q5_0_f32`. We build weights by
    /// quantizing all-ones via `BlockQ5_0::from_float`, then
    /// compare pipelined output against the direct fuel_quantized
    /// matmul on the same blocks.
    #[test]
    fn pipelined_realize_qmatmul_q5_0_against_reference() {
        use fuel_graph::QuantType;
        use fuel_quantized::{BlockQ5_0, GgmlType};
        let n = 2;
        let k = 64; // 2 blocks per row (Q5_0 vec_dot pairs blocks)
        // Quantize an all-ones [n, k] weight matrix.
        let w_f32 = vec![1.0_f32; n * k];
        let blocks_per_row = k / BlockQ5_0::BLCK_SIZE;
        let mut w_blocks = vec![BlockQ5_0::zeros(); n * blocks_per_row];
        BlockQ5_0::from_float(&w_f32, &mut w_blocks);
        // Reinterpret block slice as bytes (BlockQ5_0 is #[repr(C)]).
        let bytes_per_block = std::mem::size_of::<BlockQ5_0>();
        let w_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                w_blocks.as_ptr() as *const u8,
                w_blocks.len() * bytes_per_block,
            )
        }
        .to_vec();
        let w_storage = fuel_memory::Storage::new(
            fuel_memory::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_bytes(&w_bytes),
            ),
            DType::U32,
        );
        let act_vec: Vec<f32> = (1..=k).map(|x| x as f32).collect();
        let act_storage = fuel_memory::from_slice_cpu(&act_vec);
        // Reference: direct matmul through fuel_quantized.
        let mut ref_out = vec![0.0_f32; n];
        fuel_quantized::matmul::<BlockQ5_0>((1, k, n), &act_vec, &w_blocks, &mut ref_out)
            .expect("ref matmul");

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (act_id, w_id, mm_id) = {
            let mut g = graph.write().unwrap();
            let act = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, k]), dtype: DType::F32,
            });
            let w = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[w_bytes.len() / 4]),
                dtype: DType::U32,
            });
            let mm = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::QMATMUL,
                    fuel_graph::registry::FusedOpParams::QMatMul {
                        quant_type: QuantType::Q5_0, k, n,
                    },
                ),
                inputs: vec![act, w],
                shape: Shape::from_dims(&[1, n]),
                dtype: DType::F32,
            });
            g.set_target_backend(mm, BackendId::Cpu);
            (act, w, mm)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(act_id, Arc::new(RwLock::new(act_storage)));
        inputs.insert(w_id, Arc::new(RwLock::new(w_storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, mm_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r: &[f32] = c.as_slice().unwrap();
        // Bit-exact against same kernel via the trait, since both
        // paths run fuel_quantized::matmul<BlockQ5_0>.
        assert_eq!(r, ref_out.as_slice(),
            "pipelined Q5_0 differs from reference: got {r:?}, want {ref_out:?}");
    }

    /// E2E: QMatMul with Q6K (256-element super-block k-quant).
    /// Same idea as the Q5_0 test — bit-exact against the
    /// reference fuel_quantized::matmul<BlockQ6K>. Confirms the
    /// dispatch arm wires `qmatmul_q6k_f32` correctly.
    #[test]
    fn pipelined_realize_qmatmul_q6k_against_reference() {
        use fuel_graph::QuantType;
        use fuel_quantized::{BlockQ6K, GgmlType};
        let n = 2;
        let k = 256; // 1 super-block per row
        let w_f32 = vec![1.0_f32; n * k];
        let blocks_per_row = k / BlockQ6K::BLCK_SIZE;
        let mut w_blocks = vec![BlockQ6K::zeros(); n * blocks_per_row];
        BlockQ6K::from_float(&w_f32, &mut w_blocks);
        let bytes_per_block = std::mem::size_of::<BlockQ6K>();
        let w_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                w_blocks.as_ptr() as *const u8,
                w_blocks.len() * bytes_per_block,
            )
        }
        .to_vec();
        let w_storage = fuel_memory::Storage::new(
            fuel_memory::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_bytes(&w_bytes),
            ),
            DType::U32,
        );
        let act_vec: Vec<f32> = (1..=k).map(|x| x as f32 / 100.0).collect();
        let act_storage = fuel_memory::from_slice_cpu(&act_vec);
        let mut ref_out = vec![0.0_f32; n];
        fuel_quantized::matmul::<BlockQ6K>((1, k, n), &act_vec, &w_blocks, &mut ref_out)
            .expect("ref matmul");

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (act_id, w_id, mm_id) = {
            let mut g = graph.write().unwrap();
            let act = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, k]), dtype: DType::F32,
            });
            let w = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[w_bytes.len() / 4]),
                dtype: DType::U32,
            });
            let mm = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::QMATMUL,
                    fuel_graph::registry::FusedOpParams::QMatMul {
                        quant_type: QuantType::Q6K, k, n,
                    },
                ),
                inputs: vec![act, w],
                shape: Shape::from_dims(&[1, n]),
                dtype: DType::F32,
            });
            g.set_target_backend(mm, BackendId::Cpu);
            (act, w, mm)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(act_id, Arc::new(RwLock::new(act_storage)));
        inputs.insert(w_id, Arc::new(RwLock::new(w_storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, mm_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r: &[f32] = c.as_slice().unwrap();
        assert_eq!(r, ref_out.as_slice(),
            "pipelined Q6K differs from reference: got {r:?}, want {ref_out:?}");
    }

    /// E2E: BF16 RmsNormLastDim through the pipelined executor.
    /// Verifies that capability-driven dispatch routes the
    /// half-float norm op to the bf16-specific kernel (which
    /// accumulates in f32 internally).
    #[test]
    fn pipelined_realize_rms_norm_bf16() {
        let v: Vec<half::bf16> = [3.0_f32, 4.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let storage = fuel_memory::from_slice_cpu(&v);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 2]), dtype: DType::BF16,
            });
            let op_id = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::RMS_NORM_LAST_DIM,
                    fuel_graph::registry::FusedOpParams::RmsNormLastDim { eps: 0.0 },
                ), inputs: vec![in_id],
                shape: Shape::from_dims(&[1, 2]), dtype: DType::BF16,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::BF16);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r: &[half::bf16] = c.as_slice().unwrap();
        let rms = (12.5_f32).sqrt();
        // bf16's ~3-digit mantissa absorbs the divisor; allow ~5%.
        assert!((r[0].to_f32() - 3.0 / rms).abs() < 0.05);
        assert!((r[1].to_f32() - 4.0 / rms).abs() < 0.05);
    }

    /// E2E: BF16 matmul through the pipelined executor — proves
    /// the LLM forward-pass blocker (every transformer layer
    /// is dominated by matmul). Identity matmul on bf16 round-
    /// trips small integers exactly.
    #[test]
    fn pipelined_realize_matmul_bf16_identity() {
        let lhs_v: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let rhs_v: Vec<half::bf16> = [1.0_f32, 0.0, 0.0, 1.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let lhs = fuel_memory::from_slice_cpu(&lhs_v);
        let rhs = fuel_memory::from_slice_cpu(&rhs_v);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (l_id, r_id, mm_id) = {
            let mut g = graph.write().unwrap();
            let l = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::BF16,
            });
            let r = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::BF16,
            });
            let mm = g.push(Node {
                op: Op::MatMul, inputs: vec![l, r],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::BF16,
            });
            g.set_target_backend(mm, BackendId::Cpu);
            (l, r, mm)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(l_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(r_id, Arc::new(RwLock::new(rhs)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, mm_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::BF16);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r: &[half::bf16] = c.as_slice().unwrap();
        let result_f32: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        assert_eq!(result_f32, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// E2E: F16 sum-reduce — verifies bf16/f16 reduction dispatch
    /// works through the executor.
    #[test]
    fn pipelined_realize_sum_dim_f16() {
        let v: Vec<half::f16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter().map(|&x| half::f16::from_f32(x)).collect();
        let storage = fuel_memory::from_slice_cpu(&v);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, sum_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F16,
            });
            let sum_id = g.push(Node {
                op: Op::SumDim(1), inputs: vec![in_id],
                shape: Shape::from_dims(&[2]), dtype: DType::F16,
            });
            g.set_target_backend(sum_id, BackendId::Cpu);
            (in_id, sum_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, sum_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::F16);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r: &[half::f16] = c.as_slice().unwrap();
        let result_f32: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        assert_eq!(result_f32, vec![6.0, 15.0]);
    }

    /// E2E: BF16 elementwise add through the pipelined executor.
    #[test]
    fn pipelined_realize_add_bf16() {
        let lhs_vec: Vec<half::bf16> = [1.0_f32, 2.0, 3.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let rhs_vec: Vec<half::bf16> = [10.0_f32, 20.0, 30.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let lhs = fuel_memory::from_slice_cpu(&lhs_vec);
        let rhs = fuel_memory::from_slice_cpu(&rhs_vec);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (l_id, r_id, op_id) = {
            let mut g = graph.write().unwrap();
            let l = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::BF16,
            });
            let r = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::BF16,
            });
            let op = g.push(Node {
                op: Op::Add, inputs: vec![l, r],
                shape: Shape::from_dims(&[3]), dtype: DType::BF16,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (l, r, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(l_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(r_id, Arc::new(RwLock::new(rhs)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::BF16);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r: &[half::bf16] = c.as_slice().unwrap();
        let result_f32: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        assert_eq!(result_f32, vec![11.0, 22.0, 33.0]);
    }

    /// E2E: F16 unary chain — Const + Sqr + Sqrt — verifies F16
    /// dispatch works and the via-f32 round-trip kernels behave
    /// correctly through the executor.
    #[test]
    fn pipelined_realize_sqr_then_sqrt_f16() {
        let v: Vec<half::f16> = [1.0_f32, 4.0, 9.0]
            .iter().map(|&x| half::f16::from_f32(x)).collect();
        let storage = fuel_memory::from_slice_cpu(&v);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, sqrt_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F16,
            });
            let sqr_id = g.push(Node {
                op: Op::Sqr, inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F16,
            });
            let sqrt_id = g.push(Node {
                op: Op::Sqrt, inputs: vec![sqr_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F16,
            });
            g.set_target_backend(sqr_id, BackendId::Cpu);
            g.set_target_backend(sqrt_id, BackendId::Cpu);
            (in_id, sqrt_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, sqrt_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::F16);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r: &[half::f16] = c.as_slice().unwrap();
        let result_f32: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        // f16 has ~3 decimal digits; sqrt(sqr(x)) = |x| within rounding.
        for (got, want) in result_f32.iter().zip(&[1.0_f32, 4.0, 9.0]) {
            assert!((got - want).abs() < 0.05, "got {got}, want {want}");
        }
    }

    /// E2E: F64 sum-reduce along one dim through the pipelined
    /// executor.
    #[test]
    fn pipelined_realize_sum_dim_f64() {
        let storage = fuel_memory::from_slice_cpu(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, sum_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F64,
            });
            let sum_id = g.push(Node {
                op: Op::SumDim(1), inputs: vec![in_id],
                shape: Shape::from_dims(&[2]), dtype: DType::F64,
            });
            g.set_target_backend(sum_id, BackendId::Cpu);
            (in_id, sum_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, sum_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::F64);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f64>().unwrap(), &[6.0_f64, 15.0]);
    }

    /// E2E: F64 matmul through the pipelined executor.
    #[test]
    fn pipelined_realize_matmul_2x3_times_3x2_f64() {
        let lhs = fuel_memory::from_slice_cpu(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = fuel_memory::from_slice_cpu(&[7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (l_id, r_id, mm_id) = {
            let mut g = graph.write().unwrap();
            let l = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F64,
            });
            let r = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3, 2]), dtype: DType::F64,
            });
            let mm = g.push(Node {
                op: Op::MatMul, inputs: vec![l, r],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::F64,
            });
            g.set_target_backend(mm, BackendId::Cpu);
            (l, r, mm)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(l_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(r_id, Arc::new(RwLock::new(rhs)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, mm_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::F64);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f64>().unwrap(), &[58.0_f64, 64.0, 139.0, 154.0]);
    }

    /// E2E: F64 unary chain — Const + Sqr + Sqrt — verifies that
    /// the kernel-binding lookup picks the f64 entries when the
    /// graph nodes carry DType::F64.
    #[test]
    fn pipelined_realize_sqr_then_sqrt_f64() {
        let storage = fuel_memory::from_slice_cpu(&[1.0_f64, 4.0, 9.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, sqrt_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F64,
            });
            let sqr_id = g.push(Node {
                op: Op::Sqr, inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F64,
            });
            let sqrt_id = g.push(Node {
                op: Op::Sqrt, inputs: vec![sqr_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F64,
            });
            g.set_target_backend(sqr_id, BackendId::Cpu);
            g.set_target_backend(sqrt_id, BackendId::Cpu);
            (in_id, sqrt_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, sqrt_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::F64);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f64>().unwrap(), &[1.0_f64, 4.0, 9.0]);
    }

    /// E2E: ArgMaxDim — produces U32 output indices.
    #[test]
    fn pipelined_realize_argmax_dim() {
        // input [2, 3] = [[1, 5, 2], [9, 0, 4]]
        // argmax dim=1 → [1, 0]
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 5.0, 2.0, 9.0, 0.0, 4.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let op_id = g.push(Node {
                op: Op::ArgMaxDim(1), inputs: vec![in_id],
                shape: Shape::from_dims(&[2]), dtype: DType::U32,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        assert_eq!(guard.dtype, DType::U32);
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<u32>().unwrap(), &[1u32, 0]);
    }

    /// E2E: IndexAdd along outer dim — accumulate updates into a
    /// rank-1 base tensor at indexed positions.
    #[test]
    fn pipelined_realize_index_add_simple() {
        let base = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);
        let indices = fuel_memory::from_slice_cpu(&[0u32, 0]);
        let src = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (b_id, i_id, s_id, op_id) = {
            let mut g = graph.write().unwrap();
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let i = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::U32,
            });
            let s = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let op = g.push(Node {
                op: Op::IndexAdd { dim: 0 }, inputs: vec![b, i, s],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (b, i, s, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(b_id, Arc::new(RwLock::new(base)));
        inputs.insert(i_id, Arc::new(RwLock::new(indices)));
        inputs.insert(s_id, Arc::new(RwLock::new(src)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        // 10 + 1 + 2 = 13; 20 untouched; 30 untouched.
        assert_eq!(c.as_slice::<f32>().unwrap(), &[13.0, 20.0, 30.0]);
    }

    /// E2E: ScatterAdd along outer dim — same-rank indices, base
    /// starts as zeros, src adds values at scatter positions.
    #[test]
    fn pipelined_realize_scatter_add_outer_dim() {
        // base [3, 2] = zeros; indices [2, 2] = [[0, 1], [2, 0]];
        // src [2, 2] = [[1, 2], [3, 4]]; dim=0
        // → out = [[1, 4], [0, 2], [3, 0]]
        let base = fuel_memory::from_slice_cpu(&[0.0_f32; 6]);
        let indices = fuel_memory::from_slice_cpu(&[0u32, 1, 2, 0]);
        let src = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (b_id, i_id, s_id, op_id) = {
            let mut g = graph.write().unwrap();
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3, 2]), dtype: DType::F32,
            });
            let i = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::U32,
            });
            let s = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::F32,
            });
            let op = g.push(Node {
                op: Op::ScatterAdd { dim: 0 }, inputs: vec![b, i, s],
                shape: Shape::from_dims(&[3, 2]), dtype: DType::F32,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (b, i, s, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(b_id, Arc::new(RwLock::new(base)));
        inputs.insert(i_id, Arc::new(RwLock::new(indices)));
        inputs.insert(s_id, Arc::new(RwLock::new(src)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[1.0, 4.0, 0.0, 2.0, 3.0, 0.0]);
    }

    /// E2E: Rope through the pipelined executor. cos=0, sin=1
    /// rotates the head_dim halves with sign per the rotate_half
    /// convention.
    #[test]
    fn pipelined_realize_rope_pi_over_two() {
        // x [1, 1, 4] = [1, 2, 3, 4]. cos=[0,0,0,0], sin=[1,1,1,1].
        // Expected: [-3, -4, 1, 2].
        let x = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let cos = fuel_memory::from_slice_cpu(&[0.0_f32, 0.0, 0.0, 0.0]);
        let sin = fuel_memory::from_slice_cpu(&[1.0_f32, 1.0, 1.0, 1.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, cos_id, sin_id, r_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 4]), dtype: DType::F32,
            });
            let c = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 4]), dtype: DType::F32,
            });
            let s = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 4]), dtype: DType::F32,
            });
            let r = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::ROPE,
                    fuel_graph::registry::FusedOpParams::Rope,
                ),
                inputs: vec![x, c, s],
                shape: Shape::from_dims(&[1, 1, 4]), dtype: DType::F32,
            });
            g.set_target_backend(r, BackendId::Cpu);
            (x, c, s, r)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(x_id, Arc::new(RwLock::new(x)));
        inputs.insert(cos_id, Arc::new(RwLock::new(cos)));
        inputs.insert(sin_id, Arc::new(RwLock::new(sin)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, r_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[-3.0, -4.0, 1.0, 2.0]);
    }

    /// E2E: Gather along inner dim. Source [2, 4]; indices [2, 3];
    /// output [2, 3] = picks from each row by per-row indices.
    #[test]
    fn pipelined_realize_gather_inner_dim() {
        let source = fuel_memory::from_slice_cpu(&[
            10.0_f32, 20.0, 30.0, 40.0,
            50.0, 60.0, 70.0, 80.0,
        ]);
        let indices = fuel_memory::from_slice_cpu(&[0u32, 2, 1, 3, 0, 0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (src_id, idx_id, g_id) = {
            let mut g = graph.write().unwrap();
            let s = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 4]), dtype: DType::F32,
            });
            let i = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::U32,
            });
            let g_id = g.push(Node {
                op: Op::Gather { dim: 1 }, inputs: vec![s, i],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            g.set_target_backend(g_id, BackendId::Cpu);
            (s, i, g_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(src_id, Arc::new(RwLock::new(source)));
        inputs.insert(idx_id, Arc::new(RwLock::new(indices)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, g_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[10.0, 30.0, 20.0, 80.0, 50.0, 50.0]
        );
    }

    /// E2E: IndexSelect — embedding-table lookup. Source is a
    /// `[vocab=4, d_model=3]` table; indices are token IDs;
    /// output is `[seq=3, d_model=3]` with the picked rows.
    #[test]
    fn pipelined_realize_index_select_embedding_lookup() {
        let table = fuel_memory::from_slice_cpu(&[
            10.0_f32, 11.0, 12.0,    // row 0
            20.0, 21.0, 22.0,        // row 1
            30.0, 31.0, 32.0,        // row 2
            40.0, 41.0, 42.0,        // row 3
        ]);
        let indices = fuel_memory::from_slice_cpu(&[2u32, 0, 2]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (table_id, idx_id, sel_id) = {
            let mut g = graph.write().unwrap();
            let t = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[4, 3]), dtype: DType::F32,
            });
            let i = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::U32,
            });
            let s = g.push(Node {
                op: Op::IndexSelect { dim: 0 }, inputs: vec![t, i],
                shape: Shape::from_dims(&[3, 3]), dtype: DType::F32,
            });
            g.set_target_backend(s, BackendId::Cpu);
            (t, i, s)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(table_id, Arc::new(RwLock::new(table)));
        inputs.insert(idx_id, Arc::new(RwLock::new(indices)));
        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, sel_id, inputs).expect("realize");
        assert_eq!(result_layout.shape().dims(), &[3, 3]);
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[30.0, 31.0, 32.0, 10.0, 11.0, 12.0, 30.0, 31.0, 32.0]
        );
    }

    /// E2E: RmsNormLastDim on a 2-row input. Each row's output
    /// has unit RMS up to the eps-induced bias.
    #[test]
    fn pipelined_realize_rms_norm_last_dim() {
        let storage = fuel_memory::from_slice_cpu(&[
            3.0_f32, 4.0,    // row 0: rms = sqrt(12.5)
            6.0, 8.0,        // row 1: rms = sqrt(50.0) = 5*sqrt(2)
        ]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::F32,
            });
            let op_id = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::RMS_NORM_LAST_DIM,
                    fuel_graph::registry::FusedOpParams::RmsNormLastDim { eps: 0.0 },
                ),
                inputs: vec![in_id],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::F32,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let result: &[f32] = c.as_slice().unwrap();
        // Row 0: rms = sqrt(12.5). Output = [3, 4] / sqrt(12.5).
        let rms0 = (12.5_f32).sqrt();
        assert!((result[0] - 3.0 / rms0).abs() < 1e-6);
        assert!((result[1] - 4.0 / rms0).abs() < 1e-6);
        // Row 1: rms = sqrt(50). Output = [6, 8] / sqrt(50).
        let rms1 = (50.0_f32).sqrt();
        assert!((result[2] - 6.0 / rms1).abs() < 1e-6);
        assert!((result[3] - 8.0 / rms1).abs() < 1e-6);
    }

    /// E2E: LayerNormLastDim — each row's output has zero mean and
    /// unit variance.
    #[test]
    fn pipelined_realize_layer_norm_last_dim() {
        let storage = fuel_memory::from_slice_cpu(&[
            1.0_f32, 2.0, 3.0,
            10.0, 20.0, 30.0,
        ]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let op_id = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::LAYER_NORM_LAST_DIM,
                    fuel_graph::registry::FusedOpParams::LayerNormLastDim { eps: 0.0 },
                ),
                inputs: vec![in_id],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let result: &[f32] = c.as_slice().unwrap();
        // Each row should have mean ~0 and var ~1.
        for row in 0..2 {
            let off = row * 3;
            let sum: f32 = result[off..off + 3].iter().sum();
            let mean = sum / 3.0;
            assert!(mean.abs() < 1e-6, "row {row} mean should be 0, got {mean}");
            let var: f32 = result[off..off + 3].iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 3.0;
            assert!((var - 1.0).abs() < 1e-6, "row {row} var should be 1, got {var}");
        }
    }

    /// E2E: SoftmaxLastDim on a 2-row input. Each row should sum
    /// to 1; uniform row gives uniform output.
    #[test]
    fn pipelined_realize_softmax_last_dim() {
        // Row 0: [1, 1, 1, 1] → uniform 0.25 each
        // Row 1: [0, 0, 0, 100] → effectively a one-hot at position 3
        let storage = fuel_memory::from_slice_cpu(&[
            1.0_f32, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 100.0,
        ]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 4]), dtype: DType::F32,
            });
            let op_id = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::SOFTMAX_LAST_DIM,
                    fuel_graph::registry::FusedOpParams::SoftmaxLastDim,
                ),
                inputs: vec![in_id],
                shape: Shape::from_dims(&[2, 4]), dtype: DType::F32,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let result: &[f32] = c.as_slice().unwrap();

        // Row 0: uniform 0.25
        for v in &result[..4] {
            assert!((v - 0.25).abs() < 1e-7);
        }
        // Row 1: positions 0..3 ≈ 0, position 4 (= last column) ≈ 1
        // (e^100 dominates).
        for v in &result[4..7] {
            assert!(*v < 1e-30, "row-1 leading positions should be near 0, got {v}");
        }
        assert!(result[7] > 0.999, "row-1 last position should dominate, got {}", result[7]);
        // Each row sums to 1.
        let row0_sum: f32 = result[..4].iter().sum();
        let row1_sum: f32 = result[4..].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-6);
        assert!((row1_sum - 1.0).abs() < 1e-6);
    }

    /// E2E: Concat along inner dim — two [2, 3] tensors → [2, 6].
    #[test]
    fn pipelined_realize_concat_inner_dim() {
        let a = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = fuel_memory::from_slice_cpu(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, b_id, c_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let c = g.push(Node {
                op: Op::Concat { dim: 1 }, inputs: vec![a, b],
                shape: Shape::from_dims(&[2, 6]), dtype: DType::F32,
            });
            g.set_target_backend(c, BackendId::Cpu);
            (a, b, c)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(a_id, Arc::new(RwLock::new(a)));
        inputs.insert(b_id, Arc::new(RwLock::new(b)));
        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, c_id, inputs).expect("realize");
        assert_eq!(result_layout.shape().dims(), &[2, 6]);
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0]
        );
    }

    /// E2E: Concat with three inputs along outer dim — verifies
    /// variable-arity input handling through the executor.
    #[test]
    fn pipelined_realize_concat_three_inputs_outer() {
        let a = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0]);
        let b = fuel_memory::from_slice_cpu(&[3.0_f32, 4.0]);
        let c = fuel_memory::from_slice_cpu(&[5.0_f32, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, b_id, c_id, cat_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let c = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let cat = g.push(Node {
                op: Op::Concat { dim: 0 }, inputs: vec![a, b, c],
                shape: Shape::from_dims(&[6]), dtype: DType::F32,
            });
            g.set_target_backend(cat, BackendId::Cpu);
            (a, b, c, cat)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(a_id, Arc::new(RwLock::new(a)));
        inputs.insert(b_id, Arc::new(RwLock::new(b)));
        inputs.insert(c_id, Arc::new(RwLock::new(c)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, cat_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// E2E: AddScalar — graph emits Op::AddScalar; the executor
    /// maps it to OpKind::Affine with mul=1, add=c.
    #[test]
    fn pipelined_realize_add_scalar() {
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let op_id = g.push(Node {
                op: Op::AddScalar(10.0), inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[11.0, 12.0, 13.0]);
    }

    /// E2E: Clamp — clamp values to [-2, 2].
    #[test]
    fn pipelined_realize_clamp() {
        let storage = fuel_memory::from_slice_cpu(&[-5.0_f32, 0.5, 100.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let op_id = g.push(Node {
                op: Op::Clamp { min: -2.0, max: 2.0 }, inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[-2.0, 0.5, 2.0]);
    }

    /// E2E: Maximum — elementwise tensor max.
    #[test]
    fn pipelined_realize_maximum_elementwise() {
        let lhs_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 5.0, -3.0]);
        let rhs_storage = fuel_memory::from_slice_cpu(&[2.0_f32, 1.0, -1.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (lhs_id, rhs_id, op_id) = {
            let mut g = graph.write().unwrap();
            let lhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let rhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let op = g.push(Node {
                op: Op::Maximum, inputs: vec![lhs, rhs],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (lhs, rhs, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(lhs_id, Arc::new(RwLock::new(lhs_storage)));
        inputs.insert(rhs_id, Arc::new(RwLock::new(rhs_storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[2.0, 5.0, -1.0]);
    }

    /// E2E: Const + Const + Conv2D — the 2×2 sum-kernel test from
    /// byte_kernels driven through the pipelined executor.
    #[test]
    fn pipelined_realize_conv2d_2x2_sum_kernel() {
        // x [1, 1, 3, 3]: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        // weight [1, 1, 2, 2]: all-ones
        // → out [1, 1, 2, 2]: [[12, 16], [24, 28]]
        let x_storage = fuel_memory::from_slice_cpu(&[
            1.0_f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let w_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 1.0, 1.0, 1.0]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, w_id, c_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 3, 3]), dtype: DType::F32,
            });
            let w = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            let c = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::CONV2D,
                    fuel_graph::registry::FusedOpParams::Conv2D {
                        stride: (1, 1), padding: (0, 0), groups: 1,
                    },
                ),
                inputs: vec![x, w],
                shape: Shape::from_dims(&[1, 1, 2, 2]),
                dtype: DType::F32,
            });
            g.set_target_backend(c, BackendId::Cpu);
            (x, w, c)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(x_id, Arc::new(RwLock::new(x_storage)));
        inputs.insert(w_id, Arc::new(RwLock::new(w_storage)));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, c_id, inputs).expect("realize");
        assert_eq!(result_layout.shape().dims(), &[1, 1, 2, 2]);

        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[12.0, 16.0, 24.0, 28.0]);
    }

    /// E2E: Conv2D with bias (3 inputs).
    #[test]
    fn pipelined_realize_conv2d_with_bias() {
        let x_storage = fuel_memory::from_slice_cpu(&[
            1.0_f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let w_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 1.0, 1.0, 1.0]);
        let bias_storage = fuel_memory::from_slice_cpu(&[100.0_f32]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, w_id, b_id, c_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 3, 3]), dtype: DType::F32,
            });
            let w = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1]), dtype: DType::F32,
            });
            let c = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::CONV2D,
                    fuel_graph::registry::FusedOpParams::Conv2D {
                        stride: (1, 1), padding: (0, 0), groups: 1,
                    },
                ),
                inputs: vec![x, w, b],
                shape: Shape::from_dims(&[1, 1, 2, 2]),
                dtype: DType::F32,
            });
            g.set_target_backend(c, BackendId::Cpu);
            (x, w, b, c)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(x_id, Arc::new(RwLock::new(x_storage)));
        inputs.insert(w_id, Arc::new(RwLock::new(w_storage)));
        inputs.insert(b_id, Arc::new(RwLock::new(bias_storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, c_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[112.0, 116.0, 124.0, 128.0]);
    }

    /// E2E: Conv2D in F64 — same 2x2 sum-kernel test as F32, on doubles.
    #[test]
    fn pipelined_realize_conv2d_f64() {
        let x_storage = fuel_memory::from_slice_cpu(&[
            1.0_f64, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let w_storage = fuel_memory::from_slice_cpu(&[1.0_f64, 1.0, 1.0, 1.0]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, w_id, c_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 3, 3]), dtype: DType::F64,
            });
            let w = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F64,
            });
            let c = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::CONV2D,
                    fuel_graph::registry::FusedOpParams::Conv2D {
                        stride: (1, 1), padding: (0, 0), groups: 1,
                    },
                ),
                inputs: vec![x, w],
                shape: Shape::from_dims(&[1, 1, 2, 2]),
                dtype: DType::F64,
            });
            g.set_target_backend(c, BackendId::Cpu);
            (x, w, c)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(x_id, Arc::new(RwLock::new(x_storage)));
        inputs.insert(w_id, Arc::new(RwLock::new(w_storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, c_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f64>().unwrap(), &[12.0, 16.0, 24.0, 28.0]);
    }

    /// E2E: Conv2D in BF16 — f32-accumulator path. Tolerant compare.
    #[test]
    fn pipelined_realize_conv2d_bf16() {
        let x_data: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            .iter().map(|v| half::bf16::from_f32(*v)).collect();
        let w_data: Vec<half::bf16> = [1.0_f32, 1.0, 1.0, 1.0]
            .iter().map(|v| half::bf16::from_f32(*v)).collect();
        let x_storage = fuel_memory::from_slice_cpu(&x_data);
        let w_storage = fuel_memory::from_slice_cpu(&w_data);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, w_id, c_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 3, 3]), dtype: DType::BF16,
            });
            let w = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::BF16,
            });
            let c = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::CONV2D,
                    fuel_graph::registry::FusedOpParams::Conv2D {
                        stride: (1, 1), padding: (0, 0), groups: 1,
                    },
                ),
                inputs: vec![x, w],
                shape: Shape::from_dims(&[1, 1, 2, 2]),
                dtype: DType::BF16,
            });
            g.set_target_backend(c, BackendId::Cpu);
            (x, w, c)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(x_id, Arc::new(RwLock::new(x_storage)));
        inputs.insert(w_id, Arc::new(RwLock::new(w_storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, c_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let got: Vec<f32> = c.as_slice::<half::bf16>().unwrap()
            .iter().map(|v| v.to_f32()).collect();
        let want = [12.0_f32, 16.0, 24.0, 28.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 0.5, "got {got:?} want {want:?}");
        }
    }

    /// E2E: Conv2D in F16 — f32-accumulator path. Tolerant compare.
    #[test]
    fn pipelined_realize_conv2d_f16() {
        let x_data: Vec<half::f16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            .iter().map(|v| half::f16::from_f32(*v)).collect();
        let w_data: Vec<half::f16> = [1.0_f32, 1.0, 1.0, 1.0]
            .iter().map(|v| half::f16::from_f32(*v)).collect();
        let x_storage = fuel_memory::from_slice_cpu(&x_data);
        let w_storage = fuel_memory::from_slice_cpu(&w_data);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, w_id, c_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 3, 3]), dtype: DType::F16,
            });
            let w = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F16,
            });
            let c = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::CONV2D,
                    fuel_graph::registry::FusedOpParams::Conv2D {
                        stride: (1, 1), padding: (0, 0), groups: 1,
                    },
                ),
                inputs: vec![x, w],
                shape: Shape::from_dims(&[1, 1, 2, 2]),
                dtype: DType::F16,
            });
            g.set_target_backend(c, BackendId::Cpu);
            (x, w, c)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(x_id, Arc::new(RwLock::new(x_storage)));
        inputs.insert(w_id, Arc::new(RwLock::new(w_storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, c_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let got: Vec<f32> = c.as_slice::<half::f16>().unwrap()
            .iter().map(|v| v.to_f32()).collect();
        let want = [12.0_f32, 16.0, 24.0, 28.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 0.05, "got {got:?} want {want:?}");
        }
    }

    /// E2E: Op::PagedAttn — F32, single-head, B=1, Sq=1.
    /// Same setup as the FlashAttn smoke test, but the K/V live in a
    /// paged cache that we look up via block_table.
    /// Layout:
    ///   block_size=2, num_blocks=1, max_blocks_per_seq=1.
    ///   k_cache shape [1, 2, 1, 2] = num_blocks × block_size × Hkv × D.
    ///   k_cache[block 0, slot 0, h 0] = [1, 0]
    ///   k_cache[block 0, slot 1, h 0] = [0, 1]
    ///   v_cache values: [10, 0] / [0, 10]
    ///   block_table[b=0, logical_block 0] = 0 (physical)
    ///   context_lens[0] = 2
    ///   q[0, 0, 0] = [2, 0]
    /// Causal is implicit (q_pos = ctx_len - Sq + sq = 2 - 1 + 0 = 1, both keys admissible).
    /// Same softmax math as FlashAttn → ~[8.808, 1.192].
    #[test]
    fn pipelined_realize_paged_attn_f32() {
        let q = fuel_memory::from_slice_cpu(&[2.0_f32, 0.0]);
        let k_cache = fuel_memory::from_slice_cpu(&[1.0_f32, 0.0, 0.0, 1.0]);
        let v_cache = fuel_memory::from_slice_cpu(&[10.0_f32, 0.0, 0.0, 10.0]);
        let block_table_u32 = fuel_memory::Storage::new(
            fuel_memory::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_slice(&[0_u32]),
            ),
            DType::U32,
        );
        let context_lens_u32 = fuel_memory::Storage::new(
            fuel_memory::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_slice(&[2_u32]),
            ),
            DType::U32,
        );
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (q_id, kc_id, vc_id, bt_id, cl_id, op_id) = {
            let mut g = graph.write().unwrap();
            let q = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 1, 2]), dtype: DType::F32,
            });
            let kc = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 2, 1, 2]), dtype: DType::F32,
            });
            let vc = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 2, 1, 2]), dtype: DType::F32,
            });
            let bt = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1]), dtype: DType::U32,
            });
            let cl = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1]), dtype: DType::U32,
            });
            let op = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::PAGED_ATTN,
                    fuel_graph::registry::FusedOpParams::PagedAttn {
                        softmax_scale: 1.0, block_size: 2, softcap: None,
                    },
                ),
                inputs: vec![q, kc, vc, bt, cl],
                shape: Shape::from_dims(&[1, 1, 1, 2]), dtype: DType::F32,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (q, kc, vc, bt, cl, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(q_id, Arc::new(RwLock::new(q)));
        inputs.insert(kc_id, Arc::new(RwLock::new(k_cache)));
        inputs.insert(vc_id, Arc::new(RwLock::new(v_cache)));
        inputs.insert(bt_id, Arc::new(RwLock::new(block_table_u32)));
        inputs.insert(cl_id, Arc::new(RwLock::new(context_lens_u32)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r = c.as_slice::<f32>().unwrap();
        let expected_p0 = 2.0_f32.exp() / (2.0_f32.exp() + 1.0);
        let expected_p1 = 1.0_f32 / (2.0_f32.exp() + 1.0);
        assert!((r[0] - 10.0 * expected_p0).abs() < 1e-5,
            "row[0]: got {} expected {}", r[0], 10.0 * expected_p0);
        assert!((r[1] - 10.0 * expected_p1).abs() < 1e-5,
            "row[1]: got {} expected {}", r[1], 10.0 * expected_p1);
    }

    /// E2E: PagedAttn BF16 — same single-row test, tolerant.
    #[test]
    fn pipelined_realize_paged_attn_bf16() {
        let q_v: Vec<half::bf16> = [2.0_f32, 0.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let kc_v: Vec<half::bf16> = [1.0_f32, 0.0, 0.0, 1.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let vc_v: Vec<half::bf16> = [10.0_f32, 0.0, 0.0, 10.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let q = fuel_memory::from_slice_cpu(&q_v);
        let k_cache = fuel_memory::from_slice_cpu(&kc_v);
        let v_cache = fuel_memory::from_slice_cpu(&vc_v);
        let bt = fuel_memory::Storage::new(
            fuel_memory::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_slice(&[0_u32]),
            ),
            DType::U32,
        );
        let cl = fuel_memory::Storage::new(
            fuel_memory::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_slice(&[2_u32]),
            ),
            DType::U32,
        );
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (q_id, kc_id, vc_id, bt_id, cl_id, op_id) = {
            let mut g = graph.write().unwrap();
            let q = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 1, 2]), dtype: DType::BF16,
            });
            let kc = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 2, 1, 2]), dtype: DType::BF16,
            });
            let vc = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 2, 1, 2]), dtype: DType::BF16,
            });
            let bt = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1]), dtype: DType::U32,
            });
            let cl = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1]), dtype: DType::U32,
            });
            let op = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::PAGED_ATTN,
                    fuel_graph::registry::FusedOpParams::PagedAttn {
                        softmax_scale: 1.0, block_size: 2, softcap: None,
                    },
                ),
                inputs: vec![q, kc, vc, bt, cl],
                shape: Shape::from_dims(&[1, 1, 1, 2]), dtype: DType::BF16,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (q, kc, vc, bt, cl, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(q_id, Arc::new(RwLock::new(q)));
        inputs.insert(kc_id, Arc::new(RwLock::new(k_cache)));
        inputs.insert(vc_id, Arc::new(RwLock::new(v_cache)));
        inputs.insert(bt_id, Arc::new(RwLock::new(bt)));
        inputs.insert(cl_id, Arc::new(RwLock::new(cl)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let got: Vec<f32> = c.as_slice::<half::bf16>().unwrap().iter().map(|v| v.to_f32()).collect();
        let expected_p0 = 2.0_f32.exp() / (2.0_f32.exp() + 1.0);
        let expected_p1 = 1.0_f32 / (2.0_f32.exp() + 1.0);
        let want = [10.0 * expected_p0, 10.0 * expected_p1];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 0.5, "got {got:?} want {want:?}");
        }
    }

    /// E2E: Op::FlashAttn — F32 single-head, single-batch, no mask.
    /// q = [[2.0, 0.0]], k = [[1.0, 0.0], [0.0, 1.0]], v = [[10, 0], [0, 10]]
    /// scale = 1.0
    /// scores = q · kᵀ = [2.0, 0.0]
    /// softmax = [e^2/(e^2+1), 1/(e^2+1)] ≈ [0.8808, 0.1192]
    /// out = softmax @ v = [10*0.8808, 10*0.1192] ≈ [8.808, 1.192]
    #[test]
    fn pipelined_realize_flash_attn_f32() {
        // [B=1, H=1, S=1or2, D=2]
        let q = fuel_memory::from_slice_cpu(&[2.0_f32, 0.0]);
        let k = fuel_memory::from_slice_cpu(&[1.0_f32, 0.0, 0.0, 1.0]);
        let v = fuel_memory::from_slice_cpu(&[10.0_f32, 0.0, 0.0, 10.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (q_id, k_id, v_id, op_id) = {
            let mut g = graph.write().unwrap();
            let q = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 1, 2]), dtype: DType::F32,
            });
            let k = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            let v = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            let op = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::FLASH_ATTN,
                    fuel_graph::registry::FusedOpParams::FlashAttn {
                        softmax_scale: 1.0,
                        causal: false,
                        window_size_left: None,
                        window_size_right: None,
                        softcap: None,
                        k_len: None,
                    },
                ),
                inputs: vec![q, k, v],
                shape: Shape::from_dims(&[1, 1, 1, 2]),
                dtype: DType::F32,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (q, k, v, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(q_id, Arc::new(RwLock::new(q)));
        inputs.insert(k_id, Arc::new(RwLock::new(k)));
        inputs.insert(v_id, Arc::new(RwLock::new(v)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r = c.as_slice::<f32>().unwrap();
        let expected_p0 = 2.0_f32.exp() / (2.0_f32.exp() + 1.0);
        let expected_p1 = 1.0_f32 / (2.0_f32.exp() + 1.0);
        assert!((r[0] - 10.0 * expected_p0).abs() < 1e-5,
            "row[0]: got {} expected {}", r[0], 10.0 * expected_p0);
        assert!((r[1] - 10.0 * expected_p1).abs() < 1e-5,
            "row[1]: got {} expected {}", r[1], 10.0 * expected_p1);
    }

    /// E2E: FlashAttn with causal mask — second query position
    /// attends to both keys (positions 0 and 1), first only attends
    /// to key 0 (everything beyond is masked).
    #[test]
    fn pipelined_realize_flash_attn_causal_f32() {
        // q [1,1,2,2]: query 0 = [1, 0], query 1 = [0, 1]
        // k [1,1,2,2]: keys = [[1, 0], [0, 1]]
        // v [1,1,2,2]: values = [[5, 6], [7, 8]]
        // softmax_scale=1, causal:
        //   query 0: only key 0 admissible → out = v[0] = [5, 6]
        //   query 1: both admissible. scores = q1·k = [0, 1]
        //            softmax = [1/(e+1), e/(e+1)]
        //            out = scores · v = [(5)/(e+1) + 7e/(e+1), 6/(e+1) + 8e/(e+1)]
        let q = fuel_memory::from_slice_cpu(&[1.0_f32, 0.0, 0.0, 1.0]);
        let k = fuel_memory::from_slice_cpu(&[1.0_f32, 0.0, 0.0, 1.0]);
        let v = fuel_memory::from_slice_cpu(&[5.0_f32, 6.0, 7.0, 8.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (q_id, k_id, v_id, op_id) = {
            let mut g = graph.write().unwrap();
            let q = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            let k = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            let v = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            let op = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::FLASH_ATTN,
                    fuel_graph::registry::FusedOpParams::FlashAttn {
                        softmax_scale: 1.0, causal: true,
                        window_size_left: None, window_size_right: None,
                        softcap: None,
                        k_len: None,
                    },
                ),
                inputs: vec![q, k, v],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (q, k, v, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(q_id, Arc::new(RwLock::new(q)));
        inputs.insert(k_id, Arc::new(RwLock::new(k)));
        inputs.insert(v_id, Arc::new(RwLock::new(v)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r = c.as_slice::<f32>().unwrap();
        // Query 0 sees only key 0 → output = v[0]
        assert!((r[0] - 5.0).abs() < 1e-5, "got {}", r[0]);
        assert!((r[1] - 6.0).abs() < 1e-5, "got {}", r[1]);
        // Query 1 sees both. softmax([0, 1]) = [1/(e+1), e/(e+1)]
        let denom = (1.0_f32).exp() + 1.0;
        let expected_a = 5.0 / denom + 7.0 * (1.0_f32).exp() / denom;
        let expected_b = 6.0 / denom + 8.0 * (1.0_f32).exp() / denom;
        assert!((r[2] - expected_a).abs() < 1e-5, "row1[0]: got {} expected {}", r[2], expected_a);
        assert!((r[3] - expected_b).abs() < 1e-5, "row1[1]: got {} expected {}", r[3], expected_b);
    }

    /// E2E: FlashAttn over a fixed-**capacity** K/V with a runtime
    /// `k_len` resolved through a `SymEnv` (Phase D symbolic extents).
    /// Only the first `k_len` rows are attended; trailing "poison" rows
    /// in the capacity buffer are ignored, and the causal mask is
    /// bottom-right-aligned at offset `k_len - Sq`. This is flash decode
    /// over a persistent KV-cache: K/V are `[.., max_seq, ..]` capacity
    /// and the live prefix grows per token.
    #[test]
    fn pipelined_realize_flash_attn_dynamic_k_len() {
        use fuel_ir::{DynScalar, SymId};
        // q [1,1,2,2]; K/V capacity [1,1,4,2]; bind k_len = 3 so row 3
        // (the "poison" row) is never attended.
        let q = fuel_memory::from_slice_cpu(&[1.0_f32, 0.0,  0.0, 1.0]);
        let k = fuel_memory::from_slice_cpu(&[
            1.0_f32, 0.0,
            0.0,     1.0,
            1.0,     1.0,
            100.0,   100.0,   // poison row 3 — must be ignored (k_len=3)
        ]);
        let v = fuel_memory::from_slice_cpu(&[
            5.0_f32, 6.0,
            7.0,     8.0,
            9.0,     10.0,
            999.0,   999.0,   // poison row 3 — must be ignored
        ]);
        let sym = SymId(0);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (q_id, k_id, v_id, op_id) = {
            let mut g = graph.write().unwrap();
            let q = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            let k = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 4, 2]), dtype: DType::F32,
            });
            let v = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 4, 2]), dtype: DType::F32,
            });
            let op = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::FLASH_ATTN,
                    fuel_graph::registry::FusedOpParams::FlashAttn {
                        softmax_scale: 1.0, causal: true,
                        window_size_left: None, window_size_right: None,
                        softcap: None,
                        k_len: Some(DynScalar::Sym(sym)),
                    },
                ),
                inputs: vec![q, k, v],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (q, k, v, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(q_id, Arc::new(RwLock::new(q)));
        inputs.insert(k_id, Arc::new(RwLock::new(k)));
        inputs.insert(v_id, Arc::new(RwLock::new(v)));
        let mut env = SymEnv::new();
        env.bind(sym, 3).unwrap();
        let (result_arc, _) =
            PipelinedExecutor::realize_with_env(graph, op_id, inputs, env)
                .expect("realize_with_env");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let r = c.as_slice::<f32>().unwrap();
        let e = 1.0_f32.exp();
        // causal_offset = k_len - Sq = 3 - 2 = 1.
        // Query 0 (abs pos 1): keys {0,1}. scores [1,0] → softmax [e,1]/(e+1).
        let d0 = e + 1.0;
        let e0x = (5.0 * e + 7.0) / d0;
        let e0y = (6.0 * e + 8.0) / d0;
        // Query 1 (abs pos 2): keys {0,1,2}. scores [0,1,1] → [1,e,e]/(1+2e).
        let d1 = 1.0 + 2.0 * e;
        let e1x = (5.0 + 7.0 * e + 9.0 * e) / d1;
        let e1y = (6.0 + 8.0 * e + 10.0 * e) / d1;
        for (got, want) in r.iter().zip([e0x, e0y, e1x, e1y].iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "flash dyn k_len: got {got}, want {want} (full output {r:?}); \
                 a poison-contaminated value means the capacity rows past k_len leaked in",
            );
        }
    }

    /// A flash `k_len` symbol unbound in the `SymEnv` surfaces a typed
    /// error at realize (never a panic).
    #[test]
    fn pipelined_realize_flash_attn_dynamic_k_len_unbound_errors() {
        use fuel_ir::{DynScalar, SymId};
        let q = fuel_memory::from_slice_cpu(&[1.0_f32, 0.0, 0.0, 1.0]);
        let k = fuel_memory::from_slice_cpu(&[0.0_f32; 16]);
        let v = fuel_memory::from_slice_cpu(&[0.0_f32; 16]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (q_id, k_id, v_id, op_id) = {
            let mut g = graph.write().unwrap();
            let q = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32 });
            let k = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[1, 1, 4, 2]), dtype: DType::F32 });
            let v = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[1, 1, 4, 2]), dtype: DType::F32 });
            let op = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::FLASH_ATTN,
                    fuel_graph::registry::FusedOpParams::FlashAttn {
                        softmax_scale: 1.0, causal: true,
                        window_size_left: None, window_size_right: None,
                        softcap: None,
                        k_len: Some(DynScalar::Sym(SymId(0))),
                    },
                ),
                inputs: vec![q, k, v],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (q, k, v, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(q_id, Arc::new(RwLock::new(q)));
        inputs.insert(k_id, Arc::new(RwLock::new(k)));
        inputs.insert(v_id, Arc::new(RwLock::new(v)));
        // Empty env — the k_len symbol is unbound.
        let result = PipelinedExecutor::realize_with_env(graph, op_id, inputs, SymEnv::new());
        assert!(result.is_err(), "unbound flash k_len symbol must surface a typed error");
    }

    /// E2E: FlashAttn BF16 — same single-row test as f32, tolerant.
    #[test]
    fn pipelined_realize_flash_attn_bf16() {
        let q_v: Vec<half::bf16> = [2.0_f32, 0.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let k_v: Vec<half::bf16> = [1.0_f32, 0.0, 0.0, 1.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let v_v: Vec<half::bf16> = [10.0_f32, 0.0, 0.0, 10.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let q = fuel_memory::from_slice_cpu(&q_v);
        let k = fuel_memory::from_slice_cpu(&k_v);
        let v = fuel_memory::from_slice_cpu(&v_v);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (q_id, k_id, v_id, op_id) = {
            let mut g = graph.write().unwrap();
            let q = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 1, 2]), dtype: DType::BF16,
            });
            let k = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::BF16,
            });
            let v = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::BF16,
            });
            let op = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::FLASH_ATTN,
                    fuel_graph::registry::FusedOpParams::FlashAttn {
                        softmax_scale: 1.0, causal: false,
                        window_size_left: None, window_size_right: None,
                        softcap: None,
                        k_len: None,
                    },
                ),
                inputs: vec![q, k, v],
                shape: Shape::from_dims(&[1, 1, 1, 2]), dtype: DType::BF16,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (q, k, v, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(q_id, Arc::new(RwLock::new(q)));
        inputs.insert(k_id, Arc::new(RwLock::new(k)));
        inputs.insert(v_id, Arc::new(RwLock::new(v)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let got: Vec<f32> = c.as_slice::<half::bf16>().unwrap().iter().map(|v| v.to_f32()).collect();
        let expected_p0 = 2.0_f32.exp() / (2.0_f32.exp() + 1.0);
        let expected_p1 = 1.0_f32 / (2.0_f32.exp() + 1.0);
        let want = [10.0 * expected_p0, 10.0 * expected_p1];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 0.5, "got {got:?} want {want:?}");
        }
    }

    /// E2E: Op::FusedLinear — F32. a[1,2,3] @ b[1,3,2] + bias[2].
    /// a = [[[1,2,3],[4,5,6]]], b = [[[1,0],[0,1],[1,1]]], bias=[10,20]
    /// matmul = [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4, 5], [10, 11]]
    /// + bias = [[14, 25], [20, 31]]
    #[test]
    fn pipelined_realize_fused_linear_f32() {
        let a = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = fuel_memory::from_slice_cpu(&[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let bias = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, b_id, bias_id, op_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 2, 3]), dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 3, 2]), dtype: DType::F32,
            });
            let bias = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let op = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::FUSED_LINEAR,
                    fuel_graph::registry::FusedOpParams::FusedLinear,
                ),
                inputs: vec![a, b, bias],
                shape: Shape::from_dims(&[1, 2, 2]), dtype: DType::F32,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (a, b, bias, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(a_id, Arc::new(RwLock::new(a)));
        inputs.insert(b_id, Arc::new(RwLock::new(b)));
        inputs.insert(bias_id, Arc::new(RwLock::new(bias)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[14.0, 25.0, 20.0, 31.0]);
    }

    /// E2E: FusedLinear F64 — same shape test on doubles.
    #[test]
    fn pipelined_realize_fused_linear_f64() {
        let a = fuel_memory::from_slice_cpu(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = fuel_memory::from_slice_cpu(&[1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let bias = fuel_memory::from_slice_cpu(&[10.0_f64, 20.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, b_id, bias_id, op_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 2, 3]), dtype: DType::F64,
            });
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 3, 2]), dtype: DType::F64,
            });
            let bias = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F64,
            });
            let op = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::FUSED_LINEAR,
                    fuel_graph::registry::FusedOpParams::FusedLinear,
                ),
                inputs: vec![a, b, bias],
                shape: Shape::from_dims(&[1, 2, 2]), dtype: DType::F64,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (a, b, bias, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(a_id, Arc::new(RwLock::new(a)));
        inputs.insert(b_id, Arc::new(RwLock::new(b)));
        inputs.insert(bias_id, Arc::new(RwLock::new(bias)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f64>().unwrap(), &[14.0, 25.0, 20.0, 31.0]);
    }

    /// E2E: FusedLinear BF16 — tolerant compare.
    #[test]
    fn pipelined_realize_fused_linear_bf16() {
        let a_v: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let b_v: Vec<half::bf16> = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let bias_v: Vec<half::bf16> = [10.0_f32, 20.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let a = fuel_memory::from_slice_cpu(&a_v);
        let b = fuel_memory::from_slice_cpu(&b_v);
        let bias = fuel_memory::from_slice_cpu(&bias_v);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, b_id, bias_id, op_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 2, 3]), dtype: DType::BF16,
            });
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 3, 2]), dtype: DType::BF16,
            });
            let bias = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::BF16,
            });
            let op = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::FUSED_LINEAR,
                    fuel_graph::registry::FusedOpParams::FusedLinear,
                ),
                inputs: vec![a, b, bias],
                shape: Shape::from_dims(&[1, 2, 2]), dtype: DType::BF16,
            });
            g.set_target_backend(op, BackendId::Cpu);
            (a, b, bias, op)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(a_id, Arc::new(RwLock::new(a)));
        inputs.insert(b_id, Arc::new(RwLock::new(b)));
        inputs.insert(bias_id, Arc::new(RwLock::new(bias)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let got: Vec<f32> = c.as_slice::<half::bf16>().unwrap().iter().map(|v| v.to_f32()).collect();
        let want = [14.0_f32, 25.0, 20.0, 31.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 0.5, "got {got:?} want {want:?}");
        }
    }

    /// E2E: Op::ReduceSumTo — sum the leading axis of a [2,3] tensor.
    /// Input [[1,2,3],[4,5,6]] → output [5,7,9].
    #[test]
    fn pipelined_realize_reduce_sum_to_f32_drops_leading_axis() {
        let v = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let op_id = g.push(Node {
                op: Op::ReduceSumTo(Shape::from_dims(&[3])),
                inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(v)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[5.0, 7.0, 9.0]);
    }

    /// E2E: Op::ReduceSumTo — keep-dim with 1 in the middle.
    /// Input [2,3,4] → [2,1,4] sums along dim 1.
    #[test]
    fn pipelined_realize_reduce_sum_to_f32_keepdim_middle() {
        // [2,3,4]: layer 0 = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        //          layer 1 = [[13..16],[17..20],[21..24]]
        let mut v: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let _ = &mut v;
        let s = fuel_memory::from_slice_cpu(&v);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3, 4]), dtype: DType::F32,
            });
            let op_id = g.push(Node {
                op: Op::ReduceSumTo(Shape::from_dims(&[2, 1, 4])),
                inputs: vec![in_id],
                shape: Shape::from_dims(&[2, 1, 4]), dtype: DType::F32,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(s)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        // layer0 dim1-sum: col j = 1+5+9, 2+6+10, 3+7+11, 4+8+12 = [15,18,21,24]
        // layer1 dim1-sum: col j = 13+17+21, 14+18+22, 15+19+23, 16+20+24 = [51,54,57,60]
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[15.0, 18.0, 21.0, 24.0, 51.0, 54.0, 57.0, 60.0],
        );
    }

    /// E2E: ReduceSumTo F64 — same drop-leading-axis test on doubles.
    #[test]
    fn pipelined_realize_reduce_sum_to_f64() {
        let v = fuel_memory::from_slice_cpu(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F64,
            });
            let op_id = g.push(Node {
                op: Op::ReduceSumTo(Shape::from_dims(&[3])),
                inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F64,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(v)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
    }

    /// E2E: Op::ReduceMaxTo F32 — drop the leading axis with max-reduce.
    #[test]
    fn pipelined_realize_reduce_max_to_f32_drops_leading_axis() {
        // Input [2,3]: row 0 = [1, 7, 3], row 1 = [4, 2, 6]. Max along
        // dim 0: [4, 7, 6].
        let v = fuel_memory::from_slice_cpu(&[1.0_f32, 7.0, 3.0, 4.0, 2.0, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let op_id = g.push(Node {
                op: Op::ReduceMaxTo(Shape::from_dims(&[3])),
                inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(v)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[4.0, 7.0, 6.0]);
    }

    /// E2E: Op::ReduceMaxTo F32 — keep-dim with 1 in the trailing axis.
    /// Mirrors the SoftmaxLastDim lowering's max-side shape: input
    /// [..., last] → [..., 1].
    #[test]
    fn pipelined_realize_reduce_max_to_f32_keepdim_trailing() {
        // Input [2, 3]: row maxes = [3, 6].
        let v = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let op_id = g.push(Node {
                op: Op::ReduceMaxTo(Shape::from_dims(&[2, 1])),
                inputs: vec![in_id],
                shape: Shape::from_dims(&[2, 1]), dtype: DType::F32,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(v)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[3.0, 6.0]);
    }

    /// E2E: ReduceSumTo BF16 — tolerant compare via f32-acc.
    #[test]
    fn pipelined_realize_reduce_sum_to_bf16() {
        let v: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let s = fuel_memory::from_slice_cpu(&v);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::BF16,
            });
            let op_id = g.push(Node {
                op: Op::ReduceSumTo(Shape::from_dims(&[3])),
                inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::BF16,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(s)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let got: Vec<f32> = c.as_slice::<half::bf16>().unwrap().iter().map(|v| v.to_f32()).collect();
        let want = [5.0_f32, 7.0, 9.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 0.5, "got {got:?} want {want:?}");
        }
    }

    /// E2E: Op::ConvTranspose2D — F32 spread test.
    /// x = [[1, 2], [3, 4]] shape [1,1,2,2], all-ones kernel
    /// shape [1,1,2,2], stride=1, padding=0, dilation=1, no bias.
    /// Expected output (3x3):
    ///   [[1,  3, 2],
    ///    [4, 10, 6],
    ///    [3,  7, 4]]
    #[test]
    fn pipelined_realize_conv_transpose2d_f32() {
        let x_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let w_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 1.0, 1.0, 1.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, w_id, c_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            let w = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F32,
            });
            let c = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::CONV_TRANSPOSE2D,
                    fuel_graph::registry::FusedOpParams::ConvTranspose2D {
                        stride: (1, 1), padding: (0, 0),
                        output_padding: (0, 0), dilation: (1, 1), groups: 1,
                    },
                ),
                inputs: vec![x, w],
                shape: Shape::from_dims(&[1, 1, 3, 3]),
                dtype: DType::F32,
            });
            g.set_target_backend(c, BackendId::Cpu);
            (x, w, c)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(x_id, Arc::new(RwLock::new(x_storage)));
        inputs.insert(w_id, Arc::new(RwLock::new(w_storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, c_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0],
        );
    }

    /// E2E: ConvTranspose2D F64 — same shape test.
    #[test]
    fn pipelined_realize_conv_transpose2d_f64() {
        let x_storage = fuel_memory::from_slice_cpu(&[1.0_f64, 2.0, 3.0, 4.0]);
        let w_storage = fuel_memory::from_slice_cpu(&[1.0_f64, 1.0, 1.0, 1.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, w_id, c_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F64,
            });
            let w = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::F64,
            });
            let c = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::CONV_TRANSPOSE2D,
                    fuel_graph::registry::FusedOpParams::ConvTranspose2D {
                        stride: (1, 1), padding: (0, 0),
                        output_padding: (0, 0), dilation: (1, 1), groups: 1,
                    },
                ),
                inputs: vec![x, w],
                shape: Shape::from_dims(&[1, 1, 3, 3]),
                dtype: DType::F64,
            });
            g.set_target_backend(c, BackendId::Cpu);
            (x, w, c)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(x_id, Arc::new(RwLock::new(x_storage)));
        inputs.insert(w_id, Arc::new(RwLock::new(w_storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, c_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(
            c.as_slice::<f64>().unwrap(),
            &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0],
        );
    }

    /// E2E: ConvTranspose2D BF16 — tolerant compare via f32-acc.
    #[test]
    fn pipelined_realize_conv_transpose2d_bf16() {
        let x: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter().map(|v| half::bf16::from_f32(*v)).collect();
        let w: Vec<half::bf16> = [1.0_f32, 1.0, 1.0, 1.0]
            .iter().map(|v| half::bf16::from_f32(*v)).collect();
        let x_storage = fuel_memory::from_slice_cpu(&x);
        let w_storage = fuel_memory::from_slice_cpu(&w);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, w_id, c_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::BF16,
            });
            let w = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 1, 2, 2]), dtype: DType::BF16,
            });
            let c = g.push(Node {
                op: Op::Fused(
                    fuel_graph::registry::FusedOps::CONV_TRANSPOSE2D,
                    fuel_graph::registry::FusedOpParams::ConvTranspose2D {
                        stride: (1, 1), padding: (0, 0),
                        output_padding: (0, 0), dilation: (1, 1), groups: 1,
                    },
                ),
                inputs: vec![x, w],
                shape: Shape::from_dims(&[1, 1, 3, 3]),
                dtype: DType::BF16,
            });
            g.set_target_backend(c, BackendId::Cpu);
            (x, w, c)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(x_id, Arc::new(RwLock::new(x_storage)));
        inputs.insert(w_id, Arc::new(RwLock::new(w_storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, c_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let got: Vec<f32> = c.as_slice::<half::bf16>().unwrap().iter().map(|v| v.to_f32()).collect();
        let want = [1.0_f32, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 0.5, "got {got:?} want {want:?}");
        }
    }

    /// E2E: GQA-style matmul through the pipelined executor.
    /// lhs has 4 batch heads, rhs has 2; each rhs head is shared
    /// by 2 lhs heads. Output's batch dim follows lhs (4 heads).
    #[test]
    fn pipelined_realize_matmul_gqa() {
        // lhs [4, 1, 2]: heads 0..3 are [[1,2]], [[3,4]], [[5,6]], [[7,8]]
        // rhs [2, 2, 1]: heads 0,1 are [[1],[0]], [[0],[1]]
        // Expected out [4, 1, 1]: [[1]], [[3]], [[6]], [[8]]
        let lhs_storage = fuel_memory::from_slice_cpu(&[
            1.0_f32, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ]);
        let rhs_storage = fuel_memory::from_slice_cpu(&[
            1.0_f32, 0.0,
            0.0, 1.0,
        ]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (lhs_id, rhs_id, mm_id) = {
            let mut g = graph.write().unwrap();
            let lhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[4, 1, 2]), dtype: DType::F32,
            });
            let rhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 2, 1]), dtype: DType::F32,
            });
            let mm = g.push(Node {
                op: Op::MatMul, inputs: vec![lhs, rhs],
                shape: Shape::from_dims(&[4, 1, 1]), dtype: DType::F32,
            });
            g.set_target_backend(mm, BackendId::Cpu);
            (lhs, rhs, mm)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(lhs_id, Arc::new(RwLock::new(lhs_storage)));
        inputs.insert(rhs_id, Arc::new(RwLock::new(rhs_storage)));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, mm_id, inputs).expect("realize");
        assert_eq!(result_layout.shape().dims(), &[4, 1, 1]);

        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[1.0, 3.0, 6.0, 8.0]);
    }

    /// E2E: matmul with a transposed rhs — proves stage 3+4
    /// integration carries through the matmul path. The transpose
    /// is metadata-only; auto-Contiguize materializes the strided
    /// rhs before the matmul kernel sees it.
    #[test]
    fn pipelined_realize_matmul_with_transposed_rhs() {
        // lhs [[1, 2], [3, 4]], rhs original [[5, 6], [7, 8]]
        // rhs.T = [[5, 7], [6, 8]]
        // lhs @ rhs.T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
        //             = [[17, 23], [39, 53]]
        let lhs_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let rhs_storage = fuel_memory::from_slice_cpu(&[5.0_f32, 6.0, 7.0, 8.0]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (lhs_id, rhs_id, t_id, mm_id) = {
            let mut g = graph.write().unwrap();
            let lhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::F32,
            });
            let rhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::F32,
            });
            let t = g.push(Node {
                op: Op::Transpose, inputs: vec![rhs],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::F32,
            });
            let mm = g.push(Node {
                op: Op::MatMul, inputs: vec![lhs, t],
                shape: Shape::from_dims(&[2, 2]), dtype: DType::F32,
            });
            g.set_target_backend(mm, BackendId::Cpu);
            (lhs, rhs, t, mm)
        };
        let _ = t_id;
        let mut inputs = StorageCache::new();
        inputs.insert(lhs_id, Arc::new(RwLock::new(lhs_storage)));
        inputs.insert(rhs_id, Arc::new(RwLock::new(rhs_storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, mm_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[17.0, 23.0, 39.0, 53.0]);
    }

    /// E2E: Const + Reshape — contiguous-input reshape is zero
    /// copy. The output Storage Arc is the input Arc; the layout
    /// is contiguous in the new shape.
    #[test]
    fn pipelined_realize_reshape_zero_copy_when_contiguous() {
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, r_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let r_id = g.push(Node {
                op: Op::Reshape(Shape::from_dims(&[3, 2])),
                inputs: vec![in_id],
                shape: Shape::from_dims(&[3, 2]),
                dtype: DType::F32,
            });
            (in_id, r_id)
        };
        let mut inputs = StorageCache::new();
        let in_arc = Arc::new(RwLock::new(storage));
        inputs.insert(in_id, Arc::clone(&in_arc));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, r_id, inputs).expect("realize");

        // Zero copy — same Arc.
        assert!(Arc::ptr_eq(&result_arc, &in_arc), "contiguous reshape must zero-copy");
        assert_eq!(result_layout.shape().dims(), &[3, 2]);
        assert!(result_layout.is_contiguous());

        // Bytes are unchanged; just reinterpreted as [3, 2].
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// E2E: Const + Transpose + Reshape — reshape on a strided
    /// input auto-contiguizes the bytes. Output Arc is fresh
    /// (NOT the input Arc); the bytes are the materialized
    /// transposed layout.
    #[test]
    fn pipelined_realize_reshape_materializes_when_strided() {
        // shape [2, 3]: 1 2 3 / 4 5 6
        // Transpose → [3, 2] strided
        // Reshape → [6] (forces materialization)
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, t_id, r_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let t_id = g.push(Node {
                op: Op::Transpose, inputs: vec![in_id],
                shape: Shape::from_dims(&[3, 2]), dtype: DType::F32,
            });
            let r_id = g.push(Node {
                op: Op::Reshape(Shape::from_dims(&[6])), inputs: vec![t_id],
                shape: Shape::from_dims(&[6]), dtype: DType::F32,
            });
            (in_id, t_id, r_id)
        };
        let _ = t_id;
        let mut inputs = StorageCache::new();
        let in_arc = Arc::new(RwLock::new(storage));
        inputs.insert(in_id, Arc::clone(&in_arc));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, r_id, inputs).expect("realize");

        // Fresh Arc — auto-contiguize allocated new bytes.
        assert!(!Arc::ptr_eq(&result_arc, &in_arc));
        assert_eq!(result_layout.shape().dims(), &[6]);
        assert!(result_layout.is_contiguous());

        // Materialized transposed bytes flattened: [1, 4, 2, 5, 3, 6].
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    /// E2E: Const + Transpose + SumDim — exercises stage 3
    /// (metadata-only Transpose, strided intermediate Layout) +
    /// stage 4 (auto-Contiguize before reduce kernel) end-to-end.
    /// The transpose makes the intermediate non-contiguous; the
    /// reduce wrapper would have failed in stage 2; with stage 4's
    /// auto-Contiguize, the kernel sees the materialized contiguous
    /// transposed bytes and produces the right answer.
    #[test]
    fn pipelined_realize_transpose_then_sum_dim_e2e() {
        // shape [2, 3]: rows are [1, 2, 3], [4, 5, 6]
        // After transpose: shape [3, 2], rows are [1, 4], [2, 5], [3, 6]
        // After SumDim(1): shape [3], values [5, 7, 9]
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, t_id, sum_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let t_id = g.push(Node {
                op: Op::Transpose, inputs: vec![in_id],
                shape: Shape::from_dims(&[3, 2]), dtype: DType::F32,
            });
            let sum_id = g.push(Node {
                op: Op::SumDim(1), inputs: vec![t_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            // Only the reduce kernel runs on the backend; the
            // transpose is metadata-only and doesn't need a target.
            g.set_target_backend(sum_id, BackendId::Cpu);
            (in_id, t_id, sum_id)
        };
        let _ = t_id;
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let (result_arc, result_layout) =
            PipelinedExecutor::realize(graph, sum_id, inputs).expect("realize");

        // The reduce output is contiguous (kernel-produced).
        assert_eq!(result_layout.shape().dims(), &[3]);
        assert!(result_layout.is_contiguous());

        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let typed: &[f32] = c.as_slice().unwrap();
        assert_eq!(typed, &[5.0, 7.0, 9.0]);
    }

    /// E2E: Const + BroadcastTo + Add — broadcast intermediate
    /// auto-contiguizes for the Add kernel; the result is the
    /// expected sum.
    #[test]
    fn pipelined_realize_broadcast_then_add_e2e() {
        // shape [3]: [10, 20, 30]
        // BroadcastTo [2, 3]: [[10, 20, 30], [10, 20, 30]]
        // Plus shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
        // Result: [[11, 22, 33], [14, 25, 36]]
        let bc_input = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);
        let plus_input = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (bc_in_id, plus_in_id, b_id, add_id) = {
            let mut g = graph.write().unwrap();
            let bc_in = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let plus_in = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::BroadcastTo(Shape::from_dims(&[2, 3])),
                inputs: vec![bc_in],
                shape: Shape::from_dims(&[2, 3]),
                dtype: DType::F32,
            });
            let add = g.push(Node {
                op: Op::Add,
                inputs: vec![b, plus_in],
                shape: Shape::from_dims(&[2, 3]),
                dtype: DType::F32,
            });
            g.set_target_backend(add, BackendId::Cpu);
            (bc_in, plus_in, b, add)
        };
        let _ = b_id;
        let mut inputs = StorageCache::new();
        inputs.insert(bc_in_id, Arc::new(RwLock::new(bc_input)));
        inputs.insert(plus_in_id, Arc::new(RwLock::new(plus_input)));

        let (result_arc, _result_layout) =
            PipelinedExecutor::realize(graph, add_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let typed: &[f32] = c.as_slice().unwrap();
        assert_eq!(typed, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    /// E2E: Const + SumDim — verifies that `OpParams::Reduce`
    /// flows from the graph (input shape via `op_to_op_params`)
    /// through compile_one and reaches the reduce kernel.
    #[test]
    fn pipelined_realize_sum_dim() {
        // shape [2, 3]; reduce dim 1 → output shape [2]
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, sum_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let sum_id = g.push(Node {
                op: Op::SumDim(1), inputs: vec![in_id],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            g.set_target_backend(sum_id, BackendId::Cpu);
            (in_id, sum_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, sum_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            assert_eq!(typed, &[6.0, 15.0]);
        }
    }

    /// E2E: SumAll on a rank-3 input, exercising the all-dims branch
    /// of `op_to_op_params` (every dim reduced, rank-0 output).
    #[test]
    fn pipelined_realize_sum_all_rank3() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let storage = fuel_memory::from_slice_cpu(&data);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, sum_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3, 4]), dtype: DType::F32,
            });
            let sum_id = g.push(Node {
                op: Op::SumAll, inputs: vec![in_id],
                shape: Shape::from_dims(&[]), dtype: DType::F32,
            });
            g.set_target_backend(sum_id, BackendId::Cpu);
            (in_id, sum_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, sum_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            // 1 + 2 + ... + 24 = 300
            assert_eq!(typed, &[300.0]);
        }
    }

    /// E2E: MaxDim + MeanDim chained — verifies all four reduce
    /// OpKinds reach their wrappers via the OpParams plumbing.
    #[test]
    fn pipelined_realize_max_then_mean() {
        // shape [2, 3], MaxDim(1) → [2], MeanDim(0) → []
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 9.0, 3.0, 4.0, 2.0, 8.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, mean_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let max_id = g.push(Node {
                op: Op::MaxDim(1), inputs: vec![in_id],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let mean_id = g.push(Node {
                op: Op::MeanDim(0), inputs: vec![max_id],
                shape: Shape::from_dims(&[]), dtype: DType::F32,
            });
            g.set_target_backend(max_id, BackendId::Cpu);
            g.set_target_backend(mean_id, BackendId::Cpu);
            (in_id, mean_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, mean_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            // MaxDim(1) on [[1,9,3],[4,2,8]] = [9, 8]; MeanDim(0) = 8.5
            assert_eq!(typed, &[8.5]);
        }
    }

    /// E2E: Sigmoid + Silu — exercises two of the more compositional
    /// new unary kernels through the pipelined executor. Verifies
    /// the additional `op_to_op_kind` mappings reach the right
    /// dispatch wrappers.
    #[test]
    fn pipelined_realize_sigmoid_then_silu() {
        let storage = fuel_memory::from_slice_cpu(&[0.0_f32, 1.0, -1.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, sig_id, silu_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let sig_id = g.push(Node {
                op: Op::Sigmoid, inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            // Silu of the sigmoid output — chains to confirm cache flow.
            let silu_id = g.push(Node {
                op: Op::Silu, inputs: vec![sig_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(sig_id, BackendId::Cpu);
            g.set_target_backend(silu_id, BackendId::Cpu);
            (in_id, sig_id, silu_id)
        };
        let _ = sig_id;
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, silu_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            // sigmoid(0) = 0.5; silu(0.5) = 0.5 * sigmoid(0.5) ≈ 0.3112
            assert!((typed[0] - 0.5 * (1.0 / (1.0 + (-0.5_f32).exp()))).abs() < 1e-6);
        }
    }

    /// E2E: chained unary ops — Const + Sqr + Sqrt should be a noop
    /// for non-negative inputs. Exercises the cache reuse path.
    #[test]
    fn pipelined_realize_chained_unary_sqr_then_sqrt() {
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 4.0, 9.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, sqrt_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let sqr_id = g.push(Node {
                op: Op::Sqr, inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let sqrt_id = g.push(Node {
                op: Op::Sqrt, inputs: vec![sqr_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(sqr_id, BackendId::Cpu);
            g.set_target_backend(sqrt_id, BackendId::Cpu);
            (in_id, sqrt_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, sqrt_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            // sqrt(sqr(x)) == |x| == x for non-negative inputs.
            assert_eq!(typed, &[1.0, 4.0, 9.0]);
        }
    }

    /// Multi-stage pipelined: Const + Const + Add + Add (chain of
    /// two adds). Tests that work items are processed in topo order
    /// and intermediate results are cached and reused.
    #[test]
    fn pipelined_realize_chained_adds() {
        let a_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0]);
        let b_storage = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0]);
        let c_storage = fuel_memory::from_slice_cpu(&[100.0_f32, 200.0]);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, b_id, c_id, ab_id, abc_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2]),
                dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2]),
                dtype: DType::F32,
            });
            let c = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2]),
                dtype: DType::F32,
            });
            let ab = g.push(Node {
                op: Op::Add,
                inputs: vec![a, b],
                shape: Shape::from_dims(&[2]),
                dtype: DType::F32,
            });
            let abc = g.push(Node {
                op: Op::Add,
                inputs: vec![ab, c],
                shape: Shape::from_dims(&[2]),
                dtype: DType::F32,
            });
            g.set_target_backend(ab, BackendId::Cpu);
            g.set_target_backend(abc, BackendId::Cpu);
            (a, b, c, ab, abc)
        };

        let mut inputs = StorageCache::new();
        inputs.insert(a_id, Arc::new(RwLock::new(a_storage)));
        inputs.insert(b_id, Arc::new(RwLock::new(b_storage)));
        inputs.insert(c_id, Arc::new(RwLock::new(c_storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, abc_id, inputs).expect("realize");
        // Suppress unused warning for the intermediate id.
        let _ = ab_id;

        let guard = result_arc.read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            // (1+10) + 100 = 111;  (2+20) + 200 = 222.
            assert_eq!(typed, &[111.0, 222.0]);
        }
    }

    /// E2E: triu(diagonal=0) on a 3×3 matrix.
    /// Input:  [1 2 3 / 4 5 6 / 7 8 9]
    /// Output: [1 2 3 / 0 5 6 / 0 0 9]
    #[test]
    fn pipelined_realize_triu_3x3_diag0() {
        let storage = fuel_memory::from_slice_cpu(&[
            1.0_f32, 2.0, 3.0,
            4.0,     5.0, 6.0,
            7.0,     8.0, 9.0,
        ]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, out_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3, 3]), dtype: DType::F32,
            });
            let out_id = g.push(Node {
                op: Op::Triu { diagonal: 0 }, inputs: vec![in_id],
                shape: Shape::from_dims(&[3, 3]), dtype: DType::F32,
            });
            g.set_target_backend(out_id, BackendId::Cpu);
            (in_id, out_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, out_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let out: &[f32] = c.as_slice().unwrap();
        assert_eq!(out, &[
            1.0, 2.0, 3.0,
            0.0, 5.0, 6.0,
            0.0, 0.0, 9.0,
        ]);
    }

    /// E2E: tril(diagonal=0) on a 3×3 matrix — the canonical causal mask.
    /// Output: [1 0 0 / 4 5 0 / 7 8 9]
    #[test]
    fn pipelined_realize_tril_3x3_diag0_causal_mask() {
        let storage = fuel_memory::from_slice_cpu(&[
            1.0_f32, 2.0, 3.0,
            4.0,     5.0, 6.0,
            7.0,     8.0, 9.0,
        ]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, out_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3, 3]), dtype: DType::F32,
            });
            let out_id = g.push(Node {
                op: Op::Tril { diagonal: 0 }, inputs: vec![in_id],
                shape: Shape::from_dims(&[3, 3]), dtype: DType::F32,
            });
            g.set_target_backend(out_id, BackendId::Cpu);
            (in_id, out_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, out_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let out: &[f32] = c.as_slice().unwrap();
        assert_eq!(out, &[
            1.0, 0.0, 0.0,
            4.0, 5.0, 0.0,
            7.0, 8.0, 9.0,
        ]);
    }

    /// E2E: log_softmax over a 2×3 input. Rows are [1,2,3] and [3,2,1].
    /// log_softmax(row) = row - max - log(sum exp(row - max)).
    /// Row 0: max=3, exp(-2)+exp(-1)+exp(0) = 0.135+0.368+1.0 = 1.503;
    ///        log(1.503) ≈ 0.4076; out = [1-3-0.4076, 2-3-0.4076, 3-3-0.4076]
    ///                              = [-2.4076, -1.4076, -0.4076]
    /// Row 1 is row 0 reversed: [-0.4076, -1.4076, -2.4076].
    #[test]
    fn pipelined_realize_log_softmax_last_dim() {
        let storage = fuel_memory::from_slice_cpu(&[
            1.0_f32, 2.0, 3.0,
            3.0,     2.0, 1.0,
        ]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, out_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            let out_id = g.push(Node {
                op: Op::LogSoftmaxLastDim, inputs: vec![in_id],
                shape: Shape::from_dims(&[2, 3]), dtype: DType::F32,
            });
            g.set_target_backend(out_id, BackendId::Cpu);
            (in_id, out_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, out_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let out: &[f32] = c.as_slice().unwrap();
        let expected = [
            -2.4076059, -1.4076059, -0.40760595_f32,
            -0.40760595, -1.4076059, -2.4076059,
        ];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "log_softmax mismatch: {a} vs {b}");
        }
        // The exp() of the output should sum to 1 per row.
        for row in out.chunks(3) {
            let sum: f32 = row.iter().map(|x| x.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-5, "softmax(log_softmax) row sum != 1: {sum}");
        }
    }

    // --- side-effect roots + destructive cleanup (9c Phase B) -----

    /// Side-effect roots get merged into the realize walk even when
    /// not reachable from the user's targets. The production trigger
    /// is `Op::Release` (emitted by `ResidencyEvictionRule`); this
    /// test uses that exact shape.
    ///
    /// Graph: Const(a) + Const(b) → Add (user target); separate
    /// Const(c) → Release(c) (side-effect root, not reachable from
    /// Add). After realize_many on `[add]`, the Add should be in the
    /// output AND the Release should have fired (verified by the
    /// fact that realize succeeded — Op::Release in the walk used to
    /// fail compilation pre-Phase B+).
    #[test]
    fn pipelined_realize_merges_op_release_side_effect_root() {
        let a = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0]);
        let b = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0]);
        let c = fuel_memory::from_slice_cpu(&[100.0_f32, 200.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, b_id, c_id, add_id, release_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            // c is NOT reachable from add; its Release IS marked as
            // a side-effect root.
            let c = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let add = g.push(Node {
                op: Op::Add, inputs: vec![a, b],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            // Production shape: Op::Release on c, marked as
            // side-effect root so it fires even though no graph
            // node reads its output.
            let release = g.push(Node {
                op: Op::Release, inputs: vec![c],
                shape: Shape::from_dims(&[0]), dtype: DType::F32,
            });
            g.set_target_backend(add, BackendId::Cpu);
            g.set_target_backend(release, BackendId::Cpu);
            g.add_side_effect_root(release);
            (a, b, c, add, release)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(a_id, Arc::new(RwLock::new(a)));
        inputs.insert(b_id, Arc::new(RwLock::new(b)));
        let c_arc = Arc::new(RwLock::new(c));
        let c_arc_external = Arc::clone(&c_arc);
        inputs.insert(c_id, c_arc);

        // Only `add` is requested. The Release on c is a side-effect
        // root that must still fire.
        let out = PipelinedExecutor::realize_many(graph, &[add_id], inputs)
            .expect("realize_many");
        assert_eq!(out.len(), 1);
        let add_guard = out[0].0.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(arr) = &add_guard.inner else { panic!() };
        assert_eq!(arr.as_slice::<f32>().unwrap(), &[11.0, 22.0]);

        // The Release's destructive_input cleanup dropped the
        // cache's Arc to c. The only Arc to c that survives is the
        // external one this test holds — confirms the eviction
        // actually freed the cache's reference.
        assert_eq!(
            Arc::strong_count(&c_arc_external),
            1,
            "Op::Release should evict the cache's Arc to its source; \
             only the test's external clone should remain",
        );
        let _ = release_id;
    }

    /// `compile_one` emits a `WorkItemKind::ReleaseMarker` for
    /// `Op::Release` with `destructive_input = Some(0)`. Verify
    /// the WorkItem shape via a direct compile_one call.
    #[test]
    fn compile_one_emits_release_marker_with_destructive_input() {
        let graph_rc = Arc::new(RwLock::new(Graph::new()));
        let (src_id, release_id) = {
            let mut g = graph_rc.write().unwrap();
            let src = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let release = g.push(Node {
                op: Op::Release, inputs: vec![src],
                shape: Shape::from_dims(&[0]), dtype: DType::F32,
            });
            g.set_target_backend(release, BackendId::Cpu);
            (src, release)
        };
        let g = graph_rc.read().unwrap();
        let bindings = crate::dispatch::global_bindings();
        let mut layout_cache: HashMap<NodeId, Layout> = HashMap::new();
        let item = compile_one(&g, release_id, &mut layout_cache, &bindings, &SymEnv::default())
            .expect("compile_one Op::Release");
        assert!(matches!(item.kind, WorkItemKind::ReleaseMarker));
        assert_eq!(item.destructive_input, Some(0));
        assert_eq!(item.inputs, vec![src_id]);
        assert_eq!(item.elem_count, 0, "Release output is the zero-element marker");
    }

    // --- Op::Move (executor-unification Session 1, gap 13) --------

    /// `compile_one` emits a `WorkItemKind::Move` for `Op::Move` with
    /// `destructive_input = Some(0)` and a resolved transfer kernel
    /// (binding-table lookup at `(OpKind::Copy, [dt, dt], source
    /// backend)` — Move shares Op::Copy's data-movement kernel).
    #[test]
    fn compile_one_emits_move_with_destructive_input() {
        let graph_rc = Arc::new(RwLock::new(Graph::new()));
        let (src_id, move_id) = {
            let mut g = graph_rc.write().unwrap();
            let src = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let mv = g.push(Node {
                op: Op::Move { target: DeviceLocation::Cpu }, inputs: vec![src],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            g.set_target_backend(mv, BackendId::Cpu);
            (src, mv)
        };
        let g = graph_rc.read().unwrap();
        let bindings = crate::dispatch::global_bindings();
        let mut layout_cache: HashMap<NodeId, Layout> = HashMap::new();
        let item = compile_one(&g, move_id, &mut layout_cache, &bindings, &SymEnv::default())
            .expect("compile_one Op::Move");
        assert!(matches!(
            item.kind,
            WorkItemKind::Move { target_location: DeviceLocation::Cpu },
        ));
        assert_eq!(item.destructive_input, Some(0), "Move destroys its source");
        assert_eq!(item.inputs, vec![src_id]);
        assert!(
            item.compiled.is_some(),
            "Move resolves a transfer kernel at (OpKind::Copy, [dt, dt], source backend)",
        );
    }

    /// Move-to-same-device is the degenerate case: per the legacy
    /// `GraphExecutor` contract (`Op::Copy | Op::Move` shared arm →
    /// `backend.copy_to`) it is a plain copy — fresh storage on the
    /// same device, data intact — and the source is still evicted
    /// afterward (destructive semantics don't depend on the devices
    /// differing).
    #[test]
    fn pipelined_op_move_same_device_is_plain_copy() {
        let src = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (src_id, move_id) = {
            let mut g = graph.write().unwrap();
            let src = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let mv = g.push(Node {
                op: Op::Move { target: DeviceLocation::Cpu }, inputs: vec![src],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            g.set_target_backend(mv, BackendId::Cpu);
            (src, mv)
        };
        let src_arc = Arc::new(RwLock::new(src));
        let src_arc_external = Arc::clone(&src_arc);
        let mut inputs = StorageCache::new();
        inputs.insert(src_id, src_arc);

        let (out, layout) =
            PipelinedExecutor::realize(graph, move_id, inputs).expect("realize Op::Move");
        let guard = out.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else { panic!() };
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 4.0],
            "Op::Move output should carry input's data to target",
        );
        assert!(layout.is_contiguous());
        drop(guard);
        assert!(
            !Arc::ptr_eq(&out, &src_arc_external),
            "same-device Move is a plain copy — fresh storage, not an alias",
        );
        // The realize loop's destructive_input cleanup dropped the
        // cache's Arc to the source; only the test's external clone
        // survives.
        assert_eq!(
            Arc::strong_count(&src_arc_external),
            1,
            "Op::Move should evict the cache's Arc to its source",
        );
    }

    /// The multiple-consumer-source case: a Move must NOT strand
    /// another consumer of its source. `execution_plan` (via
    /// `derive_ordering`, keyed off `destructive_input`) pins the
    /// Move AFTER the sibling reader; the executor's post-Move cache
    /// eviction then can't break the reader. Mirrors the legacy
    /// `op_move_pinned_after_sibling_reader_via_derive_ordering`
    /// test in fuel-graph-router/tests/cross_device.rs.
    ///
    ///   a   = const [1, 2, 3, 4]
    ///   b   = relu(a)        (non-destructive sibling reader)
    ///   m   = move(a, Cpu)   (destructive — evicts a once run)
    ///   out = add(b, m)      → [2, 4, 6, 8]
    #[test]
    fn pipelined_op_move_multi_consumer_source_not_stranded() {
        let a = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, out_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Relu, inputs: vec![a],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let m = g.push(Node {
                op: Op::Move { target: DeviceLocation::Cpu }, inputs: vec![a],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let out = g.push(Node {
                op: Op::Add, inputs: vec![b, m],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            g.set_target_backend(b, BackendId::Cpu);
            g.set_target_backend(m, BackendId::Cpu);
            g.set_target_backend(out, BackendId::Cpu);
            (a, out)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(a_id, Arc::new(RwLock::new(a)));

        let (out, _) = PipelinedExecutor::realize(graph, out_id, inputs)
            .expect("realize add(relu(a), move(a))");
        let guard = out.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else { panic!() };
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[2.0, 4.0, 6.0, 8.0],
            "relu(a) must run before move(a) evicts a",
        );
    }

    /// A Move whose source is itself in the realize target set must
    /// not evict it — the caller asked for the source's storage. The
    /// destructive cleanup's `target_set` gate covers Move exactly
    /// like Release.
    #[test]
    fn pipelined_op_move_source_in_target_set_not_evicted() {
        let a = fuel_memory::from_slice_cpu(&[5.0_f32, 6.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, move_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let mv = g.push(Node {
                op: Op::Move { target: DeviceLocation::Cpu }, inputs: vec![a],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            g.set_target_backend(mv, BackendId::Cpu);
            (a, mv)
        };
        let a_arc = Arc::new(RwLock::new(a));
        let a_arc_external = Arc::clone(&a_arc);
        let mut inputs = StorageCache::new();
        inputs.insert(a_id, a_arc);

        let out = PipelinedExecutor::realize_many(graph, &[move_id, a_id], inputs)
            .expect("realize_many [move, source]");
        assert_eq!(out.len(), 2);
        let mv_guard = out[0].0.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &mv_guard.inner else { panic!() };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[5.0, 6.0]);
        assert!(
            Arc::ptr_eq(&out[1].0, &a_arc_external),
            "requested source must survive the Move's destructive cleanup",
        );
    }

    // --- realize_many (Phase 7.6 step 9c Phase A) ----------------

    /// realize_many on empty targets returns an empty Vec — no graph
    /// walk, no panic.
    #[test]
    fn pipelined_realize_many_empty_targets() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let inputs = StorageCache::new();
        let out = PipelinedExecutor::realize_many(graph, &[], inputs).expect("realize_many");
        assert!(out.is_empty());
    }

    /// realize_many with two independent target chains. Each chain's
    /// output is in the cache; the shared topo walk realizes both.
    /// Verifies parallel chain handling + return order matches input.
    #[test]
    fn pipelined_realize_many_two_independent_targets() {
        let lhs = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
        let rhs = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (lhs_id, rhs_id, add_id, mul_id) = {
            let mut g = graph.write().unwrap();
            let lhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let rhs = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let add = g.push(Node {
                op: Op::Add, inputs: vec![lhs, rhs],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let mul = g.push(Node {
                op: Op::Mul, inputs: vec![lhs, rhs],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(add, BackendId::Cpu);
            g.set_target_backend(mul, BackendId::Cpu);
            (lhs, rhs, add, mul)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(lhs_id, Arc::new(RwLock::new(lhs)));
        inputs.insert(rhs_id, Arc::new(RwLock::new(rhs)));

        // Pass mul first to confirm return order matches targets order
        // (not graph order — graph order would put add first).
        let out =
            PipelinedExecutor::realize_many(graph, &[mul_id, add_id], inputs).expect("realize_many");
        assert_eq!(out.len(), 2);

        let mul_guard = out[0].0.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &mul_guard.inner else { panic!() };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[10.0, 40.0, 90.0]);

        let add_guard = out[1].0.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &add_guard.inner else { panic!() };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[11.0, 22.0, 33.0]);
    }

    /// realize_many with the same NodeId twice — caller asking for
    /// the same output twice. Both outputs are the same Arc (cheap).
    #[test]
    fn pipelined_realize_many_duplicate_target_returns_same_arc() {
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, neg_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let neg = g.push(Node {
                op: Op::Neg, inputs: vec![in_id],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(neg, BackendId::Cpu);
            (in_id, neg)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));

        let out = PipelinedExecutor::realize_many(graph, &[neg_id, neg_id], inputs)
            .expect("realize_many");
        assert_eq!(out.len(), 2);
        assert!(
            Arc::ptr_eq(&out[0].0, &out[1].0),
            "duplicate target should share the same storage Arc",
        );
    }

    /// realize_many with shared upstream nodes evaluates the shared
    /// chunk exactly once. Two targets `f = a + b` and `g = (a + b) * a`
    /// — `a + b` is computed once.
    #[test]
    fn pipelined_realize_many_shared_subgraph_evaluates_once() {
        let a_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
        let b_storage = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, b_id, add_id, mul_id) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let add = g.push(Node {
                op: Op::Add, inputs: vec![a, b],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            let mul = g.push(Node {
                op: Op::Mul, inputs: vec![add, a],
                shape: Shape::from_dims(&[3]), dtype: DType::F32,
            });
            g.set_target_backend(add, BackendId::Cpu);
            g.set_target_backend(mul, BackendId::Cpu);
            (a, b, add, mul)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(a_id, Arc::new(RwLock::new(a_storage)));
        inputs.insert(b_id, Arc::new(RwLock::new(b_storage)));

        let out = PipelinedExecutor::realize_many(graph, &[add_id, mul_id], inputs)
            .expect("realize_many");
        assert_eq!(out.len(), 2);

        let add_guard = out[0].0.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &add_guard.inner else { panic!() };
        assert_eq!(c.as_slice::<f32>().unwrap(), &[11.0, 22.0, 33.0]);

        let mul_guard = out[1].0.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &mul_guard.inner else { panic!() };
        // (a + b) * a = [11*1, 22*2, 33*3] = [11, 44, 99].
        assert_eq!(c.as_slice::<f32>().unwrap(), &[11.0, 44.0, 99.0]);
    }

    /// E2E: masked_fill with -inf — the attention masking pattern.
    /// x = [1, 2, 3, 4]; mask = [0, 1, 0, 1]; value = -1000.
    /// out = [1, -1000, 3, -1000].
    #[test]
    fn pipelined_realize_masked_fill_attention_pattern() {
        let x_storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let mask_storage = fuel_memory::from_slice_cpu(&[0u8, 1, 0, 1]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, mask_id, out_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let mask = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[4]), dtype: DType::U8,
            });
            let out = g.push(Node {
                op: Op::MaskedFill { value: fuel_ir::Scalar::F32(-1000.0) },
                inputs: vec![x, mask],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            g.set_target_backend(out, BackendId::Cpu);
            (x, mask, out)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(x_id, Arc::new(RwLock::new(x_storage)));
        inputs.insert(mask_id, Arc::new(RwLock::new(mask_storage)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, out_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let out: &[f32] = c.as_slice().unwrap();
        assert_eq!(out, &[1.0, -1000.0, 3.0, -1000.0]);
    }

    // ---- WriteSlice (Phase 7.6 step 9c E.3.2) -------------------------------

    /// E2E: pipelined realize of `Op::WriteSlice` writing a [1, 3, 2]
    /// source slab into a [4, 3, 2] destination at axis-0 row 2 — the
    /// canonical KV-cache append shape.
    #[test]
    fn pipelined_realize_write_slice_kv_cache_append() {
        let dest_storage = fuel_memory::from_slice_cpu(&[0.0_f32; 24]);
        let src_storage = fuel_memory::from_slice_cpu(&[
            1.0_f32, 2.0,
            3.0,     4.0,
            5.0,     6.0,
        ]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (dest_id, src_id, ws_id) = {
            let mut g = graph.write().unwrap();
            let dest = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[4, 3, 2]), dtype: DType::F32,
            });
            let src = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 3, 2]), dtype: DType::F32,
            });
            let ws = g.push(Node {
                op: Op::WriteSlice { ranges: vec![(2, 3), (0, 3), (0, 2)], dyn_offset: None },
                inputs: vec![dest, src],
                shape: Shape::from_dims(&[4, 3, 2]),  // adopts dest shape
                dtype: DType::F32,
            });
            g.set_target_backend(ws, BackendId::Cpu);
            (dest, src, ws)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(dest_id, Arc::new(RwLock::new(dest_storage)));
        inputs.insert(src_id, Arc::new(RwLock::new(src_storage)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, ws_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let out: &[f32] = c.as_slice().unwrap();
        let expected = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,  // row 2 = source
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(out, &expected);
    }

    /// E2E: pipelined realize of `Op::WriteSlice` adopts the
    /// destination's Storage Arc as the output slot (in-place
    /// alias) — the resulting Arc points to the SAME bytes the
    /// destination was constructed from.
    #[test]
    fn pipelined_realize_write_slice_aliases_dest_storage() {
        let dest_storage = fuel_memory::from_slice_cpu(&[0.0_f32; 6]);
        let src_storage = fuel_memory::from_slice_cpu(&[10.0_f32, 20.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (dest_id, src_id, ws_id, dest_arc_external) = {
            let mut g = graph.write().unwrap();
            let dest = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[3, 2]), dtype: DType::F32,
            });
            let src = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 2]), dtype: DType::F32,
            });
            let ws = g.push(Node {
                op: Op::WriteSlice { ranges: vec![(1, 2), (0, 2)], dyn_offset: None },
                inputs: vec![dest, src],
                shape: Shape::from_dims(&[3, 2]),
                dtype: DType::F32,
            });
            g.set_target_backend(ws, BackendId::Cpu);
            let arc = Arc::new(RwLock::new(dest_storage));
            (dest, src, ws, arc)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(dest_id, Arc::clone(&dest_arc_external));
        inputs.insert(src_id, Arc::new(RwLock::new(src_storage)));
        let (result_arc, _) =
            PipelinedExecutor::realize(graph, ws_id, inputs).expect("realize");
        // Bytes were written into the same Arc the caller held —
        // the WriteSlice op aliases dest's Storage Arc, not allocates
        // a fresh buffer.
        assert!(
            Arc::ptr_eq(&result_arc, &dest_arc_external),
            "WriteSlice output Arc must be the same Arc as the destination's"
        );
        let guard = dest_arc_external.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let out: &[f32] = c.as_slice().unwrap();
        assert_eq!(out, &[0.0, 0.0, 10.0, 20.0, 0.0, 0.0]);
    }

    // ---- WriteSlice dynamic offset (Phase D symbolic extents) ---------------

    /// E2E: pipelined realize of `Op::WriteSlice` with a runtime start
    /// offset (`dyn_offset`) resolved through a per-pass `SymEnv`. The
    /// width-2 source slab lands at the bound offset (3), not the static
    /// `ranges[0].0` placeholder (0) — the persistent-decode KV-cache
    /// write at a per-token `cached_len`.
    #[test]
    fn pipelined_realize_write_slice_dynamic_offset() {
        use fuel_ir::{DynScalar, SymId};
        let dest_storage = fuel_memory::from_slice_cpu(&[0.0_f32; 6]);
        let src_storage = fuel_memory::from_slice_cpu(&[7.0_f32, 8.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let sym = SymId(0);
        let (dest_id, src_id, ws_id) = {
            let mut g = graph.write().unwrap();
            let dest = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[6]), dtype: DType::F32,
            });
            let src = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let ws = g.push(Node {
                // ranges[0] = (0, 2): the start is a placeholder (ignored —
                // overridden by dyn_offset); only the width (2) is read.
                op: Op::WriteSlice {
                    ranges: vec![(0, 2)],
                    dyn_offset: Some((0, DynScalar::Sym(sym))),
                },
                inputs: vec![dest, src],
                shape: Shape::from_dims(&[6]),
                dtype: DType::F32,
            });
            g.set_target_backend(ws, BackendId::Cpu);
            (dest, src, ws)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(dest_id, Arc::new(RwLock::new(dest_storage)));
        inputs.insert(src_id, Arc::new(RwLock::new(src_storage)));

        // Bind cached_len = 3: the slab must land at indices [3, 4].
        let mut env = SymEnv::new();
        env.bind(sym, 3).unwrap();

        let (result_arc, _) =
            PipelinedExecutor::realize_with_env(graph, ws_id, inputs, env)
                .expect("realize_with_env");
        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let out: &[f32] = c.as_slice().unwrap();
        assert_eq!(
            out, &[0.0, 0.0, 0.0, 7.0, 8.0, 0.0],
            "slab must land at the SymEnv-bound offset 3, not the static placeholder 0",
        );
    }

    /// A `dyn_offset` whose symbol is unbound in the `SymEnv` surfaces a
    /// typed error at realize (never a panic).
    #[test]
    fn pipelined_realize_write_slice_dynamic_offset_unbound_errors() {
        use fuel_ir::{DynScalar, SymId};
        let dest_storage = fuel_memory::from_slice_cpu(&[0.0_f32; 6]);
        let src_storage = fuel_memory::from_slice_cpu(&[7.0_f32, 8.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (dest_id, src_id, ws_id) = {
            let mut g = graph.write().unwrap();
            let dest = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[6]), dtype: DType::F32,
            });
            let src = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let ws = g.push(Node {
                op: Op::WriteSlice {
                    ranges: vec![(0, 2)],
                    dyn_offset: Some((0, DynScalar::Sym(SymId(0)))),
                },
                inputs: vec![dest, src],
                shape: Shape::from_dims(&[6]),
                dtype: DType::F32,
            });
            g.set_target_backend(ws, BackendId::Cpu);
            (dest, src, ws)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(dest_id, Arc::new(RwLock::new(dest_storage)));
        inputs.insert(src_id, Arc::new(RwLock::new(src_storage)));
        // Empty env — the symbol is unbound.
        let result =
            PipelinedExecutor::realize_with_env(graph, ws_id, inputs, SymEnv::new());
        assert!(
            result.is_err(),
            "unbound dyn_offset symbol must surface a typed error, not a panic",
        );
    }

    /// A runtime offset whose resolved slab runs past the destination
    /// capacity errors at realize (never an out-of-bounds write).
    #[test]
    fn pipelined_realize_write_slice_dynamic_offset_out_of_capacity_errors() {
        use fuel_ir::{DynScalar, SymId};
        let dest_storage = fuel_memory::from_slice_cpu(&[0.0_f32; 6]);
        let src_storage = fuel_memory::from_slice_cpu(&[7.0_f32, 8.0]);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let sym = SymId(0);
        let (dest_id, src_id, ws_id) = {
            let mut g = graph.write().unwrap();
            let dest = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[6]), dtype: DType::F32,
            });
            let src = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let ws = g.push(Node {
                op: Op::WriteSlice {
                    ranges: vec![(0, 2)],
                    dyn_offset: Some((0, DynScalar::Sym(sym))),
                },
                inputs: vec![dest, src],
                shape: Shape::from_dims(&[6]),
                dtype: DType::F32,
            });
            g.set_target_backend(ws, BackendId::Cpu);
            (dest, src, ws)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(dest_id, Arc::new(RwLock::new(dest_storage)));
        inputs.insert(src_id, Arc::new(RwLock::new(src_storage)));
        // offset 5 + width 2 = 7 > capacity 6.
        let mut env = SymEnv::new();
        env.bind(sym, 5).unwrap();
        let result = PipelinedExecutor::realize_with_env(graph, ws_id, inputs, env);
        assert!(
            result.is_err(),
            "resolved slab past destination capacity must surface a typed error",
        );
    }

    /// Regression for the PipelinedExecutor ordering-integration session.
    ///
    /// Graph: `x = Const`, `step_x = Step(x)`, `y = SigmoidInplace(x)`.
    /// `Step` and `SigmoidInplace` are both readers of `x`; `Step` is
    /// non-destructive, `SigmoidInplace` mutates `x`'s storage. The
    /// correct semantic is for `Step` to read PRE-mutation bytes.
    ///
    /// Pre-this-session (raw `topo_order_multi`): the two are siblings
    /// in topo order and the tie-break is unspecified — `SigmoidInplace`
    /// could run first, making `step_x` see post-sigmoid bytes
    /// (`[1, 1, 1, 1]` because every sigmoid output is positive).
    ///
    /// Post-this-session: `execution_plan` consults `derive_ordering`,
    /// which pins `SigmoidInplace` AFTER every other reader of `x`.
    /// `step_x` deterministically sees `x` and is `[1, 0, 1, 0]`.
    #[test]
    fn pipelined_inplace_with_multiple_readers_orders_correctly() {
        use fuel_ir::Shape;
        let data = [1.0_f32, -2.0, 3.0, -4.0];
        let src_storage = fuel_memory::from_slice_cpu(&data);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, step_id, sig_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let step = g.push(Node {
                op: Op::Step, inputs: vec![x],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let sig = g.push(Node {
                op: Op::SigmoidInplace, inputs: vec![x],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            g.set_target_backend(step, BackendId::Cpu);
            g.set_target_backend(sig, BackendId::Cpu);
            (x, step, sig)
        };

        let mut cache = StorageCache::new();
        cache.insert(x_id, Arc::new(RwLock::new(src_storage)));

        let results = PipelinedExecutor::realize_many(
            graph, &[step_id, sig_id], cache,
        ).expect("realize_many");

        // step_x must reflect pre-mutation x: sign of [1, -2, 3, -4] →
        // [1, 0, 1, 0]. If the executor ran SigmoidInplace first, step
        // would see all-positive sigmoid outputs and produce [1, 1, 1, 1].
        let step_guard = results[0].0.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &step_guard.inner else {
            panic!("expected Cpu storage for step output");
        };
        let step_out: &[f32] = c.as_slice().expect("f32 cast");
        assert_eq!(
            step_out, &[1.0_f32, 0.0, 1.0, 0.0],
            "Step must run before SigmoidInplace; got post-mutation bytes"
        );
        drop(step_guard);

        // SigmoidInplace's output Arc IS the mutated cache entry — bytes
        // approximate sigmoid([1, -2, 3, -4]).
        let sig_guard = results[1].0.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &sig_guard.inner else {
            panic!("expected Cpu storage for sigmoid output");
        };
        let sig_out: &[f32] = c.as_slice().expect("f32 cast");
        let sig_ref = [
            1.0_f32 / (1.0 + (-1.0_f32).exp()),
            1.0_f32 / (1.0 + (2.0_f32).exp()),
            1.0_f32 / (1.0 + (-3.0_f32).exp()),
            1.0_f32 / (1.0 + (4.0_f32).exp()),
        ];
        for (got, want) in sig_out.iter().zip(sig_ref.iter()) {
            assert!((got - want).abs() < 1e-6, "sigmoid: got {got}, want {want}");
        }
    }

    /// Regression for the PipelinedExecutor ordering-integration session.
    ///
    /// Canonical residual-connection pattern:
    ///   `x = Const`, `y = ReluInplace(x)`, `z = Add(y, x)`.
    ///
    /// Pre-this-session: `derive_ordering` was never run, so the cycle
    /// (Add must precede ReluInplace per ordering edge; Add must follow
    /// ReluInplace per data-flow edge through y) was undetected. The
    /// executor produced `z = relu(x) + relu(x) = [2, 0, 6, 0]` —
    /// silently wrong.
    ///
    /// Post-this-session: `insert_safety_copies` runs before
    /// `execution_plan`, inserts `Op::Copy(x) → x_safe`, rewires Add's
    /// `x` input to `x_safe`. Result: `z = relu(x) + x = [2, -2, 6, -4]`.
    #[test]
    fn pipelined_residual_connection_inserts_safety_copy() {
        use fuel_ir::Shape;
        let data = [1.0_f32, -2.0, 3.0, -4.0];
        let src_storage = fuel_memory::from_slice_cpu(&data);

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x_id, _y_id, z_id) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let y = g.push(Node {
                op: Op::ReluInplace, inputs: vec![x],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            let z = g.push(Node {
                op: Op::Add, inputs: vec![y, x],
                shape: Shape::from_dims(&[4]), dtype: DType::F32,
            });
            g.set_target_backend(y, BackendId::Cpu);
            g.set_target_backend(z, BackendId::Cpu);
            (x, y, z)
        };

        let mut cache = StorageCache::new();
        cache.insert(x_id, Arc::new(RwLock::new(src_storage)));

        let (result_arc, _) =
            PipelinedExecutor::realize(graph, z_id, cache)
                .expect("realize z — insert_safety_copies must break the cycle");

        let guard = result_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let got: &[f32] = c.as_slice().expect("f32 cast");
        assert_eq!(
            got, &[2.0_f32, -2.0, 6.0, -4.0],
            "Add must read pre-mutation x via the inserted safety copy"
        );
    }

    // ------------------------------------------------------------------
    // Item 3 end-to-end: SelectiveScan through the bundled-output
    // PipelinedExecutor path. The producer allocates bundled
    // (y_bytes + last_state_bytes), the kernel writes both slots,
    // and the bundled-tuple builder returns each via Op::View.
    // ------------------------------------------------------------------

    fn cpu_const_f32(
        graph: &Arc<RwLock<Graph>>,
        cache: &mut StorageCache,
        data:  &[f32],
        dims:  &[usize],
    ) -> NodeId {
        let id = {
            let mut g = graph.write().unwrap();
            g.push(Node {
                op:     Op::Const,
                inputs: vec![],
                shape:  Shape::from_dims(dims),
                dtype:  DType::F32,
            })
        };
        cache.insert(
            id,
            Arc::new(RwLock::new(fuel_memory::from_slice_cpu(data))),
        );
        id
    }

    /// `selective_scan_bundled` realizes both `y` and `last_state` in
    /// a single `realize_many` pass; the Views project the same
    /// producer's bundled Storage Arc and the numerics match the
    /// hand-computed single-step kernel test (b=1,s=1,d=1,k=1).
    #[test]
    fn pipelined_realize_selective_scan_bundled_produces_y_and_last_state() {
        let graph: Arc<RwLock<Graph>> = Arc::new(RwLock::new(Graph::new()));
        let mut cache = StorageCache::new();

        let u_id     = cpu_const_f32(&graph, &mut cache, &[3.0], &[1, 1, 1]);
        let delta_id = cpu_const_f32(&graph, &mut cache, &[1.0], &[1, 1, 1]);
        let a_id     = cpu_const_f32(&graph, &mut cache, &[-1.0], &[1, 1]);
        let b_id     = cpu_const_f32(&graph, &mut cache, &[2.0], &[1, 1, 1]);
        let c_id     = cpu_const_f32(&graph, &mut cache, &[0.5], &[1, 1, 1]);

        let (y_id, last_state_id) = {
            let u_t     = fuel_graph::Tensor::from_existing(Arc::clone(&graph), u_id);
            let delta_t = fuel_graph::Tensor::from_existing(Arc::clone(&graph), delta_id);
            let a_t     = fuel_graph::Tensor::from_existing(Arc::clone(&graph), a_id);
            let b_t     = fuel_graph::Tensor::from_existing(Arc::clone(&graph), b_id);
            let c_t     = fuel_graph::Tensor::from_existing(Arc::clone(&graph), c_id);
            let (y, last_state) = u_t
                .selective_scan_bundled(&delta_t, &a_t, &b_t, &c_t, /* delta_softplus */ false)
                .expect("selective_scan_bundled");
            (y.id(), last_state.id())
        };
        // Set target_backend on the producer (View nodes inherit it).
        {
            let mut g = graph.write().unwrap();
            let producer = g.node(y_id).inputs[0];
            g.set_target_backend(producer, BackendId::Cpu);
        }

        let outputs = PipelinedExecutor::realize_many(
            Arc::clone(&graph), &[y_id, last_state_id], cache,
        ).expect("realize_many");
        let (y_arc, y_layout) = outputs[0].clone();
        let (last_state_arc, last_state_layout) = outputs[1].clone();

        // Both Views project the same bundled producer Storage Arc.
        assert!(
            Arc::ptr_eq(&y_arc, &last_state_arc),
            "y and last_state Views project the SAME bundled producer Storage",
        );

        // Hand-computed numerics (single-step recurrence):
        // h[0,0,0] = exp(1.0 * -1.0) * 0 + 1.0 * 2.0 * 3.0 = 6.0
        // y[0,0,0] = h * c = 6.0 * 0.5 = 3.0
        // last_state[0,0,0] = h = 6.0
        let guard = y_arc.read().unwrap();
        let fuel_memory::BackendStorage::Cpu(c_bytes) = &guard.inner else {
            panic!("expected Cpu storage");
        };
        let typed: &[f32] = c_bytes.as_slice().expect("f32 cast");
        assert_eq!(y_layout.start_offset(), 0);
        let y_value = typed[y_layout.start_offset()];
        assert!((y_value - 3.0).abs() < 1e-5, "y expected 3.0 got {y_value}");
        assert_eq!(
            last_state_layout.start_offset(), 1,
            "slot 1 byte_offset 4 / sizeof(f32) = 1",
        );
        let ls_value = typed[last_state_layout.start_offset()];
        assert!((ls_value - 6.0).abs() < 1e-5, "last_state expected 6.0 got {ls_value}");
    }
}
