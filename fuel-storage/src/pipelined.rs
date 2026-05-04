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
//! `fuel_core_types::Storage` (the `Box<dyn DynBackendStorage>`
//! newtype). The pipelined executor uses the new
//! `fuel_storage::Storage` (BackendStorage enum + dtype). During
//! the migration the two coexist — neither is converted on the fly.
//! The pipelined executor takes pre-realized inputs as a
//! `HashMap<NodeId, Arc<RwLock<fuel_storage::Storage>>>` rather
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

use fuel_core_types::dispatch::OpKind;
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, Error, Layout, Result};
use fuel_graph::{topo_order, Graph, Node, NodeId, Op};

use crate::compiled::{compile_node, execute_compiled, CompiledNode};
use crate::dispatch::global_bindings;
use crate::kernel::{KernelBindingTable, OpParams};
use crate::Storage;

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
}

/// Pipelined executor: walks a graph, compiles each node in a
/// dedicated thread, executes the compiled stream on the calling
/// thread.
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
        // Topo order + initial layouts for the input cache entries
        // computed on the calling thread to keep the compiler
        // thread free of graph-locking responsibilities.
        let (order, mut layout_cache): (Vec<NodeId>, HashMap<NodeId, Layout>) = {
            let g = graph.read().map_err(|_| poisoned("graph lock"))?;
            let order = topo_order(&g, target);
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
            compiler_thread_body(graph_for_compiler, order_for_compiler, tx);
        });

        // Executor on this thread: consume WorkItems, gather
        // inputs from the cache, allocate outputs, call kernels,
        // populate the cache.
        let mut cache: StorageCache = inputs;
        let mut last_processed: Option<NodeId> = None;
        for item in rx {
            let item = item?;
            execute_work_item(&item, &mut cache, &mut layout_cache)?;
            last_processed = Some(item.node_id);
        }

        compiler
            .join()
            .map_err(|_| Error::Msg("compiler thread panicked".to_string()).bt())?;

        match last_processed {
            Some(id) if id == target => {}
            Some(_) | None => {
                return Err(Error::Msg(format!(
                    "PipelinedExecutor::realize: target {:?} not reached",
                    target
                ))
                .bt());
            }
        }

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
}

/// Compiler thread body. Reads each node in topo order, resolves
/// its kernel via `global_bindings()`, and pushes a `WorkItem` on
/// the channel. Sends `Err(...)` and stops on the first failure.
fn compiler_thread_body(
    graph: Arc<RwLock<Graph>>,
    order: Vec<NodeId>,
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
        let item = compile_one(&g, id, &mut layout_cache, &bindings);
        let stop_on_err = item.is_err();
        if tx.send(item).is_err() {
            return;
        }
        if stop_on_err {
            return;
        }
    }
}

/// Whether this op is a metadata-only view op — a node whose output
/// shares bytes with its sole input but reinterprets them through
/// strides + offset.
fn is_view_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Transpose | Op::Permute(_) | Op::BroadcastTo(_) | Op::Slice { .. }
    )
}

/// Compute the output Layout of a metadata-only view op from its
/// input Layout + the op variant. Returns `Err` if the op variant
/// isn't a recognized view op (caller's contract).
fn derive_view_output_layout(op: &Op, input_layout: &Layout) -> Result<Layout> {
    match op {
        Op::Transpose => {
            let rank = input_layout.shape().rank();
            if rank < 2 {
                return Err(Error::Msg(format!(
                    "Op::Transpose requires rank >= 2, input rank is {rank}",
                ))
                .bt());
            }
            input_layout.transpose(rank - 2, rank - 1)
        }
        Op::Permute(axes) => input_layout.permute(axes),
        Op::BroadcastTo(target_shape) => input_layout.broadcast_as(target_shape.clone()),
        Op::Slice { dim, start, len } => input_layout.narrow(*dim, *start, *len),
        other => Err(Error::Msg(format!(
            "derive_view_output_layout called with non-view op {other:?}",
        ))
        .bt()),
    }
}

/// Resolve one node into a `WorkItem` and update `layout_cache`
/// with the node's output layout. Three op shapes:
///
/// - `Op::Const` — adopts the entry from the input cache; layout is
///   read from the graph's side-table (or its contiguous fallback).
///
/// - Metadata-only view op — output layout is derived from the
///   input layout via [`derive_view_output_layout`]; the executor
///   adopts the input's Storage Arc (no allocation, no kernel).
///
/// - Computational op — output layout is `Layout::contiguous(node.shape)`
///   because today's kernels write contiguous output. The compiler
///   resolves the kernel ref and emits a Kernel work item.
fn compile_one(
    graph: &Graph,
    id: NodeId,
    layout_cache: &mut HashMap<NodeId, Layout>,
    bindings: &KernelBindingTable,
) -> Result<WorkItem> {
    let node = graph.node(id);
    let elem_count = node.shape.elem_count();
    let inputs = node.inputs.clone();

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
        });
    }

    if is_view_op(&node.op) {
        if inputs.len() != 1 {
            return Err(Error::Msg(format!(
                "view op {:?} expects 1 input, got {}",
                node.op,
                inputs.len(),
            ))
            .bt());
        }
        let input_layout = layout_cache.get(&inputs[0]).cloned().ok_or_else(|| {
            Error::Msg(format!(
                "view op {:?} input {:?} has no layout in compiler cache",
                node.op, inputs[0],
            ))
            .bt()
        })?;
        let output_layout = derive_view_output_layout(&node.op, &input_layout)?;
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
        let input_layout = layout_cache.get(&inputs[0]).cloned().ok_or_else(|| {
            Error::Msg(format!(
                "Op::Reshape input {:?} has no layout in compiler cache",
                inputs[0],
            ))
            .bt()
        })?;
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

    let op_params = op_to_op_params(graph, node, layout_cache)?;
    let compiled = compile_node(op_kind, node.dtype, target_backend, op_params, bindings)?;
    let output_layout = Layout::contiguous(node.shape.clone());
    layout_cache.insert(id, output_layout.clone());
    Ok(WorkItem {
        node_id: id,
        inputs,
        elem_count,
        dtype: node.dtype,
        target_backend,
        kind: WorkItemKind::Kernel,
        compiled: Some(compiled),
        output_layout,
    })
}

/// Map a `fuel_graph::Op` to a `fuel_core_types::dispatch::OpKind`.
/// Returns `None` for ops that haven't been wired into the new
/// dispatch path yet — Phase C extends this as op families migrate.
fn op_to_op_kind(op: &Op) -> Option<OpKind> {
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
        Op::SumDim(_)     => Some(OpKind::SumReduce),
        Op::MaxDim(_)     => Some(OpKind::MaxReduce),
        Op::MinDim(_)     => Some(OpKind::MinReduce),
        Op::MeanDim(_)    => Some(OpKind::MeanReduce),
        Op::SumAll        => Some(OpKind::SumReduce),
        Op::MaxAll        => Some(OpKind::MaxReduce),
        Op::MinAll        => Some(OpKind::MinReduce),
        Op::MeanAll       => Some(OpKind::MeanReduce),
        Op::MatMul        => Some(OpKind::MatMul),
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
fn op_to_op_params(
    graph: &Graph,
    node: &Node,
    layout_cache: &HashMap<NodeId, Layout>,
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
        Op::SumDim(d) | Op::MaxDim(d) | Op::MinDim(d) | Op::MeanDim(d) => {
            OpParams::Reduce {
                input_layout: input_layout(node.inputs[0]),
                dims: vec![*d],
                keepdim: false,
            }
        }
        Op::SumAll | Op::MaxAll | Op::MinAll | Op::MeanAll => {
            let il = input_layout(node.inputs[0]);
            let rank = il.shape().rank();
            OpParams::Reduce {
                input_layout: il,
                dims: (0..rank).collect(),
                keepdim: false,
            }
        }
        Op::MatMul => {
            // Strict rank-2 today: lhs [m, k] @ rhs [k, n] → [m, n].
            // Batched matmul lands later — extend this arm with a
            // batched OpParams variant when it does.
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
            if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
                return Err(Error::Msg(format!(
                    "Op::MatMul: only rank-2 inputs are wired in the new path \
                     today (got lhs rank {}, rhs rank {}); batched matmul \
                     is a follow-up — reshape or split first",
                    lhs_dims.len(),
                    rhs_dims.len(),
                ))
                .bt());
            }
            let (m, k_lhs) = (lhs_dims[0], lhs_dims[1]);
            let (k_rhs, n) = (rhs_dims[0], rhs_dims[1]);
            if k_lhs != k_rhs {
                return Err(Error::Msg(format!(
                    "Op::MatMul: contracting dims disagree — lhs is [{m}, {k_lhs}], \
                     rhs is [{k_rhs}, {n}]",
                ))
                .bt());
            }
            OpParams::Matmul { m, n, k: k_lhs }
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
fn execute_work_item(
    item: &WorkItem,
    cache: &mut StorageCache,
    layout_cache: &mut HashMap<NodeId, Layout>,
) -> Result<()> {
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
            Ok(())
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
            Ok(())
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
            Ok(())
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
            let mut input_arcs: Vec<Arc<RwLock<Storage>>> = Vec::with_capacity(item.inputs.len());
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
                let already_ok =
                    in_layout.is_contiguous() && in_layout.start_offset() == 0;
                if already_ok {
                    input_arcs.push(in_arc);
                } else {
                    let contig_arc = auto_contiguize(&in_arc, &in_layout)?;
                    input_arcs.push(contig_arc);
                }
            }

            // Allocate output on the target backend.
            let output = match item.target_backend {
                BackendId::Cpu => crate::alloc_cpu_zeroed(item.dtype, item.elem_count)?,
                other => {
                    return Err(Error::Msg(format!(
                        "PipelinedExecutor: target_backend {:?} output allocation \
                         not yet implemented (CPU is wired; GPUs extend later)",
                        other
                    ))
                    .bt());
                }
            };
            let mut output_arcs = vec![Arc::new(RwLock::new(output))];

            execute_compiled(compiled, &input_arcs, &mut output_arcs)?;

            let arc = output_arcs.into_iter().next().expect("one output");
            cache.insert(item.node_id, arc);
            layout_cache.insert(item.node_id, item.output_layout.clone());
            Ok(())
        }
    }
}

fn poisoned(what: &'static str) -> Error {
    Error::Msg(format!("PipelinedExecutor: {} poisoned", what)).bt()
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
        crate::BackendStorage::Cpu(c) => {
            let new_bytes = fuel_cpu_backend::byte_kernels::contiguize_cpu(c, layout, dtype_size)?;
            Storage::new(crate::BackendStorage::Cpu(new_bytes), dtype)
        }
        #[allow(unreachable_patterns)]
        _ => {
            return Err(Error::Msg(
                "auto_contiguize: only the CPU backend is wired today; \
                 GPU backends extend this match when their first kernel \
                 family lands"
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
    use fuel_core_types::Shape;
    use fuel_graph::Node;

    /// E2E: 3-node graph (Const + Const + Add), pre-seeded inputs,
    /// realized through the compiler+executor thread pair, returns
    /// expected sum bytes.
    #[test]
    fn pipelined_realize_const_const_add() {
        let lhs_storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
        let rhs_storage = crate::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);

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
        if let crate::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().expect("f32 cast");
            assert_eq!(typed, &[11.0, 22.0, 33.0]);
        } else {
            panic!("expected CPU output");
        }
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
            Arc::new(RwLock::new(crate::from_slice_cpu(&[1.0_f32, 2.0]))),
        );
        inputs.insert(
            rhs_id,
            Arc::new(RwLock::new(crate::from_slice_cpu(&[3.0_f32, 4.0]))),
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
        let storage = crate::from_slice_cpu(&[5.0_f32, 6.0, 7.0]);

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
        if let crate::BackendStorage::Cpu(c) = &guard.inner {
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
        let storage = crate::from_slice_cpu(&[-1.0_f32, 0.0, 0.5, -3.5, 7.25]);
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
        if let crate::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().expect("f32 cast");
            assert_eq!(typed, &[0.0, 0.0, 0.5, 0.0, 7.25]);
        }
    }

    /// E2E: Const + Const + Sub + Mul + Div — exercises three more
    /// of the freshly-migrated binary kernels in one graph. Verifies
    /// that intermediates flow through the cache as expected.
    #[test]
    fn pipelined_realize_chained_binary_ops() {
        let a_storage = crate::from_slice_cpu(&[10.0_f32, 20.0]);
        let b_storage = crate::from_slice_cpu(&[3.0_f32, 5.0]);
        let c_storage = crate::from_slice_cpu(&[2.0_f32, 4.0]);

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
        if let crate::BackendStorage::Cpu(c) = &guard.inner {
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
        let storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
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

    /// E2E: Const + Permute(rank-3 axes [2, 0, 1]) — verifies the
    /// general permute path through metadata-only adoption.
    #[test]
    fn pipelined_realize_permute_is_metadata_only() {
        // shape [2, 3, 4]; permute axes [2, 0, 1] → shape [4, 2, 3]
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let storage = crate::from_slice_cpu(&data);
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

    /// E2E: Const + Slice — slice is metadata-only; the output Arc
    /// shares bytes with the input, and the Layout's start_offset
    /// + narrowed shape reflect the slice. Stage 3 of Layout-on-Node
    /// extended to cover Op::Slice via Layout::narrow.
    #[test]
    fn pipelined_realize_slice_is_metadata_only() {
        // shape [5]; slice dim 0 from index 1 with len 3 → shape [3]
        // Source: [10, 20, 30, 40, 50]; slice → [20, 30, 40]
        let storage = crate::from_slice_cpu(&[10.0_f32, 20.0, 30.0, 40.0, 50.0]);
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
        let storage = crate::from_slice_cpu(&[10.0_f32, 20.0, 30.0, 40.0, 50.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[90.0]);
    }

    /// E2E: Const + BroadcastTo — verifies that broadcast layouts
    /// have stride 0 on the broadcast dim while sharing the input's
    /// bytes. Stage 3 of Layout-on-Node.
    #[test]
    fn pipelined_realize_broadcast_is_metadata_only() {
        // shape [3]; broadcast to [4, 3] — leading dim is 0-stride
        let storage = crate::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);
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
        let lhs_storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs_storage = crate::from_slice_cpu(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]);

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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
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
        let lhs_storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let rhs_storage = crate::from_slice_cpu(&[5.0_f32, 6.0, 7.0, 8.0]);

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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[17.0, 23.0, 39.0, 53.0]);
    }

    /// E2E: Const + Reshape — contiguous-input reshape is zero
    /// copy. The output Storage Arc is the input Arc; the layout
    /// is contiguous in the new shape.
    #[test]
    fn pipelined_realize_reshape_zero_copy_when_contiguous() {
        let storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let bc_input = crate::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);
        let plus_input = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        let typed: &[f32] = c.as_slice().unwrap();
        assert_eq!(typed, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    /// E2E: Const + SumDim — verifies that `OpParams::Reduce`
    /// flows from the graph (input shape via `op_to_op_params`)
    /// through compile_one and reaches the reduce kernel.
    #[test]
    fn pipelined_realize_sum_dim() {
        // shape [2, 3]; reduce dim 1 → output shape [2]
        let storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        if let crate::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            assert_eq!(typed, &[6.0, 15.0]);
        }
    }

    /// E2E: SumAll on a rank-3 input, exercising the all-dims branch
    /// of `op_to_op_params` (every dim reduced, rank-0 output).
    #[test]
    fn pipelined_realize_sum_all_rank3() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let storage = crate::from_slice_cpu(&data);
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
        if let crate::BackendStorage::Cpu(c) = &guard.inner {
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
        let storage = crate::from_slice_cpu(&[1.0_f32, 9.0, 3.0, 4.0, 2.0, 8.0]);
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
        if let crate::BackendStorage::Cpu(c) = &guard.inner {
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
        let storage = crate::from_slice_cpu(&[0.0_f32, 1.0, -1.0]);
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
        if let crate::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            // sigmoid(0) = 0.5; silu(0.5) = 0.5 * sigmoid(0.5) ≈ 0.3112
            assert!((typed[0] - 0.5 * (1.0 / (1.0 + (-0.5_f32).exp()))).abs() < 1e-6);
        }
    }

    /// E2E: chained unary ops — Const + Sqr + Sqrt should be a noop
    /// for non-negative inputs. Exercises the cache reuse path.
    #[test]
    fn pipelined_realize_chained_unary_sqr_then_sqrt() {
        let storage = crate::from_slice_cpu(&[1.0_f32, 4.0, 9.0]);
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
        if let crate::BackendStorage::Cpu(c) = &guard.inner {
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
        let a_storage = crate::from_slice_cpu(&[1.0_f32, 2.0]);
        let b_storage = crate::from_slice_cpu(&[10.0_f32, 20.0]);
        let c_storage = crate::from_slice_cpu(&[100.0_f32, 200.0]);

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
        if let crate::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            // (1+10) + 100 = 111;  (2+20) + 200 = 222.
            assert_eq!(typed, &[111.0, 222.0]);
        }
    }
}
