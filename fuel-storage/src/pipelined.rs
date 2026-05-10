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
    // Build the per-operand dtype list for the binding-table lookup —
    // inputs in order, then outputs. Variadic uniform-dtype ops
    // (Concat) collapse to the canonical `[T_in, T_out]` shorthand to
    // match how registrations index them.
    let dtypes = build_lookup_dtypes(graph, node);
    let compiled = compile_node(op_kind, &dtypes, target_backend, op_params, bindings)?;
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

/// Build the per-operand dtype list used as the binding-table lookup
/// key. Inputs in order, then the output. Variadic-uniform ops
/// (Concat) collapse to the canonical `[T_in, T_out]` shorthand to
/// match how those wrappers are registered (otherwise an N-way concat
/// would need N+1 distinct registrations per dtype).
fn build_lookup_dtypes(graph: &Graph, node: &Node) -> Vec<DType> {
    if matches!(node.op, Op::Concat { .. }) {
        // Concat: all inputs share node.dtype by construction.
        let in_dt = node
            .inputs
            .first()
            .map(|&id| graph.node(id).dtype)
            .unwrap_or(node.dtype);
        return vec![in_dt, node.dtype];
    }
    let mut dts: Vec<DType> = node
        .inputs
        .iter()
        .map(|&id| graph.node(id).dtype)
        .collect();
    dts.push(node.dtype);
    dts
}

/// Phase 7.6 step 3: SoftmaxLastDim flows through both the legacy
/// `Op::SoftmaxLastDim` variant and the registry-extended
/// `Op::Fused(FusedOps::SOFTMAX_LAST_DIM, _)` form. Both dispatch to
/// the same `OpKind::SoftmaxLastDim` binding-table entry; this helper
/// collapses the two shapes for op-to-OpKind/OpParams call sites.
fn op_is_softmax_last_dim(op: &Op) -> bool {
    match op {
        Op::SoftmaxLastDim => true,
        Op::Fused(fid, _) => *fid == fuel_graph::registry::FusedOps::SOFTMAX_LAST_DIM,
        _ => false,
    }
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
        Op::Pad { .. }    => Some(OpKind::Pad),
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
        Op::Conv2D { .. } => Some(OpKind::Conv2D),
        Op::ConvTranspose2D { .. } => Some(OpKind::ConvTranspose2D),
        Op::ReduceSumTo(_) => Some(OpKind::ReduceSumTo),
        Op::ReduceMaxTo(_) => Some(OpKind::ReduceMaxTo),
        Op::FusedLinear => Some(OpKind::FusedLinear),
        Op::FlashAttn { .. } => Some(OpKind::FlashAttn),
        Op::PagedAttn { .. } => Some(OpKind::PagedAttn),
        Op::AddScalar(_)  => Some(OpKind::Affine),
        Op::MulScalar(_)  => Some(OpKind::Affine),
        Op::Clamp { .. }  => Some(OpKind::ClampElementwise),
        Op::PowI(_)       => Some(OpKind::PowIElementwise),
        Op::Maximum       => Some(OpKind::MaximumElementwise),
        Op::Minimum       => Some(OpKind::MinimumElementwise),
        Op::Concat { .. } => Some(OpKind::Concat),
        Op::SoftmaxLastDim => Some(OpKind::SoftmaxLastDim),
        // Phase 7.6 step 3: route the registry-extended fused arm
        // through the same OpKind binding; the per-dtype CPU/CUDA
        // wrappers registered against OpKind::SoftmaxLastDim continue
        // to handle dispatch.
        Op::Fused(fid, _)
            if *fid == fuel_graph::registry::FusedOps::SOFTMAX_LAST_DIM =>
        {
            Some(OpKind::SoftmaxLastDim)
        }
        Op::RmsNormLastDim { .. } => Some(OpKind::RmsNormLastDim),
        Op::LayerNormLastDim { .. } => Some(OpKind::LayerNormLastDim),
        Op::IndexSelect { .. } => Some(OpKind::IndexSelect),
        Op::Gather { .. } => Some(OpKind::Gather),
        Op::Rope => Some(OpKind::Rope),
        Op::IndexAdd { .. } => Some(OpKind::IndexAdd),
        Op::ScatterAdd { .. } => Some(OpKind::ScatterAdd),
        Op::ArgMaxDim(_) => Some(OpKind::ArgMaxDim),
        Op::ArgMinDim(_) => Some(OpKind::ArgMinDim),
        Op::QMatMul { .. } => Some(OpKind::QMatMul),
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
        Op::QMatMul { quant_type, k, n } => {
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
        Op::FusedLinear => {
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
        // Phase 7.6 step 3: SoftmaxLastDim flows in either the legacy
        // `Op::SoftmaxLastDim` shape or the new
        // `Op::Fused(FusedOps::SOFTMAX_LAST_DIM, _)` shape; both share
        // the same shape contract (input == output == [..., last_dim]),
        // and the params are derived from the input layout (not the op
        // variant), so collapse to one body.
        op if op_is_softmax_last_dim(op) => {
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
            OpParams::Flip { outer_count, dim_size, inner_count }
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
                outer_count, dim_size, inner_count, shift: *shift,
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
            OpParams::CumSum { outer_count, dim_size, inner_count }
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
        Op::Rope => {
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
        Op::RmsNormLastDim { eps } | Op::LayerNormLastDim { eps } => {
            let il = input_layout(node.inputs[0]);
            let dims = il.shape().dims();
            if dims.is_empty() {
                return Err(Error::Msg(format!(
                    "Op::{:?} requires rank ≥ 1",
                    node.op,
                ))
                .bt());
            }
            let last_dim = *dims.last().unwrap();
            let outer_count: usize = dims[..dims.len() - 1].iter().product();
            OpParams::NormLastDim { outer_count, last_dim, eps: *eps }
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
            }
        }
        Op::AddScalar(c) => OpParams::Affine { mul: 1.0, add: *c },
        Op::MulScalar(c) => OpParams::Affine { mul: *c, add: 0.0 },
        Op::Clamp { min, max } => OpParams::Clamp { min: *min, max: *max },
        Op::PowI(exp) => OpParams::PowI { exp: *exp },
        Op::Conv2D { stride, padding, groups } => {
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
        Op::PagedAttn { softmax_scale, block_size, softcap } => {
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
        Op::FlashAttn { softmax_scale, causal, window_size_left, window_size_right, softcap } => {
            // Inputs[0]=q [B,Hq,Sq,D], inputs[1]=k [B,Hkv,Sk,D],
            // inputs[2]=v [B,Hkv,Sk,D], inputs[3]=alibi_slopes [Hq] (optional).
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
            OpParams::FlashAttn {
                b: q_dims[0],
                hq: q_dims[1],
                hkv: k_dims[1],
                sq: q_dims[2],
                sk: k_dims[2],
                d: q_dims[3],
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
        Op::ConvTranspose2D { stride, padding, output_padding, dilation, groups } => {
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
            //
            // We assemble a parallel `kernel_layouts` vec — the layout
            // of the bytes the kernel actually receives. After
            // auto-contiguize, that's `Layout::contiguous(shape)` for
            // the input's shape; for inputs already contiguous we use
            // the cached layout directly. Output layout comes last.
            let mut input_arcs: Vec<Arc<RwLock<Storage>>> = Vec::with_capacity(item.inputs.len());
            let mut kernel_layouts: Vec<fuel_core_types::Layout> =
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
                let already_contig =
                    in_layout.is_contiguous() && in_layout.start_offset() == 0;
                let strided_ok =
                    kernel_handles_strided && in_layout.start_offset() == 0;
                if already_contig || strided_ok {
                    input_arcs.push(in_arc);
                    kernel_layouts.push(in_layout);
                } else {
                    let contig_arc = auto_contiguize(&in_arc, &in_layout)?;
                    input_arcs.push(contig_arc);
                    kernel_layouts.push(fuel_core_types::Layout::contiguous(
                        in_layout.shape().clone(),
                    ));
                }
            }
            kernel_layouts.push(item.output_layout.clone());

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

            execute_compiled(compiled, &input_arcs, &mut output_arcs, &kernel_layouts)?;

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

    /// E2E: Const + Cast(f32→f64) — verifies cast through the
    /// pipelined executor. Output Storage has the target dtype;
    /// bytes encode the widened values.
    #[test]
    fn pipelined_realize_cast_f32_to_f64() {
        let storage = crate::from_slice_cpu(&[1.5_f32, -2.25, 100.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f64>().unwrap(), &[1.5_f64, -2.25, 100.0]);
    }

    /// E2E: Const + Cast(f32→bf16) + Cast(bf16→f32) — round trip
    /// through bf16; verifies the Cast wrapper's source-dtype
    /// dispatch (different sources hit different match arms).
    /// Inputs chosen to round-trip exactly through bf16.
    #[test]
    fn pipelined_realize_cast_round_trip_via_bf16() {
        let storage = crate::from_slice_cpu(&[1.0_f32, 2.0, -3.0, 0.5]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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

    /// E2E: batched matmul through the pipelined executor. Two
    /// batches of [2, 2] @ [2, 2]; the kernel iterates over
    /// `batch_count` and produces concatenated outputs.
    #[test]
    fn pipelined_realize_matmul_batched_2x_2x2_times_2x2() {
        let lhs_storage = crate::from_slice_cpu(&[
            1.0_f32, 2.0, 3.0, 4.0, // batch 0
            1.0, 0.0, 0.0, 1.0,     // batch 1 (identity)
        ]);
        let rhs_storage = crate::from_slice_cpu(&[
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let lhs = crate::from_slice_cpu(&[1.0_f64, 2.0, 3.0]);
        let rhs = crate::from_slice_cpu(&[10.0_f64, 20.0, 30.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let lhs = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, f32::NAN, 0.0]);
        let rhs = crate::from_slice_cpu(&[1.0_f32, 5.0, 3.0, f32::NAN, -0.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let lhs = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, f32::NAN, 0.0]);
        let rhs = crate::from_slice_cpu(&[1.0_f32, 5.0, 3.0, f32::NAN, -0.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let lhs = crate::from_slice_cpu(&[1.0_f32, 5.0, 3.0, f32::NAN, -1.0]);
        let rhs = crate::from_slice_cpu(&[2.0_f32, 5.0, 3.0, 0.0,      0.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        let mask: &[u8] = c.as_slice().expect("u8 view");
        // 1.0 < 2.0 → 1;  5.0 < 5.0 → 0 (strict);  3.0 < 3.0 → 0;
        // NaN < 0.0 → 0 (unordered);  -1.0 < 0.0 → 1.
        assert_eq!(mask, &[1, 0, 0, 0, 1]);
    }

    /// E2E: Op::Le F32 → U8 mask. Distinct from Lt at the equal slot
    /// (`5.0 <= 5.0` = 1, vs Lt's `5.0 < 5.0` = 0). NaN unordered → 0.
    #[test]
    fn pipelined_realize_le_f32_to_u8_mask() {
        let lhs = crate::from_slice_cpu(&[1.0_f32, 5.0, 3.0, f32::NAN, -1.0]);
        let rhs = crate::from_slice_cpu(&[2.0_f32, 5.0, 2.0, 0.0,      0.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        let mask: &[u8] = c.as_slice().expect("u8 view");
        // 1.0 <= 2.0 → 1;  5.0 <= 5.0 → 1 (key Lt difference);
        // 3.0 <= 2.0 → 0;  NaN <= 0.0 → 0 (unordered);  -1.0 <= 0.0 → 1.
        assert_eq!(mask, &[1, 1, 0, 0, 1]);
    }

    /// E2E: Op::Gt F32 → U8 mask. Strict-greater: equality slot is
    /// `0`. NaN unordered → `0`.
    #[test]
    fn pipelined_realize_gt_f32_to_u8_mask() {
        let lhs = crate::from_slice_cpu(&[3.0_f32, 5.0, 2.0, f32::NAN, 1.0]);
        let rhs = crate::from_slice_cpu(&[2.0_f32, 5.0, 3.0, 0.0,      0.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let lhs = crate::from_slice_cpu(&[3.0_f32, 5.0, 2.0, f32::NAN, 0.0]);
        let rhs = crate::from_slice_cpu(&[2.0_f32, 5.0, 3.0, 0.0,      0.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let lhs = crate::from_slice_cpu(&[1.0_f64, 2.0, 3.0]);
        let rhs = crate::from_slice_cpu(&[1.0_f64, 2.0, 4.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let cond_storage = crate::from_slice_cpu(&[1u8, 0, 1, 0, 1]);
        let a_storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]);
        let b_storage = crate::from_slice_cpu(&[10.0_f32, 20.0, 30.0, 40.0, 50.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let a_storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
        let b_storage = crate::from_slice_cpu(&[1.0_f32, 5.0, 3.0]);
        let pick_storage = crate::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);
        let fb_storage = crate::from_slice_cpu(&[99.0_f32, 99.0, 99.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let w_storage = crate::Storage::new(
            crate::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_bytes(&w_bytes),
            ),
            DType::U32,
        );

        let act_vec: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let act_storage = crate::from_slice_cpu(&act_vec);

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
                op: Op::QMatMul { quant_type: QuantType::Q4_0, k: 32, n: 2 },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let w_storage = crate::Storage::new(
            crate::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_bytes(&w_bytes),
            ),
            DType::U32,
        );
        let act_vec: Vec<f32> = (1..=k).map(|x| x as f32).collect();
        let act_storage = crate::from_slice_cpu(&act_vec);
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
                op: Op::QMatMul { quant_type: QuantType::Q5_0, k, n },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let w_storage = crate::Storage::new(
            crate::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_bytes(&w_bytes),
            ),
            DType::U32,
        );
        let act_vec: Vec<f32> = (1..=k).map(|x| x as f32 / 100.0).collect();
        let act_storage = crate::from_slice_cpu(&act_vec);
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
                op: Op::QMatMul { quant_type: QuantType::Q6K, k, n },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let storage = crate::from_slice_cpu(&v);
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (in_id, op_id) = {
            let mut g = graph.write().unwrap();
            let in_id = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[1, 2]), dtype: DType::BF16,
            });
            let op_id = g.push(Node {
                op: Op::RmsNormLastDim { eps: 0.0 }, inputs: vec![in_id],
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let lhs = crate::from_slice_cpu(&lhs_v);
        let rhs = crate::from_slice_cpu(&rhs_v);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let storage = crate::from_slice_cpu(&v);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let lhs = crate::from_slice_cpu(&lhs_vec);
        let rhs = crate::from_slice_cpu(&rhs_vec);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let storage = crate::from_slice_cpu(&v);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let storage = crate::from_slice_cpu(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f64>().unwrap(), &[6.0_f64, 15.0]);
    }

    /// E2E: F64 matmul through the pipelined executor.
    #[test]
    fn pipelined_realize_matmul_2x3_times_3x2_f64() {
        let lhs = crate::from_slice_cpu(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = crate::from_slice_cpu(&[7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f64>().unwrap(), &[58.0_f64, 64.0, 139.0, 154.0]);
    }

    /// E2E: F64 unary chain — Const + Sqr + Sqrt — verifies that
    /// the kernel-binding lookup picks the f64 entries when the
    /// graph nodes carry DType::F64.
    #[test]
    fn pipelined_realize_sqr_then_sqrt_f64() {
        let storage = crate::from_slice_cpu(&[1.0_f64, 4.0, 9.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f64>().unwrap(), &[1.0_f64, 4.0, 9.0]);
    }

    /// E2E: ArgMaxDim — produces U32 output indices.
    #[test]
    fn pipelined_realize_argmax_dim() {
        // input [2, 3] = [[1, 5, 2], [9, 0, 4]]
        // argmax dim=1 → [1, 0]
        let storage = crate::from_slice_cpu(&[1.0_f32, 5.0, 2.0, 9.0, 0.0, 4.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<u32>().unwrap(), &[1u32, 0]);
    }

    /// E2E: IndexAdd along outer dim — accumulate updates into a
    /// rank-1 base tensor at indexed positions.
    #[test]
    fn pipelined_realize_index_add_simple() {
        let base = crate::from_slice_cpu(&[10.0_f32, 20.0, 30.0]);
        let indices = crate::from_slice_cpu(&[0u32, 0]);
        let src = crate::from_slice_cpu(&[1.0_f32, 2.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let base = crate::from_slice_cpu(&[0.0_f32; 6]);
        let indices = crate::from_slice_cpu(&[0u32, 1, 2, 0]);
        let src = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[1.0, 4.0, 0.0, 2.0, 3.0, 0.0]);
    }

    /// E2E: Rope through the pipelined executor. cos=0, sin=1
    /// rotates the head_dim halves with sign per the rotate_half
    /// convention.
    #[test]
    fn pipelined_realize_rope_pi_over_two() {
        // x [1, 1, 4] = [1, 2, 3, 4]. cos=[0,0,0,0], sin=[1,1,1,1].
        // Expected: [-3, -4, 1, 2].
        let x = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let cos = crate::from_slice_cpu(&[0.0_f32, 0.0, 0.0, 0.0]);
        let sin = crate::from_slice_cpu(&[1.0_f32, 1.0, 1.0, 1.0]);
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
                op: Op::Rope, inputs: vec![x, c, s],
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[-3.0, -4.0, 1.0, 2.0]);
    }

    /// E2E: Gather along inner dim. Source [2, 4]; indices [2, 3];
    /// output [2, 3] = picks from each row by per-row indices.
    #[test]
    fn pipelined_realize_gather_inner_dim() {
        let source = crate::from_slice_cpu(&[
            10.0_f32, 20.0, 30.0, 40.0,
            50.0, 60.0, 70.0, 80.0,
        ]);
        let indices = crate::from_slice_cpu(&[0u32, 2, 1, 3, 0, 0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let table = crate::from_slice_cpu(&[
            10.0_f32, 11.0, 12.0,    // row 0
            20.0, 21.0, 22.0,        // row 1
            30.0, 31.0, 32.0,        // row 2
            40.0, 41.0, 42.0,        // row 3
        ]);
        let indices = crate::from_slice_cpu(&[2u32, 0, 2]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[30.0, 31.0, 32.0, 10.0, 11.0, 12.0, 30.0, 31.0, 32.0]
        );
    }

    /// E2E: RmsNormLastDim on a 2-row input. Each row's output
    /// has unit RMS up to the eps-induced bias.
    #[test]
    fn pipelined_realize_rms_norm_last_dim() {
        let storage = crate::from_slice_cpu(&[
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
                op: Op::RmsNormLastDim { eps: 0.0 },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let storage = crate::from_slice_cpu(&[
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
                op: Op::LayerNormLastDim { eps: 0.0 },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let storage = crate::from_slice_cpu(&[
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
                op: Op::SoftmaxLastDim, inputs: vec![in_id],
                shape: Shape::from_dims(&[2, 4]), dtype: DType::F32,
            });
            g.set_target_backend(op_id, BackendId::Cpu);
            (in_id, op_id)
        };
        let mut inputs = StorageCache::new();
        inputs.insert(in_id, Arc::new(RwLock::new(storage)));
        let (result_arc, _) = PipelinedExecutor::realize(graph, op_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let a = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = crate::from_slice_cpu(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0]
        );
    }

    /// E2E: Concat with three inputs along outer dim — verifies
    /// variable-arity input handling through the executor.
    #[test]
    fn pipelined_realize_concat_three_inputs_outer() {
        let a = crate::from_slice_cpu(&[1.0_f32, 2.0]);
        let b = crate::from_slice_cpu(&[3.0_f32, 4.0]);
        let c = crate::from_slice_cpu(&[5.0_f32, 6.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// E2E: AddScalar — graph emits Op::AddScalar; the executor
    /// maps it to OpKind::Affine with mul=1, add=c.
    #[test]
    fn pipelined_realize_add_scalar() {
        let storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[11.0, 12.0, 13.0]);
    }

    /// E2E: Clamp — clamp values to [-2, 2].
    #[test]
    fn pipelined_realize_clamp() {
        let storage = crate::from_slice_cpu(&[-5.0_f32, 0.5, 100.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[-2.0, 0.5, 2.0]);
    }

    /// E2E: Maximum — elementwise tensor max.
    #[test]
    fn pipelined_realize_maximum_elementwise() {
        let lhs_storage = crate::from_slice_cpu(&[1.0_f32, 5.0, -3.0]);
        let rhs_storage = crate::from_slice_cpu(&[2.0_f32, 1.0, -1.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[2.0, 5.0, -1.0]);
    }

    /// E2E: Const + Const + Conv2D — the 2×2 sum-kernel test from
    /// byte_kernels driven through the pipelined executor.
    #[test]
    fn pipelined_realize_conv2d_2x2_sum_kernel() {
        // x [1, 1, 3, 3]: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        // weight [1, 1, 2, 2]: all-ones
        // → out [1, 1, 2, 2]: [[12, 16], [24, 28]]
        let x_storage = crate::from_slice_cpu(&[
            1.0_f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let w_storage = crate::from_slice_cpu(&[1.0_f32, 1.0, 1.0, 1.0]);

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
                op: Op::Conv2D { stride: (1, 1), padding: (0, 0), groups: 1 },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[12.0, 16.0, 24.0, 28.0]);
    }

    /// E2E: Conv2D with bias (3 inputs).
    #[test]
    fn pipelined_realize_conv2d_with_bias() {
        let x_storage = crate::from_slice_cpu(&[
            1.0_f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let w_storage = crate::from_slice_cpu(&[1.0_f32, 1.0, 1.0, 1.0]);
        let bias_storage = crate::from_slice_cpu(&[100.0_f32]);

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
                op: Op::Conv2D { stride: (1, 1), padding: (0, 0), groups: 1 },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[112.0, 116.0, 124.0, 128.0]);
    }

    /// E2E: Conv2D in F64 — same 2x2 sum-kernel test as F32, on doubles.
    #[test]
    fn pipelined_realize_conv2d_f64() {
        let x_storage = crate::from_slice_cpu(&[
            1.0_f64, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let w_storage = crate::from_slice_cpu(&[1.0_f64, 1.0, 1.0, 1.0]);

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
                op: Op::Conv2D { stride: (1, 1), padding: (0, 0), groups: 1 },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f64>().unwrap(), &[12.0, 16.0, 24.0, 28.0]);
    }

    /// E2E: Conv2D in BF16 — f32-accumulator path. Tolerant compare.
    #[test]
    fn pipelined_realize_conv2d_bf16() {
        let x_data: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            .iter().map(|v| half::bf16::from_f32(*v)).collect();
        let w_data: Vec<half::bf16> = [1.0_f32, 1.0, 1.0, 1.0]
            .iter().map(|v| half::bf16::from_f32(*v)).collect();
        let x_storage = crate::from_slice_cpu(&x_data);
        let w_storage = crate::from_slice_cpu(&w_data);

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
                op: Op::Conv2D { stride: (1, 1), padding: (0, 0), groups: 1 },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let x_storage = crate::from_slice_cpu(&x_data);
        let w_storage = crate::from_slice_cpu(&w_data);

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
                op: Op::Conv2D { stride: (1, 1), padding: (0, 0), groups: 1 },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let q = crate::from_slice_cpu(&[2.0_f32, 0.0]);
        let k_cache = crate::from_slice_cpu(&[1.0_f32, 0.0, 0.0, 1.0]);
        let v_cache = crate::from_slice_cpu(&[10.0_f32, 0.0, 0.0, 10.0]);
        let block_table_u32 = crate::Storage::new(
            crate::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_slice(&[0_u32]),
            ),
            DType::U32,
        );
        let context_lens_u32 = crate::Storage::new(
            crate::BackendStorage::Cpu(
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
                op: Op::PagedAttn { softmax_scale: 1.0, block_size: 2, softcap: None },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let q = crate::from_slice_cpu(&q_v);
        let k_cache = crate::from_slice_cpu(&kc_v);
        let v_cache = crate::from_slice_cpu(&vc_v);
        let bt = crate::Storage::new(
            crate::BackendStorage::Cpu(
                fuel_cpu_backend::CpuStorageBytes::from_slice(&[0_u32]),
            ),
            DType::U32,
        );
        let cl = crate::Storage::new(
            crate::BackendStorage::Cpu(
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
                op: Op::PagedAttn { softmax_scale: 1.0, block_size: 2, softcap: None },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let q = crate::from_slice_cpu(&[2.0_f32, 0.0]);
        let k = crate::from_slice_cpu(&[1.0_f32, 0.0, 0.0, 1.0]);
        let v = crate::from_slice_cpu(&[10.0_f32, 0.0, 0.0, 10.0]);
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
                op: Op::FlashAttn {
                    softmax_scale: 1.0,
                    causal: false,
                    window_size_left: None,
                    window_size_right: None,
                    softcap: None,
                },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let q = crate::from_slice_cpu(&[1.0_f32, 0.0, 0.0, 1.0]);
        let k = crate::from_slice_cpu(&[1.0_f32, 0.0, 0.0, 1.0]);
        let v = crate::from_slice_cpu(&[5.0_f32, 6.0, 7.0, 8.0]);
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
                op: Op::FlashAttn {
                    softmax_scale: 1.0, causal: true,
                    window_size_left: None, window_size_right: None,
                    softcap: None,
                },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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

    /// E2E: FlashAttn BF16 — same single-row test as f32, tolerant.
    #[test]
    fn pipelined_realize_flash_attn_bf16() {
        let q_v: Vec<half::bf16> = [2.0_f32, 0.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let k_v: Vec<half::bf16> = [1.0_f32, 0.0, 0.0, 1.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let v_v: Vec<half::bf16> = [10.0_f32, 0.0, 0.0, 10.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let q = crate::from_slice_cpu(&q_v);
        let k = crate::from_slice_cpu(&k_v);
        let v = crate::from_slice_cpu(&v_v);
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
                op: Op::FlashAttn {
                    softmax_scale: 1.0, causal: false,
                    window_size_left: None, window_size_right: None,
                    softcap: None,
                },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let a = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = crate::from_slice_cpu(&[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let bias = crate::from_slice_cpu(&[10.0_f32, 20.0]);
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
                op: Op::FusedLinear, inputs: vec![a, b, bias],
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[14.0, 25.0, 20.0, 31.0]);
    }

    /// E2E: FusedLinear F64 — same shape test on doubles.
    #[test]
    fn pipelined_realize_fused_linear_f64() {
        let a = crate::from_slice_cpu(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = crate::from_slice_cpu(&[1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let bias = crate::from_slice_cpu(&[10.0_f64, 20.0]);
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
                op: Op::FusedLinear, inputs: vec![a, b, bias],
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let a = crate::from_slice_cpu(&a_v);
        let b = crate::from_slice_cpu(&b_v);
        let bias = crate::from_slice_cpu(&bias_v);
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
                op: Op::FusedLinear, inputs: vec![a, b, bias],
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let v = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let s = crate::from_slice_cpu(&v);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let v = crate::from_slice_cpu(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
    }

    /// E2E: Op::ReduceMaxTo F32 — drop the leading axis with max-reduce.
    #[test]
    fn pipelined_realize_reduce_max_to_f32_drops_leading_axis() {
        // Input [2,3]: row 0 = [1, 7, 3], row 1 = [4, 2, 6]. Max along
        // dim 0: [4, 7, 6].
        let v = crate::from_slice_cpu(&[1.0_f32, 7.0, 3.0, 4.0, 2.0, 6.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[4.0, 7.0, 6.0]);
    }

    /// E2E: Op::ReduceMaxTo F32 — keep-dim with 1 in the trailing axis.
    /// Mirrors the SoftmaxLastDim lowering's max-side shape: input
    /// [..., last] → [..., 1].
    #[test]
    fn pipelined_realize_reduce_max_to_f32_keepdim_trailing() {
        // Input [2, 3]: row maxes = [3, 6].
        let v = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(c.as_slice::<f32>().unwrap(), &[3.0, 6.0]);
    }

    /// E2E: ReduceSumTo BF16 — tolerant compare via f32-acc.
    #[test]
    fn pipelined_realize_reduce_sum_to_bf16() {
        let v: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter().map(|x| half::bf16::from_f32(*x)).collect();
        let s = crate::from_slice_cpu(&v);
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let x_storage = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let w_storage = crate::from_slice_cpu(&[1.0_f32, 1.0, 1.0, 1.0]);
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
                op: Op::ConvTranspose2D {
                    stride: (1, 1), padding: (0, 0),
                    output_padding: (0, 0), dilation: (1, 1), groups: 1,
                },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0],
        );
    }

    /// E2E: ConvTranspose2D F64 — same shape test.
    #[test]
    fn pipelined_realize_conv_transpose2d_f64() {
        let x_storage = crate::from_slice_cpu(&[1.0_f64, 2.0, 3.0, 4.0]);
        let w_storage = crate::from_slice_cpu(&[1.0_f64, 1.0, 1.0, 1.0]);
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
                op: Op::ConvTranspose2D {
                    stride: (1, 1), padding: (0, 0),
                    output_padding: (0, 0), dilation: (1, 1), groups: 1,
                },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let x_storage = crate::from_slice_cpu(&x);
        let w_storage = crate::from_slice_cpu(&w);
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
                op: Op::ConvTranspose2D {
                    stride: (1, 1), padding: (0, 0),
                    output_padding: (0, 0), dilation: (1, 1), groups: 1,
                },
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
        let lhs_storage = crate::from_slice_cpu(&[
            1.0_f32, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ]);
        let rhs_storage = crate::from_slice_cpu(&[
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
        let crate::BackendStorage::Cpu(c) = &guard.inner;
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
