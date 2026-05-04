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
use fuel_core_types::{DType, Error, Result};
use fuel_graph::{topo_order, Graph, NodeId, Op};

use crate::compiled::{compile_node, execute_compiled, CompiledNode};
use crate::dispatch::global_bindings;
use crate::kernel::{KernelBindingTable, OpParams};
use crate::Storage;

/// Map from NodeId to a realized Storage Arc. Used both as the
/// input cache (passed in by the caller for pre-realized leaves)
/// and as the output cache (built up during execution).
pub type StorageCache = HashMap<NodeId, Arc<RwLock<Storage>>>;

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
    /// `None` for `Op::Const` — the executor adopts the entry from
    /// the input cache rather than calling a kernel.
    /// `Some` for any computational op.
    compiled: Option<CompiledNode>,
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
    /// Returns the realized `Storage` Arc for `target`.
    /// Production-correct: errors on any unmet precondition rather
    /// than panicking.
    pub fn realize(
        graph: Arc<RwLock<Graph>>,
        target: NodeId,
        inputs: StorageCache,
    ) -> Result<Arc<RwLock<Storage>>> {
        // Topo order computed on the calling thread to keep the
        // compiler thread free of graph-locking responsibilities.
        let order: Vec<NodeId> = {
            let g = graph.read().map_err(|_| poisoned("graph lock"))?;
            topo_order(&g, target)
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
            execute_work_item(&item, &mut cache)?;
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

        cache.remove(&target).ok_or_else(|| {
            Error::Msg(format!(
                "PipelinedExecutor::realize: target slot {:?} not populated after execution",
                target
            ))
            .bt()
        })
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

    for id in order {
        let item = compile_one(&g, id, &bindings);
        let stop_on_err = item.is_err();
        if tx.send(item).is_err() {
            return;
        }
        if stop_on_err {
            return;
        }
    }
}

/// Resolve one node into a `WorkItem`. `Op::Const` produces a
/// `WorkItem` with `compiled: None` — the executor adopts the
/// pre-realized entry from the input cache.
fn compile_one(graph: &Graph, id: NodeId, bindings: &KernelBindingTable) -> Result<WorkItem> {
    let node = graph.node(id);
    let elem_count = node.shape.elem_count();
    let inputs = node.inputs.clone();

    if matches!(node.op, Op::Const) {
        return Ok(WorkItem {
            node_id: id,
            inputs,
            elem_count,
            dtype: node.dtype,
            target_backend: BackendId::Cpu,
            compiled: None,
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

    let compiled = compile_node(op_kind, node.dtype, target_backend, OpParams::None, bindings)?;
    Ok(WorkItem {
        node_id: id,
        inputs,
        elem_count,
        dtype: node.dtype,
        target_backend,
        compiled: Some(compiled),
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
        Op::MatMul        => Some(OpKind::MatMul),
        _ => None,
    }
}

/// Execute one work item. Const adoption verifies the input cache
/// has the entry; computational ops gather inputs, allocate
/// output, call the kernel, and store the output in the cache.
fn execute_work_item(item: &WorkItem, cache: &mut StorageCache) -> Result<()> {
    // Const adoption.
    let Some(compiled) = &item.compiled else {
        if cache.contains_key(&item.node_id) {
            return Ok(());
        }
        return Err(Error::Msg(format!(
            "PipelinedExecutor: Const node {:?} not in input cache",
            item.node_id
        ))
        .bt());
    };

    // Gather input Arcs from the cache.
    let input_arcs: Vec<Arc<RwLock<Storage>>> = item
        .inputs
        .iter()
        .map(|in_id| {
            cache.get(in_id).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "PipelinedExecutor: input {:?} of {:?} not realized",
                    in_id, item.node_id
                ))
                .bt()
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Allocate output on the target backend. B4 ships CPU only;
    // multi-backend allocation lands in Phase C.
    let output = match item.target_backend {
        BackendId::Cpu => crate::alloc_cpu_zeroed(item.dtype, item.elem_count)?,
        other => {
            return Err(Error::Msg(format!(
                "PipelinedExecutor: target_backend {:?} output allocation \
                 not yet implemented (B4 ships CPU only; Phase C extends)",
                other
            ))
            .bt());
        }
    };
    let mut output_arcs = vec![Arc::new(RwLock::new(output))];

    execute_compiled(compiled, &input_arcs, &mut output_arcs)?;

    let arc = output_arcs.into_iter().next().expect("one output");
    cache.insert(item.node_id, arc);
    Ok(())
}

fn poisoned(what: &'static str) -> Error {
    Error::Msg(format!("PipelinedExecutor: {} poisoned", what)).bt()
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

        let result_arc = PipelinedExecutor::realize(graph, add_id, inputs).expect("realize");

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

        let result_arc =
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

        let result_arc = PipelinedExecutor::realize(graph, relu_id, inputs).expect("realize");
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

        let result_arc = PipelinedExecutor::realize(graph, div_id, inputs).expect("realize");
        let guard = result_arc.read().unwrap();
        if let crate::BackendStorage::Cpu(c) = &guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            assert!((typed[0] - (14.0_f32 / 3.0)).abs() < 1e-6);
            assert!((typed[1] - 12.0).abs() < 1e-6);
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

        let result_arc = PipelinedExecutor::realize(graph, silu_id, inputs).expect("realize");
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

        let result_arc = PipelinedExecutor::realize(graph, sqrt_id, inputs).expect("realize");
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

        let result_arc = PipelinedExecutor::realize(graph, abc_id, inputs).expect("realize");
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
