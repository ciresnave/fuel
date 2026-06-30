//! Compiled-node representation. Phase 7.5 B4 (interface only).
//!
//! A [`CompiledNode`] is the output of dispatch resolution: a Node
//! plus its resolved kernel function pointer plus its `OpParams`.
//! The executor consumes these and runs them.
//!
//! ## Design intent: pipelined compile + execute (deferred)
//!
//! The original B-plan called for pipelined compilation: a compiler
//! thread that walks the graph in topological order, resolves
//! kernels per node, and pushes [`CompiledNode`] entries onto a
//! `crossbeam::channel`. An executor thread consumes the channel,
//! looks up inputs from the graph's storage map (blocking when
//! inputs aren't yet realized), calls the kernel, and stores the
//! output back. Time-to-first-token wins because the compiler runs
//! ahead of the executor.
//!
//! Today's "compile" step is just a `KernelBindingTable` lookup —
//! roughly nanoseconds per node. There's no meaningful work to
//! overlap with execution. Pipelining adds `std::thread` +
//! `crossbeam::channel` + lifecycle handling for zero current win.
//!
//! When compile grows beyond trivial — residency-aware planning,
//! transfer-cost minimization, kernel auto-tuning, dynamic-shape
//! specialization — pipelining lands as a follow-up. The
//! interfaces in this module ([`CompiledNode`] +
//! [`compile_node`] + [`execute_compiled`]) are designed so the
//! threaded variant slots in without changing call sites.

use std::sync::{Arc, RwLock};

use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;
#[cfg_attr(not(feature = "cuda"), allow(unused_imports))]
use fuel_ir::Error;
use fuel_ir::{DType, Layout, Result};

use crate::kernel::{KernelBindingTable, KernelCaps, KernelDTypes, KernelRef, OpParams};
use fuel_graph::{Graph, NodeId};
use fuel_memory::Storage;

/// A graph node plus its resolved kernel function pointer and
/// op-specific parameters. Produced by [`compile_node`]; consumed
/// by [`execute_compiled`].
#[derive(Debug)]
pub struct CompiledNode {
    /// The op family this node implements.
    pub op: OpKind,
    /// Per-operand dtypes used as the binding-table lookup key.
    /// Inputs in order, then outputs. For uniform-precision ops the
    /// list is `[T, T, ..., T]`; for mixed-precision ops it carries
    /// the distinguishing combo. The output dtype is always the last
    /// entry.
    pub dtypes: KernelDTypes,
    /// Which backend's kernel was selected.
    pub backend: BackendId,
    /// Resolved kernel function pointer. Looked up once at compile
    /// time; the executor calls this directly.
    pub kernel: KernelRef,
    /// Capability flags registered alongside the kernel. The executor
    /// consults `caps.strided_input` to decide whether to skip
    /// auto-Contiguize for non-contiguous inputs.
    pub caps: KernelCaps,
    /// Op-specific parameters. Most ops use `OpParams::None`;
    /// reductions / conv / slice carry their config here.
    pub op_params: OpParams,
}

impl CompiledNode {
    /// The output dtype — last entry in `dtypes`. Convenience accessor
    /// for callers that only need the output type (e.g. shape inference,
    /// allocation).
    pub fn output_dtype(&self) -> DType {
        *self
            .dtypes
            .last()
            .expect("CompiledNode::dtypes must have at least one entry (the output)")
    }
}

/// Resolve a node's kernel from the binding table and return a
/// [`CompiledNode`] ready for execution. Production-correct: surfaces
/// `NoBackendForOp` if the table doesn't have a binding for the
/// requested triple.
///
/// Today this is one HashMap lookup. When dispatch grows more
/// sophisticated (residency-aware, cost-aware, auto-tuning), the
/// added work happens here. The interface stays the same.
pub fn compile_node(
    op: OpKind,
    dtypes: &[DType],
    backend: BackendId,
    op_params: OpParams,
    bindings: &KernelBindingTable,
) -> Result<CompiledNode> {
    let (kernel, caps) = bindings.lookup_with_caps(op, dtypes, backend)?;
    Ok(CompiledNode {
        op,
        dtypes: KernelDTypes::from_slice(dtypes),
        backend,
        kernel,
        caps,
        op_params,
    })
}

/// The diagnostic `kernel_source` tag the executor dispatches for `node`,
/// derived from the SAME graph stamp + registry the executor resolves
/// through: `op_to_op_kind` + `build_lookup_dtypes` + `graph.target_backend`
/// → the first-registered binding at `(op, dtypes, backend)` (the entry
/// [`compile_node`]'s `lookup_with_caps` picks). `None` when the node has no
/// dispatch mapping (view/structural op) or no registered binding.
///
/// This is the post-realize attribution the bridge reports for the Judge
/// telemetry. Step D moved it off the plan's `AlternativeSet::winner()`: the
/// production realize path dispatches via the binding-table lookup (no plan),
/// so the first-registered binding IS the matching attribution.
pub fn dispatched_kernel_source(
    graph: &Graph,
    node: NodeId,
    bindings: &KernelBindingTable,
) -> Option<&'static str> {
    let n = graph.node(node);
    let op = crate::pipelined::op_to_op_kind(&n.op)?;
    let dtypes = crate::pipelined::build_lookup_dtypes(graph, n);
    let backend = graph.target_backend(node)?;
    bindings
        .lookup_alternatives(op, &dtypes, backend)
        .first()
        .map(|e| e.kernel_source)
}

/// Run a compiled node against the given inputs/outputs. The output
/// `Storage`s must be pre-allocated (the executor's responsibility,
/// using the node's shape + dtype).
///
/// `layouts` parallels `inputs.append(outputs)` and carries the
/// stride/shape/offset metadata that stride-aware kernels consume.
/// Today's contiguous-only kernels ignore it; the executor's
/// auto-Contiguize pass guarantees every input layout is
/// `Layout::contiguous(shape)` before this is called.
///
/// Production-correct: surfaces kernel errors as `Result`; never
/// panics on dispatch mismatch (the wrapper functions handle that).
pub fn execute_compiled(
    compiled: &CompiledNode,
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
) -> Result<CompletionHandle> {
    (compiled.kernel)(inputs, outputs, layouts, &compiled.op_params)?;
    // Step E A4b-1: the kernel has *enqueued* its work (CUDA: launched on the
    // device stream and returned without a per-op sync, per A3; CPU/Vulkan: see
    // below). Produce a completion handle FROM THE OUTPUT STORAGE — the kernel
    // reaches the device via the `Storage`, not an executor-held backend, so the
    // output's `BackendStorage` is where we find the device/stream to record a
    // signal on. The executor stores this handle and defers the wait to a
    // dependency / realize-end boundary instead of `wait`ing inline here.
    produce_pending(compiled.backend, outputs)
}

/// Step E A4b-1: build the [`CompletionHandle`] for a just-launched kernel by
/// inspecting `outputs[0]`'s backend storage.
///
/// - **CUDA** → [`CompletionHandle::Pending`] wrapping an [`Event`] recorded on
///   the output's device stream (signals when this kernel + all prior stream
///   work completes; one stream per device makes a per-node event a sufficient
///   stream marker — A4b-1 §1.2).
/// - **CPU** → [`CompletionHandle::Ready`] (the kernel computed in-process; the
///   work is already done).
/// - **Vulkan** → [`CompletionHandle::Ready`]. The per-op compute path keeps
///   RECORDING into the deferred batch and returns `Ready` during the walk (no
///   per-op handle, no eager submit — eager submission at sub-DAG boundaries is
///   A4b-4). A4b-2's async split lives at the executor's realize-end drain
///   (`drain_vulkan_pending`): it `submit_pending`s the open batch (the per-batch
///   fence) then waits it through a `VulkanCompletion` handle — byte-identical to
///   A2's atomic submit+wait, just split. The A2 lazy-batch model still carries
///   intra-walk Vulkan correctness (same-queue order + the eviction `force_flush`).
///
/// Multi-output kernels share one backing buffer (one `BackendStorage`), so
/// `outputs[0]` is the device handle for every slot — a single event covers them.
fn produce_pending(
    _backend: BackendId,
    outputs: &[Arc<RwLock<Storage>>],
) -> Result<CompletionHandle> {
    // `_backend` (= `compiled.backend`) records the *intent*; we read the
    // device truth off the output storage's variant instead, which is robust
    // even where the backend stamp and the realized storage could ever diverge.
    #[cfg(not(feature = "cuda"))]
    let _ = outputs;
    #[cfg(feature = "cuda")]
    {
        if let Some(out0) = outputs.first() {
            let guard = out0
                .read()
                .map_err(|_| Error::Msg("output storage lock poisoned recording completion".into()).bt())?;
            if let fuel_memory::BackendStorage::Cuda(cuda_bytes) = &guard.inner {
                let ev = cuda_bytes.record_completion_event()?;
                return Ok(CompletionHandle::Pending(Box::new(CudaCompletion { ev })));
            }
        }
    }
    Ok(CompletionHandle::Ready)
}

/// Step E A4b-1: a CUDA [`Completion`] over a recorded [`Event`]. `wait` is a
/// blocking host wait (`cuEventSynchronize`) on the event — which fires after
/// the producing kernel and every prior op on the (single per-device) stream.
#[cfg(feature = "cuda")]
struct CudaCompletion {
    ev: fuel_cuda_backend::Event,
}

#[cfg(feature = "cuda")]
impl Completion for CudaCompletion {
    fn wait(self: Box<Self>) -> Result<()> {
        use fuel_cuda_backend::WrapErr;
        // cuEventSynchronize: blocks the host until the event completes; surfaces
        // a sticky async kernel fault on this stream as the mapped `Err`.
        self.ev.synchronize().w()
    }
}

/// A node's completion signal, returned by [`execute_compiled`]. Step E
/// Phase A1 foundation: today every dispatch is synchronous, so this is always
/// [`CompletionHandle::Ready`] (the work finished before the call returned) and
/// callers `wait` it immediately — byte-identical to the prior `Result<()>`.
/// Phases A2/A3 add an async GPU path that returns [`CompletionHandle::Pending`]
/// carrying a device fence/event the executor defers the wait on.
///
/// `#[must_use]`: dropping a `Pending` handle without `wait`ing would leak
/// un-awaited async work (harmless for today's `Ready`, a correctness bug once
/// A2/A3 land) — callers must `wait` it or thread it to the executor.
#[must_use = "a CompletionHandle must be waited on (or threaded to the executor) — dropping a Pending handle leaks async work"]
pub enum CompletionHandle {
    /// The work finished synchronously before [`execute_compiled`] returned —
    /// [`wait`](CompletionHandle::wait) is a no-op.
    Ready,
    /// The work was enqueued asynchronously; [`wait`](CompletionHandle::wait)
    /// blocks until the carried device signal fires. (No producers until A2/A3.)
    Pending(Box<dyn Completion>),
}

impl CompletionHandle {
    /// Block until this node's work has finished. No-op for [`Ready`].
    ///
    /// [`Ready`]: CompletionHandle::Ready
    pub fn wait(self) -> Result<()> {
        match self {
            CompletionHandle::Ready => Ok(()),
            CompletionHandle::Pending(c) => c.wait(),
        }
    }
}

/// An async-dispatch completion signal — a backend's wrapper over a CUDA event
/// / Vulkan fence so the executor can defer the wait past the submit point
/// (Step E Phase A2/A3). No implementors today; all dispatch is synchronous.
pub trait Completion: Send {
    /// Block until the enqueued work this handle represents has finished.
    fn wait(self: Box<Self>) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::register_cpu_kernels;

    /// E2E with the compile/execute split: register CPU kernels,
    /// compile a node, execute it. Same shape as B5's e2e test but
    /// goes through the CompiledNode interface.
    #[test]
    fn compile_then_execute_add_f32() {
        let mut bindings = KernelBindingTable::new();
        register_cpu_kernels(&mut bindings);

        let compiled = compile_node(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            OpParams::None,
            &bindings,
        )
        .expect("compile");

        assert_eq!(compiled.op, OpKind::AddElementwise);
        assert_eq!(compiled.output_dtype(), DType::F32);
        assert_eq!(compiled.backend, BackendId::Cpu);

        let lhs = fuel_memory::from_slice_cpu(&[5.0_f32, 6.0, 7.0]);
        let rhs = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
        let out = fuel_memory::alloc_cpu_zeroed(DType::F32, 3).unwrap();
        let inputs = vec![Arc::new(RwLock::new(lhs)), Arc::new(RwLock::new(rhs))];
        let mut outputs = vec![Arc::new(RwLock::new(out))];

        let layout_3 = Layout::contiguous(fuel_ir::Shape::from(vec![3]));
        let layouts = vec![layout_3.clone(), layout_3.clone(), layout_3];
        execute_compiled(&compiled, &inputs, &mut outputs, &layouts)
            .expect("execute")
            .wait()
            .expect("wait");

        let out_guard = outputs[0].read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &out_guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            assert_eq!(typed, &[6.0, 8.0, 10.0]);
        }
    }

    /// Step E A4b-1 behavior-preservation: a CPU kernel's `execute_compiled`
    /// returns `CompletionHandle::Ready` — `produce_pending` must NOT turn the
    /// (synchronous, in-process) CPU path into a `Pending` handle. The CUDA arm
    /// (Pending) is covered by the live `cuda_async_realize_live` gate.
    #[test]
    fn execute_compiled_cpu_is_ready_not_pending() {
        let mut bindings = KernelBindingTable::new();
        register_cpu_kernels(&mut bindings);
        let compiled = compile_node(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            OpParams::None,
            &bindings,
        )
        .expect("compile");

        let lhs = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0]);
        let rhs = fuel_memory::from_slice_cpu(&[3.0_f32, 4.0]);
        let out = fuel_memory::alloc_cpu_zeroed(DType::F32, 2).unwrap();
        let inputs = vec![Arc::new(RwLock::new(lhs)), Arc::new(RwLock::new(rhs))];
        let mut outputs = vec![Arc::new(RwLock::new(out))];
        let l = Layout::contiguous(fuel_ir::Shape::from(vec![2]));
        let layouts = vec![l.clone(), l.clone(), l];

        let handle = execute_compiled(&compiled, &inputs, &mut outputs, &layouts)
            .expect("execute");
        assert!(
            matches!(handle, CompletionHandle::Ready),
            "CPU execute_compiled must return Ready (A4b-1 behavior-preserving)",
        );
        handle.wait().expect("wait");
    }

    /// Step E A1: the completion-handle contract — `Ready.wait()` is a no-op
    /// success; `Pending(c).wait()` delegates to the boxed `Completion`.
    #[test]
    fn completion_handle_ready_is_noop_pending_delegates() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc as StdArc;

        CompletionHandle::Ready.wait().expect("Ready.wait() is Ok");

        struct Flag(StdArc<AtomicBool>);
        impl Completion for Flag {
            fn wait(self: Box<Self>) -> Result<()> {
                self.0.store(true, Ordering::SeqCst);
                Ok(())
            }
        }
        let waited = StdArc::new(AtomicBool::new(false));
        CompletionHandle::Pending(Box::new(Flag(waited.clone())))
            .wait()
            .expect("Pending.wait() is Ok");
        assert!(
            waited.load(Ordering::SeqCst),
            "Pending.wait() must run the boxed Completion",
        );
    }

    /// compile_node surfaces NoBackendForOp on missing binding.
    #[test]
    fn compile_node_errors_on_missing_binding() {
        let bindings = KernelBindingTable::new();  // empty
        let result = compile_node(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            OpParams::None,
            &bindings,
        );
        assert!(result.is_err());
    }
}
