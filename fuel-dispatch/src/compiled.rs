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
use fuel_ir::{DType, Layout, Result};

use crate::kernel::{KernelBindingTable, KernelCaps, KernelDTypes, KernelRef, OpParams};
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
) -> Result<()> {
    (compiled.kernel)(inputs, outputs, layouts, &compiled.op_params)
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
        execute_compiled(&compiled, &inputs, &mut outputs, &layouts).expect("execute");

        let out_guard = outputs[0].read().unwrap();
        if let fuel_memory::BackendStorage::Cpu(c) = &out_guard.inner {
            let typed: &[f32] = c.as_slice().unwrap();
            assert_eq!(typed, &[6.0, 8.0, 10.0]);
        }
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
