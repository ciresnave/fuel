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

use fuel_core_types::dispatch::OpKind;
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, Result};

use crate::kernel::{KernelBindingTable, KernelRef, OpParams};
use crate::Storage;

/// A graph node plus its resolved kernel function pointer and
/// op-specific parameters. Produced by [`compile_node`]; consumed
/// by [`execute_compiled`].
#[derive(Debug)]
pub struct CompiledNode {
    /// The op family this node implements.
    pub op: OpKind,
    /// The node's dtype (output dtype; per-input dtypes match the
    /// inputs' Storage::dtype).
    pub dtype: DType,
    /// Which backend's kernel was selected.
    pub backend: BackendId,
    /// Resolved kernel function pointer. Looked up once at compile
    /// time; the executor calls this directly.
    pub kernel: KernelRef,
    /// Op-specific parameters. Most ops use `OpParams::None`;
    /// reductions / conv / slice carry their config here.
    pub op_params: OpParams,
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
    dtype: DType,
    backend: BackendId,
    op_params: OpParams,
    bindings: &KernelBindingTable,
) -> Result<CompiledNode> {
    let kernel = bindings.lookup(op, dtype, backend)?;
    Ok(CompiledNode {
        op,
        dtype,
        backend,
        kernel,
        op_params,
    })
}

/// Run a compiled node against the given inputs/outputs. The output
/// `Storage`s must be pre-allocated (the executor's responsibility,
/// using the node's shape + dtype).
///
/// Production-correct: surfaces kernel errors as `Result`; never
/// panics on dispatch mismatch (the wrapper functions handle that).
pub fn execute_compiled(
    compiled: &CompiledNode,
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
) -> Result<()> {
    (compiled.kernel)(inputs, outputs, &compiled.op_params)
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
            DType::F32,
            BackendId::Cpu,
            OpParams::None,
            &bindings,
        )
        .expect("compile");

        assert_eq!(compiled.op, OpKind::AddElementwise);
        assert_eq!(compiled.dtype, DType::F32);
        assert_eq!(compiled.backend, BackendId::Cpu);

        let lhs = crate::from_slice_cpu(&[5.0_f32, 6.0, 7.0]);
        let rhs = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
        let out = crate::alloc_cpu_zeroed(DType::F32, 3).unwrap();
        let inputs = vec![Arc::new(RwLock::new(lhs)), Arc::new(RwLock::new(rhs))];
        let mut outputs = vec![Arc::new(RwLock::new(out))];

        execute_compiled(&compiled, &inputs, &mut outputs).expect("execute");

        let out_guard = outputs[0].read().unwrap();
        if let crate::BackendStorage::Cpu(c) = &out_guard.inner {
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
            DType::F32,
            BackendId::Cpu,
            OpParams::None,
            &bindings,
        );
        assert!(result.is_err());
    }
}
