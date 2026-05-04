//! Kernel reference + op parameters + binding table. Phase 7.5 B1+B5.
//!
//! [`KernelRef`] is the uniform function-pointer type that every
//! dispatch wrapper matches. Backend-specific typed kernels live in
//! their backend crates (e.g. `fuel_cpu_backend::byte_kernels`); the
//! *wrapper* functions here in fuel-storage bridge the dispatch-
//! erased `Storage` to those typed kernels by matching on
//! `BackendStorage::Cpu(...)` etc.
//!
//! [`OpParams`] is the typed extras bag — one variant per op family
//! that needs auxiliary data beyond inputs and outputs. Most
//! elementwise ops use `OpParams::None`; reductions carry their
//! reduce dims; conv2d carries kernel/stride/padding; etc.
//!
//! [`KernelBindingTable`] (B5) maps `(OpKind, DType, BackendId) ->
//! KernelRef`. Backends register their dispatch wrappers via the
//! `dispatch::register_*` functions; op-builder methods consult the
//! table at DAG construction time using `Graph::target_backend(id)`
//! as the BackendId key.
//!
//! ## Architecture (cycle-avoidance)
//!
//! - Backend crates (`fuel-cpu-backend`, …): typed kernels on their
//!   concrete storage types (`CpuStorageBytes`, …). No fuel-storage
//!   dependency.
//! - fuel-storage (this crate): dispatch wrappers that match
//!   `BackendStorage::Cpu(...)`, extract the typed storage, and
//!   call the backend's typed kernel. KernelBindingTable lives here.
//! - Backend crates depend on fuel-storage? No — only the wrappers
//!   do. fuel-storage already depends on backend crates for variant
//!   types; this round-trip closes naturally.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use fuel_core_types::conv::{ParamsConv1D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use fuel_core_types::dispatch::OpKind;
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, Error, Layout, Result};

use crate::Storage;

/// Uniform function-pointer signature for per-backend op kernels.
///
/// Inputs are passed as a slice of `&Storage` to handle multi-input
/// ops (binary, ternary, custom). Outputs are passed as a slice of
/// `&mut Storage` to handle multi-output ops (topk, var_mean,
/// custom). Most ops use single-element slices.
///
/// `OpParams` carries op-family-specific extras (reduce dims,
/// conv2d geometry, etc.). Most kernels match a specific variant;
/// mismatches are programming bugs that the dispatch resolver
/// must prevent.
///
/// **Output Storage is pre-allocated** by the executor before the
/// kernel is called. Kernels write into the pre-allocated bytes;
/// they never allocate.
///
/// **Production-correct**: kernels return `Result`, never panic.
pub type KernelRef = fn(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    params: &OpParams,
) -> Result<()>;

/// Typed extras bag passed to kernels via [`KernelRef`]. One variant
/// per op family that needs auxiliary parameters; `None` for
/// parameter-less ops (most elementwise unary and binary).
#[derive(Debug, Clone)]
pub enum OpParams {
    /// Op needs no auxiliary parameters. Used by elementwise
    /// unary (relu, neg, sqr, …), elementwise binary (add, mul,
    /// sub, div, …), shape-only ops (reshape, transpose), etc.
    None,

    /// Reduction (sum, max, mean, …) along specific dims. Carries
    /// the input tensor's [`Layout`] because
    /// [`Storage`](crate::Storage) only holds bytes + dtype — the
    /// kernel needs the shape (and, eventually, strides) to walk
    /// the input multi-index. Today's CPU reduce kernels assume
    /// contiguous layout and use only `input_layout.shape()`; the
    /// strided case lands when stage 4 inserts an auto-Contiguize
    /// before non-contiguous inputs.
    ///
    /// `dims` is the sorted list of dims to reduce; `keepdim`
    /// controls whether reduced dims are retained as size-1 in
    /// the output (today fuel-graph never asks for keepdim, but
    /// the field is reserved for the future).
    Reduce {
        input_layout: Layout,
        dims: Vec<usize>,
        keepdim: bool,
    },

    /// Matrix multiplication. Carries the dimensions explicitly
    /// because [`Storage`](crate::Storage) only holds bytes + dtype;
    /// the kernel needs the batch shape and `(m, n, k)` to walk
    /// inputs and outputs.
    ///
    /// Shape contract: lhs `[..lhs_batch.., m, k]` @
    /// rhs `[..rhs_batch.., k, n]` → out `[..lhs_batch.., m, n]`.
    /// Per-axis the batch dims must either match or follow GQA-style
    /// divisibility (`lhs_dim > rhs_dim && lhs_dim % rhs_dim == 0`)
    /// — the kernel maps each lhs batch slot to the corresponding
    /// rhs slot via `rhs_axis_idx = lhs_axis_idx / n_rep_axis`.
    ///
    /// Equal-batch case: `lhs_batch_dims == rhs_batch_dims`. Rank-2
    /// case: both batch vectors are empty. Both work uniformly.
    ///
    /// Inputs are guaranteed contiguous by the executor's
    /// auto-Contiguize pass; transpose flags don't appear here
    /// because `Op::Transpose` is its own metadata-only op in
    /// fuel-graph.
    Matmul {
        lhs_batch_dims: Vec<usize>,
        rhs_batch_dims: Vec<usize>,
        m: usize,
        n: usize,
        k: usize,
    },

    /// 1D convolution geometry (forward path).
    Conv1D(ParamsConv1D),

    /// 2D convolution geometry (forward path). Carries the tuple-shaped
    /// stride/padding that fuel-graph's `Op::Conv2D` uses (asymmetric
    /// supported), plus the input/weight/output shapes the kernel
    /// needs to walk the multi-index. Storage holds only bytes +
    /// dtype, so spatial shapes flow through OpParams.
    ///
    /// Inputs: `x` shape `[N, Cin, Hin, Win]`, `weight` shape
    /// `[Cout, Cin/groups, Kh, Kw]`, optional `bias` shape `[Cout]`.
    /// Output: `[N, Cout, Hout, Wout]`.
    Conv2D {
        x_shape: [usize; 4],
        w_shape: [usize; 4],
        out_shape: [usize; 4],
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    },

    /// 1D transposed-convolution geometry.
    ConvTranspose1D(ParamsConvTranspose1D),

    /// 2D transposed-convolution geometry.
    ConvTranspose2D(ParamsConvTranspose2D),

    /// Slice along a single dim with explicit start/end/step. The
    /// dim is implicit in the input Layout's relabeling for
    /// multi-dim slice; this variant covers the simple case.
    Slice {
        dim: usize,
        start: usize,
        end: usize,
        step: usize,
    },

    /// Pad with a constant value along one or more dims.
    Pad {
        /// Per-dim (left, right) padding pairs.
        padding: Vec<(usize, usize)>,
        /// Constant fill value, encoded as a little-endian byte
        /// pattern matching the output dtype's `size_in_bytes()`.
        /// Kernels reinterpret per their dtype.
        fill_bytes: Vec<u8>,
    },

    /// Cast input dtype → target dtype. The target lives on the
    /// output Storage's `dtype` field; this variant signals the
    /// op family without requiring a re-read.
    Cast,

    /// Affine transformation `y = mul * x + add`. Used by
    /// `Tensor::affine` / `scale_and_shift`.
    Affine {
        mul: f64,
        add: f64,
    },
}

// =============================================================================
// Phase 7.5 B5 — kernel binding table
// =============================================================================

/// Maps `(OpKind, DType, BackendId)` triples to dispatch wrapper
/// functions. Built once at backend registration time (typically a
/// process-wide `OnceLock` though that's not enforced here for
/// testability), consulted at execute time via lookup.
///
/// Backends register their wrappers via the `dispatch::register_*`
/// functions in this crate (e.g.
/// [`crate::dispatch::register_cpu_kernels`]).
#[derive(Default)]
pub struct KernelBindingTable {
    bindings: HashMap<(OpKind, DType, BackendId), KernelRef>,
}

impl KernelBindingTable {
    pub fn new() -> Self {
        Self { bindings: HashMap::new() }
    }

    /// Register a dispatch wrapper for `(op, dtype, backend)`.
    /// Idempotent: re-registering with the same key overwrites.
    pub fn register(&mut self, op: OpKind, dtype: DType, backend: BackendId, kernel: KernelRef) {
        self.bindings.insert((op, dtype, backend), kernel);
    }

    /// Look up the wrapper for `(op, dtype, backend)`. Returns
    /// [`Error::NoBackendForOp`] if none registered (production-
    /// correct: no panic). The error includes diagnostic data.
    pub fn lookup(&self, op: OpKind, dtype: DType, backend: BackendId) -> Result<KernelRef> {
        self.bindings.get(&(op, dtype, backend)).copied().ok_or_else(|| {
            let available_backends: Vec<BackendId> = self
                .bindings
                .keys()
                .map(|(_, _, b)| *b)
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            let supported_combinations: Vec<(BackendId, OpKind, DType)> = self
                .bindings
                .keys()
                .map(|(o, d, b)| (*b, *o, *d))
                .collect();
            Error::NoBackendForOp {
                op,
                dtype,
                available_backends,
                supported_combinations,
            }
            .bt()
        })
    }

    /// Total bindings registered.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Empty binding table.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

impl std::fmt::Debug for KernelBindingTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelBindingTable")
            .field("bindings_count", &self.bindings.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::DType;

    /// Smoke: KernelRef can be constructed and stored.
    #[test]
    fn kernel_ref_stores_function_pointer() {
        fn dummy_kernel(
            _inputs: &[Arc<RwLock<Storage>>],
            _outputs: &mut [Arc<RwLock<Storage>>],
            _params: &OpParams,
        ) -> Result<()> {
            Ok(())
        }
        let k: KernelRef = dummy_kernel;
        // Function pointer is Copy + Clone.
        let _k2 = k;
        let _k3: KernelRef = k;
    }

    /// Smoke: OpParams variants construct cleanly and Debug-format.
    #[test]
    fn op_params_variants_construct() {
        let _ = OpParams::None;
        let _ = OpParams::Reduce {
            input_layout: Layout::contiguous(fuel_core_types::Shape::from_dims(&[4, 8])),
            dims: vec![0, 1],
            keepdim: false,
        };
        let _ = OpParams::Matmul {
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            m: 4,
            n: 8,
            k: 16,
        };
        let _ = OpParams::Slice { dim: 0, start: 0, end: 10, step: 1 };
        let _ = OpParams::Cast;
        let _ = OpParams::Affine { mul: 2.0, add: 1.0 };
        // Debug format compiles.
        let _ = format!("{:?}", OpParams::None);
    }

    /// Smoke: a hand-constructed kernel that allocates the output
    /// Storage outside this crate would be:
    ///   1. allocate output via fuel_storage::alloc_cpu_zeroed
    ///   2. wrap inputs as &[Arc<RwLock<Storage>>]
    ///   3. call the kernel
    /// Phase B5 lands the first such migration; B1 just type-checks
    /// the surface.
    #[test]
    fn kernel_ref_can_be_called() {
        fn copy_kernel(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _params: &OpParams,
        ) -> Result<()> {
            // Simplest real kernel: copy bytes from inputs[0] to outputs[0].
            let in_arc = &inputs[0];
            let out_arc = &outputs[0];
            let in_guard = in_arc.read().unwrap();
            let mut out_guard = out_arc.write().unwrap();
            let in_bytes = match &in_guard.inner {
                crate::BackendStorage::Cpu(s) => s.bytes(),
                #[allow(unreachable_patterns)]
                _ => return Ok(()),
            };
            match &mut out_guard.inner {
                crate::BackendStorage::Cpu(s) => {
                    s.bytes_mut().copy_from_slice(in_bytes);
                }
                #[allow(unreachable_patterns)]
                _ => {}
            }
            Ok(())
        }

        let input = crate::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let output = crate::alloc_cpu_zeroed(DType::F32, 4).unwrap();
        let inputs = vec![Arc::new(RwLock::new(input))];
        let mut outputs = vec![Arc::new(RwLock::new(output))];

        let k: KernelRef = copy_kernel;
        k(&inputs, &mut outputs, &OpParams::None).unwrap();

        // Output bytes match input.
        let out_guard = outputs[0].read().unwrap();
        if let crate::BackendStorage::Cpu(s) = &out_guard.inner {
            let typed: &[f32] = s.as_slice().unwrap();
            assert_eq!(typed, &[1.0, 2.0, 3.0, 4.0]);
        }
    }
}
