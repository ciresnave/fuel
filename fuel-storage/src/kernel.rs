//! Kernel reference + op parameters. Phase 7.5 B1.
//!
//! [`KernelRef`] is the uniform function-pointer type that every
//! per-backend kernel implementation matches. Phase B's dispatch
//! resolver picks one at DAG construction time and stores it on the
//! graph node; the executor walks the graph calling stored kernel
//! references without per-op match-on-dtype.
//!
//! [`OpParams`] is the typed extras bag — one variant per op family
//! that needs auxiliary data beyond inputs and outputs. Most
//! elementwise ops use `OpParams::None`; reductions carry their
//! reduce dims; conv2d carries kernel/stride/padding; etc.
//!
//! ## Status
//!
//! B1 (this commit): type definitions only. Nothing yet wires
//! KernelRef into Node or invokes it. Subsequent B sub-phases:
//! - B2: Node gains `kernel: Option<KernelRef>` field.
//! - B3: dispatch resolver `(op, dtype, registry) -> KernelRef`.
//! - B4: pipelined compilation (channels + threads).
//! - B5: first op family migrated through the new path.

use std::sync::Arc;
use std::sync::RwLock;

use fuel_core_types::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use fuel_core_types::Result;

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

    /// Reduction (sum, max, mean, …) along specific dims with
    /// optional keepdim semantics.
    Reduce {
        dims: Vec<usize>,
        keepdim: bool,
    },

    /// Matrix multiplication with optional transpose flags.
    /// Single-batch and batched matmul share this variant; batch
    /// shape lives on the Storage's Layout.
    Matmul {
        transpose_lhs: bool,
        transpose_rhs: bool,
    },

    /// 1D convolution geometry (forward path).
    Conv1D(ParamsConv1D),

    /// 2D convolution geometry (forward path).
    Conv2D(ParamsConv2D),

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
            dims: vec![0, 1],
            keepdim: false,
        };
        let _ = OpParams::Matmul {
            transpose_lhs: false,
            transpose_rhs: true,
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
