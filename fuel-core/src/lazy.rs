//! Phase 6a bridge: a lazy-computation-graph tensor that wraps
//! [`fuel_graph::Tensor`] and presents it through an API compatible
//! with fuel-core's eager [`Tensor`](crate::Tensor).
//!
//! # Purpose
//!
//! The Phase 6 architectural pivot moves fuel from eager execution to a
//! lazy computation graph. End state: `fuel_core::Tensor` *is* a
//! `fuel_graph::Tensor` and every model in `fuel-transformers` runs
//! through the lazy backend without per-model porting.
//!
//! The bridge is the intermediate stage. [`LazyTensor`] is a wrapper
//! around [`fuel_graph::Tensor`] that exposes the fuel-core-style
//! method API (`.add()`, `.mul()`, `.matmul()`, `.relu()`, `.shape()`,
//! `.to_vec0()`, `.to_vec1()`, ...) so callers can gradually migrate
//! from eager to lazy one function at a time. Each method appends a
//! node to the underlying [`fuel_graph::Graph`]; nothing runs until
//! you call [`LazyTensor::realize_f32`] or a sibling.
//!
//! This is NOT intended as a permanent user-facing type. It's the
//! scaffolding that makes the final merge incremental: each
//! `fuel-transformers` model can be converted to `LazyTensor` in a
//! separate PR, and once they all compile against the wrapper, the
//! type alias flips and `fuel_core::Tensor` becomes the lazy variant.
//!
//! # What's here today
//!
//! A minimal but real subset: constructors from `Vec<f32>`/`Vec<f64>`
//! and friends, shape/dtype inspection, the element-wise arithmetic
//! and unary ops most models use, matmul, softmax, layer_norm,
//! rms_norm, and realization to a typed `Vec`. Everything routes
//! through `fuel_graph::Tensor` underneath.
//!
//! Missing: autograd integration via `fuel_core::Var`, the
//! `backward()` / `apply_op*` convenience methods, safetensors
//! loading directly into `LazyTensor`s, and many of the niche
//! methods on `fuel_core::Tensor`. All of these are additive
//! extensions — they do not require changes to the bridge's
//! structural design.

use crate::{DType, Shape};
use std::sync::Arc;

/// A lazy tensor that builds a `fuel_graph::Graph` as its methods are
/// called. Cheap to clone — the underlying `fuel_graph::Tensor` is a
/// cheap handle pair `(Rc<RefCell<Graph>>, NodeId)`, so cloning just
/// bumps the `Rc` and copies the id.
#[derive(Clone, Debug)]
pub struct LazyTensor {
    inner: fuel_graph::Tensor,
}

impl LazyTensor {
    // ---- constructors ----

    /// Build an `f32` lazy tensor from flat data and a shape.
    ///
    /// `data` takes `impl Into<Arc<[f32]>>` so both `Vec<f32>` and
    /// `Arc<[f32]>` callers work without conversion. Pass an `Arc`
    /// when you already have one (e.g. model weights loaded once at
    /// startup) to avoid any copy.
    pub fn from_f32(data: impl Into<Arc<[f32]>>, shape: impl Into<Shape>) -> Self {
        Self {
            inner: fuel_graph::Tensor::from_f32(data, shape),
        }
    }

    /// Build an `f64` lazy tensor.
    pub fn from_f64(data: impl Into<Arc<[f64]>>, shape: impl Into<Shape>) -> Self {
        Self {
            inner: fuel_graph::Tensor::from_f64(data, shape),
        }
    }

    /// Build a `bf16` lazy tensor.
    pub fn from_bf16(data: impl Into<Arc<[half::bf16]>>, shape: impl Into<Shape>) -> Self {
        Self {
            inner: fuel_graph::Tensor::from_bf16(data, shape),
        }
    }

    /// Build an `f16` lazy tensor.
    pub fn from_f16(data: impl Into<Arc<[half::f16]>>, shape: impl Into<Shape>) -> Self {
        Self {
            inner: fuel_graph::Tensor::from_f16(data, shape),
        }
    }

    /// Build a `u32` (index) lazy tensor. Used for gather/scatter/
    /// index_select and similar discrete operations.
    pub fn from_u32(data: impl Into<Arc<[u32]>>, shape: impl Into<Shape>) -> Self {
        Self {
            inner: fuel_graph::Tensor::from_u32(data, shape),
        }
    }

    /// Build a const tensor of the same dtype and graph as `self`.
    /// This is the most convenient way to attach new input data to an
    /// existing computation.
    pub fn const_f32_like(
        &self,
        data: impl Into<Arc<[f32]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        Self {
            inner: self.inner.const_f32_like(data, shape),
        }
    }

    /// Unwrap the underlying `fuel_graph::Tensor`. Used by callers that
    /// need to drop down to the graph layer for operations the bridge
    /// does not yet expose.
    pub fn into_graph_tensor(self) -> fuel_graph::Tensor {
        self.inner
    }

    /// Borrow the underlying `fuel_graph::Tensor`.
    pub fn graph_tensor(&self) -> &fuel_graph::Tensor {
        &self.inner
    }

    /// Wrap an existing `fuel_graph::Tensor` in a `LazyTensor`. Useful
    /// when you have code that already builds a graph and want to
    /// present its outputs through this API.
    pub fn from_graph_tensor(t: fuel_graph::Tensor) -> Self {
        Self { inner: t }
    }

    // ---- shape / dtype inspection ----

    /// The tensor's shape.
    pub fn shape(&self) -> Shape {
        self.inner.shape()
    }

    /// The tensor's dtype.
    pub fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    /// The tensor's rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.inner.shape().dims().len()
    }

    /// The tensor's shape as a `&[usize]`. Convenience for callers who
    /// want to read dims without borrowing the shape.
    pub fn dims(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    /// Total element count.
    pub fn elem_count(&self) -> usize {
        self.inner.shape().elem_count()
    }

    // ---- arithmetic (element-wise, strict shape) ----

    /// Element-wise addition. Shapes must match.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.add(&other.inner),
        }
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.sub(&other.inner),
        }
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.mul(&other.inner),
        }
    }

    /// Element-wise division.
    pub fn div(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.div(&other.inner),
        }
    }

    // ---- broadcast-aware arithmetic ----

    /// Element-wise addition with auto-broadcasting.
    pub fn broadcast_add(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.broadcast_add(&other.inner),
        }
    }

    /// Element-wise subtraction with auto-broadcasting.
    pub fn broadcast_sub(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.broadcast_sub(&other.inner),
        }
    }

    /// Element-wise multiplication with auto-broadcasting.
    pub fn broadcast_mul(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.broadcast_mul(&other.inner),
        }
    }

    /// Element-wise division with auto-broadcasting.
    pub fn broadcast_div(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.broadcast_div(&other.inner),
        }
    }

    // ---- unary ----

    /// Element-wise negation.
    pub fn neg(&self) -> Self {
        Self { inner: self.inner.neg() }
    }

    /// Element-wise square.
    pub fn sqr(&self) -> Self {
        Self { inner: self.inner.sqr() }
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Self {
        Self { inner: self.inner.sqrt() }
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> Self {
        Self { inner: self.inner.exp() }
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Self {
        Self { inner: self.inner.log() }
    }

    /// Rectified linear unit.
    pub fn relu(&self) -> Self {
        Self { inner: self.inner.relu() }
    }

    /// SiLU / Swish activation.
    pub fn silu(&self) -> Self {
        Self { inner: self.inner.silu() }
    }

    /// GELU activation (tanh approximation).
    pub fn gelu(&self) -> Self {
        Self { inner: self.inner.gelu() }
    }

    /// Logistic sigmoid.
    pub fn sigmoid(&self) -> Self {
        Self { inner: self.inner.sigmoid() }
    }

    /// Hyperbolic tangent.
    pub fn tanh(&self) -> Self {
        Self { inner: self.inner.tanh() }
    }

    // ---- linear algebra & shape ----

    /// N-D batched matrix multiply with automatic rank-2 broadcasting.
    pub fn matmul(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.matmul(&other.inner),
        }
    }

    /// Transpose the last two dims (any rank ≥ 2).
    pub fn transpose(&self) -> Self {
        Self {
            inner: self.inner.transpose(),
        }
    }

    /// Permute axes by the given ordering.
    pub fn permute(&self, axes: &[usize]) -> Self {
        Self {
            inner: self.inner.permute(axes),
        }
    }

    /// Reshape to a new shape with matching element count.
    pub fn reshape(&self, shape: impl Into<Shape>) -> Self {
        Self {
            inner: self.inner.reshape(shape),
        }
    }

    /// Broadcast to a larger shape.
    pub fn broadcast_to(&self, shape: impl Into<Shape>) -> Self {
        Self {
            inner: self.inner.broadcast_to(shape),
        }
    }

    /// Slice (narrow) along `dim`: take elements `[start, start+len)`.
    pub fn slice(&self, dim: usize, start: usize, len: usize) -> Self {
        Self {
            inner: self.inner.slice(dim, start, len),
        }
    }

    /// Concatenate two tensors along `dim`.
    pub fn concat(&self, other: &Self, dim: usize) -> Self {
        Self {
            inner: self.inner.concat(&other.inner, dim),
        }
    }

    /// Add a scalar to every element.
    pub fn add_scalar(&self, c: f64) -> Self {
        Self {
            inner: self.inner.add_scalar(c),
        }
    }

    /// Multiply every element by a scalar.
    pub fn mul_scalar(&self, c: f64) -> Self {
        Self {
            inner: self.inner.mul_scalar(c),
        }
    }

    /// Argmax along a dim, returning a U32 tensor with the reduced
    /// dim removed. Non-differentiable.
    pub fn argmax_dim(&self, dim: usize) -> Self {
        Self {
            inner: self.inner.argmax_dim(dim),
        }
    }

    /// Realize as a `u32` (index) `Vec`.
    pub fn realize_u32(&self) -> Vec<u32> {
        fuel_reference_backend::exec::realize(&self.inner)
            .into_u32()
            .into_vec()
    }

    // ---- reductions ----

    /// Sum of all elements, producing a scalar.
    pub fn sum_all(&self) -> Self {
        Self { inner: self.inner.sum_all() }
    }

    /// Arithmetic mean of all elements, producing a scalar.
    pub fn mean_all(&self) -> Self {
        Self { inner: self.inner.mean_all() }
    }

    /// Sum along a single dimension (dim removed from output).
    pub fn sum_dim(&self, dim: usize) -> Self {
        Self {
            inner: self.inner.sum_dim(dim),
        }
    }

    /// Mean along a single dimension.
    pub fn mean_dim(&self, dim: usize) -> Self {
        Self {
            inner: self.inner.mean_dim(dim),
        }
    }

    // ---- compositions ----

    /// Softmax along the last dim.
    pub fn softmax_last_dim(&self) -> Self {
        Self {
            inner: self.inner.softmax_last_dim(),
        }
    }

    /// LayerNorm along the last dim with the given epsilon.
    pub fn layer_norm_last_dim(&self, eps: f64) -> Self {
        Self {
            inner: self.inner.layer_norm_last_dim(eps),
        }
    }

    /// RmsNorm along the last dim (LLaMA's normalization).
    pub fn rms_norm_last_dim(&self, eps: f64) -> Self {
        Self {
            inner: self.inner.rms_norm_last_dim(eps),
        }
    }

    /// Apply rotary position embeddings. See [`fuel_graph::Tensor::rope`].
    pub fn rope(&self, base: f64, start_pos: usize) -> Self {
        Self {
            inner: self.inner.rope(base, start_pos),
        }
    }

    /// Apply RoPE using caller-supplied `cos` and `sin` tables so they
    /// can be shared across many layers. See
    /// [`fuel_graph::Tensor::rope_with_tables`].
    pub fn rope_with_tables(&self, cos: &Self, sin: &Self) -> Self {
        Self {
            inner: self.inner.rope_with_tables(&cos.inner, &sin.inner),
        }
    }

    // ---- indexing ----

    /// Pick slices along `dim` using a 1-D U32 index tensor.
    pub fn index_select(&self, dim: usize, indices: &Self) -> Self {
        Self {
            inner: self.inner.index_select(dim, &indices.inner),
        }
    }

    /// N-D gather along `dim` using a U32 index tensor with the same
    /// shape as the output.
    pub fn gather(&self, dim: usize, indices: &Self) -> Self {
        Self {
            inner: self.inner.gather(dim, &indices.inner),
        }
    }

    // ---- dtype ----

    /// Cast to a different dtype.
    pub fn cast(&self, dtype: DType) -> Self {
        Self {
            inner: self.inner.cast(dtype),
        }
    }

    // ---- realization (the bridge to the reference backend) ----

    /// Realize this tensor as an `f32` `Vec` using the **fast** CPU
    /// executor from `fuel-graph-cpu`. This uses the `gemm` crate for
    /// matrix multiply (typically 50-200× faster than the reference
    /// backend's naive matmul) and delegates all other ops to the
    /// reference implementations.
    ///
    /// This is the default entry point for production workloads. For
    /// the textbook-correct oracle path, use [`realize_f32_reference`]
    /// instead.
    pub fn realize_f32(&self) -> Vec<f32> {
        fuel_graph_cpu::realize_f32(&self.inner).into_vec()
    }

    /// Realize as an `f64` `Vec` using the fast CPU executor.
    pub fn realize_f64(&self) -> Vec<f64> {
        fuel_graph_cpu::realize_f64(&self.inner).into_vec()
    }

    /// Realize as a `bf16` `Vec`. Note that the fast executor falls
    /// back to the reference matmul for `bf16` inputs since `gemm`
    /// does not provide a bf16 path; cast to f32 first for speed.
    pub fn realize_bf16(&self) -> Vec<half::bf16> {
        fuel_graph_cpu::realize_bf16(&self.inner).into_vec()
    }

    /// Realize as an `f16` `Vec`. Same note as `realize_bf16` about
    /// matmul — cast to f32 for the fast path.
    pub fn realize_f16(&self) -> Vec<half::f16> {
        fuel_graph_cpu::realize_f16(&self.inner).into_vec()
    }

    /// Realize using the reference backend directly — slow but
    /// textbook-correct. Used as an oracle when validating new ops,
    /// comparing backends, or debugging numerical discrepancies.
    pub fn realize_f32_reference(&self) -> Vec<f32> {
        fuel_reference_backend::exec::realize_f32(&self.inner).into_vec()
    }

    /// Realize on a CUDA GPU. The executor uploads const nodes (model
    /// weights) on first use and caches them, so repeated calls on
    /// the same executor amortize the H2D transfer.
    ///
    /// Requires the `cuda` feature.
    #[cfg(feature = "cuda")]
    pub fn realize_f32_cuda(
        &self,
        executor: &mut fuel_graph_cuda::CudaGraphExecutor,
    ) -> Vec<f32> {
        executor.realize_f32(&self.inner).into_vec()
    }
}

/// Realize a batch of `LazyTensor` roots in a single fast-executor
/// walk. All tensors must belong to the same underlying graph. The
/// return value has one `Vec<f32>` per input, in matching order.
///
/// This is the executor-level primitive behind the KV-cache forward
/// pass: one graph call produces logits plus every layer's updated
/// K/V tensors, avoiding n separate walks that would each recompute
/// the shared prefix.
pub fn realize_many_f32(tensors: &[&LazyTensor]) -> Vec<Vec<f32>> {
    let inner: Vec<&fuel_graph::Tensor> = tensors.iter().map(|t| &t.inner).collect();
    fuel_graph_cpu::realize_many_f32(&inner)
        .into_iter()
        .map(|t| t.into_vec())
        .collect()
}

/// CUDA variant of [`realize_many_f32`]. Uses the given executor's
/// device and weight cache.
#[cfg(feature = "cuda")]
pub fn realize_many_f32_cuda(
    tensors: &[&LazyTensor],
    executor: &mut fuel_graph_cuda::CudaGraphExecutor,
) -> Vec<Vec<f32>> {
    let inner: Vec<&fuel_graph::Tensor> = tensors.iter().map(|t| &t.inner).collect();
    executor
        .realize_many_f32(&inner)
        .into_iter()
        .map(|t| t.into_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn constructors_wrap_graph_tensor_correctly() {
        let t = LazyTensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.shape().dims(), &[3]);
        assert_eq!(t.rank(), 1);
        assert_eq!(t.elem_count(), 3);
    }

    #[test]
    fn add_builds_add_node_in_underlying_graph() {
        let a = LazyTensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        let c = a.add(&b);
        assert_eq!(c.shape().dims(), &[3]);
        // All three tensors share one underlying graph (by Rc cloning
        // via const_f32_like / add).
        assert!(std::rc::Rc::ptr_eq(
            c.graph_tensor().graph(),
            a.graph_tensor().graph(),
        ));
    }

    #[test]
    fn chained_lazy_method_call_builds_sensible_graph() {
        // Exercise a small pipeline typical of what an early LLaMA
        // port would write: RmsNorm → matmul → RMS-style residual.
        // We just verify the shapes thread through cleanly and the
        // final tensor is consistent.
        let x = LazyTensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
        );
        let w = x.const_f32_like(
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            Shape::from_dims(&[3, 3]),
        );
        let y = x.rms_norm_last_dim(1e-6).matmul(&w).relu();
        assert_eq!(y.shape().dims(), &[2, 3]);
        assert_eq!(y.dtype(), DType::F32);
    }

    #[test]
    fn rope_through_lazy_wrapper() {
        // Verify the RoPE builder is reachable through LazyTensor.
        let x = LazyTensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::from_dims(&[2, 4]),
        );
        let y = x.rope(10000.0, 0);
        assert_eq!(y.shape().dims(), &[2, 4]);
        assert_eq!(y.dtype(), DType::F32);
    }

    #[test]
    fn cast_switches_dtype_through_wrapper() {
        let x = LazyTensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let y = x.cast(DType::F64);
        assert_eq!(y.dtype(), DType::F64);
        assert_eq!(y.shape().dims(), &[3]);
    }

    #[test]
    fn indexing_builds_correct_output_shape() {
        let data = LazyTensor::from_f32(vec![1.0; 12], Shape::from_dims(&[3, 4]));
        let idx = data.const_u32_like(vec![0, 2, 1], Shape::from_dims(&[3]));
        let out = data.index_select(0, &idx);
        assert_eq!(out.shape().dims(), &[3, 4]);
    }

    // ---- Bridge realization tests ----

    #[test]
    fn realize_f32_executes_the_underlying_graph() {
        // The moment of truth: build a graph through LazyTensor and
        // then realize it end-to-end. (a + b) * a for a = [1, 2, 3],
        // b = [4, 5, 6] should yield [5, 14, 27].
        let a = LazyTensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        let c = a.add(&b).mul(&a);
        let result = c.realize_f32();
        assert_eq!(result, vec![5.0, 14.0, 27.0]);
    }

    #[test]
    fn realize_f32_matmul_hand_computed() {
        // Classic 2x3 @ 3x2 matmul through the bridge.
        let a = LazyTensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
        );
        let b = a.const_f32_like(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::from_dims(&[3, 2]),
        );
        let c = a.matmul(&b);
        assert_eq!(c.shape().dims(), &[2, 2]);
        assert_eq!(c.realize_f32(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn fast_and_reference_agree_on_medium_matmul() {
        // 64 × 96 @ 96 × 32 — bigger than anything we could hand-check
        // but small enough to verify every element. The fast path
        // goes through gemm, the reference path uses the naive triple
        // loop; results should agree within float-rounding tolerance.
        let m = 64;
        let k = 96;
        let n = 32;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.013).cos()).collect();
        let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[m, k]));
        let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n]));
        let c = a.matmul(&b);
        let fast = c.realize_f32();
        let reference = c.realize_f32_reference();
        assert_eq!(fast.len(), reference.len());
        for (i, (&f, &r)) in fast.iter().zip(&reference).enumerate() {
            // Accept either absolute or relative tolerance — gemm's
            // blocked accumulation order differs from the naive triple
            // loop, so values near zero can have large relative diffs
            // on tiny absolute diffs. Both bounds are loose enough for
            // float-noise but tight enough to catch real bugs.
            let diff = (f - r).abs();
            let rel = if r.abs() > 1e-6 { diff / r.abs() } else { 0.0 };
            assert!(
                diff < 1e-4 || rel < 1e-3,
                "at index {i}: fast={f}, reference={r}, diff={diff}, rel={rel}",
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_add_mul() {
        let a = LazyTensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        let c = a.add(&b).mul(&a);
        let cpu_result = c.realize_f32();
        let mut executor = fuel_graph_cuda::CudaGraphExecutor::for_device(0).unwrap();
        let cuda_result = c.realize_f32_cuda(&mut executor);
        assert_eq!(cpu_result, cuda_result);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_matmul() {
        let a = LazyTensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
        );
        let b = a.const_f32_like(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::from_dims(&[3, 2]),
        );
        let c = a.matmul(&b);
        let cpu = c.realize_f32();
        let mut exe = fuel_graph_cuda::CudaGraphExecutor::for_device(0).unwrap();
        let cuda = c.realize_f32_cuda(&mut exe);
        assert_eq!(cpu.len(), cuda.len());
        for (i, (a, b)) in cpu.iter().zip(cuda.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "matmul[{i}]: cpu={a}, cuda={b}",
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_broadcast_matmul() {
        // Rank-3 × rank-2 matmul (what the transformer forward does).
        // The graph auto-broadcasts the rank-2 to rank-3.
        let x = LazyTensor::from_f32(
            (0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 4]),
        );
        let w = x.const_f32_like(
            (0..8).map(|i| i as f32 * 0.2).collect::<Vec<_>>(),
            Shape::from_dims(&[4, 2]),
        );
        let y = x.matmul(&w);
        let cpu = y.realize_f32();
        let mut exe = fuel_graph_cuda::CudaGraphExecutor::for_device(0).unwrap();
        let cuda = y.realize_f32_cuda(&mut exe);
        assert_eq!(cpu.len(), cuda.len());
        for (i, (&a, &b)) in cpu.iter().zip(cuda.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "bcast_mm[{i}]: cpu={a}, cuda={b}");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_permute() {
        let x = LazyTensor::from_f32(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 2, 3, 4]),
        );
        let y = x.permute(&[0, 2, 1, 3]);
        let cpu = y.realize_f32();
        let mut exe = fuel_graph_cuda::CudaGraphExecutor::for_device(0).unwrap();
        let cuda = y.realize_f32_cuda(&mut exe);
        assert_eq!(cpu, cuda, "permute mismatch");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_softmax() {
        let x = LazyTensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
        );
        let y = x.softmax_last_dim();
        let cpu = y.realize_f32();
        let mut exe = fuel_graph_cuda::CudaGraphExecutor::for_device(0).unwrap();
        let cuda = y.realize_f32_cuda(&mut exe);
        assert_eq!(cpu.len(), cuda.len());
        for (i, (&a, &b)) in cpu.iter().zip(cuda.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "softmax[{i}]: cpu={a}, cuda={b}");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_concat_slice() {
        let a = LazyTensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]));
        let b = a.const_f32_like(vec![5.0, 6.0, 7.0, 8.0], Shape::from_dims(&[2, 2]));
        let cat = a.concat(&b, 1); // [2, 4]
        let sliced = cat.slice(1, 1, 2); // [2, 2]
        let cpu = sliced.realize_f32();
        let mut exe = fuel_graph_cuda::CudaGraphExecutor::for_device(0).unwrap();
        let cuda = sliced.realize_f32_cuda(&mut exe);
        assert_eq!(cpu, cuda, "concat+slice mismatch");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_rms_norm() {
        let x = LazyTensor::from_f32(
            (0..8).map(|i| i as f32 * 0.5 - 1.5).collect::<Vec<_>>(),
            Shape::from_dims(&[2, 4]),
        );
        let y = x.rms_norm_last_dim(1e-5);
        let cpu = y.realize_f32();
        let mut exe = fuel_graph_cuda::CudaGraphExecutor::for_device(0).unwrap();
        let cuda = y.realize_f32_cuda(&mut exe);
        assert_eq!(cpu.len(), cuda.len());
        for (i, (&a, &b)) in cpu.iter().zip(cuda.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "rms_norm[{i}]: cpu={a}, cuda={b}");
        }
    }

    #[test]
    fn realize_f64_through_bridge() {
        let a = LazyTensor::from_f64(vec![1.5, 2.5, 3.5], Shape::from_dims(&[3]));
        let b = a.mul(&a);
        assert_eq!(b.realize_f64(), vec![2.25, 6.25, 12.25]);
    }

    #[test]
    fn lazy_tensor_mini_llama_block_forward() {
        // A minimal LLaMA-style attention-only "block" built entirely
        // through LazyTensor. No training, just the forward pass:
        //
        //   h = x + (RmsNorm(x) @ W_qkv → split Q/K/V → RoPE → attention → out proj)
        //
        // This is the sanity check that every LLaMA primitive is
        // reachable through the bridge's API, and that the bridge's
        // realize_f32 call actually runs the whole thing.
        let seq = 3;
        let d_head = 4; // must be even for RoPE
        let num_heads = 2;
        let d_model = num_heads * d_head; // 8

        // Fake input: [1, seq, d_model]
        let x_data: Vec<f32> = (0..seq * d_model).map(|i| i as f32 * 0.01).collect();
        let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[1, seq, d_model]));

        // Fake weights (just identities for simplicity — makes the
        // test easy to verify output finiteness without needing to
        // hand-compute).
        let w_q =
            x.const_f32_like(identity_matrix(d_model), Shape::from_dims(&[d_model, d_model]));
        let w_k =
            x.const_f32_like(identity_matrix(d_model), Shape::from_dims(&[d_model, d_model]));
        let w_v =
            x.const_f32_like(identity_matrix(d_model), Shape::from_dims(&[d_model, d_model]));
        let w_o =
            x.const_f32_like(identity_matrix(d_model), Shape::from_dims(&[d_model, d_model]));

        // RmsNorm → Q/K/V projection (auto-broadcasting matmul).
        let x_norm = x.rms_norm_last_dim(1e-6);
        let q = x_norm.matmul(&w_q);
        let k = x_norm.matmul(&w_k);
        let v = x_norm.matmul(&w_v);

        // Split heads: [1, seq, 8] → [1, seq, 2, 4] → [1, 2, seq, 4]
        let q_h = q
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);
        let k_h = k
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);
        let v_h = v
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head]))
            .permute(&[0, 2, 1, 3]);

        // RoPE on Q and K.
        let q_r = q_h.rope(10000.0, 0);
        let k_r = k_h.rope(10000.0, 0);

        // Scaled dot-product attention.
        let k_t = k_r.transpose();
        let scores = q_r.matmul(&k_t);
        let attn = scores.softmax_last_dim();
        let attn_v = attn.matmul(&v_h);

        // Merge heads + output projection.
        let merged = attn_v
            .permute(&[0, 2, 1, 3])
            .reshape(Shape::from_dims(&[1, seq, d_model]));
        let attn_out = merged.matmul(&w_o);
        let h = x.add(&attn_out);

        // Realize end-to-end through the bridge.
        let result = h.realize_f32();
        assert_eq!(result.len(), seq * d_model);
        for &v in &result {
            assert!(v.is_finite(), "bridge-based LLaMA block output non-finite: {v}");
        }
    }

    /// Build an identity matrix of size `n × n` in row-major layout.
    fn identity_matrix(n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; n * n];
        for i in 0..n {
            out[i * n + i] = 1.0;
        }
        out
    }
}

// Helper method on the wrapper that we didn't include above because the
// main struct's `impl` block was getting long. Kept in its own small
// `impl` for readability.
impl LazyTensor {
    /// Build a second const U32 (index) tensor on the same graph.
    pub fn const_u32_like(
        &self,
        data: impl Into<Arc<[u32]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        Self {
            inner: self.inner.const_u32_like(data, shape),
        }
    }
}

// ---- safetensors integration -----------------------------------------------

impl LazyTensor {
    /// Build a `LazyTensor` from raw little-endian bytes as they appear
    /// in a safetensors file, plus a dtype and shape. Row-major layout
    /// is assumed. The byte count must match `shape.elem_count() *
    /// dtype_bytes`.
    ///
    /// This is the low-level loader. Prefer [`from_safetensors_view`]
    /// if you already have a `safetensors::TensorView` in hand.
    ///
    /// Supported dtypes today: `F32`, `F64`, `BF16`, `F16`, `U32`.
    /// Integer types other than `U32` are rejected to keep the
    /// surface small; add them when a real model needs them.
    pub fn from_safetensors_bytes(
        bytes: &[u8],
        dtype: safetensors::Dtype,
        shape: &[usize],
    ) -> crate::Result<Self> {
        use safetensors::Dtype;
        let shape_obj = Shape::from_dims(shape);
        let elem_count = shape_obj.elem_count();

        let check_len = |expected: usize| -> crate::Result<()> {
            if bytes.len() != expected {
                crate::bail!(
                    "from_safetensors_bytes: expected {expected} bytes for dtype {dtype:?} \
                     and shape {shape:?}, got {}",
                    bytes.len(),
                );
            }
            Ok(())
        };

        match dtype {
            Dtype::F32 => {
                check_len(elem_count * 4)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.chunks_exact(4) {
                    data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(Self::from_f32(data, shape_obj))
            }
            Dtype::F64 => {
                check_len(elem_count * 8)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.chunks_exact(8) {
                    let arr: [u8; 8] = chunk.try_into().unwrap();
                    data.push(f64::from_le_bytes(arr));
                }
                Ok(Self::from_f64(data, shape_obj))
            }
            Dtype::BF16 => {
                check_len(elem_count * 2)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.chunks_exact(2) {
                    let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                    data.push(half::bf16::from_bits(raw));
                }
                Ok(Self::from_bf16(data, shape_obj))
            }
            Dtype::F16 => {
                check_len(elem_count * 2)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.chunks_exact(2) {
                    let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                    data.push(half::f16::from_bits(raw));
                }
                Ok(Self::from_f16(data, shape_obj))
            }
            Dtype::U32 => {
                check_len(elem_count * 4)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.chunks_exact(4) {
                    data.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(Self::from_u32(data, shape_obj))
            }
            other => crate::bail!(
                "from_safetensors_bytes: unsupported dtype {other:?} — extend LazyTensor's \
                 safetensors loader to handle it",
            ),
        }
    }

    /// Build a `LazyTensor` from a `safetensors::TensorView`. This is
    /// the most natural entry point when iterating over a
    /// [`crate::safetensors::MmapedSafetensors`] or similar.
    pub fn from_safetensors_view(
        view: &safetensors::tensor::TensorView<'_>,
    ) -> crate::Result<Self> {
        Self::from_safetensors_bytes(view.data(), view.dtype(), view.shape())
    }
}

// ---- LLaMA model assembly --------------------------------------------------

/// Hyperparameters for a LLaMA-style transformer model.
///
/// Field names follow the conventional LLaMA nomenclature:
/// - `dim` is the model hidden dimension (often written `d_model`).
/// - `n_heads` is the number of attention query heads.
/// - `n_kv_heads` is the number of key/value heads. Equal to `n_heads`
///   for standard multi-head attention; smaller (e.g. `n_heads / 4`)
///   for Grouped Query Attention (GQA). LLaMA 2 onwards uses GQA.
/// - `head_dim` is the per-head feature dimension (`dim / n_heads`).
/// - `ffn_dim` is the hidden dimension of the SwiGLU feed-forward
///   network, conventionally around `4 × dim` with some rounding.
/// - `norm_eps` is the epsilon of the RmsNorm layers.
/// - `rope_base` is the frequency base for rotary position embeddings
///   (`10_000` in original LLaMA, `500_000` in LLaMA 3).
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub dim:        usize,
    pub n_layers:   usize,
    pub n_heads:    usize,
    pub n_kv_heads: usize,
    pub head_dim:   usize,
    pub ffn_dim:    usize,
    pub norm_eps:   f64,
    pub rope_base:  f64,
}

impl LlamaConfig {
    /// Parse a LlamaConfig from a Hugging Face `config.json` string.
    ///
    /// Maps HF's field names to ours:
    /// - `hidden_size` → `dim`
    /// - `num_hidden_layers` → `n_layers`
    /// - `num_attention_heads` → `n_heads`
    /// - `num_key_value_heads` → `n_kv_heads` (falls back to `n_heads`
    ///   when absent, for older configs without GQA)
    /// - `intermediate_size` → `ffn_dim`
    /// - `vocab_size` → `vocab_size`
    /// - `rms_norm_eps` → `norm_eps`
    /// - `rope_theta` → `rope_base` (defaults to 10000 when absent)
    /// - `head_dim` is taken directly when present, or computed as
    ///   `hidden_size / num_attention_heads` otherwise.
    pub fn from_hf_json_str(json: &str) -> crate::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| crate::Error::Msg(format!("parsing config.json: {e}")))?;

        let get_usize = |key: &str| -> crate::Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| {
                    crate::Error::Msg(format!("config.json: missing/invalid field {key:?}"))
                })
        };
        let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };

        let vocab_size = get_usize("vocab_size")?;
        let dim = get_usize("hidden_size")?;
        let n_layers = get_usize("num_hidden_layers")?;
        let n_heads = get_usize("num_attention_heads")?;
        let n_kv_heads = v
            .get("num_key_value_heads")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(n_heads);
        let ffn_dim = get_usize("intermediate_size")?;
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(dim / n_heads);
        let norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-5);
        let rope_base = get_f64("rope_theta").unwrap_or(10_000.0);

        Ok(LlamaConfig {
            vocab_size,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            norm_eps,
            rope_base,
        })
    }
}

/// Per-layer weights of a LLaMA transformer block. All tensors are
/// stored as `Arc<[f32]>` so they can be loaded once and shared across
/// every forward pass with zero copy — each call to
/// [`LlamaModel::forward`] clones the `Arc` (a refcount bump) when it
/// builds fresh const nodes for this layer.
///
/// LLaMA proper has no biases anywhere in the attention block, so the
/// `*_bias` fields are `None` for LLaMA family models. Qwen2 and a few
/// related architectures do add biases on Q/K/V (but not on the output
/// projection), so the loader stores them here when the safetensors
/// file contains them.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// `[dim, dim]` query projection.
    pub attn_q: Arc<[f32]>,
    /// `[dim]` query projection bias (Qwen2-style; LLaMA has none).
    pub attn_q_bias: Option<Arc<[f32]>>,
    /// `[dim, dim]` key projection.
    pub attn_k: Arc<[f32]>,
    /// `[kv_dim]` key projection bias.
    pub attn_k_bias: Option<Arc<[f32]>>,
    /// `[dim, dim]` value projection.
    pub attn_v: Arc<[f32]>,
    /// `[kv_dim]` value projection bias.
    pub attn_v_bias: Option<Arc<[f32]>>,
    /// `[dim, dim]` output projection.
    pub attn_o: Arc<[f32]>,
    /// `[dim, ffn_dim]` gate projection for SwiGLU.
    pub ffn_gate: Arc<[f32]>,
    /// `[dim, ffn_dim]` up projection for SwiGLU.
    pub ffn_up: Arc<[f32]>,
    /// `[ffn_dim, dim]` down projection for SwiGLU.
    pub ffn_down: Arc<[f32]>,
    /// `[dim]` RmsNorm gain for the pre-attention norm.
    pub attn_norm_gain: Arc<[f32]>,
    /// `[dim]` RmsNorm gain for the pre-FFN norm.
    pub ffn_norm_gain: Arc<[f32]>,
}

/// Top-level weights: token embedding table, per-layer weights, final
/// norm gain, and output projection (which may be tied to the embedding
/// or a separate matrix).
#[derive(Debug, Clone)]
pub struct LlamaWeights {
    /// `[vocab_size, dim]` token embedding table.
    pub token_embedding: Arc<[f32]>,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
    /// `[dim]` RmsNorm gain for the final norm before the output head.
    pub final_norm_gain: Arc<[f32]>,
    /// `[dim, vocab_size]` output projection (a.k.a. `lm_head`).
    pub output: Arc<[f32]>,
}

/// A LLaMA-style transformer model assembled via `LazyTensor`. Holds
/// config + weights as plain vectors; each `forward` call rebuilds a
/// graph using those vectors as `Const` leaves.
///
/// This lives in `fuel_core::lazy` rather than `fuel_transformers`
/// because it was built directly on top of the Phase 6a bridge
/// primitives and predates the migration of `fuel_transformers`'
/// existing model code onto `LazyTensor`. Once that migration lands,
/// this code will move back to `fuel-transformers::models::llama`.
#[derive(Debug, Clone)]
pub struct LlamaModel {
    pub config:  LlamaConfig,
    pub weights: LlamaWeights,
}

impl LlamaModel {
    /// Run a forward pass from a sequence of token IDs and return the
    /// final logits as a `LazyTensor` of shape `[1, seq_len, vocab_size]`.
    /// Call `.realize_f32()` on the result to materialize them.
    ///
    /// `start_pos` offsets the RoPE frequencies — use `0` for the
    /// first forward call of a conversation and the previous total
    /// token count for each subsequent decode step when using a KV
    /// cache. The current implementation does NOT use a KV cache
    /// internally; it recomputes the full attention each call. Adding
    /// a KV cache is orthogonal plumbing that doesn't change the graph
    /// structure.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> LazyTensor {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert_eq!(cfg.n_heads * cfg.head_dim, cfg.dim, "LlamaConfig: n_heads * head_dim must equal dim");

        // Embedding lookup: build a token embedding const tensor +
        // a U32 index tensor + index_select along dim 0.
        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
        );
        let token_ids =
            embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        // index_select(0, token_ids) produces [seq, dim]. Reshape to
        // [1, seq, dim] for the downstream attention code.
        let mut h = embed
            .index_select(0, &token_ids)
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]));

        // Share RoPE cos/sin across all layers — see the matching
        // comment in `forward_with_cache`.
        let (cos_data, sin_data) = fuel_graph::build_rope_tables(
            cfg.rope_base,
            start_pos,
            seq,
            cfg.head_dim,
        );
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        // Chain through all decoder layers.
        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin);
        }

        // Final norm (affine RmsNorm).
        let h_norm = apply_affine_rms_norm(
            &h,
            &weights.final_norm_gain,
            cfg.dim,
            cfg.norm_eps,
        );
        // Output projection to vocab logits.
        let w_out = h_norm.const_f32_like(
            weights.output.clone(),
            Shape::from_dims(&[cfg.dim, cfg.vocab_size]),
        );
        h_norm.matmul(&w_out)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> LazyTensor {
        let cfg = &self.config;
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;

        // Pre-attention RmsNorm with affine gain.
        let x_norm = apply_affine_rms_norm(x, &layer.attn_norm_gain, cfg.dim, cfg.norm_eps);

        // Project to Q, K, V using auto-broadcasting matmul.
        // Under GQA, W_k and W_v have fewer output features (kv_dim
        // instead of dim) because there are fewer key/value heads.
        let w_q = x.const_f32_like(layer.attn_q.clone(), Shape::from_dims(&[cfg.dim, cfg.dim]));
        let w_k = x.const_f32_like(layer.attn_k.clone(), Shape::from_dims(&[cfg.dim, kv_dim]));
        let w_v = x.const_f32_like(layer.attn_v.clone(), Shape::from_dims(&[cfg.dim, kv_dim]));
        let w_o = x.const_f32_like(layer.attn_o.clone(), Shape::from_dims(&[cfg.dim, cfg.dim]));
        let q = apply_optional_bias(x_norm.matmul(&w_q), layer.attn_q_bias.as_ref(), cfg.dim);
        let k = apply_optional_bias(x_norm.matmul(&w_k), layer.attn_k_bias.as_ref(), kv_dim);
        let v = apply_optional_bias(x_norm.matmul(&w_v), layer.attn_v_bias.as_ref(), kv_dim);

        // Split heads.
        // Q: [batch, seq, dim] → [batch, seq, n_heads, head_dim] → [batch, n_heads, seq, head_dim]
        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);
        // K/V: [batch, seq, kv_dim] → [batch, seq, n_kv_heads, head_dim] → [batch, n_kv_heads, seq, head_dim]
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);

        // RoPE on Q and K (applied per-head; V is NOT rotated). Uses
        // caller-supplied cos/sin so all layers share a single pair
        // of const nodes.
        let q_r = q_h.rope_with_tables(rope_cos, rope_sin);
        let k_r = k_h.rope_with_tables(rope_cos, rope_sin);

        // If GQA (n_kv_heads < n_heads), replicate each KV head
        // `n_heads / n_kv_heads` times along the head dim so Q and K/V
        // have the same number of heads for the attention matmul. We
        // expand via reshape + broadcast_to + reshape:
        //
        //   [batch, n_kv_heads, seq, head_dim]
        //     → reshape [batch, n_kv_heads, 1, seq, head_dim]
        //     → broadcast_to [batch, n_kv_heads, n_rep, seq, head_dim]
        //     → reshape [batch, n_heads, seq, head_dim]
        //
        // When n_kv_heads == n_heads (standard MHA), n_rep == 1 and
        // these reshape/broadcast steps are no-ops in effect.
        let (k_r, v_h) = if cfg.n_kv_heads == cfg.n_heads {
            (k_r, v_h)
        } else {
            assert_eq!(
                cfg.n_heads % cfg.n_kv_heads,
                0,
                "GQA: n_heads ({}) must be divisible by n_kv_heads ({})",
                cfg.n_heads,
                cfg.n_kv_heads,
            );
            let n_rep = cfg.n_heads / cfg.n_kv_heads;
            let expand = |t: LazyTensor| -> LazyTensor {
                t.reshape(Shape::from_dims(&[
                    batch,
                    cfg.n_kv_heads,
                    1,
                    seq,
                    cfg.head_dim,
                ]))
                .broadcast_to(Shape::from_dims(&[
                    batch,
                    cfg.n_kv_heads,
                    n_rep,
                    seq,
                    cfg.head_dim,
                ]))
                .reshape(Shape::from_dims(&[
                    batch,
                    cfg.n_heads,
                    seq,
                    cfg.head_dim,
                ]))
            };
            (expand(k_r), expand(v_h))
        };

        // Scaled dot-product attention with a causal mask. LLaMA was
        // trained with a strict lower-triangular mask — without it,
        // each prefill token's hidden state is contaminated by future
        // tokens, and the model produces garbage.
        let k_t = k_r.transpose();
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t);
        let mut mask_data = vec![0.0_f32; seq * seq];
        for q in 0..seq {
            for k in (q + 1)..seq {
                mask_data[q * seq + k] = f32::NEG_INFINITY;
            }
        }
        let mask =
            x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));
        let scores_scaled = LazyTensor {
            inner: scores.inner.mul_scalar(scale),
        };
        let scores_masked = scores_scaled.broadcast_add(&mask);
        let attn = scores_masked.softmax_last_dim();
        let attn_v = attn.matmul(&v_h);

        // Merge heads + output projection.
        let merged = attn_v
            .permute(&[0, 2, 1, 3])
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]));
        let attn_out = merged.matmul(&w_o);

        // First residual connection.
        let h1 = x.add(&attn_out);

        // Pre-FFN RmsNorm with affine gain.
        let h1_norm = apply_affine_rms_norm(&h1, &layer.ffn_norm_gain, cfg.dim, cfg.norm_eps);

        // SwiGLU FFN.
        let w_gate = x.const_f32_like(
            layer.ffn_gate.clone(),
            Shape::from_dims(&[cfg.dim, cfg.ffn_dim]),
        );
        let w_up = x.const_f32_like(
            layer.ffn_up.clone(),
            Shape::from_dims(&[cfg.dim, cfg.ffn_dim]),
        );
        let w_down = x.const_f32_like(
            layer.ffn_down.clone(),
            Shape::from_dims(&[cfg.ffn_dim, cfg.dim]),
        );
        let gate = h1_norm.matmul(&w_gate);
        let up = h1_norm.matmul(&w_up);
        let swiglu = gate.silu().mul(&up);
        let ffn_out = swiglu.matmul(&w_down);

        // Second residual connection.
        h1.add(&ffn_out)
    }

    /// Variant of [`apply_layer`] that also exposes the fresh K and V
    /// tensors so the caller can persist them to a KV cache, and that
    /// prepends cached keys/values in front of the fresh ones before
    /// the attention matmul.
    ///
    /// Returns `(output, fresh_k_post_rope, fresh_v)` where the two
    /// fresh tensors have shape `[batch, n_kv_heads, seq, head_dim]`
    /// — the layout [`LlamaKVCache::append_layer`] expects.
    fn apply_layer_with_cache(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        layer_cache: &LayerKVCache,
        cached_len: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> (LazyTensor, LazyTensor, LazyTensor) {
        let cfg = &self.config;
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let total_seq = cached_len + seq;

        let x_norm = apply_affine_rms_norm(x, &layer.attn_norm_gain, cfg.dim, cfg.norm_eps);

        let w_q = x.const_f32_like(layer.attn_q.clone(), Shape::from_dims(&[cfg.dim, cfg.dim]));
        let w_k = x.const_f32_like(layer.attn_k.clone(), Shape::from_dims(&[cfg.dim, kv_dim]));
        let w_v = x.const_f32_like(layer.attn_v.clone(), Shape::from_dims(&[cfg.dim, kv_dim]));
        let w_o = x.const_f32_like(layer.attn_o.clone(), Shape::from_dims(&[cfg.dim, cfg.dim]));
        let q = apply_optional_bias(x_norm.matmul(&w_q), layer.attn_q_bias.as_ref(), cfg.dim);
        let k = apply_optional_bias(x_norm.matmul(&w_k), layer.attn_k_bias.as_ref(), kv_dim);
        let v = apply_optional_bias(x_norm.matmul(&w_v), layer.attn_v_bias.as_ref(), kv_dim);

        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);

        // RoPE uses caller-supplied cos/sin tables so that all 22+
        // layers in a forward pass share a single pair of const nodes
        // rather than each rebuilding their own. The caller computes
        // the tables for `(rope_base, cached_len, seq, head_dim)` once
        // before dispatching any layer.
        let q_r = q_h.rope_with_tables(rope_cos, rope_sin);
        let k_r = k_h.rope_with_tables(rope_cos, rope_sin);

        // Keys/values to persist to the external cache — the pre-concat,
        // pre-GQA-expansion variants, exactly matching the cache layout.
        let fresh_k = k_r.clone();
        let fresh_v = v_h.clone();

        // Prepend cached K/V in front of the fresh ones along the seq
        // dim (dim 2). When `cached_len == 0` we skip the concat so the
        // first forward pass is structurally identical to non-cached
        // forward(), which makes the prefill path bitwise-comparable.
        let (full_k, full_v) = if cached_len > 0 {
            let cached_shape =
                Shape::from_dims(&[batch, cfg.n_kv_heads, cached_len, cfg.head_dim]);
            let cached_k = x.const_f32_like(layer_cache.k.clone(), cached_shape.clone());
            let cached_v = x.const_f32_like(layer_cache.v.clone(), cached_shape);
            (cached_k.concat(&fresh_k, 2), cached_v.concat(&fresh_v, 2))
        } else {
            (fresh_k.clone(), fresh_v.clone())
        };

        // GQA: after prepending cached K/V, expand to `n_heads` before
        // the attention matmul.
        let (full_k, full_v) = if cfg.n_kv_heads == cfg.n_heads {
            (full_k, full_v)
        } else {
            assert_eq!(cfg.n_heads % cfg.n_kv_heads, 0);
            let n_rep = cfg.n_heads / cfg.n_kv_heads;
            let expand = |t: LazyTensor| -> LazyTensor {
                t.reshape(Shape::from_dims(&[
                    batch,
                    cfg.n_kv_heads,
                    1,
                    total_seq,
                    cfg.head_dim,
                ]))
                .broadcast_to(Shape::from_dims(&[
                    batch,
                    cfg.n_kv_heads,
                    n_rep,
                    total_seq,
                    cfg.head_dim,
                ]))
                .reshape(Shape::from_dims(&[
                    batch,
                    cfg.n_heads,
                    total_seq,
                    cfg.head_dim,
                ]))
            };
            (expand(full_k), expand(full_v))
        };

        let k_t = full_k.transpose();
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t);

        // Additive causal mask. The query at position `q` (fresh-index
        // 0..seq) lives at absolute position `cached_len + q`, and may
        // only attend to keys at absolute positions ≤ cached_len + q.
        // Build the mask as `[1, 1, seq, total_seq]` — zeros where
        // allowed, `-inf` where not — and broadcast-add it to scores
        // before softmax. During decode (seq=1) this is a row of all
        // zeros so it's a no-op; during prefill it's a standard
        // lower-triangular mask over the fresh block with the cached
        // prefix fully visible.
        let mut mask_data = vec![0.0_f32; seq * total_seq];
        for q in 0..seq {
            let abs_q = cached_len + q;
            for k in (abs_q + 1)..total_seq {
                mask_data[q * total_seq + k] = f32::NEG_INFINITY;
            }
        }
        let mask = x.const_f32_like(
            mask_data,
            Shape::from_dims(&[1, 1, seq, total_seq]),
        );
        let scores_scaled = LazyTensor {
            inner: scores.inner.mul_scalar(scale),
        };
        let scores_masked = scores_scaled.broadcast_add(&mask);
        let attn = scores_masked.softmax_last_dim();
        let attn_v = attn.matmul(&full_v);

        let merged = attn_v
            .permute(&[0, 2, 1, 3])
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]));
        let attn_out = merged.matmul(&w_o);

        let h1 = x.add(&attn_out);
        let h1_norm = apply_affine_rms_norm(&h1, &layer.ffn_norm_gain, cfg.dim, cfg.norm_eps);

        let w_gate = x.const_f32_like(
            layer.ffn_gate.clone(),
            Shape::from_dims(&[cfg.dim, cfg.ffn_dim]),
        );
        let w_up = x.const_f32_like(
            layer.ffn_up.clone(),
            Shape::from_dims(&[cfg.dim, cfg.ffn_dim]),
        );
        let w_down = x.const_f32_like(
            layer.ffn_down.clone(),
            Shape::from_dims(&[cfg.ffn_dim, cfg.dim]),
        );
        let gate = h1_norm.matmul(&w_gate);
        let up = h1_norm.matmul(&w_up);
        let swiglu = gate.silu().mul(&up);
        let ffn_out = swiglu.matmul(&w_down);

        (h1.add(&ffn_out), fresh_k, fresh_v)
    }

    /// Run a forward pass that consumes a growing KV cache. The model
    /// prepends each layer's previously cached K/V in front of the
    /// freshly computed K/V before attention, and afterwards materializes
    /// the fresh K/V (along with the logits) in a single executor walk
    /// so the cache can be grown in place.
    ///
    /// Returns only the **last-position** logits — shape `[vocab_size]`
    /// — since the caller almost always wants to sample from that
    /// single slice, and realizing the full `[1, seq, vocab]` logits
    /// tensor would be the single largest allocation of the decode
    /// step for large vocabs.
    pub fn forward_with_cache(
        &self,
        tokens: &[u32],
        cache: &mut LlamaKVCache,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        let cached_len = cache.cached_len;

        assert_eq!(
            cache.layers.len(),
            cfg.n_layers,
            "forward_with_cache: cache layer count {} does not match model n_layers {}",
            cache.layers.len(),
            cfg.n_layers,
        );
        assert!(seq > 0, "forward_with_cache: cannot forward zero tokens");

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
        );
        let token_ids =
            embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0, &token_ids)
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]));

        // Build RoPE cos/sin tables once and share them across every
        // decoder layer. Every layer applies RoPE with the same
        // `(rope_base, cached_len, seq, head_dim)`, so doing this at
        // layer scope wastes ~n_layers × O(seq·head_dim) const nodes
        // plus their downstream reshape/broadcast chains.
        let (cos_data, sin_data) = fuel_graph::build_rope_tables(
            cfg.rope_base,
            cached_len,
            seq,
            cfg.head_dim,
        );
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        let mut fresh_ks: Vec<LazyTensor> = Vec::with_capacity(cfg.n_layers);
        let mut fresh_vs: Vec<LazyTensor> = Vec::with_capacity(cfg.n_layers);

        for (li, layer) in weights.layers.iter().enumerate() {
            let (new_h, fk, fv) = self.apply_layer_with_cache(
                &h,
                layer,
                &cache.layers[li],
                cached_len,
                &rope_cos,
                &rope_sin,
            );
            h = new_h;
            fresh_ks.push(fk);
            fresh_vs.push(fv);
        }

        let h_norm = apply_affine_rms_norm(
            &h,
            &weights.final_norm_gain,
            cfg.dim,
            cfg.norm_eps,
        );
        let w_out = h_norm.const_f32_like(
            weights.output.clone(),
            Shape::from_dims(&[cfg.dim, cfg.vocab_size]),
        );
        let logits = h_norm.matmul(&w_out);

        // Slice out logits for the last sequence position only.
        let last_pos = seq - 1;
        let last_logits = logits
            .slice(1, last_pos, 1)
            .reshape(Shape::from_dims(&[cfg.vocab_size]));

        // Build root list: [last_logits, k_0..k_N, v_0..v_N] and realize
        // all of them in a single executor walk.
        let mut roots: Vec<&LazyTensor> = Vec::with_capacity(1 + 2 * cfg.n_layers);
        roots.push(&last_logits);
        for fk in &fresh_ks {
            roots.push(fk);
        }
        for fv in &fresh_vs {
            roots.push(fv);
        }
        let realized = realize_many_f32(&roots);
        Self::unpack_kv_cache(realized, cache, cfg.n_layers, seq)
    }

    /// CUDA variant of the cached forward pass. Identical graph, but
    /// realized on GPU via the `CudaGraphExecutor`.
    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_cuda(
        &self,
        tokens: &[u32],
        cache: &mut LlamaKVCache,
        executor: &mut fuel_graph_cuda::CudaGraphExecutor,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        let cached_len = cache.cached_len;

        assert_eq!(cache.layers.len(), cfg.n_layers);
        assert!(seq > 0);

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
        );
        let token_ids =
            embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0, &token_ids)
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]));

        let (cos_data, sin_data) = fuel_graph::build_rope_tables(
            cfg.rope_base, cached_len, seq, cfg.head_dim,
        );
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        let mut fresh_ks: Vec<LazyTensor> = Vec::with_capacity(cfg.n_layers);
        let mut fresh_vs: Vec<LazyTensor> = Vec::with_capacity(cfg.n_layers);

        for (li, layer) in weights.layers.iter().enumerate() {
            let (new_h, fk, fv) = self.apply_layer_with_cache(
                &h, layer, &cache.layers[li], cached_len,
                &rope_cos, &rope_sin,
            );
            h = new_h;
            fresh_ks.push(fk);
            fresh_vs.push(fv);
        }

        let h_norm = apply_affine_rms_norm(
            &h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps,
        );
        let w_out = h_norm.const_f32_like(
            weights.output.clone(),
            Shape::from_dims(&[cfg.dim, cfg.vocab_size]),
        );
        let logits = h_norm.matmul(&w_out);

        let last_pos = seq - 1;
        let last_logits = logits
            .slice(1, last_pos, 1)
            .reshape(Shape::from_dims(&[cfg.vocab_size]));

        let mut roots: Vec<&LazyTensor> = Vec::with_capacity(1 + 2 * cfg.n_layers);
        roots.push(&last_logits);
        for fk in &fresh_ks { roots.push(fk); }
        for fv in &fresh_vs { roots.push(fv); }

        let realized = realize_many_f32_cuda(&roots, executor);
        Self::unpack_kv_cache(realized, cache, cfg.n_layers, seq)
    }

    fn unpack_kv_cache(
        mut realized: Vec<Vec<f32>>,
        cache: &mut LlamaKVCache,
        n_layers: usize,
        seq: usize,
    ) -> Vec<f32> {
        let logits_vec = realized.remove(0);
        let fresh_k_vecs: Vec<Vec<f32>> = realized.drain(..n_layers).collect();
        let fresh_v_vecs: Vec<Vec<f32>> = realized.drain(..n_layers).collect();

        for (li, (fk_vec, fv_vec)) in fresh_k_vecs
            .into_iter()
            .zip(fresh_v_vecs.into_iter())
            .enumerate()
        {
            cache.append_layer(li, &fk_vec, &fv_vec, seq);
        }
        cache.cached_len += seq;

        logits_vec
    }
}

/// Per-layer KV cache: contiguous `f32` storage for the keys and values
/// the layer has seen so far, laid out as `[n_kv_heads, cached_len,
/// head_dim]` (batch is always 1 for the current decode loop).
#[derive(Debug, Clone, Default)]
pub struct LayerKVCache {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Whole-model KV cache for a [`LlamaModel`]. One [`LayerKVCache`] per
/// decoder layer plus the shared `cached_len` counter. Rebuild a fresh
/// cache between independent generations — the cache's shape is tied
/// to a specific prompt prefix.
#[derive(Debug, Clone)]
pub struct LlamaKVCache {
    pub layers:     Vec<LayerKVCache>,
    pub cached_len: usize,
    n_kv_heads:     usize,
    head_dim:       usize,
}

impl LlamaKVCache {
    /// Build an empty cache sized for a given model config.
    pub fn new(config: &LlamaConfig) -> Self {
        Self {
            layers: (0..config.n_layers)
                .map(|_| LayerKVCache::default())
                .collect(),
            cached_len: 0,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
        }
    }

    /// Append `seq` freshly computed keys and values for a given layer.
    /// Both `fresh_k` and `fresh_v` must be laid out as
    /// `[1, n_kv_heads, seq, head_dim]` (batch=1), matching the shape
    /// the graph produced them in.
    fn append_layer(&mut self, layer_idx: usize, fresh_k: &[f32], fresh_v: &[f32], seq: usize) {
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;
        let cached_len = self.cached_len;
        let new_len = cached_len + seq;

        debug_assert_eq!(fresh_k.len(), n_kv * seq * hd);
        debug_assert_eq!(fresh_v.len(), n_kv * seq * hd);

        let cache = &mut self.layers[layer_idx];
        let mut new_k = Vec::with_capacity(n_kv * new_len * hd);
        let mut new_v = Vec::with_capacity(n_kv * new_len * hd);
        for h in 0..n_kv {
            let old_off = h * cached_len * hd;
            let old_end = old_off + cached_len * hd;
            new_k.extend_from_slice(&cache.k[old_off..old_end]);
            new_v.extend_from_slice(&cache.v[old_off..old_end]);
            let new_off = h * seq * hd;
            let new_end = new_off + seq * hd;
            new_k.extend_from_slice(&fresh_k[new_off..new_end]);
            new_v.extend_from_slice(&fresh_v[new_off..new_end]);
        }
        cache.k = new_k;
        cache.v = new_v;
    }
}

/// Broadcast-add a 1-D bias along the last axis of `x`, or return
/// `x` unchanged if `bias` is `None`. Used for the Qwen2-style
/// Q/K/V attention biases — LLaMA's `bias` is always `None`, so the
/// early return makes this a no-op for the LLaMA path.
fn apply_optional_bias(
    x: LazyTensor,
    bias: Option<&Arc<[f32]>>,
    last_dim: usize,
) -> LazyTensor {
    match bias {
        None => x,
        Some(b) => {
            assert_eq!(
                b.len(),
                last_dim,
                "apply_optional_bias: bias length {} does not match last_dim {last_dim}",
                b.len(),
            );
            let b_t = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[last_dim]));
            x.broadcast_add(&b_t)
        }
    }
}

/// RmsNorm with a learned per-channel gain, applied along the last dim.
/// This is the affine version used by LLaMA: `y = (x / rms) * gain`.
///
/// `gain` is taken as `&Arc<[f32]>` so we can clone it into the const
/// node without copying the underlying slice — the same refcount-bump
/// pattern used for every other weight in the model.
fn apply_affine_rms_norm(
    x: &LazyTensor,
    gain: &Arc<[f32]>,
    dim: usize,
    eps: f64,
) -> LazyTensor {
    assert_eq!(gain.len(), dim, "apply_affine_rms_norm: gain length must equal dim");
    let normalized = x.rms_norm_last_dim(eps);
    let gain_t = x.const_f32_like(Arc::clone(gain), Shape::from_dims(&[dim]));
    normalized.broadcast_mul(&gain_t)
}

// ---- HuggingFace Hub and safetensors weight loading ----------------------

/// Load a tensor by name from a `MmapedSafetensors` as a flat
/// `Vec<f32>`, converting from whatever dtype the file stores it in.
/// Handles `F32`, `F64`, `BF16`, and `F16` — the dtypes real LLaMA
/// weights use on disk. Returns an error for unsupported dtypes.
fn load_tensor_as_f32(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
) -> crate::Result<Vec<f32>> {
    use safetensors::Dtype;
    let view = st.get(name)?;
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F64 => {
            let mut out = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.chunks_exact(8) {
                let arr: [u8; 8] = chunk.try_into().unwrap();
                out.push(f64::from_le_bytes(arr) as f32);
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::bf16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        Dtype::F16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        other => crate::bail!(
            "load_tensor_as_f32: unsupported dtype {other:?} for tensor {name:?}",
        ),
    }
}

/// Load a tensor by name and physically transpose it from `[out, in]`
/// (HuggingFace layout) to `[in, out]` (fuel-graph's layout for
/// `x @ W` where `W` is `[in, out]`). Linear-layer weights in HF
/// transformers are stored as `[out_features, in_features]`, so every
/// call to this function is effectively "give me that matrix as I'd
/// use it in `matmul`."
fn load_transposed_matrix(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<f32>> {
    let flat = load_tensor_as_f32(st, name)?;
    if flat.len() != out_features * in_features {
        crate::bail!(
            "load_transposed_matrix: tensor {name:?} has {} elements, expected {} ({out_features} × {in_features})",
            flat.len(),
            out_features * in_features,
        );
    }
    // HF layout: flat[i * in_features + j] is W[i, j] for (out i, in j).
    // Target layout: out[j * out_features + i] so that indexing `[j, i]`
    // in row-major gives the same W[i, j] — i.e. out has shape [in, out].
    let mut out = vec![0.0_f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            out[j * out_features + i] = flat[i * in_features + j];
        }
    }
    Ok(out)
}

impl LlamaWeights {
    /// Load all LLaMA weights from one or more memory-mapped safetensors
    /// files using the HuggingFace naming convention (the same names
    /// you see in any `pytorch_model.bin.index.json` or
    /// `model.safetensors.index.json` for a LLaMA-architecture model).
    ///
    /// Expected names:
    /// - `model.embed_tokens.weight` → token embedding (kept as-is)
    /// - `model.layers.{i}.self_attn.q_proj.weight` (transposed)
    /// - `model.layers.{i}.self_attn.k_proj.weight` (transposed)
    /// - `model.layers.{i}.self_attn.v_proj.weight` (transposed)
    /// - `model.layers.{i}.self_attn.o_proj.weight` (transposed)
    /// - `model.layers.{i}.mlp.gate_proj.weight` (transposed)
    /// - `model.layers.{i}.mlp.up_proj.weight` (transposed)
    /// - `model.layers.{i}.mlp.down_proj.weight` (transposed)
    /// - `model.layers.{i}.input_layernorm.weight` (per-channel gain)
    /// - `model.layers.{i}.post_attention_layernorm.weight` (per-channel gain)
    /// - `model.norm.weight` → final RmsNorm gain
    /// - `lm_head.weight` → output projection (transposed)
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &LlamaConfig,
    ) -> crate::Result<Self> {
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let token_embedding = load_tensor_as_f32(st, "model.embed_tokens.weight")?;
        if token_embedding.len() != cfg.vocab_size * cfg.dim {
            crate::bail!(
                "embed_tokens: {} elements, expected {} ({}×{})",
                token_embedding.len(),
                cfg.vocab_size * cfg.dim,
                cfg.vocab_size,
                cfg.dim,
            );
        }

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let attn_q = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.self_attn.q_proj.weight"),
                cfg.dim,
                cfg.dim,
            )?;
            let attn_k = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.self_attn.k_proj.weight"),
                kv_dim,
                cfg.dim,
            )?;
            let attn_v = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.self_attn.v_proj.weight"),
                kv_dim,
                cfg.dim,
            )?;
            let attn_o = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.self_attn.o_proj.weight"),
                cfg.dim,
                cfg.dim,
            )?;
            let ffn_gate = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.mlp.gate_proj.weight"),
                cfg.ffn_dim,
                cfg.dim,
            )?;
            let ffn_up = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.mlp.up_proj.weight"),
                cfg.ffn_dim,
                cfg.dim,
            )?;
            let ffn_down = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.mlp.down_proj.weight"),
                cfg.dim,
                cfg.ffn_dim,
            )?;
            let attn_norm_gain = load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.input_layernorm.weight"),
            )?;
            let ffn_norm_gain = load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.post_attention_layernorm.weight"),
            )?;
            // Qwen2-style biases on Q/K/V. LLaMA has no biases at all,
            // so these will return `Err` for LLaMA weights and we
            // store `None`. We don't bail — a missing bias is a
            // legitimate architectural variation, not an error.
            let attn_q_bias = load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.self_attn.q_proj.bias"),
            )
            .ok()
            .map(Arc::from);
            let attn_k_bias = load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.self_attn.k_proj.bias"),
            )
            .ok()
            .map(Arc::from);
            let attn_v_bias = load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.self_attn.v_proj.bias"),
            )
            .ok()
            .map(Arc::from);
            layers.push(LayerWeights {
                attn_q:         Arc::from(attn_q),
                attn_q_bias,
                attn_k:         Arc::from(attn_k),
                attn_k_bias,
                attn_v:         Arc::from(attn_v),
                attn_v_bias,
                attn_o:         Arc::from(attn_o),
                ffn_gate:       Arc::from(ffn_gate),
                ffn_up:         Arc::from(ffn_up),
                ffn_down:       Arc::from(ffn_down),
                attn_norm_gain: Arc::from(attn_norm_gain),
                ffn_norm_gain:  Arc::from(ffn_norm_gain),
            });
        }

        let final_norm_gain = load_tensor_as_f32(st, "model.norm.weight")?;
        // `lm_head.weight` is `[vocab_size, dim]` in HF layout; we want
        // `[dim, vocab_size]` for `h @ W_out`. Fall back to tied
        // embeddings (`lm_head.weight` absent → reuse embed_tokens) for
        // models that tie input/output weights.
        let output: Vec<f32> =
            match load_transposed_matrix(st, "lm_head.weight", cfg.vocab_size, cfg.dim) {
                Ok(w) => w,
                Err(_) => {
                    // Tied weights: transpose embed_tokens.
                    let mut transposed = vec![0.0_f32; cfg.dim * cfg.vocab_size];
                    for i in 0..cfg.vocab_size {
                        for j in 0..cfg.dim {
                            transposed[j * cfg.vocab_size + i] =
                                token_embedding[i * cfg.dim + j];
                        }
                    }
                    transposed
                }
            };

        Ok(LlamaWeights {
            token_embedding: Arc::from(token_embedding),
            layers,
            final_norm_gain: Arc::from(final_norm_gain),
            output:          Arc::from(output),
        })
    }
}

/// A small wrapper around `tokenizers::Tokenizer` tuned for the
/// chat-generation workflow: encode a prompt into token IDs, decode
/// token IDs back into a string, find the model's end-of-sequence
/// token. Lives next to LlamaModel in the same module so a decode
/// loop can keep both under one import.
pub struct LlamaTokenizer {
    inner: tokenizers::Tokenizer,
    eos_id: Option<u32>,
}

impl LlamaTokenizer {
    /// Load a tokenizer from a `tokenizer.json` on disk.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> crate::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| crate::Error::Msg(format!("loading tokenizer: {e}")))?;
        // LLaMA 3 uses `<|end_of_text|>` as EOS; LLaMA 2 uses `</s>`;
        // Qwen2 chat models use `<|im_end|>`. Try each in order and
        // take whichever the vocab has.
        let eos_id = ["<|end_of_text|>", "</s>", "<|eot_id|>", "<|im_end|>"]
            .iter()
            .find_map(|s| inner.token_to_id(s));
        Ok(Self { inner, eos_id })
    }

    /// Load a tokenizer from a HuggingFace repo. Downloads
    /// `tokenizer.json` and calls [`from_file`].
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let path = repo
            .get("tokenizer.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub tokenizer.json: {e}")))?;
        Self::from_file(path)
    }

    /// Encode a prompt into token IDs. `add_special_tokens=true`
    /// prepends the model's BOS token (for LLaMA, `<|begin_of_text|>`).
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> crate::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| crate::Error::Msg(format!("tokenizer encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a slice of token IDs back into a string.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> crate::Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| crate::Error::Msg(format!("tokenizer decode: {e}")))
    }

    /// The model's end-of-sequence token ID, if one was identified.
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }
}

/// Sampling strategy for decode loops.
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Greedy: always pick the highest-probability token.
    Greedy,
    /// Temperature-scaled sampling with a deterministic seed. `temp`
    /// is the softmax temperature (`1.0` is unscaled, `0.0` is
    /// effectively greedy, higher values spread probability mass).
    /// The seed makes sampling reproducible.
    Temperature { temp: f32, seed: u64 },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        SamplingStrategy::Greedy
    }
}

impl LlamaModel {
    /// Run greedy or temperature-sampled token generation for
    /// `max_new_tokens` steps starting from `prompt_tokens`. Returns
    /// the full sequence including the prompt.
    ///
    /// This is the minimum viable decode loop: each iteration runs a
    /// full forward pass on the entire sequence so far (no KV cache),
    /// slices out the logits for the last position, samples the next
    /// token, and appends. It stops early if the sampled token equals
    /// `eos_id`.
    ///
    /// Without a KV cache this is O(n²) in sequence length — fine for
    /// a correctness demo, way too slow for production. A cached
    /// decode loop is mechanical to add once the graph layer grows
    /// persistent state.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
    ) -> crate::Result<Vec<u32>> {
        self.generate_streaming(prompt_tokens, max_new_tokens, strategy, eos_id, |_| {})
    }

    /// Same contract as [`generate`], but invokes `on_token` once per
    /// freshly sampled token (prompt tokens are NOT emitted). Used by
    /// the CLI runner to print tokens as they're produced instead of
    /// waiting for the full sequence. Returns the full token sequence
    /// including the prompt once generation finishes or EOS is hit.
    pub fn generate_streaming(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        mut on_token: impl FnMut(u32),
    ) -> crate::Result<Vec<u32>> {
        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut rng_state: u64 = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };

        // Prefill: one forward pass over the full prompt, populating
        // the KV cache for every layer. This is the only O(prompt²)
        // matmul in the whole generation.
        let mut cache = LlamaKVCache::new(&self.config);
        let mut last_logits = self.forward_with_cache(&tokens, &mut cache);

        for _ in 0..max_new_tokens {
            let next = sample_logits(&last_logits, strategy, &mut rng_state);
            tokens.push(next);
            on_token(next);
            if let Some(eos) = eos_id {
                if next == eos {
                    break;
                }
            }
            // Decode step: feed just the one new token. The cache does
            // the work of making this O(total_seq) instead of O(total_seq²).
            last_logits = self.forward_with_cache(&[next], &mut cache);
        }
        Ok(tokens)
    }
}

/// CUDA variant of generate_streaming. Identical decode loop but
/// each forward pass runs through the CUDA graph executor.
#[cfg(feature = "cuda")]
impl LlamaModel {
    pub fn generate_streaming_cuda(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        executor: &mut fuel_graph_cuda::CudaGraphExecutor,
        mut on_token: impl FnMut(u32),
    ) -> crate::Result<Vec<u32>> {
        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut rng_state: u64 = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };

        let mut cache = LlamaKVCache::new(&self.config);
        let mut last_logits =
            self.forward_with_cache_cuda(&tokens, &mut cache, executor);

        for _ in 0..max_new_tokens {
            let next = sample_logits(&last_logits, strategy, &mut rng_state);
            tokens.push(next);
            on_token(next);
            if let Some(eos) = eos_id {
                if next == eos {
                    break;
                }
            }
            last_logits =
                self.forward_with_cache_cuda(&[next], &mut cache, executor);
        }
        Ok(tokens)
    }
}

/// Pick the next token from a logits vector using the configured
/// sampling strategy. Pulled out of `generate` so both the cached and
/// future non-cached callers can share it.
pub fn sample_logits(
    logits: &[f32],
    strategy: SamplingStrategy,
    rng_state: &mut u64,
) -> u32 {
    match strategy {
        SamplingStrategy::Greedy => {
            let (i, _) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("sample_logits: empty logits");
            i as u32
        }
        SamplingStrategy::Temperature { temp, .. } => {
            // Stable softmax over optionally temperature-scaled logits,
            // then a deterministic multinomial draw.
            let inv_temp = if temp == 0.0 { 1.0 } else { 1.0 / temp as f32 };
            let scaled: Vec<f32> = logits.iter().map(|&x| x * inv_temp).collect();
            let max = scaled
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let probs: Vec<f32> = exp.iter().map(|&x| x / sum).collect();
            sample_multinomial(&probs, rng_state)
        }
    }
}

/// Sample a categorical distribution using a small deterministic LCG.
/// Takes `probs` (assumed to sum to ~1) and a mutable RNG state,
/// returns a sampled index.
fn sample_multinomial(probs: &[f32], rng_state: &mut u64) -> u32 {
    // Advance the LCG and turn it into a u01 uniform.
    *rng_state = rng_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = (*rng_state >> 32) as f32 / u32::MAX as f32;
    let mut cumulative = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if u <= cumulative {
            return i as u32;
        }
    }
    // Floating-point slop: fall through to the last index.
    (probs.len() - 1) as u32
}

impl LlamaModel {
    /// Download a LLaMA-architecture model from the HuggingFace Hub and
    /// return a fully assembled `LlamaModel`. Uses `hf_hub::sync` for
    /// the downloads — blocking, with the usual `~/.cache/huggingface`
    /// caching semantics.
    ///
    /// `repo_id` is the HuggingFace repo name in the usual form
    /// (e.g. `"meta-llama/Meta-Llama-3-8B"`). Gated models require
    /// `HF_TOKEN` or a prior `huggingface-cli login`.
    ///
    /// This call downloads:
    /// - `config.json` — the model config
    /// - `model.safetensors.index.json` OR `model.safetensors` —
    ///   depending on whether the model is sharded
    /// - every shard in the index (if sharded)
    ///
    /// It does NOT download the tokenizer or any other files. Wire the
    /// tokenizer separately via `hf_hub::api::sync::ApiRepo::get`.
    ///
    /// For a 70B model this function will download ~150GB. The cache
    /// is persistent so subsequent calls are instant.
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());

        // 1. config.json
        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub config.json: {e}")))?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config = LlamaConfig::from_hf_json_str(&config_str)?;

        // 2. Weight file(s). Try sharded layout first, fall back to single file.
        let weight_paths: Vec<std::path::PathBuf> = match repo.get("model.safetensors.index.json") {
            Ok(index_path) => {
                let index_str = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_str)
                    .map_err(|e| crate::Error::Msg(format!("parsing index: {e}")))?;
                let weight_map = index
                    .get("weight_map")
                    .and_then(|x| x.as_object())
                    .ok_or_else(|| {
                        crate::Error::Msg("index.json: missing weight_map".into())
                    })?;
                let mut unique = std::collections::HashSet::new();
                for v in weight_map.values() {
                    if let Some(s) = v.as_str() {
                        unique.insert(s.to_string());
                    }
                }
                let mut paths: Vec<std::path::PathBuf> = Vec::new();
                for shard_name in unique {
                    let p = repo.get(&shard_name).map_err(|e| {
                        crate::Error::Msg(format!("hf-hub {shard_name}: {e}"))
                    })?;
                    paths.push(p);
                }
                paths
            }
            Err(_) => {
                // Single-shard model.
                let p = repo
                    .get("model.safetensors")
                    .map_err(|e| crate::Error::Msg(format!("hf-hub model.safetensors: {e}")))?;
                vec![p]
            }
        };

        // 3. Memory-map the safetensors files and load the weights.
        let st = unsafe {
            crate::safetensors::MmapedSafetensors::multi(&weight_paths)
        }?;
        let weights = LlamaWeights::load_from_mmapped(&st, &config)?;

        Ok(LlamaModel { config, weights })
    }
}

// ---- Gemma 2 model assembly -------------------------------------------------

/// Hyperparameters for a Gemma 2 model.
///
/// Key differences from LLaMA:
/// - `head_dim` is decoupled from `dim / n_heads`
/// - GeGLU activation instead of SwiGLU
/// - Embedding scaled by `sqrt(dim)`
/// - Four RmsNorms per layer (pre+post attention, pre+post FFN)
/// - RmsNorm offset: `(gain + 1) * normalized`
/// - Attention logit softcapping before softmax
/// - Final logit softcapping after output projection
/// - Alternating sliding-window and full attention layers
/// - `query_pre_attn_scalar` for attention scale
#[derive(Debug, Clone)]
pub struct Gemma2Config {
    pub vocab_size:             usize,
    pub dim:                    usize,
    pub n_layers:               usize,
    pub n_heads:                usize,
    pub n_kv_heads:             usize,
    pub head_dim:               usize,
    pub ffn_dim:                usize,
    pub norm_eps:               f64,
    pub rope_base:              f64,
    pub query_pre_attn_scalar:  f64,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub sliding_window:         Option<usize>,
}

impl Gemma2Config {
    pub fn from_hf_json_str(json: &str) -> crate::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| crate::Error::Msg(format!("parsing config.json: {e}")))?;

        let get_usize = |key: &str| -> crate::Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| {
                    crate::Error::Msg(format!("config.json: missing/invalid field {key:?}"))
                })
        };
        let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };

        let vocab_size = get_usize("vocab_size")?;
        let dim = get_usize("hidden_size")?;
        let n_layers = get_usize("num_hidden_layers")?;
        let n_heads = get_usize("num_attention_heads")?;
        let n_kv_heads = v
            .get("num_key_value_heads")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(n_heads);
        let ffn_dim = get_usize("intermediate_size")?;
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(dim / n_heads);
        let norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-6);
        let rope_base = get_f64("rope_theta").unwrap_or(10000.0);
        let query_pre_attn_scalar = get_f64("query_pre_attn_scalar")
            .unwrap_or(head_dim as f64);
        let attn_logit_softcapping = get_f64("attn_logit_softcapping");
        let final_logit_softcapping = get_f64("final_logit_softcapping");
        let sliding_window = v
            .get("sliding_window")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize);

        Ok(Self {
            vocab_size,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            norm_eps,
            rope_base,
            query_pre_attn_scalar,
            attn_logit_softcapping,
            final_logit_softcapping,
            sliding_window,
        })
    }
}

/// Per-layer weights for a Gemma 2 transformer block.
///
/// Four norm gains (pre/post attention + pre/post FFN) instead of LLaMA's two.
#[derive(Debug, Clone)]
pub struct Gemma2LayerWeights {
    pub attn_q:                  Arc<[f32]>,
    pub attn_k:                  Arc<[f32]>,
    pub attn_v:                  Arc<[f32]>,
    pub attn_o:                  Arc<[f32]>,
    pub ffn_gate:                Arc<[f32]>,
    pub ffn_up:                  Arc<[f32]>,
    pub ffn_down:                Arc<[f32]>,
    pub input_layernorm:         Arc<[f32]>,
    pub post_attention_layernorm: Arc<[f32]>,
    pub pre_feedforward_layernorm: Arc<[f32]>,
    pub post_feedforward_layernorm: Arc<[f32]>,
}

/// Top-level weights for a Gemma 2 model.
#[derive(Debug, Clone)]
pub struct Gemma2Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers:          Vec<Gemma2LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
}

pub struct Gemma2Model {
    pub config:  Gemma2Config,
    pub weights: Gemma2Weights,
}

/// Gemma's RmsNorm: `(gain + 1) * (x / rms)`. The `+1` centers the
/// initial gain around 1 (HF initializes to 0 rather than 1).
fn apply_gemma_rms_norm(
    x: &LazyTensor,
    gain: &Arc<[f32]>,
    dim: usize,
    eps: f64,
) -> LazyTensor {
    let normalized = x.rms_norm_last_dim(eps);
    let gain_t = x.const_f32_like(Arc::clone(gain), Shape::from_dims(&[dim]));
    let gain_plus_one = gain_t.add_scalar(1.0);
    normalized.broadcast_mul(&gain_plus_one)
}

/// Softcapping: `tanh(x / cap) * cap`. Bounds values to `[-cap, cap]`.
fn softcap(x: &LazyTensor, cap: f64) -> LazyTensor {
    x.mul_scalar(1.0 / cap).tanh().mul_scalar(cap)
}

impl Gemma2Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> LazyTensor {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;

        // Embedding + scale by sqrt(dim).
        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
        );
        let token_ids =
            embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0, &token_ids)
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))
            .mul_scalar((cfg.dim as f64).sqrt());

        // Shared RoPE tables.
        let (cos_data, sin_data) = fuel_graph::build_rope_tables(
            cfg.rope_base,
            start_pos,
            seq,
            cfg.head_dim,
        );
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        for (li, layer) in weights.layers.iter().enumerate() {
            h = self.apply_layer(&h, layer, li, start_pos, &rope_cos, &rope_sin);
        }

        // Final norm.
        let h_norm = apply_gemma_rms_norm(
            &h,
            &weights.final_norm_gain,
            cfg.dim,
            cfg.norm_eps,
        );

        // Output projection (tied embeddings — transpose embed_tokens).
        // embed_tokens is [vocab_size, dim]; for `h @ W` we need [dim, vocab_size].
        // We store the already-transposed version.
        let w_out_data = {
            let e = weights.token_embedding.as_ref();
            let mut t = vec![0.0_f32; cfg.dim * cfg.vocab_size];
            for i in 0..cfg.vocab_size {
                for j in 0..cfg.dim {
                    t[j * cfg.vocab_size + i] = e[i * cfg.dim + j];
                }
            }
            t
        };
        let w_out = h_norm.const_f32_like(
            w_out_data,
            Shape::from_dims(&[cfg.dim, cfg.vocab_size]),
        );
        let logits = h_norm.matmul(&w_out);

        // Final logit softcapping.
        match cfg.final_logit_softcapping {
            Some(cap) if cap > 0.0 => softcap(&logits, cap),
            _ => logits,
        }
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &Gemma2LayerWeights,
        layer_idx: usize,
        start_pos: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> LazyTensor {
        let cfg = &self.config;
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let qk_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;

        // Pre-attention RmsNorm (Gemma offset style).
        let x_norm = apply_gemma_rms_norm(x, &layer.input_layernorm, cfg.dim, cfg.norm_eps);

        // Q/K/V projections. Note: Gemma 2 projects to head_dim * n_heads
        // which may differ from dim when head_dim != dim/n_heads.
        let w_q = x.const_f32_like(layer.attn_q.clone(), Shape::from_dims(&[cfg.dim, qk_dim]));
        let w_k = x.const_f32_like(layer.attn_k.clone(), Shape::from_dims(&[cfg.dim, kv_dim]));
        let w_v = x.const_f32_like(layer.attn_v.clone(), Shape::from_dims(&[cfg.dim, kv_dim]));
        let w_o = x.const_f32_like(layer.attn_o.clone(), Shape::from_dims(&[qk_dim, cfg.dim]));
        let q = x_norm.matmul(&w_q);
        let k = x_norm.matmul(&w_k);
        let v = x_norm.matmul(&w_v);

        // Split heads.
        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);

        // RoPE.
        let q_r = q_h.rope_with_tables(rope_cos, rope_sin);
        let k_r = k_h.rope_with_tables(rope_cos, rope_sin);

        // GQA expansion.
        let (k_r, v_h) = if cfg.n_kv_heads == cfg.n_heads {
            (k_r, v_h)
        } else {
            assert_eq!(cfg.n_heads % cfg.n_kv_heads, 0);
            let n_rep = cfg.n_heads / cfg.n_kv_heads;
            let expand = |t: LazyTensor| -> LazyTensor {
                t.reshape(Shape::from_dims(&[batch, cfg.n_kv_heads, 1, seq, cfg.head_dim]))
                    .broadcast_to(Shape::from_dims(&[batch, cfg.n_kv_heads, n_rep, seq, cfg.head_dim]))
                    .reshape(Shape::from_dims(&[batch, cfg.n_heads, seq, cfg.head_dim]))
            };
            (expand(k_r), expand(v_h))
        };

        // Attention with query_pre_attn_scalar and optional softcapping.
        let k_t = k_r.transpose();
        let scale = 1.0 / cfg.query_pre_attn_scalar.sqrt();
        let scores = q_r.matmul(&k_t);
        let scores_scaled = scores.mul_scalar(scale);

        // Attention logit softcapping.
        let scores_capped = match cfg.attn_logit_softcapping {
            Some(cap) if cap > 0.0 => softcap(&scores_scaled, cap),
            _ => scores_scaled,
        };

        // Causal mask (with optional sliding window on alternating layers).
        let use_sliding = cfg.sliding_window.is_some() && layer_idx % 2 == 0;
        let window = cfg.sliding_window.unwrap_or(usize::MAX);
        let mut mask_data = vec![0.0_f32; seq * seq];
        for q_pos in 0..seq {
            for k_pos in 0..seq {
                let abs_q = start_pos + q_pos;
                let abs_k = start_pos + k_pos;
                let causal_ok = abs_k <= abs_q;
                let window_ok = !use_sliding || abs_q.saturating_sub(abs_k) < window;
                if !(causal_ok && window_ok) {
                    mask_data[q_pos * seq + k_pos] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));
        let scores_masked = scores_capped.broadcast_add(&mask);
        let attn = scores_masked.softmax_last_dim();
        let attn_v = attn.matmul(&v_h);

        // Merge heads + output projection.
        let merged = attn_v
            .permute(&[0, 2, 1, 3])
            .reshape(Shape::from_dims(&[batch, seq, qk_dim]));
        let attn_out = merged.matmul(&w_o);

        // Post-attention RmsNorm (Gemma has this; LLaMA does not).
        let attn_out_norm = apply_gemma_rms_norm(
            &attn_out,
            &layer.post_attention_layernorm,
            cfg.dim,
            cfg.norm_eps,
        );

        // First residual.
        let h1 = x.add(&attn_out_norm);

        // Pre-FFN RmsNorm.
        let h1_norm = apply_gemma_rms_norm(
            &h1,
            &layer.pre_feedforward_layernorm,
            cfg.dim,
            cfg.norm_eps,
        );

        // GeGLU FFN (GELU activation instead of SiLU).
        let w_gate = x.const_f32_like(layer.ffn_gate.clone(), Shape::from_dims(&[cfg.dim, cfg.ffn_dim]));
        let w_up = x.const_f32_like(layer.ffn_up.clone(), Shape::from_dims(&[cfg.dim, cfg.ffn_dim]));
        let w_down = x.const_f32_like(layer.ffn_down.clone(), Shape::from_dims(&[cfg.ffn_dim, cfg.dim]));
        let gate = h1_norm.matmul(&w_gate);
        let up = h1_norm.matmul(&w_up);
        let geglu = gate.gelu().mul(&up);
        let ffn_out = geglu.matmul(&w_down);

        // Post-FFN RmsNorm.
        let ffn_out_norm = apply_gemma_rms_norm(
            &ffn_out,
            &layer.post_feedforward_layernorm,
            cfg.dim,
            cfg.norm_eps,
        );

        // Second residual.
        h1.add(&ffn_out_norm)
    }
}

impl Gemma2Weights {
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Gemma2Config,
    ) -> crate::Result<Self> {
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let qk_dim = cfg.n_heads * cfg.head_dim;
        let token_embedding = load_tensor_as_f32(st, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let attn_q = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.self_attn.q_proj.weight"),
                qk_dim,
                cfg.dim,
            )?;
            let attn_k = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.self_attn.k_proj.weight"),
                kv_dim,
                cfg.dim,
            )?;
            let attn_v = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.self_attn.v_proj.weight"),
                kv_dim,
                cfg.dim,
            )?;
            let attn_o = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.self_attn.o_proj.weight"),
                cfg.dim,
                qk_dim,
            )?;
            let ffn_gate = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.mlp.gate_proj.weight"),
                cfg.ffn_dim,
                cfg.dim,
            )?;
            let ffn_up = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.mlp.up_proj.weight"),
                cfg.ffn_dim,
                cfg.dim,
            )?;
            let ffn_down = load_transposed_matrix(
                st,
                &format!("model.layers.{i}.mlp.down_proj.weight"),
                cfg.dim,
                cfg.ffn_dim,
            )?;
            let input_layernorm = load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.input_layernorm.weight"),
            )?;
            let post_attention_layernorm = load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.post_attention_layernorm.weight"),
            )?;
            let pre_feedforward_layernorm = load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.pre_feedforward_layernorm.weight"),
            )?;
            let post_feedforward_layernorm = load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.post_feedforward_layernorm.weight"),
            )?;
            layers.push(Gemma2LayerWeights {
                attn_q:                    Arc::from(attn_q),
                attn_k:                    Arc::from(attn_k),
                attn_v:                    Arc::from(attn_v),
                attn_o:                    Arc::from(attn_o),
                ffn_gate:                  Arc::from(ffn_gate),
                ffn_up:                    Arc::from(ffn_up),
                ffn_down:                  Arc::from(ffn_down),
                input_layernorm:           Arc::from(input_layernorm),
                post_attention_layernorm:  Arc::from(post_attention_layernorm),
                pre_feedforward_layernorm: Arc::from(pre_feedforward_layernorm),
                post_feedforward_layernorm: Arc::from(post_feedforward_layernorm),
            });
        }

        let final_norm_gain = load_tensor_as_f32(st, "model.norm.weight")?;

        Ok(Gemma2Weights {
            token_embedding: Arc::from(token_embedding),
            layers,
            final_norm_gain: Arc::from(final_norm_gain),
        })
    }
}

impl Gemma2Model {
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub config.json: {e}")))?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config = Gemma2Config::from_hf_json_str(&config_str)?;

        let weight_paths: Vec<std::path::PathBuf> =
            match repo.get("model.safetensors.index.json") {
                Ok(index_path) => {
                    let index_str = std::fs::read_to_string(&index_path)?;
                    let index: serde_json::Value = serde_json::from_str(&index_str)
                        .map_err(|e| crate::Error::Msg(format!("parsing index: {e}")))?;
                    let weight_map = index
                        .get("weight_map")
                        .and_then(|x| x.as_object())
                        .ok_or_else(|| crate::Error::Msg("index: missing weight_map".into()))?;
                    let mut unique = std::collections::HashSet::new();
                    for v in weight_map.values() {
                        if let Some(s) = v.as_str() {
                            unique.insert(s.to_string());
                        }
                    }
                    let mut paths = Vec::new();
                    for name in &unique {
                        paths.push(
                            repo.get(name)
                                .map_err(|e| crate::Error::Msg(format!("hf-hub {name}: {e}")))?,
                        );
                    }
                    paths
                }
                Err(_) => {
                    vec![repo
                        .get("model.safetensors")
                        .map_err(|e| {
                            crate::Error::Msg(format!("hf-hub model.safetensors: {e}"))
                        })?]
                }
            };

        let st = unsafe {
            crate::safetensors::MmapedSafetensors::multi(&weight_paths)
        }?;
        let weights = Gemma2Weights::load_from_mmapped(&st, &config)?;

        Ok(Gemma2Model { config, weights })
    }
}

#[cfg(test)]
mod hub_tests {
    use super::*;

    #[test]
    fn parse_llama3_style_hf_config() {
        // A minimal LLaMA 3 8B config.json. Real values from the
        // Hugging Face card; we just check the parser maps every field
        // correctly.
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "head_dim": 128,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16"
        }"#;
        let cfg = LlamaConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.dim, 4096);
        assert_eq!(cfg.ffn_dim, 14336);
        assert_eq!(cfg.n_layers, 32);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 8); // GQA
        assert_eq!(cfg.vocab_size, 128256);
        assert!((cfg.norm_eps - 1e-5).abs() < 1e-12);
        assert!((cfg.rope_base - 500_000.0).abs() < 1e-6);
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn parse_legacy_llama_config_defaults_to_mha() {
        // Older LLaMA 1 configs don't have `num_key_value_heads` or
        // `rope_theta`. The parser should fall back to non-GQA and
        // rope base 10000.
        let json = r#"{
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5
        }"#;
        let cfg = LlamaConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.n_kv_heads, cfg.n_heads);
        assert_eq!(cfg.head_dim, 64 / 4);
        assert!((cfg.rope_base - 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn parse_rejects_missing_required_fields() {
        let json = r#"{"hidden_size": 64}"#;
        let result = LlamaConfig::from_hf_json_str(json);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod generate_tests {
    use super::*;

    /// Same tiny-weight helper as the llama_tests module, duplicated
    /// here to keep these tests self-contained.
    fn make_tiny_weights(cfg: &LlamaConfig) -> LlamaWeights {
        let mut s: u32 = 9999;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            let v: Vec<f32> = (0..n).map(|_| next()).collect();
            Arc::from(v)
        };
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| LayerWeights {
                    attn_q:         vec_of(cfg.dim * cfg.dim),
                    attn_q_bias:    None,
                    attn_k:         vec_of(cfg.dim * kv_dim),
                    attn_k_bias:    None,
                    attn_v:         vec_of(cfg.dim * kv_dim),
                    attn_v_bias:    None,
                    attn_o:         vec_of(cfg.dim * cfg.dim),
                    ffn_gate:       vec_of(cfg.dim * cfg.ffn_dim),
                    ffn_up:         vec_of(cfg.dim * cfg.ffn_dim),
                    ffn_down:       vec_of(cfg.ffn_dim * cfg.dim),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain:  Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output:          vec_of(cfg.dim * cfg.vocab_size),
        }
    }

    /// Qwen2-style tiny weights: same shapes as LLaMA plus Q/K/V
    /// biases. Used to verify the bias path is wired through both
    /// `forward` and `forward_with_cache`.
    fn make_tiny_weights_with_qkv_bias(cfg: &LlamaConfig) -> LlamaWeights {
        let mut w = make_tiny_weights(cfg);
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        for layer in &mut w.layers {
            layer.attn_q_bias = Some(Arc::from(vec![0.01_f32; cfg.dim]));
            layer.attn_k_bias = Some(Arc::from(vec![0.01_f32; kv_dim]));
            layer.attn_v_bias = Some(Arc::from(vec![0.01_f32; kv_dim]));
        }
        w
    }

    #[test]
    fn qwen2_style_bias_changes_forward_output_but_keeps_it_finite() {
        // Build two identical tiny LLaMAs: one with all-None biases,
        // one with small nonzero biases. The bias-bearing model must
        // still produce finite logits and must produce a different
        // argmax (otherwise the bias code is dead).
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   2,
            n_heads:    4,
            n_kv_heads: 2,
            head_dim:   2,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let tokens = [1_u32, 2, 3];
        let no_bias = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let with_bias = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights_with_qkv_bias(&cfg),
        };
        let no_bias_logits = no_bias
            .forward(&tokens, 0)
            .slice(1, tokens.len() - 1, 1)
            .reshape(Shape::from_dims(&[cfg.vocab_size]))
            .realize_f32();
        let with_bias_logits = with_bias
            .forward(&tokens, 0)
            .slice(1, tokens.len() - 1, 1)
            .reshape(Shape::from_dims(&[cfg.vocab_size]))
            .realize_f32();
        for &v in &with_bias_logits {
            assert!(v.is_finite(), "with-bias logit is non-finite: {v}");
        }
        let any_different = no_bias_logits
            .iter()
            .zip(with_bias_logits.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            any_different,
            "bias had no effect — check that apply_optional_bias is actually called",
        );
    }

    #[test]
    fn qwen2_style_bias_cached_matches_non_cached_generate() {
        // Same correctness bar as the LLaMA version: greedy generation
        // via the cached path must match a non-cached greedy loop.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   2,
            n_heads:    4,
            n_kv_heads: 2,
            head_dim:   2,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let model = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights_with_qkv_bias(&cfg),
        };
        let prompt = [1_u32, 2, 3];
        let max_new = 4;

        // Non-cached reference loop.
        let mut ref_tokens = prompt.to_vec();
        for _ in 0..max_new {
            let logits = model.forward(&ref_tokens, 0);
            let last_pos = ref_tokens.len() - 1;
            let last = logits
                .slice(1, last_pos, 1)
                .reshape(Shape::from_dims(&[cfg.vocab_size]))
                .realize_f32();
            let next = last
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            ref_tokens.push(next);
        }

        let cached = model
            .generate(&prompt, max_new, SamplingStrategy::Greedy, None)
            .unwrap();
        assert_eq!(cached, ref_tokens);
    }

    #[test]
    fn generate_greedy_appends_tokens() {
        // Run greedy generation for 4 steps from a 3-token prompt on a
        // tiny model. Output sequence should be 3+4=7 tokens, all
        // valid vocab indices.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   2,
            n_heads:    2,
            n_kv_heads: 2,
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let model = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let out = model
            .generate(&[1, 2, 3], 4, SamplingStrategy::Greedy, None)
            .unwrap();
        assert_eq!(out.len(), 7);
        for &t in &out {
            assert!(
                (t as usize) < cfg.vocab_size,
                "sampled token {t} out of vocab",
            );
        }
    }

    #[test]
    fn generate_temperature_is_deterministic_with_seed() {
        // Two runs with the same seed must produce identical output.
        let cfg = LlamaConfig {
            vocab_size: 8,
            dim:        8,
            n_layers:   1,
            n_heads:    2,
            n_kv_heads: 2,
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let model = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let strategy = SamplingStrategy::Temperature { temp: 1.0, seed: 42 };
        let a = model.generate(&[0, 1], 3, strategy, None).unwrap();
        let b = model.generate(&[0, 1], 3, strategy, None).unwrap();
        assert_eq!(a, b, "seeded sampling must be deterministic");
    }

    #[test]
    fn generate_stops_early_on_eos() {
        // Construct a tiny model and pick whatever token greedy
        // selects at step 1 as our "eos". The second call then must
        // stop after exactly one new token (since the first new token
        // equals eos).
        let cfg = LlamaConfig {
            vocab_size: 8,
            dim:        8,
            n_layers:   1,
            n_heads:    2,
            n_kv_heads: 2,
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let model = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [1_u32, 2];
        // First: generate one step without eos to see which token
        // greedy picks.
        let baseline = model
            .generate(&prompt, 1, SamplingStrategy::Greedy, None)
            .unwrap();
        let picked = *baseline.last().unwrap();
        // Second: generate with that token as eos. Should stop after
        // appending it (length = prompt + 1).
        let with_eos = model
            .generate(&prompt, 10, SamplingStrategy::Greedy, Some(picked))
            .unwrap();
        assert_eq!(with_eos.len(), prompt.len() + 1);
        assert_eq!(*with_eos.last().unwrap(), picked);
    }

    #[test]
    fn forward_with_cache_matches_forward_on_prefill() {
        // Single forward pass on the full prompt via the cached path
        // must produce the same last-position logits as a non-cached
        // forward followed by the same last-position slice.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   2,
            n_heads:    2,
            n_kv_heads: 2,
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let model = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let tokens = [1_u32, 2, 3, 4];

        // Non-cached: full forward, slice last position.
        let logits_full = model.forward(&tokens, 0);
        let last_pos = tokens.len() - 1;
        let expected = logits_full
            .slice(1, last_pos, 1)
            .reshape(Shape::from_dims(&[cfg.vocab_size]))
            .realize_f32();

        // Cached: forward_with_cache already returns the last-position slice.
        let mut cache = LlamaKVCache::new(&cfg);
        let actual = model.forward_with_cache(&tokens, &mut cache);

        assert_eq!(cache.cached_len, tokens.len());
        assert_eq!(actual.len(), expected.len());
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "logit[{i}]: cached={a}, non-cached={b}",
            );
        }
    }

    #[test]
    fn generate_with_cache_matches_non_cached_generate() {
        // Greedy generation must produce the same token sequence
        // whether or not the KV cache is in use. Uses an internal
        // non-cached reference loop so this test does not depend on
        // the public `generate` still having a non-cached path.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   2,
            n_heads:    2,
            n_kv_heads: 2,
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let model = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [1_u32, 2, 3];
        let max_new = 5;

        // Reference: non-cached greedy loop.
        let mut ref_tokens = prompt.to_vec();
        for _ in 0..max_new {
            let logits = model.forward(&ref_tokens, 0);
            let last_pos = ref_tokens.len() - 1;
            let last = logits
                .slice(1, last_pos, 1)
                .reshape(Shape::from_dims(&[cfg.vocab_size]))
                .realize_f32();
            let next = last
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            ref_tokens.push(next);
        }

        // Cached: the public generate() routine.
        let cached = model
            .generate(&prompt, max_new, SamplingStrategy::Greedy, None)
            .unwrap();

        assert_eq!(cached, ref_tokens);
    }

    #[test]
    fn forward_with_cache_decode_step_matches_full_forward() {
        // Prefill a 3-token prompt, run one decode step through the
        // cache, and confirm the last-position logits match what the
        // non-cached forward() would produce for the same 4-token
        // sequence. This is the real correctness bar: concatenation of
        // cached K/V with fresh K/V must exactly equal computing K/V
        // from scratch over the whole seen sequence.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   2,
            n_heads:    4,
            n_kv_heads: 2, // exercise GQA (n_rep = 2)
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        // dim / n_heads * n_heads check: dim=8, head_dim*n_heads=16 ≠ 8.
        // Adjust dim so n_heads * head_dim == dim.
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let next_token = 4_u32;

        // Non-cached reference: full forward over all 4 tokens, slice last.
        let full = [prompt[0], prompt[1], prompt[2], next_token];
        let full_logits = model.forward(&full, 0);
        let last_pos = full.len() - 1;
        let expected = full_logits
            .slice(1, last_pos, 1)
            .reshape(Shape::from_dims(&[cfg.vocab_size]))
            .realize_f32();

        // Cached: prefill with prompt, then one decode step with the new token.
        let mut cache = LlamaKVCache::new(&cfg);
        let _prefill_logits = model.forward_with_cache(&prompt, &mut cache);
        let actual = model.forward_with_cache(&[next_token], &mut cache);

        assert_eq!(cache.cached_len, full.len());
        assert_eq!(actual.len(), expected.len());
        // Tolerance is slightly looser than the prefill-match test:
        // the cached decode path builds K/V via a `concat(cached,
        // fresh)` along the seq dim, which causes `gemm` to accumulate
        // along a contiguous dimension in a different order than the
        // single-tensor path does. That's a standard O(ε) FP drift on
        // matmul, not a correctness bug — the greedy-tokens test
        // elsewhere confirms it never crosses an argmax boundary on
        // these weights.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / a.abs().max(b.abs()).max(1e-6);
            assert!(
                diff < 5e-3 || rel < 1e-2,
                "logit[{i}]: cached={a}, non-cached={b}, diff={diff}",
            );
        }
    }

    #[test]
    fn sample_multinomial_respects_distribution() {
        // Heavy-loaded distribution: 99% on index 0. The sampler
        // should pick 0 almost always.
        let probs = vec![0.99_f32, 0.005, 0.005];
        let mut state: u64 = 12345;
        let mut counts = [0_usize; 3];
        for _ in 0..1000 {
            let idx = sample_multinomial(&probs, &mut state) as usize;
            counts[idx] += 1;
        }
        assert!(counts[0] > 900, "expected ≥900 samples on index 0, got {}", counts[0]);
    }
}

#[cfg(test)]
mod gqa_tests {
    use super::*;

    /// Build tiny GQA weights for forward-pass tests.
    fn make_tiny_weights(cfg: &LlamaConfig) -> LlamaWeights {
        let mut s: u32 = 5678;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            let v: Vec<f32> = (0..n).map(|_| next()).collect();
            Arc::from(v)
        };
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| LayerWeights {
                    attn_q:         vec_of(cfg.dim * cfg.dim),
                    attn_q_bias:    None,
                    attn_k:         vec_of(cfg.dim * kv_dim),
                    attn_k_bias:    None,
                    attn_v:         vec_of(cfg.dim * kv_dim),
                    attn_v_bias:    None,
                    attn_o:         vec_of(cfg.dim * cfg.dim),
                    ffn_gate:       vec_of(cfg.dim * cfg.ffn_dim),
                    ffn_up:         vec_of(cfg.dim * cfg.ffn_dim),
                    ffn_down:       vec_of(cfg.ffn_dim * cfg.dim),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain:  Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output:          vec_of(cfg.dim * cfg.vocab_size),
        }
    }

    #[test]
    fn llama_forward_with_gqa_matches_llama3_ratio() {
        // A Llama-3-sized head ratio in miniature: n_heads = 4,
        // n_kv_heads = 1. Every query head shares the single K and V
        // head via broadcast. Most interesting because it's the
        // extreme case (n_rep = 4).
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   1,
            n_heads:    4,
            n_kv_heads: 1,
            head_dim:   2,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel { config: cfg.clone(), weights };

        let tokens = vec![0_u32, 1, 2];
        let logits = model.forward(&tokens, 0);
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        let realized = logits.realize_f32();
        for &v in &realized {
            assert!(v.is_finite(), "GQA logit non-finite: {v}");
        }
    }

    #[test]
    fn llama_forward_with_2to1_gqa_ratio() {
        // n_heads = 4, n_kv_heads = 2 (classic GQA 2:1 ratio).
        let cfg = LlamaConfig {
            vocab_size: 8,
            dim:        8,
            n_layers:   2,
            n_heads:    4,
            n_kv_heads: 2,
            head_dim:   2,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel { config: cfg.clone(), weights };
        let tokens = vec![1_u32, 3];
        let logits = model.forward(&tokens, 0).realize_f32();
        assert_eq!(logits.len(), 1 * 2 * cfg.vocab_size);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }
}

#[cfg(test)]
mod llama_tests {
    use super::*;
    use crate::Shape;

    /// Build a set of tiny LLaMA weights filled with deterministic,
    /// small, "random" values. Respects `cfg.n_kv_heads` so GQA-style
    /// shapes come out correctly.
    fn make_tiny_weights(cfg: &LlamaConfig) -> LlamaWeights {
        let mut s: u32 = 2024;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            let v: Vec<f32> = (0..n).map(|_| next()).collect();
            Arc::from(v)
        };
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| LayerWeights {
                    attn_q:         vec_of(cfg.dim * cfg.dim),
                    attn_q_bias:    None,
                    attn_k:         vec_of(cfg.dim * kv_dim),
                    attn_k_bias:    None,
                    attn_v:         vec_of(cfg.dim * kv_dim),
                    attn_v_bias:    None,
                    attn_o:         vec_of(cfg.dim * cfg.dim),
                    ffn_gate:       vec_of(cfg.dim * cfg.ffn_dim),
                    ffn_up:         vec_of(cfg.dim * cfg.ffn_dim),
                    ffn_down:       vec_of(cfg.ffn_dim * cfg.dim),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain:  Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output:          vec_of(cfg.dim * cfg.vocab_size),
        }
    }

    #[test]
    fn llama_forward_produces_correct_logit_shape() {
        // A 2-layer 8-dim 2-head LLaMA. Not trained, just checking
        // that the graph builds, runs, and emits a logit tensor of the
        // expected shape.
        let cfg = LlamaConfig {
            vocab_size: 32,
            dim:        8,
            n_layers:   2,
            n_heads:    2,
            n_kv_heads: 2,
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel { config: cfg.clone(), weights };

        let tokens: Vec<u32> = vec![5, 12, 0, 7];
        let logits = model.forward(&tokens, 0);
        assert_eq!(logits.shape().dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn llama_forward_realizes_to_finite_logits() {
        // Same config, smaller vocab for faster realization. The
        // output must be finite across the full [1, seq, vocab] tensor.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   2,
            n_heads:    2,
            n_kv_heads: 2,
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel { config: cfg.clone(), weights };

        let tokens = vec![1_u32, 2, 3];
        let logits = model.forward(&tokens, 0);
        let logits_vec = logits.realize_f32();
        assert_eq!(logits_vec.len(), 1 * 3 * cfg.vocab_size);
        for &v in &logits_vec {
            assert!(v.is_finite(), "llama logit non-finite: {v}");
        }
    }

    #[test]
    fn llama_forward_is_relative_position_invariant() {
        // RoPE has a specific and well-known property: the attention
        // scores depend only on *relative* position differences, not
        // absolute positions. That means the forward output of a
        // LlamaModel on the same input sequence should be (modulo
        // floating-point noise) independent of `start_pos`, even
        // though the Q and K vectors themselves change.
        //
        // This test enforces the property: start_pos=0 and
        // start_pos=10 on the same token sequence must produce
        // identical logits. It's both a validation that RoPE is
        // implemented correctly AND a documented invariant for any
        // caller building a KV-cached decode loop — the cache has to
        // track absolute positions, not relative, because relative
        // differences change as new tokens arrive.
        let cfg = LlamaConfig {
            vocab_size: 8,
            dim:        8,
            n_layers:   1,
            n_heads:    2,
            n_kv_heads: 2,
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel { config: cfg.clone(), weights };

        let tokens = vec![2_u32, 4];
        let l0 = model.forward(&tokens, 0).realize_f32();
        let l10 = model.forward(&tokens, 10).realize_f32();
        // Relative-position invariance: the two should match exactly.
        assert_eq!(
            l0, l10,
            "RoPE attention should be invariant to start_pos for a fixed input",
        );
    }

    #[test]
    fn llama_forward_argmax_selects_a_token_id() {
        // Predict next-token by argmax over the last position's logits.
        // Not testing correctness (weights are random); just that the
        // predicted ID is a valid vocabulary index. This is the
        // decode-step primitive a sampling loop would call.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   2,
            n_heads:    2,
            n_kv_heads: 2,
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel { config: cfg.clone(), weights };

        let tokens = vec![3_u32, 1, 4, 1, 5];
        let logits = model.forward(&tokens, 0);
        // Take last-position slice and argmax over vocab dim, all
        // through the LazyTensor bridge API.
        let last = logits.slice(1, tokens.len() - 1, 1); // [1, 1, vocab]
        let last_flat = last.reshape(Shape::from_dims(&[cfg.vocab_size]));
        let predicted_ids = last_flat.argmax_dim(0).realize_u32();
        assert_eq!(predicted_ids.len(), 1);
        let pred = predicted_ids[0];
        assert!(
            (pred as usize) < cfg.vocab_size,
            "argmax should return a valid vocab index",
        );
    }
}

#[cfg(test)]
mod gemma2_tests {
    use super::*;

    fn make_tiny_gemma2_config() -> Gemma2Config {
        Gemma2Config {
            vocab_size:             16,
            dim:                    8,
            n_layers:               2,
            n_heads:                4,
            n_kv_heads:             2,
            head_dim:               4,
            ffn_dim:                16,
            norm_eps:               1e-6,
            rope_base:              10000.0,
            query_pre_attn_scalar:  4.0,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            sliding_window:         Some(4),
        }
    }

    fn make_tiny_gemma2_weights(cfg: &Gemma2Config) -> Gemma2Weights {
        let mut s: u32 = 7777;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            let v: Vec<f32> = (0..n).map(|_| next()).collect();
            Arc::from(v)
        };
        let qk_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        Gemma2Weights {
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| Gemma2LayerWeights {
                    attn_q:                    vec_of(cfg.dim * qk_dim),
                    attn_k:                    vec_of(cfg.dim * kv_dim),
                    attn_v:                    vec_of(cfg.dim * kv_dim),
                    attn_o:                    vec_of(qk_dim * cfg.dim),
                    ffn_gate:                  vec_of(cfg.dim * cfg.ffn_dim),
                    ffn_up:                    vec_of(cfg.dim * cfg.ffn_dim),
                    ffn_down:                  vec_of(cfg.ffn_dim * cfg.dim),
                    input_layernorm:           Arc::from(vec![0.0_f32; cfg.dim]),
                    post_attention_layernorm:  Arc::from(vec![0.0_f32; cfg.dim]),
                    pre_feedforward_layernorm: Arc::from(vec![0.0_f32; cfg.dim]),
                    post_feedforward_layernorm: Arc::from(vec![0.0_f32; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![0.0_f32; cfg.dim]),
        }
    }

    #[test]
    fn gemma2_forward_produces_finite_logits() {
        let cfg = make_tiny_gemma2_config();
        let model = Gemma2Model {
            config:  cfg.clone(),
            weights: make_tiny_gemma2_weights(&cfg),
        };
        let logits = model.forward(&[1, 2, 3], 0);
        let v = logits.realize_f32();
        assert_eq!(v.len(), 1 * 3 * cfg.vocab_size);
        for &x in &v {
            assert!(x.is_finite(), "logit is non-finite: {x}");
        }
    }

    #[test]
    fn gemma2_softcapping_bounds_logits() {
        let cfg = make_tiny_gemma2_config();
        let model = Gemma2Model {
            config:  cfg.clone(),
            weights: make_tiny_gemma2_weights(&cfg),
        };
        let logits = model.forward(&[1, 2, 3], 0);
        let v = logits.realize_f32();
        let cap = cfg.final_logit_softcapping.unwrap() as f32;
        for &x in &v {
            assert!(
                x.abs() <= cap + 1e-3,
                "logit {x} exceeds softcap {cap}",
            );
        }
    }

    #[test]
    fn gemma2_config_parses_hf_format() {
        let json = r#"{
            "architectures": ["Gemma2ForCausalLM"],
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "num_hidden_layers": 26,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 256000,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "query_pre_attn_scalar": 256,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "sliding_window": 4096
        }"#;
        let cfg = Gemma2Config::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.dim, 2304);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.n_kv_heads, 4);
        assert_eq!(cfg.vocab_size, 256000);
        assert_eq!(cfg.sliding_window, Some(4096));
        assert!((cfg.query_pre_attn_scalar - 256.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod safetensors_bridge_tests {
    use super::*;

    #[test]
    fn from_safetensors_bytes_round_trip_f32() {
        // Build a tensor, serialize it as little-endian f32 bytes,
        // then deserialize via the bridge. Should get back the same
        // values.
        let original: Vec<f32> = vec![1.0, -2.5, 3.25, 0.0];
        let mut bytes = Vec::with_capacity(original.len() * 4);
        for &v in &original {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let t = LazyTensor::from_safetensors_bytes(
            &bytes,
            safetensors::Dtype::F32,
            &[4],
        )
        .unwrap();
        assert_eq!(t.shape().dims(), &[4]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.realize_f32(), original);
    }

    #[test]
    fn from_safetensors_bytes_round_trip_bf16() {
        let original_f32: Vec<f32> = vec![0.5, -1.0, 2.0, 4.0];
        let bf16_vec: Vec<half::bf16> =
            original_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let mut bytes = Vec::with_capacity(bf16_vec.len() * 2);
        for b in &bf16_vec {
            bytes.extend_from_slice(&b.to_bits().to_le_bytes());
        }
        let t = LazyTensor::from_safetensors_bytes(
            &bytes,
            safetensors::Dtype::BF16,
            &[4],
        )
        .unwrap();
        assert_eq!(t.dtype(), DType::BF16);
        // Values that round-trip exactly through bf16 should come back
        // unchanged.
        let realized = t.realize_bf16();
        assert_eq!(realized, bf16_vec);
    }

    #[test]
    fn from_safetensors_bytes_rejects_wrong_byte_count() {
        // Shape says 3 elements, but we pass 4 bytes (1 f32 = 4 bytes
        // so 3 elements would need 12 bytes).
        let bad_bytes = vec![0_u8; 4];
        let result = LazyTensor::from_safetensors_bytes(
            &bad_bytes,
            safetensors::Dtype::F32,
            &[3],
        );
        assert!(result.is_err());
    }
}
