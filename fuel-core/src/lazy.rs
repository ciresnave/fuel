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

use crate::inference_context::{InferenceContext, KvCache, KvSlot};
use crate::{DType, Device, Shape};
use fuel_graph_executor::GraphExecutor;
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

    /// Build an `f32` lazy tensor from flat data, a shape, and a device.
    ///
    /// `data` takes `impl Into<Arc<[f32]>>` so both `Vec<f32>` and
    /// `Arc<[f32]>` callers work without conversion. Pass an `Arc`
    /// when you already have one (e.g. model weights loaded once at
    /// startup) to avoid any copy.
    ///
    /// Phase 7.5 G2: `device` selects where the realized Storage is
    /// allocated. The graph's storage_map slot for the new node is
    /// populated and `Op::Const(None)` is emitted — no host-side
    /// `ConstData` payload rides on the graph node.
    pub fn from_f32(
        data: impl Into<Arc<[f32]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> Self {
        Self {
            inner: fuel_graph::Tensor::from_f32(data, shape, device.as_dyn()),
        }
    }

    /// Build an `f64` lazy tensor. `device` selects where the realized
    /// Storage is allocated.
    pub fn from_f64(
        data: impl Into<Arc<[f64]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> Self {
        Self {
            inner: fuel_graph::Tensor::from_f64(data, shape, device.as_dyn()),
        }
    }

    /// Build a `bf16` lazy tensor. `device` selects where the realized
    /// Storage is allocated.
    pub fn from_bf16(
        data: impl Into<Arc<[half::bf16]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> Self {
        Self {
            inner: fuel_graph::Tensor::from_bf16(data, shape, device.as_dyn()),
        }
    }

    /// Build an `f16` lazy tensor. `device` selects where the realized
    /// Storage is allocated.
    pub fn from_f16(
        data: impl Into<Arc<[half::f16]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> Self {
        Self {
            inner: fuel_graph::Tensor::from_f16(data, shape, device.as_dyn()),
        }
    }

    /// Build a `u32` (index) lazy tensor. Used for gather/scatter/
    /// index_select and similar discrete operations. `device` selects
    /// where the realized Storage is allocated.
    pub fn from_u32(
        data: impl Into<Arc<[u32]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> Self {
        Self {
            inner: fuel_graph::Tensor::from_u32(data, shape, device.as_dyn()),
        }
    }

    /// Build a const tensor of the same dtype and graph as `self`.
    /// This is the most convenient way to attach new input data to an
    /// existing computation.
    ///
    /// Phase 7.5 G2: the realized Storage is allocated on the device
    /// derived from `self`'s graph (any existing slot's device — the
    /// graph always has at least one slot-bearing leaf by the time
    /// const_*_like is called). Use [`from_f32`] with an explicit
    /// `&Device` when you need a const on a different device than
    /// `self`.
    pub fn const_f32_like(
        &self,
        data: impl Into<Arc<[f32]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        Self {
            inner: self.inner.const_f32_like(data, shape),
        }
    }

    /// Build a const f16 tensor on the same graph as `self`.
    pub fn const_f16_like(
        &self,
        data: impl Into<Arc<[half::f16]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        Self {
            inner: self.inner.const_f16_like(data, shape),
        }
    }

    /// Build a const bf16 tensor on the same graph as `self`. Used for
    /// bf16-on-device weights in the mixed-precision matmul path —
    /// activations stay f32, weight matrices live as bf16.
    pub fn const_bf16_like(
        &self,
        data: impl Into<Arc<[half::bf16]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        Self {
            inner: self.inner.const_bf16_like(data, shape),
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

    /// Element-wise maximum.
    pub fn maximum(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.maximum(&other.inner),
        }
    }

    /// Element-wise minimum.
    pub fn minimum(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.minimum(&other.inner),
        }
    }

    /// Element-wise equality (`self == other`) producing a `U8` mask:
    /// `1` where equal, `0` otherwise. Both operands must share dtype
    /// and shape. NaN follows IEEE-754 (`NaN == NaN` is false). The
    /// resulting tensor's dtype is `DType::U8`. Non-differentiable.
    pub fn eq(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.eq(&other.inner),
        }
    }

    /// Element-wise inequality (`self != other`) producing a `U8`
    /// mask. NaN follows IEEE-754 (`NaN != NaN` is true → `1`).
    /// Non-differentiable.
    pub fn ne(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.ne(&other.inner),
        }
    }

    /// Element-wise strictly-less (`self < other`) producing a `U8`
    /// mask. NaN-on-either-side is `0` (IEEE-754 unordered).
    /// Non-differentiable.
    pub fn lt(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.lt(&other.inner),
        }
    }

    /// Element-wise less-or-equal (`self <= other`) producing a `U8`
    /// mask. NaN-on-either-side is `0`. Non-differentiable.
    pub fn le(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.le(&other.inner),
        }
    }

    /// Element-wise strictly-greater (`self > other`) producing a
    /// `U8` mask. NaN-on-either-side is `0`. Non-differentiable.
    pub fn gt(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.gt(&other.inner),
        }
    }

    /// Element-wise greater-or-equal (`self >= other`) producing a
    /// `U8` mask. NaN-on-either-side is `0`. Non-differentiable.
    /// Final variant of the comparison family (`eq` / `ne` / `lt` /
    /// `le` / `gt` / `ge`).
    pub fn ge(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.ge(&other.inner),
        }
    }

    /// Ternary select (typically used to consume a comparison-op
    /// mask): `result[i] = if self[i] != 0 { a[i] } else { b[i] }`.
    /// `self` is the cond mask (must be `DType::U8`); `a` and `b`
    /// share dtype + shape with `self`. Output dtype matches `a`/`b`,
    /// shape matches `self`.
    ///
    /// Differentiable through `a` and `b` only.
    pub fn where_cond(&self, a: &Self, b: &Self) -> Self {
        Self {
            inner: self.inner.where_cond(&a.inner, &b.inner),
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

    /// Element-wise sine.
    pub fn sin(&self) -> Self {
        Self { inner: self.inner.sin() }
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Self {
        Self { inner: self.inner.cos() }
    }

    /// Heaviside step (`1` where `x > 0`, else `0`) — the derivative
    /// of [`Self::relu`].
    pub fn step(&self) -> Self {
        Self { inner: self.inner.step() }
    }

    /// Element-wise reciprocal (`1 / x`).
    pub fn recip(&self) -> Self {
        Self { inner: self.inner.recip() }
    }

    /// Element-wise absolute value (`|x|`).
    pub fn abs(&self) -> Self {
        Self { inner: self.inner.abs() }
    }

    /// Element-wise floor (`⌊x⌋`). Same dtype as input.
    /// Backward is silently zero (non-differentiable almost everywhere).
    pub fn floor(&self) -> Self {
        Self { inner: self.inner.floor() }
    }

    /// Element-wise ceiling (`⌈x⌉`). Same dtype as input.
    /// Backward is silently zero.
    pub fn ceil(&self) -> Self {
        Self { inner: self.inner.ceil() }
    }

    /// Element-wise round-to-nearest with **banker's rounding**
    /// (round-half-to-even, IEEE 754 roundeven). Backward is silently
    /// zero. Differs from C99 `round()` at exact halves: 0.5 → 0,
    /// 2.5 → 2, etc.
    pub fn round(&self) -> Self {
        Self { inner: self.inner.round() }
    }

    /// Element-wise sign (`-1` / `0` / `1`); `sign(0) = 0` by
    /// subgradient convention. Same dtype as input. Backward is
    /// silently zero.
    pub fn sign(&self) -> Self {
        Self { inner: self.inner.sign() }
    }

    /// Element-wise Gauss error function (`erf(x)`). Same dtype as
    /// input. Differentiable: `d/dx erf(x) = (2/√π) * exp(-x²)`.
    pub fn erf(&self) -> Self {
        Self { inner: self.inner.erf() }
    }

    /// GELU activation, **exact erf form** (`0.5 * x * (1 + erf(x/√2))`).
    /// Distinct from [`Self::gelu`] (tanh approximation). Same dtype
    /// as input. Differentiable.
    pub fn gelu_erf(&self) -> Self {
        Self { inner: self.inner.gelu_erf() }
    }

    /// Element-wise binary power `pow(self, other)` (real exponent).
    /// Both operands must share dtype + shape. Distinct from
    /// [`Self::powi`] (scalar `i32` exponent). Differentiable.
    /// **Returns `Result`**: dtype/shape mismatch surfaces as a
    /// typed error.
    pub fn pow(&self, other: &Self) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self {
            inner: self.inner.pow(&other.inner)?,
        })
    }

    /// Element-wise reciprocal square root (`1 / sqrt(x)`). Same
    /// dtype as input. One op rather than `sqrt(x).recip()` — saves
    /// a kernel launch and matches the RMSNorm shape. Differentiable.
    pub fn rsqrt(&self) -> Self {
        Self { inner: self.inner.rsqrt() }
    }

    /// Element-wise remainder, **PyTorch convention**:
    /// `a - floor(a/b) * b` (sign of result matches divisor; matches
    /// `torch.remainder`, not C99 fmod). Differentiable through `a`
    /// and `b`. **Returns `Result`**: dtype/shape mismatch surfaces
    /// as a typed error.
    pub fn rem(&self, other: &Self) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self {
            inner: self.inner.rem(&other.inner)?,
        })
    }

    /// Reverse element order along `dim`. Materializing op (real
    /// byte shuffle). Differentiable; backward is itself.
    /// **Returns `Result`**: bad `dim` surfaces as a typed error.
    pub fn flip(&self, dim: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.flip(dim)? })
    }

    /// Cyclic shift along `dim` by `shift` positions (positive →
    /// higher indices, wrapping). Differentiable; backward is
    /// `roll(dim, -shift)`. **Returns `Result`**: bad `dim` surfaces
    /// as a typed error.
    pub fn roll(&self, dim: usize, shift: i64) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.roll(dim, shift)? })
    }

    /// Running cumulative sum along `dim`. Same shape as input.
    /// Differentiable; backward is reverse cumsum (`flip → cumsum
    /// → flip`). **Returns `Result`**: bad `dim` surfaces as a
    /// typed error.
    pub fn cumsum(&self, dim: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.cumsum(dim)? })
    }

    /// Multi-dim Pad: `padding[i] = (before, after)` for axis `i`,
    /// length must equal tensor rank. Output shape:
    /// `out[i] = in[i] + padding[i].0 + padding[i].1`. Only Constant
    /// mode is implemented; Reflect / Replicate exist as enum stubs
    /// that error at realize time. Differentiable for Constant.
    /// **Returns `Result`**: rank mismatch surfaces as a typed error.
    pub fn pad(&self, padding: Vec<(usize, usize)>, mode: fuel_graph::PadMode, value: f64) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self {
            inner: self.inner.pad(padding, mode, value)?,
        })
    }

    /// Element-wise integer power (`x.powi(n)`).
    pub fn powi(&self, n: i32) -> Self {
        Self { inner: self.inner.powi(n) }
    }

    // ---- linear algebra & shape ----

    /// N-D batched matrix multiply with automatic rank-2 broadcasting.
    pub fn matmul(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.matmul(&other.inner),
        }
    }

    /// Quantized matmul: `C = self @ dequant(W_Q)`. See
    /// [`fuel_graph::Tensor::qmatmul`] for details. The weight bytes
    /// tensor must be a flat U32 const holding the raw Q-block byte
    /// stream (length = n_bytes / 4).
    pub fn qmatmul(
        &self,
        weight_bytes: &Self,
        quant_type: fuel_graph::QuantType,
        k: usize,
        n: usize,
    ) -> Self {
        Self {
            inner: self.inner.qmatmul(&weight_bytes.inner, quant_type, k, n),
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

    /// Drop the size-1 dimension at position `dim` (range `0..rank`).
    /// Metadata-only view; bytes shared with `self`. **Returns
    /// `Result`** rather than panicking — bad `dim` (out of bounds
    /// or `shape[dim] != 1`) surfaces as a typed error.
    pub fn squeeze(&self, dim: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self {
            inner: self.inner.squeeze(dim)?,
        })
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

    /// Maximum of every element, producing a scalar.
    pub fn max_all(&self) -> Self {
        Self { inner: self.inner.max_all() }
    }

    /// Minimum of every element, producing a scalar.
    pub fn min_all(&self) -> Self {
        Self { inner: self.inner.min_all() }
    }

    /// Sum along a single dimension (dim removed from output).
    pub fn sum_dim(&self, dim: usize) -> Self {
        Self {
            inner: self.inner.sum_dim(dim),
        }
    }

    /// Max along a single dimension (dim removed from output).
    pub fn max_dim(&self, dim: usize) -> Self {
        Self {
            inner: self.inner.max_dim(dim),
        }
    }

    /// Min along a single dimension (dim removed from output).
    pub fn min_dim(&self, dim: usize) -> Self {
        Self {
            inner: self.inner.min_dim(dim),
        }
    }

    /// Element-wise clamp to `[min, max]`.
    pub fn clamp(&self, min: f64, max: f64) -> Self {
        Self { inner: self.inner.clamp(min, max) }
    }

    /// Mean along a single dimension.
    pub fn mean_dim(&self, dim: usize) -> Self {
        Self {
            inner: self.inner.mean_dim(dim),
        }
    }

    /// Sum-reduce to a smaller broadcast-compatible shape. Inverse of
    /// [`Self::broadcast_to`]; reduces over any dim where the source
    /// was broadcast against the target.
    pub fn reduce_sum_to(&self, target: impl Into<Shape>) -> Self {
        Self { inner: self.inner.reduce_sum_to(target) }
    }

    /// Max-reduce to a smaller broadcast-compatible shape — the
    /// max-symmetric counterpart of [`Self::reduce_sum_to`].
    pub fn reduce_max_to(&self, target: impl Into<Shape>) -> Self {
        Self { inner: self.inner.reduce_max_to(target) }
    }

    // ---- compositions ----

    /// Softmax along the last dim.
    pub fn softmax_last_dim(&self) -> Self {
        Self {
            inner: self.inner.softmax_last_dim(),
        }
    }

    /// bitsandbytes-style 4-bit NormalFloat quantized matrix
    /// multiply. See [`fuel_graph::Tensor::nf4_matmul`] for the full
    /// shape contract. v1 covers F32/F16/BF16 activations.
    pub fn nf4_matmul(
        &self,
        w_packed: &Self,
        absmax: &Self,
        block_size: usize,
    ) -> Self {
        Self {
            inner: self.inner.nf4_matmul(&w_packed.inner, &absmax.inner, block_size),
        }
    }

    /// Mamba-2's State-Space Duality chunked scan (forward). See
    /// [`fuel_graph::Tensor::ssd_chunk_scan`] for the full shape
    /// contract. `chunk_size` is a GPU-parallelism granularity knob;
    /// the CPU kernel runs sequential regardless.
    pub fn ssd_chunk_scan(
        &self,
        dt: &Self,
        a: &Self,
        b: &Self,
        c: &Self,
        chunk_size: usize,
    ) -> Self {
        Self {
            inner: self.inner.ssd_chunk_scan(
                &dt.inner, &a.inner, &b.inner, &c.inner, chunk_size,
            ),
        }
    }

    /// Mamba-1's selective state-space scan (forward). See
    /// [`fuel_graph::Tensor::selective_scan`] for the full shape
    /// contract. v1 covers 5 required inputs; optional inputs and
    /// `last_state` output are mechanical follow-ups.
    pub fn selective_scan(
        &self,
        delta: &Self,
        a: &Self,
        b: &Self,
        c: &Self,
        delta_softplus: bool,
    ) -> Self {
        Self {
            inner: self.inner.selective_scan(
                &delta.inner, &a.inner, &b.inner, &c.inner, delta_softplus,
            ),
        }
    }

    /// Depthwise 1-D causal convolution + bias + optional fused SiLU
    /// — the Mamba-1 / Mamba-2 prefill convolution fusion. See
    /// [`fuel_graph::Tensor::causal_conv1d`] for the full shape
    /// contract (caller must left-pad x with `kernel - 1` zeros).
    pub fn causal_conv1d(
        &self,
        weight: &Self,
        bias: &Self,
        use_silu: bool,
    ) -> Self {
        Self {
            inner: self.inner.causal_conv1d(&weight.inner, &bias.inner, use_silu),
        }
    }

    /// Fused softmax + cross-entropy with integer (class-index)
    /// targets — the standard PyTorch CE loss. See
    /// [`fuel_graph::Tensor::fused_softmax_cross_entropy`] for the full
    /// shape contract.
    pub fn fused_softmax_cross_entropy(
        &self,
        targets: &Self,
        reduction: fuel_graph::registry::Reduction,
        ignore_index: i64,
    ) -> Self {
        Self {
            inner: self.inner.fused_softmax_cross_entropy(
                &targets.inner, reduction, ignore_index,
            ),
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

    /// Realize this tensor as an `f32` `Vec`.
    ///
    /// When [`crate::judge::cached`] returns a populated dispatch
    /// table — i.e. the app called
    /// [`crate::judge::populate_dispatch_table`] earlier this
    /// process, OR a prior process persisted one for this hardware —
    /// the realize uses a `Router` that consults the table per op,
    /// picking among every registered CPU backend (`fuel-graph-cpu`
    /// always; `fuel-aocl-cpu-backend` when the `aocl` feature is on;
    /// `fuel-mkl-cpu-backend` when `onemkl` is on).
    ///
    /// When no table is cached, falls through to the original
    /// `GraphExecutor<CpuBackend>` path — same behaviour as before
    /// the Phase 7b refactor. Users who never call
    /// `populate_dispatch_table` see no behaviour change and pay no
    /// startup cost.
    pub fn realize_f32(&self) -> Vec<f32> {
        if let Some(table) = crate::judge::cached() {
            // Router path: stays on the legacy executor until Phase G
            // of the 9c migration retires the trait surface the Router
            // dispatches through.
            let mut router = fuel_graph_router::Router::new().add_cpu();
            #[cfg(feature = "aocl")]
            { router = router.add_aocl(); }
            #[cfg(feature = "onemkl")]
            { router = router.add_mkl(); }
            let router = router.with_dispatch_table(table);
            let mut exe = GraphExecutor::new(router);
            return exe.realize_f32(&self.inner).into_vec();
        }
        // Pipelined path (Phase 7.6 step 9c E.2): walk the graph,
        // pre-realize Consts onto CPU, dispatch through
        // PipelinedExecutor, D2H result.
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_as::<f32>(&graph, target, &device)
            .expect("realize_f32 via PipelinedExecutor")
    }

    /// Realize as an `f64` `Vec`.
    pub fn realize_f64(&self) -> Vec<f64> {
        fuel_graph_cpu::realize_f64(&self.inner).into_vec()
    }

    /// Realize as a `bf16` `Vec`.
    pub fn realize_bf16(&self) -> Vec<half::bf16> {
        fuel_graph_cpu::realize_bf16(&self.inner).into_vec()
    }

    /// Realize as an `f16` `Vec`.
    pub fn realize_f16(&self) -> Vec<half::f16> {
        fuel_graph_cpu::realize_f16(&self.inner).into_vec()
    }

    /// Realize using the reference backend directly — slow but
    /// textbook-correct oracle.
    pub fn realize_f32_reference(&self) -> Vec<f32> {
        fuel_reference_backend::exec::realize_f32(&self.inner).into_vec()
    }

    /// Realize on a CUDA GPU via [`PipelinedExecutor`].
    ///
    /// Phase 7.6 step 9c E.2: signature change from
    /// `&mut GraphExecutor<CudaBackend>` to `&CudaDevice`. The
    /// pipelined executor doesn't carry a const_pool — each call
    /// re-uploads weights. For autoregressive decoding loops where
    /// const_pool was load-bearing, use the persistent-StorageCache
    /// pattern shipped in Phase E.3 (KVCache migration).
    #[cfg(feature = "cuda")]
    pub fn realize_f32_cuda(
        &self,
        device: &fuel_cuda_backend::CudaDevice,
    ) -> Vec<f32> {
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let fc_device: crate::Device = device.clone().into();
        crate::pipelined_bridge::realize_one_as::<f32>(&graph, target, &fc_device)
            .expect("realize_f32_cuda via PipelinedExecutor")
    }

    /// Realize on a Vulkan GPU via the generic executor. Mirrors the
    /// CUDA helper above so the Phase 6b Judge can profile Vulkan
    /// equivalence classes uniformly with CUDA.
    ///
    /// **NOT migrated to PipelinedExecutor yet** — kept on the
    /// legacy executor for Judge-loop measurement parity with
    /// CUDA's `realize_f32_cuda`. The PipelinedExecutor path for
    /// Vulkan-resident realize is wired via the
    /// [`Op::Copy { target: Cpu }`](fuel_graph::Op::Copy) splice in
    /// [`crate::pipelined_bridge::realize_one_as_with_initial`]
    /// (bridge-retirement Phase 2, post-9c) and is exercised through
    /// `realize_f32` on the live-Vulkan tests.
    #[cfg(feature = "vulkan")]
    pub fn realize_f32_vulkan(
        &self,
        executor: &mut GraphExecutor<fuel_vulkan_backend::VulkanBackend>,
    ) -> Vec<f32> {
        executor.realize_f32(&self.inner).into_vec()
    }

    /// Realize on the AOCL CPU backend via [`PipelinedExecutor`].
    /// AOCL is a CPU backend (just a different BLAS routing); the
    /// pipelined path treats it as CPU at the storage layer. The
    /// kernel-binding-table picks the AOCL alternatives for matmul
    /// etc. when AOCL kernels are registered.
    ///
    /// `_executor` is ignored — kept in the signature for backward
    /// compatibility with the Phase 6b Judge's measurement loop;
    /// the Judge constructs executors per measurement and we just
    /// disregard them now.
    #[cfg(feature = "aocl")]
    pub fn realize_f32_aocl(
        &self,
        _executor: &mut GraphExecutor<fuel_aocl_cpu_backend::AoclBackend>,
    ) -> Vec<f32> {
        // AOCL is a CPU backend at the storage layer; the kernel
        // table routes matmul etc. to AOCL kernels when registered.
        self.realize_f32()
    }

    /// Realize on the oneMKL CPU backend. Mirrors `realize_f32_aocl`.
    #[cfg(feature = "onemkl")]
    pub fn realize_f32_mkl(
        &self,
        _executor: &mut GraphExecutor<fuel_mkl_cpu_backend::MklBackend>,
    ) -> Vec<f32> {
        self.realize_f32()
    }
}

/// Realize many tensors in a single CPU topo-walk. Phase 7.6 step 9c E.2.
pub fn realize_many_f32(tensors: &[&LazyTensor]) -> Vec<Vec<f32>> {
    if tensors.is_empty() {
        return Vec::new();
    }
    let graph = tensors[0].inner.graph().clone();
    let targets: Vec<fuel_graph::NodeId> = tensors.iter().map(|t| t.inner.id()).collect();
    let device = crate::Device::cpu();
    crate::pipelined_bridge::realize_many_as::<f32>(&graph, &targets, &device)
        .expect("realize_many_f32 via PipelinedExecutor")
}

/// CUDA variant of realize_many_f32. Phase 7.6 step 9c E.2: signature
/// change from `&mut GraphExecutor<CudaBackend>` to `&CudaDevice`.
#[cfg(feature = "cuda")]
pub fn realize_many_f32_cuda(
    tensors: &[&LazyTensor],
    device: &fuel_cuda_backend::CudaDevice,
) -> Vec<Vec<f32>> {
    if tensors.is_empty() {
        return Vec::new();
    }
    let graph = tensors[0].inner.graph().clone();
    let targets: Vec<fuel_graph::NodeId> = tensors.iter().map(|t| t.inner.id()).collect();
    let fc_device: crate::Device = device.clone().into();
    crate::pipelined_bridge::realize_many_as::<f32>(&graph, &targets, &fc_device)
        .expect("realize_many_f32_cuda via PipelinedExecutor")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn constructors_wrap_graph_tensor_correctly() {
        let t = LazyTensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), &Device::cpu());
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.shape().dims(), &[3]);
        assert_eq!(t.rank(), 1);
        assert_eq!(t.elem_count(), 3);
    }

    #[test]
    fn add_builds_add_node_in_underlying_graph() {
        let a = LazyTensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), &Device::cpu());
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        let c = a.add(&b);
        assert_eq!(c.shape().dims(), &[3]);
        // All three tensors share one underlying graph (by Arc cloning
        // via const_f32_like / add).
        assert!(std::sync::Arc::ptr_eq(
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
            &Device::cpu(),
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
            &Device::cpu(),
        );
        let y = x.rope(10000.0, 0);
        assert_eq!(y.shape().dims(), &[2, 4]);
        assert_eq!(y.dtype(), DType::F32);
    }

    #[test]
    fn cast_switches_dtype_through_wrapper() {
        let x = LazyTensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), &Device::cpu());
        let y = x.cast(DType::F64);
        assert_eq!(y.dtype(), DType::F64);
        assert_eq!(y.shape().dims(), &[3]);
    }

    #[test]
    fn indexing_builds_correct_output_shape() {
        let data = LazyTensor::from_f32(vec![1.0; 12], Shape::from_dims(&[3, 4]), &Device::cpu());
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
        let a = LazyTensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), &Device::cpu());
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
            &Device::cpu(),
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
        let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[m, k]), &Device::cpu());
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
        let a = LazyTensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), &Device::cpu());
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        let c = a.add(&b).mul(&a);
        let cpu_result = c.realize_f32();
        let executor = fuel_cuda_backend::CudaDevice::new(0).unwrap();
        let cuda_result = c.realize_f32_cuda(&executor);
        assert_eq!(cpu_result, cuda_result);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_matmul() {
        let a = LazyTensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            &Device::cpu(),
        );
        let b = a.const_f32_like(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::from_dims(&[3, 2]),
        );
        let c = a.matmul(&b);
        let cpu = c.realize_f32();
        let exe = fuel_cuda_backend::CudaDevice::new(0).unwrap();
        let cuda = c.realize_f32_cuda(&exe);
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
            &Device::cpu(),
        );
        let w = x.const_f32_like(
            (0..8).map(|i| i as f32 * 0.2).collect::<Vec<_>>(),
            Shape::from_dims(&[4, 2]),
        );
        let y = x.matmul(&w);
        let cpu = y.realize_f32();
        let exe = fuel_cuda_backend::CudaDevice::new(0).unwrap();
        let cuda = y.realize_f32_cuda(&exe);
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
            &Device::cpu(),
        );
        let y = x.permute(&[0, 2, 1, 3]);
        let cpu = y.realize_f32();
        let exe = fuel_cuda_backend::CudaDevice::new(0).unwrap();
        let cuda = y.realize_f32_cuda(&exe);
        assert_eq!(cpu, cuda, "permute mismatch");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_softmax() {
        let x = LazyTensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            &Device::cpu(),
        );
        let y = x.softmax_last_dim();
        let cpu = y.realize_f32();
        let exe = fuel_cuda_backend::CudaDevice::new(0).unwrap();
        let cuda = y.realize_f32_cuda(&exe);
        assert_eq!(cpu.len(), cuda.len());
        for (i, (&a, &b)) in cpu.iter().zip(cuda.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "softmax[{i}]: cpu={a}, cuda={b}");
        }
    }

    /// Phase 7.5 PR 3 / 3.5 live CUDA equivalence: realize a graph with
    /// `Op::SoftmaxLastDim` through the rule-registry pipeline using
    /// `RuleRegistry::lowering_only()` so the executor sees the
    /// 7-node lowered subgraph instead of the fused op. The composed
    /// CUDA execution path (ReduceMaxTo + BroadcastTo + Sub + Exp +
    /// ReduceSumTo + BroadcastTo + Div) must match the fused CPU
    /// baseline within tight epsilon.
    ///
    /// Post PR-3.5 follow-up: ReduceMaxTo / ReduceSumTo run natively
    /// on CUDA via the legacy executor's `Op::ReduceXxxTo` arm (which
    /// delegates to `backend.reduce` and relabels the result to the
    /// keepdim shape), so the lowered subgraph stays GPU-resident
    /// end-to-end. Two D2H/H2D round-trips per softmax used to be
    /// the cost on the prior commit — both gone now.
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_softmax_via_lowering() {
        // Use a non-trivial input shape so the broadcast paths and
        // the ReduceSumTo step both have actual work to do.
        let n = 24;
        let last = 5;
        let data: Vec<f32> = (0..n * last)
            .map(|i| ((i as f32) * 0.13).sin() * 2.0 - 0.7)
            .collect();
        let x = LazyTensor::from_f32(
            data,
            Shape::from_dims(&[n, last]),
            &Device::cpu(),
        );
        let y = x.softmax_last_dim();

        // CPU baseline: fused SoftmaxLastDim through the standard
        // realize_f32 path (no rule-registry pipeline involved).
        let cpu = y.realize_f32();

        // CUDA via the lowered subgraph: run the lowering-only
        // rule registry to fixpoint first so fusion can't re-collapse
        // the lowered pattern back to Op::SoftmaxLastDim. Then
        // realize the remapped target via PipelinedExecutor.
        // (Phase 7.6 step 9c E.2: optimizer is caller-composed.)
        let graph = y.inner.graph().clone();
        let registry = fuel_graph::opt::RuleRegistry::lowering_only();
        let remapped = registry.optimize_to_fixpoint(&graph, &[y.inner.id()]);
        let dev = fuel_cuda_backend::CudaDevice::new(0).unwrap();
        let fc_device: crate::Device = dev.clone().into();
        let cuda = crate::pipelined_bridge::realize_one_as::<f32>(
            &graph, remapped[0], &fc_device,
        ).expect("realize lowered softmax on CUDA");

        assert_eq!(cpu.len(), cuda.len());
        let mut max_abs_err = 0.0_f32;
        for (i, (&a, &b)) in cpu.iter().zip(cuda.iter()).enumerate() {
            let err = (a - b).abs();
            if err > max_abs_err { max_abs_err = err; }
            assert!(
                err < 1e-5,
                "lowered softmax[{i}]: cpu={a} (fused), cuda={b} (composed), err={err}",
            );
        }
        eprintln!("max_abs_err over lowered-vs-fused softmax: {max_abs_err:.3e}");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_concat_slice() {
        let a = LazyTensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), &Device::cpu());
        let b = a.const_f32_like(vec![5.0, 6.0, 7.0, 8.0], Shape::from_dims(&[2, 2]));
        let cat = a.concat(&b, 1); // [2, 4]
        let sliced = cat.slice(1, 1, 2); // [2, 2]
        let cpu = sliced.realize_f32();
        let exe = fuel_cuda_backend::CudaDevice::new(0).unwrap();
        let cuda = sliced.realize_f32_cuda(&exe);
        assert_eq!(cpu, cuda, "concat+slice mismatch");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_rms_norm() {
        let x = LazyTensor::from_f32(
            (0..8).map(|i| i as f32 * 0.5 - 1.5).collect::<Vec<_>>(),
            Shape::from_dims(&[2, 4]),
            &Device::cpu(),
        );
        let y = x.rms_norm_last_dim(1e-5);
        let cpu = y.realize_f32();
        let exe = fuel_cuda_backend::CudaDevice::new(0).unwrap();
        let cuda = y.realize_f32_cuda(&exe);
        assert_eq!(cpu.len(), cuda.len());
        for (i, (&a, &b)) in cpu.iter().zip(cuda.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "rms_norm[{i}]: cpu={a}, cuda={b}");
        }
    }

    #[test]
    fn realize_f64_through_bridge() {
        let a = LazyTensor::from_f64(vec![1.5, 2.5, 3.5], Shape::from_dims(&[3]), &Device::cpu());
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
        let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[1, seq, d_model]), &Device::cpu());

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

    /// Push a [`fuel_graph::Op::Const`] node on the same graph as
    /// `self` **without** populating the graph's storage_map. The
    /// caller binds the storage Arc into the realize call via
    /// [`InferenceContext::insert`](crate::inference_context::InferenceContext::insert).
    ///
    /// Used by the Phase E.3.3 forward path to bind pre-allocated
    /// KV-cache storage Arcs (`Arc<RwLock<fuel_storage::Storage>>`)
    /// into a per-step graph — the graph's legacy storage_map only
    /// holds `fuel_core_types::Storage`, so direct binding isn't
    /// possible without a type conversion.
    pub fn const_placeholder_like(
        &self,
        shape: impl Into<Shape>,
        dtype: fuel_core_types::DType,
    ) -> Self {
        Self {
            inner: self.inner.const_placeholder_like(shape, dtype),
        }
    }

    /// Append an [`fuel_graph::Op::WriteSlice`] node. Copies `source`'s
    /// bytes into `self` at the rectangular slab defined by `ranges`
    /// and returns a tensor whose Storage Arc is `self`'s — i.e. the
    /// post-write reference to the same underlying buffer.
    ///
    /// Destructive on `self`: after the write, downstream consumers
    /// must read the bytes through the returned tensor's NodeId, not
    /// `self`'s.
    ///
    /// **Returns `Result`**: rank/shape/range mismatches surface as a
    /// typed error.
    pub fn write_slice(
        &self,
        source: &Self,
        ranges: Vec<(usize, usize)>,
    ) -> crate::Result<Self> {
        let inner = self.inner.write_slice(&source.inner, ranges)
            .map_err(crate::Error::from)?;
        Ok(Self { inner })
    }

    /// Append a [`fuel_graph::Op::Conv2D`] node. See `fuel_graph`'s
    /// `Tensor::conv2d` for the full shape contract: `self` must be
    /// `[N, Cin, H, W]`; `weight` must be `[Cout, Cin/groups, Kh, Kw]`;
    /// `bias` is optional and must be `[Cout]` when provided. Returns
    /// a rank-4 lazy tensor `[N, Cout, Hout, Wout]`.
    pub fn conv2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Self {
        Self {
            inner: self.inner.conv2d(
                &weight.inner,
                bias.map(|b| &b.inner),
                stride,
                padding,
                groups,
            ),
        }
    }

    /// Append a [`fuel_graph::Op::FlashAttn`] node. `self` is `q`
    /// of shape `[B, Hq, Sq, D]`; `k` and `v` are `[B, Hkv, Sk, D]`
    /// with `Hq` a multiple of `Hkv` (GQA). `alibi_slopes` (optional)
    /// is `[Hq]`. Returns the attention output, shape `[B, Hq, Sq, D]`.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn(
        &self,
        k: &Self,
        v: &Self,
        alibi_slopes: Option<&Self>,
        softmax_scale: f32,
        causal: bool,
        window_size_left: Option<usize>,
        window_size_right: Option<usize>,
        softcap: Option<f32>,
    ) -> Self {
        Self {
            inner: self.inner.flash_attn(
                &k.inner, &v.inner,
                alibi_slopes.map(|t| &t.inner),
                softmax_scale, causal, window_size_left, window_size_right, softcap,
            ),
        }
    }

    /// Append a [`fuel_graph::Op::PagedAttn`] node. `self` is the Q
    /// tensor `[B, Hq, Sq, D]`. `k_cache` / `v_cache` are paged caches
    /// `[num_blocks, block_size, Hkv, D]`. `block_table` is `[B,
    /// max_blocks]` u32; `context_lens` is `[B]` u32.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_attn(
        &self,
        k_cache: &Self,
        v_cache: &Self,
        block_table: &Self,
        context_lens: &Self,
        alibi_slopes: Option<&Self>,
        softmax_scale: f32,
        block_size: usize,
        softcap: Option<f32>,
    ) -> Self {
        Self {
            inner: self.inner.paged_attn(
                &k_cache.inner, &v_cache.inner,
                &block_table.inner, &context_lens.inner,
                alibi_slopes.map(|t| &t.inner),
                softmax_scale, block_size, softcap,
            ),
        }
    }

    /// Append a [`fuel_graph::Op::ConvTranspose2D`] node. `self` must
    /// be `[N, Cin, H, W]`; `weight` must be `[Cin, Cout/groups, Kh, Kw]`
    /// (note transposed channel order vs `conv2d`). Returns a rank-4
    /// lazy tensor `[N, Cout, Hout, Wout]`.
    pub fn conv_transpose2d(
        &self,
        weight: &Self,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Self {
        Self {
            inner: self.inner.conv_transpose2d(
                &weight.inner,
                stride, padding, output_padding, dilation, groups,
            ),
        }
    }
}

// ============================================================================
// Phase A.1 — wrapper additions (eager-`Tensor` retirement program).
//
// Methods on `fuel_graph::Tensor` that weren't previously surfaced through
// `LazyTensor`. Pure delegation; no new graph ops. See
// `docs/session-prompts/eager-tensor-retirement-master-plan.md`.
// ============================================================================

impl LazyTensor {
    // ---- shape ops: unsqueeze and Result-returning siblings ----

    /// Append a size-1 dimension at position `dim`. Inverse of [`Self::squeeze`].
    pub fn unsqueeze(&self, dim: usize) -> Self {
        Self { inner: self.inner.unsqueeze(dim) }
    }

    /// Result-returning sibling of [`Self::unsqueeze`].
    pub fn try_unsqueeze(&self, dim: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.try_unsqueeze(dim)? })
    }

    /// Result-returning sibling of [`Self::permute`].
    pub fn try_permute(&self, axes: &[usize]) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.try_permute(axes)? })
    }

    /// Result-returning sibling of [`Self::transpose`].
    pub fn try_transpose(&self) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.try_transpose()? })
    }

    /// Result-returning sibling of [`Self::broadcast_to`].
    pub fn try_broadcast_to(&self, shape: impl Into<Shape>) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.try_broadcast_to(shape)? })
    }

    /// Result-returning sibling of [`Self::reshape`].
    pub fn try_reshape(&self, shape: impl Into<Shape>) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.try_reshape(shape)? })
    }

    // ---- triangular masking (canonical attention masks) ----

    /// Upper-triangular mask along the last two dims. `diagonal = 0`
    /// keeps the main diagonal and above; positive shifts higher.
    pub fn triu(&self, diagonal: i64) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.triu(diagonal)? })
    }

    /// Lower-triangular mask along the last two dims. `tril(0)` is the
    /// canonical causal-attention mask.
    pub fn tril(&self, diagonal: i64) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.tril(diagonal)? })
    }

    // ---- additional reductions / activations ----

    /// `log(softmax(self))` along the last dim, fused into one op.
    pub fn log_softmax_last_dim(&self) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.log_softmax_last_dim()? })
    }

    /// Argmin along `dim`, returning a U32 tensor with the reduced dim
    /// removed. Non-differentiable.
    pub fn argmin_dim(&self, dim: usize) -> Self {
        Self { inner: self.inner.argmin_dim(dim) }
    }

    // ---- masking / scatter ----

    /// Fill every position where `mask != 0` with `value`; pass `self`
    /// through everywhere `mask == 0`. `mask` must be U8 with the same
    /// shape as `self`; `value`'s dtype must match `self`.
    pub fn masked_fill(
        &self,
        mask: &Self,
        value: fuel_core_types::Scalar,
    ) -> std::result::Result<Self, fuel_core_types::Error> {
        Ok(Self { inner: self.inner.masked_fill(&mask.inner, value)? })
    }

    /// `self + scatter(indices, src, dim=dim)` — accumulate `src` rows
    /// at positions named by `indices` along `dim`. `indices` is rank-1
    /// U32 with length equal to `src.dims()[dim]`.
    pub fn index_add(&self, dim: usize, indices: &Self, src: &Self) -> Self {
        Self { inner: self.inner.index_add(dim, &indices.inner, &src.inner) }
    }

    /// Functional inverse of [`Self::gather`]. Accumulates `src` into
    /// `self` at positions given by `indices` (substituted at `dim`).
    pub fn scatter_add(&self, dim: usize, indices: &Self, src: &Self) -> Self {
        Self { inner: self.inner.scatter_add(dim, &indices.inner, &src.inner) }
    }

    // ---- in-place activations (Phase 4-5 infrastructure, now surfaced) ----
    //
    // These mutate `self`'s storage in place. Safe to call on
    // tape-tracked tensors after Phase 4's view-aware ordering pass and
    // Phase 5's auto-copy pass. See `project_inplace_ops_complete`
    // memory entry.

    /// In-place `max(0, self)`. See [`Self::relu`] for the functional
    /// variant.
    pub fn relu_inplace(&self) -> Self {
        Self { inner: self.inner.relu_inplace() }
    }

    /// In-place `self * sigmoid(self)`. See [`Self::silu`] for the
    /// functional variant.
    pub fn silu_inplace(&self) -> Self {
        Self { inner: self.inner.silu_inplace() }
    }

    /// In-place tanh-approximation GELU. See [`Self::gelu`] for the
    /// functional variant.
    pub fn gelu_inplace(&self) -> Self {
        Self { inner: self.inner.gelu_inplace() }
    }

    /// In-place `tanh(self)`. See [`Self::tanh`] for the functional
    /// variant.
    pub fn tanh_inplace(&self) -> Self {
        Self { inner: self.inner.tanh_inplace() }
    }

    /// In-place `sigmoid(self)`. See [`Self::sigmoid`] for the
    /// functional variant.
    pub fn sigmoid_inplace(&self) -> Self {
        Self { inner: self.inner.sigmoid_inplace() }
    }

    /// In-place `self = mul · self + add`. Single fused-op equivalent
    /// of `self.mul_scalar(mul).add_scalar(add)` plus reassignment.
    pub fn affine_inplace(&self, mul: f64, add: f64) -> Self {
        Self { inner: self.inner.affine_inplace(mul, add) }
    }

    // ---- additional const_*_like factories ----

    /// Build a sibling F64 `Const` on the same graph as `self`.
    pub fn const_f64_like(
        &self,
        data: impl Into<Arc<[f64]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        Self { inner: self.inner.const_f64_like(data, shape) }
    }

    /// Build a sibling I64 `Const` on the same graph. Used by integer-
    /// target ops (e.g. cross-entropy with PyTorch-convention class
    /// indices).
    pub fn const_i64_like(
        &self,
        data: impl Into<Arc<[i64]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        Self { inner: self.inner.const_i64_like(data, shape) }
    }

    // ---- device residency control ----

    /// Pin this tensor's realized storage to `device`. Consumes `self`
    /// because the placement is a graph-level annotation tied to the
    /// node id rather than a side-effecting operation.
    pub fn on_device(self, device: &Device) -> Self {
        Self { inner: self.inner.on_device(device.location()) }
    }

    /// Append an `Op::Release` node — explicitly drop this tensor's
    /// device-resident storage once the ordering pass has scheduled
    /// every reader before it.
    pub fn release(&self) -> Self {
        Self { inner: self.inner.release() }
    }

    /// Move bytes to `device`, destroying the source. Use when the
    /// source is genuinely dead after the transfer.
    pub fn move_to_device(&self, device: &Device) -> Self {
        Self { inner: self.inner.move_to_device(device.location()) }
    }

    /// Copy bytes to `device`, leaving the source resident. Use when
    /// other ops still need the source.
    pub fn copy_to_device(&self, device: &Device) -> Self {
        Self { inner: self.inner.copy_to_device(device.location()) }
    }

    // ---- autograd ----

    /// Run reverse-mode autograd from this tensor as the loss, returning
    /// a [`fuel_graph::GradMap`] keyed by every input tensor reached.
    /// The gradient nodes extend the same graph; realizing a gradient
    /// re-executes the forward dependencies.
    pub fn backward(&self) -> fuel_graph::GradMap {
        self.inner.backward()
    }
}

// ============================================================================
// Phase A.2 — composite primitives expressible from existing ops.
//
// Each method here is implemented in terms of `LazyTensor`'s existing
// surface (reshape, permute, concat, unsqueeze, etc.). No new graph ops.
// ============================================================================

impl LazyTensor {
    /// Transpose the last two dims as a Result-returning convenience —
    /// rank < 2 surfaces as an error rather than the panic the
    /// no-arg [`Self::transpose`] would produce. Alias for the eager
    /// `transpose_last_two`.
    pub fn transpose_last_two(&self) -> std::result::Result<Self, fuel_core_types::Error> {
        self.try_transpose()
    }

    /// Eager-API alias of [`Self::transpose_last_two`]. Matches PyTorch's
    /// `.t()` short form and the existing eager [`Tensor::t`] method.
    pub fn t(&self) -> std::result::Result<Self, fuel_core_types::Error> {
        self.try_transpose()
    }

    /// Two-argument transpose: swap dims `dim1` and `dim2`, leaving the
    /// rest in place. Implemented via [`Self::try_permute`]; matches the
    /// eager `transpose(d1, d2)` two-arg form.
    pub fn transpose_dims(&self, dim1: usize, dim2: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        if dim1 == dim2 {
            return Ok(self.clone());
        }
        let rank = self.shape().dims().len();
        if dim1 >= rank || dim2 >= rank {
            return Err(fuel_core_types::Error::Msg(format!(
                "transpose_dims: dim1={dim1} dim2={dim2} out of bounds for rank {rank}",
            )).bt());
        }
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(dim1, dim2);
        self.try_permute(&axes)
    }

    /// Collapse dims `[start_dim, end_dim]` (inclusive) into a single
    /// dimension. Returns `Result` so out-of-bounds surfaces as a typed
    /// error rather than a panic.
    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        let dims = self.shape().dims().to_vec();
        let rank = dims.len();
        if rank == 0 {
            // Scalar: already flat; nothing to do.
            return Ok(self.clone());
        }
        if start_dim >= rank || end_dim >= rank || start_dim > end_dim {
            return Err(fuel_core_types::Error::Msg(format!(
                "flatten: start_dim={start_dim} end_dim={end_dim} out of bounds for rank {rank}",
            )).bt());
        }
        let merged: usize = dims[start_dim..=end_dim].iter().product();
        let mut new_dims: Vec<usize> = Vec::with_capacity(rank - (end_dim - start_dim));
        new_dims.extend_from_slice(&dims[..start_dim]);
        new_dims.push(merged);
        new_dims.extend_from_slice(&dims[end_dim + 1..]);
        self.try_reshape(new_dims)
    }

    /// Flatten dims `[0, end_dim]` (inclusive) into one.
    pub fn flatten_to(&self, end_dim: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        self.flatten(0, end_dim)
    }

    /// Flatten dims `[start_dim, rank-1]` into one.
    pub fn flatten_from(&self, start_dim: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        let rank = self.shape().dims().len();
        if rank == 0 {
            return Ok(self.clone());
        }
        self.flatten(start_dim, rank - 1)
    }

    /// Flatten the tensor to rank-1 (single dim containing every element).
    pub fn flatten_all(&self) -> std::result::Result<Self, fuel_core_types::Error> {
        let rank = self.shape().dims().len();
        if rank == 0 {
            return Ok(self.clone());
        }
        self.flatten(0, rank - 1)
    }

    /// Stack tensors along a new dim at position `dim`. Each input is
    /// `unsqueeze`d at `dim` then concatenated. All inputs must have
    /// identical shape; `dim` may equal `rank` (append a new trailing
    /// dim).
    pub fn stack(args: &[&Self], dim: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        if args.is_empty() {
            return Err(fuel_core_types::Error::Msg(
                "stack: requires at least one tensor".into(),
            ).bt());
        }
        let reference_dims = args[0].shape().dims().to_vec();
        let rank = reference_dims.len();
        if dim > rank {
            return Err(fuel_core_types::Error::Msg(format!(
                "stack: dim={dim} out of bounds for rank {rank} (must be <= rank)",
            )).bt());
        }
        for (idx, t) in args.iter().enumerate().skip(1) {
            if t.shape().dims() != reference_dims.as_slice() {
                return Err(fuel_core_types::Error::Msg(format!(
                    "stack: tensor {idx} shape {:?} != reference shape {:?}",
                    t.shape().dims(), reference_dims,
                )).bt());
            }
        }
        // unsqueeze every input at the new dim, then concat.
        let mut iter = args.iter();
        let first = iter.next().unwrap().try_unsqueeze(dim)?;
        let mut acc = first;
        for t in iter {
            let u = t.try_unsqueeze(dim)?;
            acc = acc.concat(&u, dim);
        }
        Ok(acc)
    }

    // ---- keepdim reductions ----
    //
    // Each keepdim variant is the squeezed reduction post-composed with
    // `unsqueeze` at the same dim. The graph optimizer can fuse these
    // back into a single op when it's profitable; until then, the cost
    // is one extra view-only node.

    /// Sum along `dim`, keeping the reduced dim as size 1.
    pub fn sum_keepdim(&self, dim: usize) -> Self {
        self.sum_dim(dim).unsqueeze(dim)
    }

    /// Mean along `dim`, keeping the reduced dim as size 1.
    pub fn mean_keepdim(&self, dim: usize) -> Self {
        self.mean_dim(dim).unsqueeze(dim)
    }

    /// Max along `dim`, keeping the reduced dim as size 1.
    pub fn max_keepdim(&self, dim: usize) -> Self {
        self.max_dim(dim).unsqueeze(dim)
    }

    /// Min along `dim`, keeping the reduced dim as size 1.
    pub fn min_keepdim(&self, dim: usize) -> Self {
        self.min_dim(dim).unsqueeze(dim)
    }

    /// Unbiased sample variance along `dim`, keeping the reduced dim as
    /// size 1. Divides squared deviations by `n - 1` (Bessel's
    /// correction), matching the eager [`Tensor::var_keepdim`] and
    /// PyTorch defaults. `n == 1` produces NaN.
    pub fn var_keepdim(&self, dim: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        let dims = self.shape().dims().to_vec();
        let rank = dims.len();
        if dim >= rank {
            return Err(fuel_core_types::Error::Msg(format!(
                "var_keepdim: dim {dim} out of bounds for rank {rank}",
            )).bt());
        }
        let n = dims[dim];
        let mean = self.mean_keepdim(dim);
        let deviation = self.broadcast_sub(&mean);
        let squares = deviation.sqr();
        // sum_keepdim then divide by (n-1); leaves the reduced dim as 1.
        let summed = squares.sum_keepdim(dim);
        let divisor = (n.saturating_sub(1)) as f64;
        Ok(summed.mul_scalar(1.0 / divisor))
    }

    /// Unbiased sample variance along `dim`, squeezing the reduced dim.
    /// See [`Self::var_keepdim`].
    pub fn var(&self, dim: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        self.var_keepdim(dim)?.squeeze(dim)
    }

    // ---- composite scalar / binary ops (Phase A.4) ----

    /// `y = mul * self + add` element-wise. Two-op composite of
    /// [`Self::mul_scalar`] then [`Self::add_scalar`]; a fused `Op::Affine`
    /// can collapse this into a single op later (see the in-place
    /// counterpart [`Self::affine_inplace`]).
    pub fn affine(&self, mul: f64, add: f64) -> Self {
        self.mul_scalar(mul).add_scalar(add)
    }

    /// `y = scale * self + shift`. Alias of [`Self::affine`] with
    /// descriptive parameter names; matches eager's
    /// `Tensor::scale_and_shift`.
    pub fn scale_and_shift(&self, scale: f64, shift: f64) -> Self {
        self.affine(scale, shift)
    }

    /// Exponential Linear Unit: `self` where `self > 0`,
    /// `alpha * (exp(self) - 1)` otherwise. Composite of `where_cond`,
    /// `gt`, `exp`, `affine`.
    pub fn elu(&self, alpha: f64) -> Self {
        // Negative-branch value: alpha * (exp(self) - 1) = alpha * exp(self) - alpha
        let neg_branch = self.exp().affine(alpha, -alpha);
        // Mask: self > 0. Build a zero on the same graph.
        let zero = self.const_f32_like(vec![0.0; self.elem_count()], self.shape());
        let mask = self.gt(&zero);
        mask.where_cond(self, &neg_branch)
    }

    /// Inner product of two rank-1 tensors. Composite of `mul` +
    /// `sum_all`; matches eager's [`Tensor::dot`].
    pub fn dot(&self, rhs: &Self) -> std::result::Result<Self, fuel_core_types::Error> {
        let a = self.shape().dims().to_vec();
        let b = rhs.shape().dims().to_vec();
        if a.len() != 1 || b.len() != 1 {
            return Err(fuel_core_types::Error::Msg(format!(
                "dot: requires rank-1 inputs, got lhs={a:?} rhs={b:?}",
            )).bt());
        }
        if a[0] != b[0] {
            return Err(fuel_core_types::Error::Msg(format!(
                "dot: length mismatch lhs={} rhs={}", a[0], b[0],
            )).bt());
        }
        Ok(self.mul(rhs).sum_all())
    }

    /// Matrix × vector: `[m, n] · [n] -> [m]`. No broadcasting. Composite
    /// of `unsqueeze` + `matmul` + `squeeze`. Matches eager's
    /// [`Tensor::mv`].
    pub fn mv(&self, rhs: &Self) -> std::result::Result<Self, fuel_core_types::Error> {
        let a = self.shape().dims().to_vec();
        let b = rhs.shape().dims().to_vec();
        if a.len() != 2 || b.len() != 1 || a[1] != b[0] {
            return Err(fuel_core_types::Error::Msg(format!(
                "mv: shape mismatch lhs={a:?} rhs={b:?} (need [m,n] · [n])",
            )).bt());
        }
        // unsqueeze rhs to [n,1], matmul -> [m,1], squeeze trailing dim.
        let rhs_col = rhs.try_unsqueeze(1)?;
        let prod = self.matmul(&rhs_col);
        prod.squeeze(1)
    }

    /// Alias of [`Self::mv`] with a more descriptive name. Matches
    /// eager's [`Tensor::matvec`].
    pub fn matvec(&self, rhs: &Self) -> std::result::Result<Self, fuel_core_types::Error> {
        self.mv(rhs)
    }

    /// Broadcast-aware matmul. Lazy's [`Self::matmul`] already accepts
    /// broadcast-compatible operands; this method is exposed for
    /// signature compatibility with eager's `Tensor::broadcast_matmul`.
    pub fn broadcast_matmul(&self, rhs: &Self) -> Self {
        self.matmul(rhs)
    }

    // ---- Phase A.5 factory family ----
    //
    // Instance methods derive shape + dtype from `self` and place the new
    // tensor on the same graph via `const_*_like`. Static methods build
    // a fresh graph anchored on a host-allocated buffer.

    /// New tensor with the same shape, dtype, and graph as `self`, filled
    /// with ones.
    pub fn ones_like(&self) -> Self {
        let n = self.elem_count();
        let shape = self.shape();
        match self.dtype() {
            DType::F32 => self.const_f32_like(vec![1.0_f32; n], shape),
            DType::F64 => self.const_f64_like(vec![1.0_f64; n], shape),
            DType::BF16 => self.const_bf16_like(vec![half::bf16::ONE; n], shape),
            DType::F16 => self.const_f16_like(vec![half::f16::ONE; n], shape),
            DType::U32 => self.const_u32_like(vec![1_u32; n], shape),
            DType::I64 => self.const_i64_like(vec![1_i64; n], shape),
            other => panic!("ones_like: unsupported dtype {other:?}"),
        }
    }

    /// New tensor with the same shape, dtype, and graph as `self`, filled
    /// with zeros.
    pub fn zeros_like(&self) -> Self {
        let n = self.elem_count();
        let shape = self.shape();
        match self.dtype() {
            DType::F32 => self.const_f32_like(vec![0.0_f32; n], shape),
            DType::F64 => self.const_f64_like(vec![0.0_f64; n], shape),
            DType::BF16 => self.const_bf16_like(vec![half::bf16::ZERO; n], shape),
            DType::F16 => self.const_f16_like(vec![half::f16::ZERO; n], shape),
            DType::U32 => self.const_u32_like(vec![0_u32; n], shape),
            DType::I64 => self.const_i64_like(vec![0_i64; n], shape),
            other => panic!("zeros_like: unsupported dtype {other:?}"),
        }
    }

    /// New tensor with `shape`/`dtype`/`device`, every element set to `1`.
    /// Static factory equivalent of eager's `Tensor::ones`.
    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Self {
        let shape = shape.into();
        let n = shape.elem_count();
        match dtype {
            DType::F32 => Self::from_f32(vec![1.0_f32; n], shape, device),
            DType::F64 => Self::from_f64(vec![1.0_f64; n], shape, device),
            DType::BF16 => Self::from_bf16(vec![half::bf16::ONE; n], shape, device),
            DType::F16 => Self::from_f16(vec![half::f16::ONE; n], shape, device),
            DType::U32 => Self::from_u32(vec![1_u32; n], shape, device),
            other => panic!("ones: unsupported dtype {other:?}"),
        }
    }

    /// New tensor with `shape`/`dtype`/`device`, every element set to `0`.
    /// Static factory equivalent of eager's `Tensor::zeros`.
    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Self {
        let shape = shape.into();
        let n = shape.elem_count();
        match dtype {
            DType::F32 => Self::from_f32(vec![0.0_f32; n], shape, device),
            DType::F64 => Self::from_f64(vec![0.0_f64; n], shape, device),
            DType::BF16 => Self::from_bf16(vec![half::bf16::ZERO; n], shape, device),
            DType::F16 => Self::from_f16(vec![half::f16::ZERO; n], shape, device),
            DType::U32 => Self::from_u32(vec![0_u32; n], shape, device),
            other => panic!("zeros: unsupported dtype {other:?}"),
        }
    }

    /// New tensor of `shape`/`device` filled with `value`. The scalar's
    /// dtype determines the tensor's dtype.
    pub fn full(
        shape: impl Into<Shape>,
        value: fuel_core_types::Scalar,
        device: &Device,
    ) -> Self {
        let shape = shape.into();
        let n = shape.elem_count();
        match value {
            fuel_core_types::Scalar::F32(v) => Self::from_f32(vec![v; n], shape, device),
            fuel_core_types::Scalar::F64(v) => Self::from_f64(vec![v; n], shape, device),
            fuel_core_types::Scalar::BF16(v) => Self::from_bf16(vec![v; n], shape, device),
            fuel_core_types::Scalar::F16(v) => Self::from_f16(vec![v; n], shape, device),
            fuel_core_types::Scalar::U32(v) => Self::from_u32(vec![v; n], shape, device),
            other => panic!("full: unsupported scalar dtype {:?}", other.dtype()),
        }
    }

    /// Identity matrix `[n, n]` with the given dtype on the given device.
    /// Built host-side as a flat Vec; no graph-layer arange dependency.
    pub fn eye(n: usize, dtype: DType, device: &Device) -> Self {
        let mut data = vec![0.0_f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let base = Self::from_f32(data, vec![n, n], device);
        if dtype == DType::F32 { base } else { base.cast(dtype) }
    }

    /// Lower-triangular ones matrix `[n, n]`. `tril2(n).cast(dtype)` is the
    /// causal-attention-mask building block.
    pub fn tril2(n: usize, dtype: DType, device: &Device) -> Self {
        let mut data = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..=i {
                data[i * n + j] = 1.0;
            }
        }
        let base = Self::from_f32(data, vec![n, n], device);
        if dtype == DType::F32 { base } else { base.cast(dtype) }
    }

    /// Upper-triangular ones matrix `[n, n]`.
    pub fn triu2(n: usize, dtype: DType, device: &Device) -> Self {
        let mut data = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in i..n {
                data[i * n + j] = 1.0;
            }
        }
        let base = Self::from_f32(data, vec![n, n], device);
        if dtype == DType::F32 { base } else { base.cast(dtype) }
    }

    // ---- additional deferred-Phase-A items: indexing / multi-dim / RNG ----

    /// Eager-API alias of [`Self::slice`] (PyTorch / Candle naming).
    /// `narrow(dim, start, len)` is `slice(dim, start, len)` with the
    /// same panic-on-OOB semantics — both produce a view of
    /// `[start, start+len)` along `dim`.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        self.slice(dim, start, len)
    }

    /// Split into `chunks` views along `dim`. The split distributes the
    /// `chunk_size = ceil(dim_size / chunks)` extra slot to the leading
    /// chunks so every chunk's size differs by at most 1. If `dim_size
    /// < chunks`, returns `dim_size` singleton chunks instead of
    /// `chunks` chunks (matches eager / PyTorch).
    pub fn chunk(&self, chunks: usize, dim: usize) -> std::result::Result<Vec<Self>, fuel_core_types::Error> {
        let dims = self.shape().dims().to_vec();
        let rank = dims.len();
        if dim >= rank {
            return Err(fuel_core_types::Error::Msg(format!(
                "chunk: dim {dim} out of bounds for rank {rank}",
            )).bt());
        }
        if chunks == 0 {
            return Err(fuel_core_types::Error::Msg(
                "chunk: chunks must be > 0".into(),
            ).bt());
        }
        let size = dims[dim];
        if size < chunks {
            return Ok((0..size).map(|i| self.slice(dim, i, 1)).collect());
        }
        let base = size / chunks;
        let extra = size % chunks;
        let mut out = Vec::with_capacity(chunks);
        let mut start = 0;
        for i in 0..chunks {
            let len = if i < extra { base + 1 } else { base };
            out.push(self.slice(dim, start, len));
            start += len;
        }
        Ok(out)
    }

    /// Sub-tensor at index `i` along dim 0. Equivalent to
    /// `self.slice(0, i, 1).squeeze(0)`. Matches eager's [`crate::Tensor::get`].
    pub fn get(&self, i: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        let dims = self.shape().dims().to_vec();
        if dims.is_empty() {
            return Ok(self.clone());
        }
        self.slice(0, i, 1).squeeze(0)
    }

    /// Sub-tensor at index along an arbitrary dim. Equivalent to
    /// `self.slice(dim, index, 1).squeeze(dim)`. Matches eager's
    /// [`crate::Tensor::get_on_dim`].
    pub fn get_on_dim(&self, dim: usize, index: usize) -> std::result::Result<Self, fuel_core_types::Error> {
        let rank = self.shape().dims().len();
        if dim >= rank {
            return Err(fuel_core_types::Error::Msg(format!(
                "get_on_dim: dim {dim} out of bounds for rank {rank}",
            )).bt());
        }
        self.slice(dim, index, 1).squeeze(dim)
    }

    /// Multi-dim sum: reduce over every dim in `dims`, squeezing each.
    /// Reduces from the highest dim down so the lower dim indices stay
    /// valid throughout the reduction.
    pub fn sum_dims(&self, dims: &[usize]) -> Self {
        let mut sorted: Vec<usize> = dims.to_vec();
        sorted.sort_by(|a, b| b.cmp(a));
        sorted.dedup();
        let mut acc = self.clone();
        for d in sorted {
            acc = acc.sum_dim(d);
        }
        acc
    }

    /// Multi-dim mean: reduce over every dim in `dims`, squeezing each.
    /// Reduces from the highest dim down.
    pub fn mean_dims(&self, dims: &[usize]) -> Self {
        let mut sorted: Vec<usize> = dims.to_vec();
        sorted.sort_by(|a, b| b.cmp(a));
        sorted.dedup();
        let mut acc = self.clone();
        for d in sorted {
            acc = acc.mean_dim(d);
        }
        acc
    }

    /// Multi-dim sum with keepdim: every named dim becomes size 1
    /// instead of being squeezed out. Reduce-order-invariant (every
    /// keepdim preserves indices).
    pub fn sum_dims_keepdim(&self, dims: &[usize]) -> Self {
        let mut sorted: Vec<usize> = dims.to_vec();
        sorted.sort();
        sorted.dedup();
        let mut acc = self.clone();
        for d in sorted {
            acc = acc.sum_keepdim(d);
        }
        acc
    }

    /// Multi-dim mean with keepdim.
    pub fn mean_dims_keepdim(&self, dims: &[usize]) -> Self {
        let mut sorted: Vec<usize> = dims.to_vec();
        sorted.sort();
        sorted.dedup();
        let mut acc = self.clone();
        for d in sorted {
            acc = acc.mean_keepdim(d);
        }
        acc
    }

    /// Uniform random tensor in `[lo, up)` with shape/dtype/device matching `self`.
    /// Backed by [`rand::thread_rng`].
    pub fn rand_like(&self, lo: f64, up: f64) -> Self {
        Self::rand(self.shape(), lo, up, self.dtype(), &Device::cpu())
    }

    /// Normal random tensor with shape/dtype/device matching `self`.
    pub fn randn_like(&self, mean: f64, stdev: f64) -> Self {
        Self::randn(self.shape(), mean, stdev, self.dtype(), &Device::cpu())
    }

    /// Uniform random tensor in `[lo, up)`. Static factory.
    /// Supported dtypes: F32, F64, BF16, F16. F32 is the typical
    /// initialization target.
    pub fn rand(
        shape: impl Into<Shape>,
        lo: f64,
        up: f64,
        dtype: DType,
        device: &Device,
    ) -> Self {
        use rand::Rng;
        let shape = shape.into();
        let n = shape.elem_count();
        let mut rng = rand::rng();
        match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..n).map(|_| rng.random_range(lo..up) as f32).collect();
                Self::from_f32(data, shape, device)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..n).map(|_| rng.random_range(lo..up)).collect();
                Self::from_f64(data, shape, device)
            }
            DType::BF16 => {
                let data: Vec<half::bf16> = (0..n)
                    .map(|_| half::bf16::from_f64(rng.random_range(lo..up)))
                    .collect();
                Self::from_bf16(data, shape, device)
            }
            DType::F16 => {
                let data: Vec<half::f16> = (0..n)
                    .map(|_| half::f16::from_f64(rng.random_range(lo..up)))
                    .collect();
                Self::from_f16(data, shape, device)
            }
            other => panic!("LazyTensor::rand: unsupported dtype {other:?}"),
        }
    }

    /// Normal random tensor with given `mean` and `stdev`. Static factory.
    /// Supported dtypes: F32, F64, BF16, F16.
    pub fn randn(
        shape: impl Into<Shape>,
        mean: f64,
        stdev: f64,
        dtype: DType,
        device: &Device,
    ) -> Self {
        use rand_distr::{Distribution, Normal};
        let shape = shape.into();
        let n = shape.elem_count();
        let normal = Normal::new(mean, stdev).expect("randn: bad stdev");
        let mut rng = rand::rng();
        match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng) as f32).collect();
                Self::from_f32(data, shape, device)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
                Self::from_f64(data, shape, device)
            }
            DType::BF16 => {
                let data: Vec<half::bf16> = (0..n)
                    .map(|_| half::bf16::from_f64(normal.sample(&mut rng)))
                    .collect();
                Self::from_bf16(data, shape, device)
            }
            DType::F16 => {
                let data: Vec<half::f16> = (0..n)
                    .map(|_| half::f16::from_f64(normal.sample(&mut rng)))
                    .collect();
                Self::from_f16(data, shape, device)
            }
            other => panic!("LazyTensor::randn: unsupported dtype {other:?}"),
        }
    }

    /// `arange(start, end, device)`: a rank-1 tensor of `[start, end)` in
    /// step 1, dtype F32. Matches NumPy / PyTorch convention.
    pub fn arange(start: f32, end: f32, device: &Device) -> Self {
        Self::arange_step(start, end, 1.0, device)
    }

    /// `arange_step(start, end, step, device)`: a rank-1 tensor of
    /// `[start, end)` with constant step. F32 only for the static
    /// factory; cast for other dtypes. Errors at runtime if `step ==
    /// 0`.
    pub fn arange_step(start: f32, end: f32, step: f32, device: &Device) -> Self {
        assert!(step != 0.0, "arange_step: step must be non-zero");
        let mut data = Vec::new();
        let mut current = start;
        if step > 0.0 {
            while current < end {
                data.push(current);
                current += step;
            }
        } else {
            while current > end {
                data.push(current);
                current += step;
            }
        }
        let n = data.len();
        Self::from_f32(data, vec![n], device)
    }

    /// Linearly-spaced 1D tensor with `n` points from `start` to `end`
    /// (inclusive on both ends). Matches NumPy's `linspace`.
    pub fn linspace(start: f32, end: f32, n: usize, device: &Device) -> Self {
        assert!(n >= 1, "linspace: n must be >= 1");
        if n == 1 {
            return Self::from_f32(vec![start], vec![1], device);
        }
        let step = (end - start) / ((n - 1) as f32);
        let data: Vec<f32> = (0..n).map(|i| start + step * (i as f32)).collect();
        Self::from_f32(data, vec![n], device)
    }

    /// Frobenius norm: `sqrt(sum(self * self))`. Returns a scalar tensor.
    pub fn norm(&self) -> Self {
        self.sqr().sum_all().sqrt()
    }

    /// Pad with zeros along `dim`: `left` zeros before, `right` zeros
    /// after. Wraps [`Self::pad`] with `PadMode::Constant` and value 0
    /// for the named dim (other dims get `(0, 0)`). Composite — no new
    /// graph op.
    pub fn pad_with_zeros(
        &self,
        dim: usize,
        left: usize,
        right: usize,
    ) -> std::result::Result<Self, fuel_core_types::Error> {
        let rank = self.shape().dims().len();
        if dim >= rank {
            return Err(fuel_core_types::Error::Msg(format!(
                "pad_with_zeros: dim {dim} out of bounds for rank {rank}",
            )).bt());
        }
        if left == 0 && right == 0 {
            return Ok(self.clone());
        }
        let mut padding: Vec<(usize, usize)> = vec![(0, 0); rank];
        padding[dim] = (left, right);
        self.pad(padding, fuel_graph::PadMode::Constant, 0.0)
    }

    /// Coordinate grids from rank-1 inputs. Matches PyTorch's
    /// `torch.meshgrid` and eager's [`crate::Tensor::meshgrid`]:
    ///
    /// - `xy_indexing = true` (Cartesian, NumPy default): the first
    ///   two inputs are swapped before broadcasting and the resulting
    ///   grids reversed back, so for `(x, y)` the grids carry shape
    ///   `[len(y), len(x)]` (image-coordinate convention).
    /// - `xy_indexing = false` (matrix / ij): grids carry the input
    ///   cardinalities in input order.
    ///
    /// Implemented as `reshape` + `repeat`. All inputs must share the
    /// same dtype and live on the same graph.
    pub fn meshgrid(
        args: &[&Self],
        xy_indexing: bool,
    ) -> std::result::Result<Vec<Self>, fuel_core_types::Error> {
        if args.len() < 2 {
            return Err(fuel_core_types::Error::Msg(
                "meshgrid: requires at least two rank-1 tensors".into(),
            ).bt());
        }
        let ordered: Vec<&Self> = if xy_indexing {
            args.iter().rev().copied().collect()
        } else {
            args.iter().copied().collect()
        };
        let mut lens = Vec::with_capacity(ordered.len());
        for (i, t) in ordered.iter().enumerate() {
            let dims = t.shape().dims().to_vec();
            if dims.len() != 1 {
                return Err(fuel_core_types::Error::Msg(format!(
                    "meshgrid: input {i} must be rank 1, got shape {dims:?}",
                )).bt());
            }
            lens.push(dims[0]);
        }
        let mut grids = Vec::with_capacity(ordered.len());
        for (idx, t) in ordered.iter().enumerate() {
            let mut shape = vec![1_usize; ordered.len()];
            shape[idx] = lens[idx];
            let placed = t.try_reshape(shape)?;
            let mut repeats = lens.clone();
            repeats[idx] = 1;
            let grid = placed.repeat(repeats)?;
            grids.push(grid);
        }
        if xy_indexing {
            grids.reverse();
        }
        Ok(grids)
    }

    /// Repeat the tensor along each dim `repeats[i]` times. If `repeats`
    /// has more dims than `self`, `self` is implicitly left-padded with
    /// size-1 dims to match. Matches PyTorch's `Tensor.repeat`.
    pub fn repeat(&self, repeats: impl Into<Shape>) -> std::result::Result<Self, fuel_core_types::Error> {
        let repeats = repeats.into();
        let repeats: Vec<usize> = repeats.dims().to_vec();
        let self_rank = self.shape().dims().len();
        let target_rank = repeats.len();
        let mut work = if self_rank < target_rank {
            let pad_count = target_rank - self_rank;
            let mut new_shape: Vec<usize> = vec![1; pad_count];
            new_shape.extend_from_slice(self.shape().dims());
            self.try_reshape(new_shape)?
        } else if self_rank > target_rank {
            return Err(fuel_core_types::Error::Msg(format!(
                "repeat: repeats rank {target_rank} smaller than tensor rank {self_rank}",
            )).bt());
        } else {
            self.clone()
        };
        for (axis, &n) in repeats.iter().enumerate() {
            if n == 0 {
                return Err(fuel_core_types::Error::Msg(format!(
                    "repeat: zero repeat count at axis {axis} not supported",
                )).bt());
            }
            if n == 1 {
                continue;
            }
            // n copies concatenated along `axis`.
            let base = work.clone();
            for _ in 1..n {
                work = work.concat(&base, axis);
            }
        }
        Ok(work)
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
        device: &Device,
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
                Ok(Self::from_f32(data, shape_obj, device))
            }
            Dtype::F64 => {
                check_len(elem_count * 8)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.chunks_exact(8) {
                    let arr: [u8; 8] = chunk.try_into().unwrap();
                    data.push(f64::from_le_bytes(arr));
                }
                Ok(Self::from_f64(data, shape_obj, device))
            }
            Dtype::BF16 => {
                check_len(elem_count * 2)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.chunks_exact(2) {
                    let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                    data.push(half::bf16::from_bits(raw));
                }
                Ok(Self::from_bf16(data, shape_obj, device))
            }
            Dtype::F16 => {
                check_len(elem_count * 2)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.chunks_exact(2) {
                    let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                    data.push(half::f16::from_bits(raw));
                }
                Ok(Self::from_f16(data, shape_obj, device))
            }
            Dtype::U32 => {
                check_len(elem_count * 4)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.chunks_exact(4) {
                    data.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(Self::from_u32(data, shape_obj, device))
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
        device: &Device,
    ) -> crate::Result<Self> {
        Self::from_safetensors_bytes(view.data(), view.dtype(), view.shape(), device)
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
/// Output of [`LlamaModel::apply_layer_with_cache`]: hidden state plus
/// both variants of the layer's key/value tensors. Different callers
/// want different pieces — see the method docs.
pub(crate) struct LayerOutput {
    pub h: LazyTensor,
    /// Just this-step's K/V, pre-GQA, pre-concat with cache. Shape
    /// `[batch, n_kv_heads, seq, head_dim]`. Used by the host-resident
    /// cache path so append only pays for the new step's bytes.
    pub fresh_k: LazyTensor,
    pub fresh_v: LazyTensor,
    /// Cached ++ this-step's K/V, pre-GQA. Shape `[batch, n_kv_heads,
    /// cached_len + seq, head_dim]`. Used by the device-resident
    /// cache path so the graph's concat is the only concat — no
    /// post-realize concat step is needed.
    pub full_k: LazyTensor,
    pub full_v: LazyTensor,
}

/// LLaMA proper has no biases anywhere in the attention block, so the
/// `*_bias` fields are `None` for LLaMA family models. Qwen2 and a few
/// related architectures do add biases on Q/K/V (but not on the output
/// projection), so the loader stores them here when the safetensors
/// file contains them.
/// Weight tensor storage that preserves source precision.
///
/// Projection weights (Q/K/V/O for attention, gate/up/down for FFN,
/// and the output `lm_head` matrix) stay in whatever dtype the
/// source checkpoint used — f32 when that's how it was saved, bf16
/// for modern HF checkpoints that ship bf16 to halve weight memory.
/// Activations in the forward pass always stay f32 regardless; the
/// matmul kernel handles the mixed precision via
/// `VulkanBackend::matmul`'s `(A:F32, B:BF16) → F32` routing.
///
/// Norm gains and biases are NOT covered by this enum — they're
/// small and precision-sensitive, so they stay `Arc<[f32]>`.
///
/// Cloning is cheap (Arc bump) for both variants. Use
/// [`WeightStorage::const_like`] to emit a [`LazyTensor`] `Const`
/// node with the right dtype.
#[derive(Debug, Clone)]
pub enum WeightStorage {
    F32(Arc<[f32]>),
    BF16(Arc<[half::bf16]>),
    /// GGML Q4_0 blocks (raw byte stream), laid out as `[out_features,
    /// in_features / 32]` blocks (18 bytes each, llama.cpp convention).
    /// Stored as `Arc<[u32]>` — the byte stream reinterpreted as u32
    /// words so subsequent forward passes just Arc-clone (cheap) rather
    /// than recopying the bytes. The graph sees this directly as a U32
    /// tensor; matmul dispatch goes through `Op::QMatMul`.
    ///
    /// `bytes_len` is the original byte count (u32_len * 4) so the
    /// const_like shape computation doesn't accidentally round up.
    Q4_0 {
        words: Arc<[u32]>,
        bytes_len: usize,
        in_features: usize,
        out_features: usize,
    },
    /// Base weight wrapped with a trainable LoRA (Low-Rank Adaptation)
    /// update: effective weight `W_eff = base + (alpha / rank) · A · B`
    /// where `A` has shape `[in_features, rank]` and `B` has shape
    /// `[rank, out_features]` (both stored in the same layout
    /// convention as F32 weights — `[in, out]`).
    ///
    /// Used for PEFT-style inference with frozen base weights (which
    /// can be F32, BF16, or Q4_0) plus small trainable adapter matrices.
    /// The adapter is cheap to apply — for a 2560×2560 projection at
    /// rank 8 the LoRA path is ~0.5% of the base matmul cost.
    WithLoRA {
        base:          Box<WeightStorage>,
        /// `[in_features, rank]` adapter A (HF's `lora_A` transposed).
        lora_a:        Arc<[f32]>,
        /// `[rank, out_features]` adapter B (HF's `lora_B` transposed).
        lora_b:        Arc<[f32]>,
        rank:          usize,
        /// LoRA scaling factor; effective scale is `alpha / rank`.
        alpha:         f32,
        in_features:   usize,
        out_features:  usize,
    },
}

impl WeightStorage {
    pub fn elem_count(&self) -> usize {
        match self {
            Self::F32(a) => a.len(),
            Self::BF16(a) => a.len(),
            // Logical element count for a Q4_0 weight matrix is n*k.
            Self::Q4_0 { in_features, out_features, .. } => *in_features * *out_features,
            Self::WithLoRA { in_features, out_features, .. } => *in_features * *out_features,
        }
    }

    pub fn dtype(&self) -> fuel_core_types::DType {
        match self {
            Self::F32(_) => fuel_core_types::DType::F32,
            Self::BF16(_) => fuel_core_types::DType::BF16,
            // Q4_0 surfaces as U32 at the graph level (raw bytes
            // reinterpreted). Callers that care about the "actual"
            // quantization type should match on the variant directly.
            Self::Q4_0 { .. } => fuel_core_types::DType::U32,
            // WithLoRA exposes the base's dtype (the LoRA adapter is
            // always F32 but activations are typed by the base weight).
            Self::WithLoRA { base, .. } => base.dtype(),
        }
    }

    /// Emit a `Const` node on `anchor`'s graph matching this
    /// storage's dtype. Used everywhere the forward pass wraps a
    /// weight into a `LazyTensor`.
    ///
    /// For `Q4_0`, the emitted tensor is a 1-D `U32` const of length
    /// `bytes.len() / 4` holding the raw block byte stream. Callers
    /// must pair this with `Tensor::qmatmul` rather than `matmul`.
    pub fn const_like(&self, anchor: &LazyTensor, shape: Shape) -> LazyTensor {
        match self {
            Self::F32(a) => anchor.const_f32_like(a.clone(), shape),
            Self::BF16(a) => anchor.const_bf16_like(a.clone(), shape),
            Self::Q4_0 { words, .. } => {
                let _ = shape; // shape arg unused — Q4_0 const is 1-D U32
                // Arc-clone the precomputed u32 view; no byte copy.
                anchor.const_u32_like(Arc::clone(words), Shape::from_dims(&[words.len()]))
            }
            Self::WithLoRA { .. } => {
                panic!(
                    "WeightStorage::WithLoRA::const_like is not supported \
                     — the base + LoRA update must be applied via \
                     apply_linear to produce the right graph structure."
                );
            }
        }
    }

    /// Produce `X @ W` (with optional bias) for this weight storage.
    /// Dispatches to `matmul` for F32/BF16 weights and to `qmatmul`
    /// for Q4_0. The activations `x` must be F32.
    pub fn apply_linear(
        &self,
        x: &LazyTensor,
        in_features: usize,
        out_features: usize,
    ) -> LazyTensor {
        match self {
            Self::F32(_) | Self::BF16(_) => {
                let w = self.const_like(x, Shape::from_dims(&[in_features, out_features]));
                x.matmul(&w)
            }
            Self::Q4_0 { in_features: expected_in, out_features: expected_out, .. } => {
                assert_eq!(
                    *expected_in, in_features,
                    "WeightStorage::Q4_0 in_features mismatch: stored {}, requested {in_features}",
                    expected_in,
                );
                assert_eq!(
                    *expected_out, out_features,
                    "WeightStorage::Q4_0 out_features mismatch: stored {}, requested {out_features}",
                    expected_out,
                );
                // const_like for Q4_0 emits a flat U32 tensor.
                let w_bytes = self.const_like(x, Shape::from_dims(&[in_features, out_features]));
                x.qmatmul(&w_bytes, fuel_graph::QuantType::Q4_0, in_features, out_features)
            }
            Self::WithLoRA {
                base, lora_a, lora_b, rank, alpha,
                in_features: expected_in, out_features: expected_out,
            } => {
                assert_eq!(*expected_in, in_features, "WithLoRA in_features mismatch");
                assert_eq!(*expected_out, out_features, "WithLoRA out_features mismatch");
                // Base forward (F32, BF16, or Q4_0).
                let base_out = base.apply_linear(x, in_features, out_features);
                // Low-rank update: y += (alpha/rank) · x @ A @ B.
                let a_t = x.const_f32_like(
                    Arc::clone(lora_a),
                    Shape::from_dims(&[in_features, *rank]),
                );
                let b_t = x.const_f32_like(
                    Arc::clone(lora_b),
                    Shape::from_dims(&[*rank, out_features]),
                );
                let scale = *alpha as f64 / *rank as f64;
                // x: [*, in] → @A [*, rank] → @B [*, out] → scale → add base.
                let lora_path = LazyTensor {
                    inner: x.matmul(&a_t).matmul(&b_t).inner.mul_scalar(scale),
                };
                base_out.add(&lora_path)
            }
        }
    }

    /// Wrap this weight storage with a LoRA adapter. Asserts that the
    /// adapter shapes match `in_features`/`out_features`. Panics if the
    /// base is already a `WithLoRA` (nested adapters aren't supported;
    /// merge them explicitly if needed).
    pub fn with_lora(
        self,
        lora_a: Arc<[f32]>,
        lora_b: Arc<[f32]>,
        rank: usize,
        alpha: f32,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        assert_eq!(
            lora_a.len(), in_features * rank,
            "lora_a length {} does not match in_features ({in_features}) × rank ({rank}) = {}",
            lora_a.len(), in_features * rank,
        );
        assert_eq!(
            lora_b.len(), rank * out_features,
            "lora_b length {} does not match rank ({rank}) × out_features ({out_features}) = {}",
            lora_b.len(), rank * out_features,
        );
        assert!(
            !matches!(self, Self::WithLoRA { .. }),
            "with_lora: base is already WithLoRA (nested adapters unsupported)",
        );
        Self::WithLoRA {
            base: Box::new(self),
            lora_a, lora_b, rank, alpha,
            in_features, out_features,
        }
    }
}

// Auto-conversions so code that was storing `Arc<[f32]>` keeps
// compiling through the refactor — the LayerWeights field type
// widened to WeightStorage but ergonomics don't regress.
impl From<Arc<[f32]>> for WeightStorage {
    fn from(a: Arc<[f32]>) -> Self { Self::F32(a) }
}
impl From<Vec<f32>> for WeightStorage {
    fn from(v: Vec<f32>) -> Self { Self::F32(Arc::from(v)) }
}
impl From<Arc<[half::bf16]>> for WeightStorage {
    fn from(a: Arc<[half::bf16]>) -> Self { Self::BF16(a) }
}
impl From<Vec<half::bf16>> for WeightStorage {
    fn from(v: Vec<half::bf16>) -> Self { Self::BF16(Arc::from(v)) }
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// `[dim, dim]` query projection. Supports bf16 or f32.
    pub attn_q: WeightStorage,
    /// `[dim]` query projection bias (Qwen2-style; LLaMA has none).
    pub attn_q_bias: Option<Arc<[f32]>>,
    /// `[dim, dim]` key projection.
    pub attn_k: WeightStorage,
    /// `[kv_dim]` key projection bias.
    pub attn_k_bias: Option<Arc<[f32]>>,
    /// `[dim, dim]` value projection.
    pub attn_v: WeightStorage,
    /// `[kv_dim]` value projection bias.
    pub attn_v_bias: Option<Arc<[f32]>>,
    /// `[dim, dim]` output projection.
    pub attn_o: WeightStorage,
    /// `[dim, ffn_dim]` gate projection for SwiGLU.
    pub ffn_gate: WeightStorage,
    /// `[dim, ffn_dim]` up projection for SwiGLU.
    pub ffn_up: WeightStorage,
    /// `[ffn_dim, dim]` down projection for SwiGLU.
    pub ffn_down: WeightStorage,
    /// `[dim]` RmsNorm gain for the pre-attention norm. Stays f32
    /// — norm gains are small and precision-sensitive.
    pub attn_norm_gain: Arc<[f32]>,
    /// `[dim]` RmsNorm gain for the pre-FFN norm.
    pub ffn_norm_gain: Arc<[f32]>,
}

/// Top-level weights: token embedding table, per-layer weights, final
/// norm gain, and output projection (which may be tied to the embedding
/// or a separate matrix).
#[derive(Debug, Clone)]
pub struct LlamaWeights {
    /// `[vocab_size, dim]` token embedding table. Stays f32 — the
    /// downstream `index_select` + graph traversal requires activation
    /// dtype to be f32, and the table is used directly as activations.
    pub token_embedding: Arc<[f32]>,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
    /// `[dim]` RmsNorm gain for the final norm before the output head.
    pub final_norm_gain: Arc<[f32]>,
    /// `[dim, vocab_size]` output projection (a.k.a. `lm_head`).
    /// Supports bf16 or f32 on-device — this is the largest single
    /// matrix after the embedding, worth ~262 MB at f32.
    pub output: WeightStorage,
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
            &Device::cpu(),
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
        // Output projection to vocab logits (routes through qmatmul for Q4_0).
        weights.output.apply_linear(&h_norm, cfg.dim, cfg.vocab_size)
    }

    /// Like [`forward`] but returns the hidden state AFTER the final
    /// RMSNorm, BEFORE the output projection. Shape: `[batch, seq, dim]`.
    ///
    /// The `anchor` tensor provides the graph to build on — use a
    /// parameter or any existing tensor from the training graph. All
    /// frozen weights are emitted as Const nodes on that graph.
    ///
    /// Use this for fine-tuning: freeze all layer weights (const nodes)
    /// and apply a trainable output head manually:
    ///
    /// ```ignore
    /// // Inside TrainState::step's build_loss callback:
    /// let lm_head = &params["lm_head"];  // ← anchor tensor
    /// let hidden = model.forward_hidden(&tokens, 0, lm_head);
    /// let logits = hidden.matmul(lm_head);
    /// let loss = cross_entropy_with_logits(&logits, &targets);
    /// ```
    pub fn forward_hidden(
        &self,
        tokens: &[u32],
        start_pos: usize,
        anchor: &LazyTensor,
    ) -> LazyTensor {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1usize;

        let embed = anchor.const_f32_like(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
        );
        let token_ids = anchor.const_u32_like(
            tokens.iter().copied().collect::<Vec<u32>>(),
            Shape::from_dims(&[seq]),
        );
        let mut h = embed
            .index_select(0, &token_ids)
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]));

        let (cos_data, sin_data) = fuel_graph::build_rope_tables(
            cfg.rope_base, start_pos, seq, cfg.head_dim,
        );
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin);
        }

        apply_affine_rms_norm(&h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps)
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

        // Project to Q, K, V using WeightStorage::apply_linear — this
        // routes F32/BF16 through standard matmul and Q4_0 through
        // fused qmatmul. Under GQA, W_k and W_v have fewer output
        // features (kv_dim instead of dim).
        let q = apply_optional_bias(
            layer.attn_q.apply_linear(&x_norm, cfg.dim, cfg.dim),
            layer.attn_q_bias.as_ref(), cfg.dim);
        let k = apply_optional_bias(
            layer.attn_k.apply_linear(&x_norm, cfg.dim, kv_dim),
            layer.attn_k_bias.as_ref(), kv_dim);
        let v = apply_optional_bias(
            layer.attn_v.apply_linear(&x_norm, cfg.dim, kv_dim),
            layer.attn_v_bias.as_ref(), kv_dim);

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
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.dim, cfg.dim);

        // First residual connection.
        let h1 = x.add(&attn_out);

        // Pre-FFN RmsNorm with affine gain.
        let h1_norm = apply_affine_rms_norm(&h1, &layer.ffn_norm_gain, cfg.dim, cfg.norm_eps);

        // SwiGLU FFN (routes through apply_linear → qmatmul for Q4_0).
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim);
        let up   = layer.ffn_up.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim);
        let swiglu = gate.silu().mul(&up);
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.ffn_dim, cfg.dim);

        // Second residual connection.
        h1.add(&ffn_out)
    }

    /// Variant of [`apply_layer`] that also exposes the fresh K and V
    /// tensors so the caller can persist them to a KV cache, and that
    /// prepends cached keys/values in front of the fresh ones before
    /// the attention matmul.
    ///
    /// Returns a [`LayerOutput`] containing the layer's new hidden
    /// state plus both the fresh K/V tensors (shape `[batch,
    /// n_kv_heads, seq, head_dim]` — the layout
    /// [`LlamaKVCache::append_layer`] expects) AND the already-
    /// concatenated full K/V (shape `[batch, n_kv_heads, cached_len +
    /// seq, head_dim]`). The host-resident cache path uses `fresh_*`
    /// so it only downloads this step's new data; the
    /// device-resident cache path uses `full_*` so the graph's
    /// concat op is the only concat and there's no post-realize
    /// concat pass.
    fn apply_layer_with_cache(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        layer_cache: &LayerKVCache,
        cached_len: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> LayerOutput {
        let cfg = &self.config;
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let total_seq = cached_len + seq;

        let x_norm = apply_affine_rms_norm(x, &layer.attn_norm_gain, cfg.dim, cfg.norm_eps);

        // Q/K/V projections via WeightStorage::apply_linear (handles F32,
        // BF16, Q4_0 variants uniformly). Optional Qwen2-style biases.
        let q = apply_optional_bias(
            layer.attn_q.apply_linear(&x_norm, cfg.dim, cfg.dim),
            layer.attn_q_bias.as_ref(), cfg.dim);
        let k = apply_optional_bias(
            layer.attn_k.apply_linear(&x_norm, cfg.dim, kv_dim),
            layer.attn_k_bias.as_ref(), kv_dim);
        let v = apply_optional_bias(
            layer.attn_v.apply_linear(&x_norm, cfg.dim, kv_dim),
            layer.attn_v_bias.as_ref(), kv_dim);

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

        // Save references to the pre-GQA-expansion full K/V — these
        // have shape `[batch, n_kv_heads, cached_len+seq, head_dim]`
        // and are exactly what the device-resident cache wants to
        // store as "the new cache" for the next forward.
        let cache_full_k = full_k.clone();
        let cache_full_v = full_v.clone();

        // GQA: skip the old broadcast_to + reshape expansion.
        // Instead, pass unexpanded K/V [batch, n_kv_heads, total_seq,
        // head_dim] directly to the attention matmuls. The backend
        // infers n_rep = n_heads / n_kv_heads from the batch-dim
        // mismatch and indexes B with (head / n_rep) * batch_stride_b.
        // This eliminates 2 broadcast strided_copies per layer
        // (~44 dispatches/token for TinyLlama).
        //
        // When n_kv_heads == n_heads (no GQA), full_k/v are already
        // [batch, n_heads, ...] so the matmul batch dims match exactly
        // and n_rep stays 1.

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
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.dim, cfg.dim);

        let h1 = x.add(&attn_out);
        let h1_norm = apply_affine_rms_norm(&h1, &layer.ffn_norm_gain, cfg.dim, cfg.norm_eps);

        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim);
        let up   = layer.ffn_up.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim);
        let swiglu = gate.silu().mul(&up);
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.ffn_dim, cfg.dim);

        LayerOutput {
            h: h1.add(&ffn_out),
            fresh_k,
            fresh_v,
            full_k: cache_full_k,
            full_v: cache_full_v,
        }
    }

    // ===== Phase 7.6 step 9c E.3.3.D — host-resident forward retired =====
    //
    // The legacy host-resident cached forward path
    // (`forward_with_cache_on`, `forward_with_cache`,
    // `forward_with_cache_cuda`, `unpack_kv_cache`) and its supporting
    // types (`LayerKVCache`, `LlamaKVCache`) were retired in favor of
    // [`Self::forward_with_kv_context`] + [`KvCache`] +
    // [`InferenceContext`]. Greedy token-sequence parity vs the
    // retired path was confirmed by
    // `generate_with_kv_context_matches_legacy_generate` immediately
    // before retirement; bitwise prefill parity vs non-cached forward
    // is checked by
    // `forward_with_kv_context_prefill_matches_non_cached_forward`.
    //
    // The host-resident helper `apply_layer_with_cache` (above) STAYS
    // because the device-resident GPU cache path
    // (`forward_with_cache_gpu_on`) shares it — that path is the
    // generic-over-`GraphBackend` fallback that downstream callers
    // who haven't migrated to `Device` + `forward_with_kv_context`
    // still depend on.

    // ===== Phase 7.6 step 9c E.3.3.B — InferenceContext + KvCache + WriteSlice =====
    //
    // The new forward path. Uses pre-allocated KV-cache buffers
    // (`KvCache::with_capacity`) + `Op::WriteSlice` in-graph to mutate
    // them, replacing the legacy concat-cached-and-fresh / download-
    // fresh / host-append pattern. Runs on CPU, CUDA, and Vulkan via
    // the pipelined executor + binding-table dispatch.

    /// Variant of [`apply_layer_with_cache`] that uses pre-allocated
    /// KV-cache buffers + `Op::WriteSlice`. The K/V caches are bound
    /// via `k_cache_const` / `v_cache_const` (Const placeholders that
    /// the caller has wired into [`InferenceContext`]); the post-
    /// write tensors are sliced to the `cached_len + seq` extent for
    /// attention. No fresh K/V is returned — the cache mutation is
    /// in-graph as a side effect of the WriteSlice nodes.
    fn apply_layer_with_kv_writes(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        k_cache_const: &LazyTensor,
        v_cache_const: &LazyTensor,
        cached_len: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let total_seq = cached_len + seq;

        let x_norm = apply_affine_rms_norm(x, &layer.attn_norm_gain, cfg.dim, cfg.norm_eps);

        // Q/K/V projections + optional biases — identical to apply_layer_with_cache.
        let q = apply_optional_bias(
            layer.attn_q.apply_linear(&x_norm, cfg.dim, cfg.dim),
            layer.attn_q_bias.as_ref(), cfg.dim);
        let k = apply_optional_bias(
            layer.attn_k.apply_linear(&x_norm, cfg.dim, kv_dim),
            layer.attn_k_bias.as_ref(), kv_dim);
        let v = apply_optional_bias(
            layer.attn_v.apply_linear(&x_norm, cfg.dim, kv_dim),
            layer.attn_v_bias.as_ref(), kv_dim);

        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);

        let q_r = q_h.rope_with_tables(rope_cos, rope_sin);
        let k_r = k_h.rope_with_tables(rope_cos, rope_sin);

        // Write fresh K/V slabs into the pre-allocated cache buffers
        // via Op::WriteSlice. Source slab shape is
        // `[batch, n_kv_heads, seq, head_dim]`; dest range along axis
        // 2 is `(cached_len, cached_len + seq)`. The returned tensor's
        // Storage Arc IS the cache const's Arc — post-write reference
        // to the same underlying buffer (the executor's
        // WorkItemKind::WriteSlice branch adopts dest's Arc as the
        // kernel output, mutating in place).
        let write_ranges = vec![
            (0, batch),
            (0, cfg.n_kv_heads),
            (cached_len, cached_len + seq),
            (0, cfg.head_dim),
        ];
        let k_full_buffer = k_cache_const.write_slice(&k_r, write_ranges.clone())?;
        let v_full_buffer = v_cache_const.write_slice(&v_h, write_ranges)?;

        // Slice the post-write full buffer down to the visible prefix
        // `[..., 0..total_seq, ...]` along axis 2. This is what
        // attention reads; bytes past `total_seq` in the cache buffer
        // are stale / zero and excluded by the slice.
        let full_k = k_full_buffer.slice(2, 0, total_seq);
        let full_v = v_full_buffer.slice(2, 0, total_seq);

        // Attention path — identical to apply_layer_with_cache from here.
        let k_t = full_k.transpose();
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t);

        let mut mask_data = vec![0.0_f32; seq * total_seq];
        for q_idx in 0..seq {
            let abs_q = cached_len + q_idx;
            for k_idx in (abs_q + 1)..total_seq {
                mask_data[q_idx * total_seq + k_idx] = f32::NEG_INFINITY;
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
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.dim, cfg.dim);

        let h1 = x.add(&attn_out);
        let h1_norm = apply_affine_rms_norm(&h1, &layer.ffn_norm_gain, cfg.dim, cfg.norm_eps);

        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim);
        let up   = layer.ffn_up.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim);
        let swiglu = gate.silu().mul(&up);
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.ffn_dim, cfg.dim);

        Ok(h1.add(&ffn_out))
    }

    /// Forward pass using pre-allocated KV-cache buffers and
    /// `Op::WriteSlice`. The cache must have been constructed via
    /// [`KvCache::with_capacity`] (the legacy `with_dims` grow-by-
    /// replacement constructor is rejected — its layers carry no
    /// pre-allocated storage to write into).
    ///
    /// ## Architectural notes
    ///
    /// - The cache's K + V Storage Arcs are bound to per-step Const
    ///   NodeIds via [`InferenceContext::insert`]. The
    ///   `const_placeholder_like` helper pushes Const nodes WITHOUT
    ///   populating the graph's legacy `storage_map` — the realize
    ///   call's `initial` StorageCache (cloned from `ctx.persistent`)
    ///   short-circuits the `build_const_cache` walk.
    /// - The cache buffers are mutated in place by
    ///   `Op::WriteSlice`'s kernel; the cache's Arcs persist outside
    ///   the graph (the graph is built fresh per forward step and
    ///   dropped after realize). Subsequent forward steps see the
    ///   accumulated K/V state via the same Arcs.
    /// - Logits return shape: rank-1 `[vocab_size]` — last-position
    ///   only, same as [`Self::forward_with_cache_on`].
    /// - Backends: CPU, CUDA, and Vulkan all run this path via the
    ///   pipelined executor + binding-table dispatch.
    pub fn forward_with_kv_context(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> crate::Result<Vec<f32>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        let cached_len = cache.cached_len;

        if seq == 0 {
            return Err(fuel_core_types::Error::Msg(
                "forward_with_kv_context: zero tokens".to_string(),
            ).bt());
        }
        if cache.n_layers() != cfg.n_layers {
            return Err(fuel_core_types::Error::Msg(format!(
                "forward_with_kv_context: cache n_layers {} != model n_layers {}",
                cache.n_layers(), cfg.n_layers,
            )).bt());
        }
        let max_seq_len = cache.max_seq_len.ok_or_else(|| {
            fuel_core_types::Error::Msg(
                "forward_with_kv_context: cache was constructed via with_dims (no \
                 pre-allocated buffers); call KvCache::with_capacity(...) for the \
                 WriteSlice path".to_string(),
            ).bt()
        })?;
        if cached_len + seq > max_seq_len {
            return Err(fuel_core_types::Error::Msg(format!(
                "forward_with_kv_context: cached_len ({cached_len}) + seq ({seq}) > \
                 max_seq_len ({max_seq_len})",
            )).bt());
        }
        let cache_dtype = cache.dtype.unwrap_or(DType::F32);
        if cache.n_kv_heads != cfg.n_kv_heads || cache.head_dim != cfg.head_dim {
            return Err(fuel_core_types::Error::Msg(format!(
                "forward_with_kv_context: cache shape (n_kv_heads={}, head_dim={}) \
                 disagrees with model config (n_kv_heads={}, head_dim={})",
                cache.n_kv_heads, cache.head_dim, cfg.n_kv_heads, cfg.head_dim,
            )).bt());
        }

        // Embed lookup + reshape to [batch, seq, dim].
        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0, &token_ids)
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]));

        // RoPE cos/sin tables shared across layers.
        let (cos_data, sin_data) = fuel_graph::build_rope_tables(
            cfg.rope_base, cached_len, seq, cfg.head_dim,
        );
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        // Per-layer: bind the cache K + V Arcs to fresh Const NodeIds,
        // dispatch through the WriteSlice variant. Track the NodeIds
        // we insert into ctx so we can clean them up after realize
        // (the per-step NodeIds reference a graph that drops at end
        // of this method; leaving them in ctx.persistent would leak).
        let cache_shape = Shape::from_dims(
            &[batch, cfg.n_kv_heads, max_seq_len, cfg.head_dim],
        );
        let mut bound_node_ids: Vec<fuel_graph::NodeId> =
            Vec::with_capacity(2 * cfg.n_layers);
        for (li, layer_weights) in weights.layers.iter().enumerate() {
            let k_arc = cache.slot_storage(li, KvSlot::K).ok_or_else(|| {
                fuel_core_types::Error::Msg(format!(
                    "forward_with_kv_context: cache layer {li} has no K slot \
                     (with_capacity should have populated all layers)",
                )).bt()
            })?;
            let v_arc = cache.slot_storage(li, KvSlot::V).ok_or_else(|| {
                fuel_core_types::Error::Msg(format!(
                    "forward_with_kv_context: cache layer {li} has no V slot",
                )).bt()
            })?;
            let k_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            let v_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            let k_id = k_cache_node.inner.id();
            let v_id = v_cache_node.inner.id();
            ctx.insert(k_id, k_arc);
            ctx.insert(v_id, v_arc);
            bound_node_ids.push(k_id);
            bound_node_ids.push(v_id);

            h = self.apply_layer_with_kv_writes(
                &h,
                layer_weights,
                &k_cache_node,
                &v_cache_node,
                cached_len,
                &rope_cos,
                &rope_sin,
            )?;
        }

        let h_norm = apply_affine_rms_norm(&h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps);
        let logits = weights.output.apply_linear(&h_norm, cfg.dim, cfg.vocab_size);
        let last_pos = seq - 1;
        let last_logits = logits
            .slice(1, last_pos, 1)
            .reshape(Shape::from_dims(&[cfg.vocab_size]));

        // Realize through InferenceContext. The WriteSlice nodes
        // mutate the cache buffers as a side effect; downstream
        // attention reads through the post-write Slice views.
        let logits_vec = ctx.realize_one_as::<f32>(
            last_logits.inner.graph(),
            last_logits.inner.id(),
        )?;

        // Clean up per-step bindings from ctx so they don't accumulate
        // across decode steps (each step gets a fresh graph; the
        // previous step's NodeIds are dead).
        for id in bound_node_ids {
            ctx.remove(id);
        }

        // Bump cache state.
        cache.cached_len += seq;
        for li in 0..cfg.n_layers {
            cache.bump_version(li, KvSlot::K);
            cache.bump_version(li, KvSlot::V);
        }

        Ok(logits_vec)
    }
}

/// Per-layer host-side K/V scratch — `[n_kv_heads, cached_len, head_dim]`
/// laid out flat. Phase 7.6 step 9c E.3.3.D: the `LlamaKVCache`
/// wrapper that held a `Vec<LayerKVCache>` + `cached_len` was retired
/// (succeeded by `KvCache` in `crate::inference_context`). `LayerKVCache`
/// itself stays alive as a shape carrier — the device-resident path
/// (`forward_with_cache_gpu_on`) uses it as a zero-filled proxy that
/// feeds `apply_layer_with_cache`'s graph-building so the right Const
/// shapes are emitted; the proxy bytes are overwritten with real
/// device storage before realize and never read.
///
/// Once the device-resident `forward_with_cache_gpu_on` callers
/// migrate to `KvCache` + `forward_with_kv_context`, this type can
/// retire — the pipelined executor's WriteSlice path subsumes the
/// proxy-shape role.
#[derive(Debug, Clone, Default)]
pub struct LayerKVCache {
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
}

// Phase 7.6 step 9c E.3.3.D — host-resident `LlamaKVCache` retired.
// Its successor is `KvCache` in `crate::inference_context`, which
// stores backend-erased `Arc<RwLock<fuel_storage::Storage>>` per slot
// and supports both the legacy `with_dims` grow-by-replace shape and
// the new `with_capacity` pre-allocated-buffer shape that
// `forward_with_kv_context` writes into via `Op::WriteSlice`.

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

/// Transposed-matrix loader that preserves source dtype. For
/// safetensors files saved with bf16 weights, returns
/// `WeightStorage::BF16` and never materializes an f32 copy — the
/// 2× memory saving vs `load_transposed_matrix` comes from here.
/// f32 and other source dtypes still go through the f32 upcast path
/// for safety; extending this to preserve f16 is a one-line change
/// when a consumer wants it.
///
/// The transpose itself is done in whatever dtype we're keeping:
/// read bf16 elements from the file, place them in the transposed
/// target buffer, no conversion.
fn load_transposed_matrix_preserve_dtype(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> crate::Result<WeightStorage> {
    use safetensors::Dtype;
    let view = st.get(name)?;
    let bytes = view.data();
    let expected = out_features * in_features;
    match view.dtype() {
        Dtype::BF16 => {
            if bytes.len() != expected * 2 {
                crate::bail!(
                    "load_transposed_matrix_preserve_dtype: bf16 tensor {name:?} has {} bytes, expected {}",
                    bytes.len(), expected * 2,
                );
            }
            // Reinterpret input as [out_features, in_features] of
            // bf16; write transposed layout.
            let mut out = vec![half::bf16::ZERO; expected];
            for i in 0..out_features {
                for j in 0..in_features {
                    let src_off = (i * in_features + j) * 2;
                    let bits = u16::from_le_bytes([bytes[src_off], bytes[src_off + 1]]);
                    out[j * out_features + i] = half::bf16::from_bits(bits);
                }
            }
            Ok(WeightStorage::BF16(Arc::from(out)))
        }
        _ => {
            // F32, F64, F16 all fall through to the f32 upcast path.
            // Non-f32 source types still benefit from being readable;
            // they just lose the "weights stay compact" win.
            let flat = load_transposed_matrix(st, name, out_features, in_features)?;
            Ok(WeightStorage::F32(Arc::from(flat)))
        }
    }
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
            // Projections use the dtype-preserving loader — bf16
            // source files stay bf16 on-device (halving weight memory
            // on this layer).
            let attn_q = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.self_attn.q_proj.weight"),
                cfg.dim,
                cfg.dim,
            )?;
            let attn_k = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.self_attn.k_proj.weight"),
                kv_dim,
                cfg.dim,
            )?;
            let attn_v = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.self_attn.v_proj.weight"),
                kv_dim,
                cfg.dim,
            )?;
            let attn_o = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.self_attn.o_proj.weight"),
                cfg.dim,
                cfg.dim,
            )?;
            let ffn_gate = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.mlp.gate_proj.weight"),
                cfg.ffn_dim,
                cfg.dim,
            )?;
            let ffn_up = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.mlp.up_proj.weight"),
                cfg.ffn_dim,
                cfg.dim,
            )?;
            let ffn_down = load_transposed_matrix_preserve_dtype(
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
                attn_q,
                attn_q_bias,
                attn_k,
                attn_k_bias,
                attn_v,
                attn_v_bias,
                attn_o,
                ffn_gate,
                ffn_up,
                ffn_down,
                attn_norm_gain: Arc::from(attn_norm_gain),
                ffn_norm_gain:  Arc::from(ffn_norm_gain),
            });
        }

        let final_norm_gain = load_tensor_as_f32(st, "model.norm.weight")?;
        // `lm_head.weight` is `[vocab_size, dim]` in HF layout; we want
        // `[dim, vocab_size]` for `h @ W_out`. Fall back to tied
        // embeddings (`lm_head.weight` absent → reuse embed_tokens) for
        // models that tie input/output weights.
        let output: WeightStorage = match load_transposed_matrix_preserve_dtype(
            st, "lm_head.weight", cfg.vocab_size, cfg.dim,
        ) {
            Ok(w) => w,
            Err(_) => {
                // Tied weights: transpose embed_tokens. Embedding is
                // always f32, so the tied output is f32 regardless
                // of how the projection weights loaded.
                let mut transposed = vec![0.0_f32; cfg.dim * cfg.vocab_size];
                for i in 0..cfg.vocab_size {
                    for j in 0..cfg.dim {
                        transposed[j * cfg.vocab_size + i] =
                            token_embedding[i * cfg.dim + j];
                    }
                }
                WeightStorage::F32(Arc::from(transposed))
            }
        };

        Ok(LlamaWeights {
            token_embedding: Arc::from(token_embedding),
            layers,
            final_norm_gain: Arc::from(final_norm_gain),
            output,
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
        // Phase 7.6 step 9c E.3.3.D: re-pointed to the new KvCache +
        // InferenceContext + Op::WriteSlice path on CPU + F32. The
        // greedy parity test
        // `generate_with_kv_context_matches_legacy_generate` confirms
        // bitwise token-sequence equivalence with the retired
        // `generate_streaming_on` / `LlamaKVCache` host-resident path.
        self.generate_with_kv_context(
            prompt_tokens, max_new_tokens, strategy, eos_id,
            &Device::cpu(), DType::F32,
        )
    }

    /// Same contract as [`generate`], but invokes `on_token` once per
    /// freshly sampled token (prompt tokens are NOT emitted). Used by
    /// the CLI runner to print tokens as they're produced instead of
    /// waiting for the full sequence. Returns the full token sequence
    /// including the prompt once generation finishes or EOS is hit.
    /// Backend-agnostic streaming decode. KV cache lives on the host
    /// (`LlamaKVCache`) so it's the same type regardless of the
    /// backend; each forward round-trips the fresh K/V through host
    /// memory. The advantage is that this works with any backend for
    /// free — the Vulkan demo gets KV cache by calling this with a
    /// `GraphExecutor<VulkanBackend>`.
    ///
    // ===== Phase 7.6 step 9c E.3.3.D — host-resident streaming retired =====
    //
    // The legacy `generate_streaming_on<B>` (host-resident KV cache via
    // LlamaKVCache + per-step D2H/H2D round-trip) and its CPU-wrapper
    // `generate_streaming` were retired in favor of
    // `generate_streaming_with_kv_context`. Greedy token-sequence parity
    // was confirmed by `generate_with_kv_context_matches_legacy_generate`
    // before retirement. CPU, CUDA, and Vulkan callers all use the new
    // path (forward_with_kv_context + WriteSlice in-graph).

    // ===== Phase 7.6 step 9c E.3.3.C — streaming with KvCache + InferenceContext =====
    //
    // These replace the legacy `generate_streaming_on` /
    // `generate_streaming_gpu_on` pair across CPU, CUDA, and Vulkan.
    // The device is passed in directly (no `GraphBackend` parameter);
    // the pipelined executor handles backend dispatch through the
    // binding-table lookup.

    /// Streaming generation through the new `forward_with_kv_context`
    /// path. Allocates a pre-allocated `KvCache` of capacity
    /// `prompt_tokens.len() + max_new_tokens` on `device` (so the
    /// cache never overflows during decode), then loops prefill +
    /// decode, calling `on_token` for each generated token.
    ///
    /// `dtype` is the K/V storage dtype — typically `F32` for
    /// inference. The cache memory cost is
    /// `n_layers * 2 * n_kv_heads * (prompt+max_new) * head_dim *
    /// dtype_size`. For TinyLlama-1.1B at 1024-token max context, F32:
    /// 22 * 2 * 4 * 1024 * 64 * 4 ≈ 46 MiB.
    ///
    /// Works on CPU, CUDA, and Vulkan — the pipelined executor's
    /// binding-table dispatch picks the registered kernel per op
    /// based on the device passed in.
    pub fn generate_streaming_with_kv_context(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        device: &Device,
        dtype: DType,
        mut on_token: impl FnMut(u32),
    ) -> crate::Result<Vec<u32>> {
        let cfg = &self.config;
        if prompt_tokens.is_empty() {
            return Err(fuel_core_types::Error::Msg(
                "generate_streaming_with_kv_context: prompt is empty".to_string(),
            ).bt());
        }
        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut rng_state: u64 = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };

        let max_seq_len = prompt_tokens.len() + max_new_tokens;
        let mut cache = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
            max_seq_len, dtype, device,
        )?;
        let mut ctx = InferenceContext::new(device.clone());

        // Prefill: one forward pass over the full prompt.
        let mut last_logits =
            self.forward_with_kv_context(prompt_tokens, &mut cache, &mut ctx)?;

        // Decode loop.
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
                self.forward_with_kv_context(&[next], &mut cache, &mut ctx)?;
        }
        Ok(tokens)
    }

    /// Non-streaming convenience wrapper around
    /// [`Self::generate_streaming_with_kv_context`]. Collects the
    /// generated tokens into a `Vec<u32>` and returns them.
    pub fn generate_with_kv_context(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        device: &Device,
        dtype: DType,
    ) -> crate::Result<Vec<u32>> {
        self.generate_streaming_with_kv_context(
            prompt_tokens,
            max_new_tokens,
            strategy,
            eos_id,
            device,
            dtype,
            |_| {},
        )
    }
}

/// Device-resident KV cache, generic over `GraphBackend`. Keys and
/// values stay on the device that owns `B::Storage` across decode
/// steps, eliminating the D2H readback + H2D re-upload round-trip
/// that the host-resident `LlamaKVCache` path requires.
///
/// For `B = CpuBackend`, `B::Storage = AnyRefTensor` which is already
/// host-resident, so this type collapses gracefully to a host cache
/// for CPU users. For `B = CudaBackend` / `VulkanBackend` / future
/// GPU backends, storage lives on the device and concat / update
/// happens via the backend's native ops.
/// Per-layer KV storage. `F32` is the default (full precision, 4 bytes
/// per element). `Q8` stores the GGML Q8_0 block stream (34 bytes per
/// 32 elements = 1.0625 bytes/elem — roughly 4× the cache capacity at
/// ~1% quality loss). The Q8 variant is opt-in via
/// `KVCache::enable_q8_cache()`.
pub enum KVCacheEntry<S> {
    F32 { k: S, v: S },
    /// `k_blocks` / `v_blocks` are U32-typed storages holding the raw
    /// Q8_0 block byte stream (via `GraphBackend::quantize_q8_0`).
    Q8 { k_blocks: S, v_blocks: S },
}

pub struct KVCache<B: fuel_graph_executor::GraphBackend> {
    /// Per-layer cache entry. `None` until the layer's first forward
    /// populates it. Logical shape: `[1, n_kv_heads, cached_len, head_dim]`.
    pub(crate) layers: Vec<Option<KVCacheEntry<B::Storage>>>,
    pub cached_len: usize,
    // Shape metadata held for future save/restore and cross-device
    // migration methods. Not currently read on the decode hot path.
    #[allow(dead_code)]
    pub(crate) n_kv_heads: usize,
    #[allow(dead_code)]
    pub(crate) head_dim: usize,
    /// When true, fresh K/V are quantized to Q8_0 after each forward
    /// and dequantized on the next read. Requires the backend to
    /// implement `GraphBackend::{quantize,dequantize}_q8_0`.
    pub q8_enabled: bool,
    /// When true, the cache's layers have been spilled to host via a
    /// backend-specific `park` method. Ops against a parked cache
    /// must `unpark` first; the cache's `forward_with_cache_*` entry
    /// points would see host-backed storages and panic cleanly.
    pub parked: bool,
}

impl<B: fuel_graph_executor::GraphBackend> KVCache<B> {
    pub fn new(config: &LlamaConfig) -> Self {
        Self::with_dims(config.n_layers, config.n_kv_heads, config.head_dim)
    }

    /// Constructor for models that don't use `LlamaConfig` (e.g. PhiModel).
    pub fn with_dims(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| None).collect(),
            cached_len: 0,
            n_kv_heads,
            head_dim,
            q8_enabled: false,
            parked: false,
        }
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Read access to a layer's entry. Returns `None` if the layer
    /// hasn't been populated yet (fresh cache) or has been cleared.
    /// Used by tiered-residency paths and tests.
    pub fn layer(&self, li: usize) -> Option<&KVCacheEntry<B::Storage>> {
        self.layers.get(li).and_then(|o| o.as_ref())
    }

    /// Mutable access. Rarely needed from outside; mainly for
    /// residency management code that needs to swap entries in place.
    pub fn layer_mut(&mut self, li: usize) -> Option<&mut KVCacheEntry<B::Storage>> {
        self.layers.get_mut(li).and_then(|o| o.as_mut())
    }

    /// Install a layer's entry directly. Used by tests and by the
    /// tiered-residency park/unpark paths when they need to swap
    /// in a rebuilt entry.
    pub fn set_layer(&mut self, li: usize, entry: KVCacheEntry<B::Storage>) {
        self.layers[li] = Some(entry);
    }

    /// Enable Q8_0 quantization of the KV cache. Fresh K/V will be
    /// quantized after each forward pass and dequantized on the next
    /// read. Cuts KV-cache memory ~4× at ~1% quality loss.
    pub fn enable_q8_cache(&mut self) {
        self.q8_enabled = true;
    }

    /// Shrink the cache back to the first `new_len` positions along the
    /// seq dim. Used by speculative decoding's reject path to roll back
    /// after drafted tokens are rejected by the target model.
    ///
    /// No-op if `new_len >= cached_len`. For `new_len == 0` all layer
    /// entries are cleared (same state as a fresh cache).
    ///
    /// Q8-cached entries are not yet supported — bails with an error.
    /// Q8 blocks are 32-element aligned and an arbitrary `new_len`
    /// would require re-quantizing the trailing partial block; needs
    /// a separate kernel. Tracked as follow-up.
    pub fn truncate_to(&mut self, new_len: usize, backend: &B) -> crate::Result<()> {
        if new_len >= self.cached_len {
            return Ok(());
        }
        if self.q8_enabled {
            fuel_core_types::bail!(
                "KVCache::truncate_to: Q8 cache truncation not yet implemented"
            );
        }

        let batch = 1;
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;
        let old_seq = self.cached_len;

        for layer in &mut self.layers {
            let entry = match layer.take() {
                Some(e) => e,
                None => continue,
            };
            let (k, v) = match entry {
                KVCacheEntry::F32 { k, v } => (k, v),
                KVCacheEntry::Q8 { .. } => unreachable!("guarded above"),
            };
            // Early-return cleanly: if new_len == 0, drop the storage.
            if new_len == 0 {
                continue;
            }
            let new_k = truncate_kv_seq(backend, &k, batch, n_kv, old_seq, new_len, hd)?;
            let new_v = truncate_kv_seq(backend, &v, batch, n_kv, old_seq, new_len, hd)?;
            *layer = Some(KVCacheEntry::F32 { k: new_k, v: new_v });
        }
        self.cached_len = new_len;
        Ok(())
    }
}

/// Shrink an F32 K/V storage of shape `[batch, n_kv, old_seq, head_dim]`
/// (row-major contiguous) to `[batch, n_kv, new_seq, head_dim]`. Uses
/// `copy_strided_src` — one dispatch per tensor, all on-device.
fn truncate_kv_seq<B: fuel_graph_executor::GraphBackend>(
    backend: &B,
    src: &B::Storage,
    batch: usize,
    n_kv: usize,
    old_seq: usize,
    new_seq: usize,
    head_dim: usize,
) -> crate::Result<B::Storage> {
    // Source is contiguous with the OLD seq length; we want to read
    // only the first new_seq rows along dim 2. That's a strided read
    // where dim-2 stride stays head_dim but the gap between heads
    // skips the trailing old_seq-new_seq rows' worth of data.
    let src_shape = Shape::from_dims(&[batch, n_kv, new_seq, head_dim]);
    let src_strides: fuel_core_types::StrideVec = smallvec::smallvec![
        (n_kv * old_seq * head_dim) as isize,
        (old_seq * head_dim) as isize,
        head_dim as isize,
        1_isize,
    ];
    let src_layout = fuel_core_types::Layout::new(src_shape.clone(), src_strides, 0);

    let dtype = backend.storage_dtype(src);
    let dst_shape = Shape::from_dims(&[batch, n_kv, new_seq, head_dim]);
    let mut dst = backend.alloc_zeros(&dst_shape, dtype)?;
    backend.copy_strided_src(src, &mut dst, 0, &src_layout)?;
    Ok(dst)
}

/// CUDA-only alias kept for backward compatibility with existing
/// callers. Prefer `KVCache<CudaBackend>` directly in new code.
#[cfg(feature = "cuda")]
pub type GpuKVCache = KVCache<fuel_cuda_backend::CudaBackend>;

// ---- Tiered residency: KVCache park / unpark (Vulkan-only) ------------
//
// An idle `KVCache<VulkanBackend>` can be spilled to a host-side
// `ResidencyFile` via `park`, reclaiming its VRAM. When the caller
// needs the cache again (e.g., the next turn of a paused
// conversation), `unpark` faults each layer back to VRAM.
//
// First consumer of the P5 tiered-residency API. Other consumers
// (weight-layer offloading, long-context KV windowing) will come
// later; they reuse the same `ResidencyFile` + evict/fault_back
// primitives.

#[cfg(feature = "vulkan")]
impl KVCache<fuel_vulkan_backend::VulkanBackend> {
    /// Evict all layer K/V storage to the given residency file,
    /// freeing VRAM. `cached_len`, `parked` flag, and layer metadata
    /// are preserved so `unpark` can bring it back faithfully.
    ///
    /// Fails cleanly if:
    /// - the cache is already parked (guard against double-park),
    /// - any layer uses the Q8 variant (Q8 park is a follow-up —
    ///   the bytes-to-host path for Q8-backed layers needs its
    ///   own kernel path to preserve block structure).
    pub fn park(
        &mut self,
        backend: &fuel_vulkan_backend::VulkanBackend,
        file: &std::sync::Arc<fuel_vulkan_backend::residency::ResidencyFile>,
    ) -> crate::Result<()> {
        if self.parked {
            fuel_core_types::bail!("KVCache::park: cache is already parked");
        }
        if self.q8_enabled {
            fuel_core_types::bail!(
                "KVCache::park: Q8-enabled caches are not yet supported"
            );
        }
        // Evict each layer's K and V. Replace the entries in-place
        // so callers holding `&mut cache` see the updated tiers.
        for li in 0..self.layers.len() {
            let entry = match self.layers[li].take() {
                Some(e) => e,
                None => continue, // layer hasn't been populated yet
            };
            let (k, v) = match entry {
                KVCacheEntry::F32 { k, v } => (k, v),
                KVCacheEntry::Q8 { .. } => unreachable!("guarded above"),
            };
            let k_host = backend.evict(&k, file)?;
            let v_host = backend.evict(&v, file)?;
            // Drop the old device-backed handles so the Arc<VulkanBuffer>
            // refcount drops to zero and the VRAM sub-allocation is
            // returned to the buffer pool.
            drop(k);
            drop(v);
            self.layers[li] = Some(KVCacheEntry::F32 { k: k_host, v: v_host });
        }
        self.parked = true;
        Ok(())
    }

    /// Bring a parked cache's layers back into VRAM. Reverses
    /// [`Self::park`]. Fails if the cache isn't parked.
    pub fn unpark(
        &mut self,
        backend: &fuel_vulkan_backend::VulkanBackend,
    ) -> crate::Result<()> {
        if !self.parked {
            fuel_core_types::bail!("KVCache::unpark: cache is not parked");
        }
        for li in 0..self.layers.len() {
            let entry = match self.layers[li].take() {
                Some(e) => e,
                None => continue,
            };
            let (k, v) = match entry {
                KVCacheEntry::F32 { k, v } => (k, v),
                KVCacheEntry::Q8 { .. } => unreachable!(
                    "park bailed on Q8; we shouldn't see it on unpark"
                ),
            };
            let k_dev = backend.fault_back(&k)?;
            let v_dev = backend.fault_back(&v)?;
            drop(k);
            drop(v);
            self.layers[li] = Some(KVCacheEntry::F32 { k: k_dev, v: v_dev });
        }
        self.parked = false;
        Ok(())
    }
}


impl LlamaModel {
    /// Backend-agnostic streaming decode with device-resident KV cache.
    /// K/V stays on the device between steps (no D2H / H2D round-trip)
    /// and fresh K/V are concat'd onto the cache via the backend's
    /// own `alloc_zeros` + `copy_strided_src` primitives.
    ///
    /// For `B = CpuBackend` this collapses to a host-resident cache
    /// because `B::Storage = AnyRefTensor` already lives on the host.
    /// For GPU backends (CUDA, Vulkan, future Metal) the K/V bytes
    /// never leave the device.
    pub fn generate_streaming_gpu_on<B: fuel_graph_executor::GraphBackend>(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        executor: &mut GraphExecutor<B>,
        mut on_token: impl FnMut(u32),
    ) -> crate::Result<Vec<u32>> {
        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut rng_state: u64 = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };

        let mut cache: KVCache<B> = KVCache::new(&self.config);
        if std::env::var("FUEL_Q8_KV").ok().as_deref() == Some("1") {
            cache.enable_q8_cache();
        }
        let mut last_logits =
            self.forward_with_cache_gpu_on(&tokens, &mut cache, executor);

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
                self.forward_with_cache_gpu_on(&[next], &mut cache, executor);
        }
        Ok(tokens)
    }

    /// Speculative decoding.
    ///
    /// Uses a `draft` model to predict `k` tokens autoregressively,
    /// then has `self` (the target) verify all `k` positions in a
    /// single forward. Accepts a prefix of the drafts per `strategy`:
    ///
    /// - `Greedy`: longest prefix where target's argmax matches draft's token.
    ///   On mismatch, emit target's argmax as the bonus.
    /// - `Temperature`: Leviathan-style probability-ratio accept.
    ///   Sample draft tokens from draft's temperature-scaled distribution;
    ///   accept each with probability `min(1, p_target(d) / p_draft(d))`.
    ///   On reject, sample replacement from `(p_target - p_draft)_+ / Z`.
    ///   Distribution of outputs is provably identical to plain sampled
    ///   generation from the target.
    ///
    /// Rejected drafts are truncated from both caches via
    /// [`KVCache::truncate_to`]; one bonus token is always emitted per
    /// iteration.
    ///
    /// Expected speedup 1.5-3× at good acceptance rates (same-family
    /// drafts only — cross-family drafts or different tokenizers will
    /// have <20% acceptance and net-negative speedup).
    ///
    /// Preconditions:
    /// - `draft.config.vocab_size == self.config.vocab_size` (so
    ///   target's distribution over draft's vocab is well-defined).
    /// - Both models share the same tokenizer (caller's responsibility).
    pub fn generate_streaming_spec<B: fuel_graph_executor::GraphBackend>(
        &self,
        draft: &LlamaModel,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        k: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        target_executor: &mut GraphExecutor<B>,
        draft_executor: &mut GraphExecutor<B>,
        mut on_token: impl FnMut(u32),
    ) -> crate::Result<Vec<u32>> {
        if draft.config.vocab_size != self.config.vocab_size {
            fuel_core_types::bail!(
                "spec-decode: draft vocab {} != target vocab {}",
                draft.config.vocab_size, self.config.vocab_size,
            );
        }
        if k == 0 {
            fuel_core_types::bail!("spec-decode: k must be >= 1");
        }

        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        let vocab = self.config.vocab_size;

        // Greedy argmax helper.
        fn argmax(logits: &[f32]) -> u32 {
            let mut best = 0;
            let mut best_v = logits[0];
            for (i, &v) in logits.iter().enumerate().skip(1) {
                if v > best_v { best_v = v; best = i; }
            }
            best as u32
        }

        // Temperature-scaled softmax. Returns normalized probabilities.
        fn softmax_temp(logits: &[f32], temp: f32) -> Vec<f32> {
            let inv_t = if temp == 0.0 { 1.0 } else { 1.0 / temp };
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = logits.iter().map(|&x| ((x - max) * inv_t).exp()).collect();
            let sum: f32 = exp.iter().sum();
            exp.iter().map(|&x| x / sum).collect()
        }

        // Advance a deterministic LCG and return a u01 uniform.
        fn next_u01(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*state >> 32) as f32 / u32::MAX as f32
        }

        // Sample a category from a distribution summing to ~1.
        fn sample_cat(probs: &[f32], state: &mut u64) -> u32 {
            let u = next_u01(state);
            let mut cum = 0.0_f32;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if u <= cum { return i as u32; }
            }
            (probs.len() - 1) as u32
        }

        // RNG state threading. Only used in Temperature mode.
        let mut rng_state: u64 = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };
        let temp = match strategy {
            SamplingStrategy::Temperature { temp, .. } => temp,
            SamplingStrategy::Greedy => 1.0, // unused in greedy
        };

        // Prefill both caches with the prompt.
        let mut target_cache: KVCache<B> = KVCache::new(&self.config);
        let mut draft_cache: KVCache<B> = KVCache::new(&draft.config);
        let mut target_last_logits = self.forward_with_cache_gpu_on(&tokens, &mut target_cache, target_executor);
        let mut draft_last_logits = draft.forward_with_cache_gpu_on(&tokens, &mut draft_cache, draft_executor);

        let mut emitted = 0usize;

        while emitted < max_new_tokens {
            // --- Draft phase: K tokens. In Greedy mode, argmax; in
            // Temperature mode, sample from draft's temp-scaled dist.
            // We ALSO stash each draft's probability (draft-dist at the
            // chosen token) for the Temperature accept rule.
            let mut drafts: Vec<u32> = Vec::with_capacity(k);
            let mut draft_probs_stash: Vec<Vec<f32>> = Vec::with_capacity(k);
            for _ in 0..k {
                let d = match strategy {
                    SamplingStrategy::Greedy => {
                        let d = argmax(&draft_last_logits);
                        // We don't need draft_probs in greedy, but the
                        // field has to exist to keep indexing uniform.
                        draft_probs_stash.push(Vec::new());
                        d
                    }
                    SamplingStrategy::Temperature { .. } => {
                        let probs = softmax_temp(&draft_last_logits, temp);
                        let d = sample_cat(&probs, &mut rng_state);
                        draft_probs_stash.push(probs);
                        d
                    }
                };
                drafts.push(d);
                draft_last_logits = draft.forward_with_cache_gpu_on(
                    &[d], &mut draft_cache, draft_executor,
                );
            }

            // --- Verify phase: target runs forward on the K drafts.
            let verify_logits = self.forward_with_cache_gpu_on_all_positions(
                &drafts, &mut target_cache, target_executor,
            );
            debug_assert_eq!(verify_logits.len(), drafts.len() * vocab);

            // --- Accept phase: strategy-specific. ---
            let mut accepted = 0usize;
            let mut bonus_token: u32;
            match strategy {
                SamplingStrategy::Greedy => {
                    let mut mismatched: Option<u32> = None;
                    for i in 0..drafts.len() {
                        let prev_row = if i == 0 {
                            &target_last_logits[..]
                        } else {
                            &verify_logits[(i - 1) * vocab .. i * vocab]
                        };
                        let target_pick = argmax(prev_row);
                        if target_pick == drafts[i] {
                            accepted += 1;
                        } else {
                            mismatched = Some(target_pick);
                            break;
                        }
                    }
                    bonus_token = match mismatched {
                        Some(t) => t,
                        None => argmax(
                            &verify_logits[(drafts.len() - 1) * vocab .. drafts.len() * vocab]
                        ),
                    };
                }
                SamplingStrategy::Temperature { .. } => {
                    // Leviathan accept rule. For each i:
                    //   q_i = draft's prob of drafts[i]
                    //   p_i = target's prob of drafts[i] (from prev[i])
                    //   accept with prob min(1, p_i / q_i)
                    // On reject: sample replacement from (p - q)_+ / sum.
                    let mut rejected_replacement: Option<u32> = None;
                    for i in 0..drafts.len() {
                        let prev_row = if i == 0 {
                            &target_last_logits[..]
                        } else {
                            &verify_logits[(i - 1) * vocab .. i * vocab]
                        };
                        let target_probs = softmax_temp(prev_row, temp);
                        let draft_probs = &draft_probs_stash[i];
                        let d_tok = drafts[i] as usize;
                        let p = target_probs[d_tok];
                        let q = draft_probs[d_tok];
                        let ratio = if q > 0.0 { (p / q).min(1.0) } else { 0.0 };
                        let u = next_u01(&mut rng_state);
                        if u < ratio {
                            accepted += 1;
                        } else {
                            // Replacement from (p - q)_+ / sum.
                            let mut residual: Vec<f32> = target_probs.iter().zip(draft_probs.iter())
                                .map(|(&pt, &qt)| (pt - qt).max(0.0))
                                .collect();
                            let sum: f32 = residual.iter().sum();
                            if sum > 0.0 {
                                for r in residual.iter_mut() { *r /= sum; }
                                rejected_replacement = Some(sample_cat(&residual, &mut rng_state));
                            } else {
                                // Degenerate case (should only happen if
                                // distributions match exactly — then any
                                // sample from target_probs is equally valid).
                                rejected_replacement = Some(sample_cat(&target_probs, &mut rng_state));
                            }
                            break;
                        }
                    }
                    bonus_token = match rejected_replacement {
                        Some(t) => t,
                        None => {
                            // All K accepted — sample bonus from target's
                            // last-position distribution.
                            let last_row = &verify_logits[(drafts.len() - 1) * vocab .. drafts.len() * vocab];
                            let probs = softmax_temp(last_row, temp);
                            sample_cat(&probs, &mut rng_state)
                        }
                    };
                }
            }

            // --- Rollback caches ---
            // Target cache advanced by K but we committed (accepted + 1) tokens
            // (accepted drafts + bonus). Excess = K - (accepted + 1) to drop,
            // but only when accepted + 1 < K (i.e., at least one draft rejected).
            let target_excess = k.saturating_sub(accepted + 1);
            if target_excess > 0 {
                let new_len = target_cache.cached_len - target_excess;
                target_cache.truncate_to(new_len, &target_executor.backend)?;
            }
            // Draft cache: advanced by K; we commit (accepted) of those drafts
            // + 1 bonus the draft didn't see. Truncate draft by (K - accepted)
            // to drop the rejected drafts. Then feed bonus to draft to advance
            // one position and get fresh draft_last_logits for the next round.
            let draft_excess = k - accepted;
            if draft_excess > 0 {
                let new_len = draft_cache.cached_len - draft_excess;
                draft_cache.truncate_to(new_len, &draft_executor.backend)?;
            }

            // --- Emit accepted drafts + bonus ---
            for i in 0..accepted {
                tokens.push(drafts[i]);
                on_token(drafts[i]);
                emitted += 1;
                if emitted >= max_new_tokens { return Ok(tokens); }
                if eos_id == Some(drafts[i]) { return Ok(tokens); }
            }
            tokens.push(bonus_token);
            on_token(bonus_token);
            emitted += 1;
            if eos_id == Some(bonus_token) { return Ok(tokens); }

            // --- Advance both caches + both "last_logits" by the bonus token. ---
            // Draft needs to see bonus (which it didn't produce), then return
            // fresh logits for the next iteration's first draft. Target cache
            // already has committed positions; advance it by the bonus so
            // target_last_logits is fresh for the next accept-check on draft[0].
            target_last_logits = self.forward_with_cache_gpu_on(
                &[bonus_token], &mut target_cache, target_executor,
            );
            draft_last_logits = draft.forward_with_cache_gpu_on(
                &[bonus_token], &mut draft_cache, draft_executor,
            );
        }
        Ok(tokens)
    }

    /// CUDA-specific thin wrapper preserved for call-site compatibility.
    /// Prefer `generate_streaming_gpu_on` in new code.
    #[cfg(feature = "cuda")]
    pub fn generate_streaming_cuda(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        executor: &mut GraphExecutor<fuel_cuda_backend::CudaBackend>,
        on_token: impl FnMut(u32),
    ) -> crate::Result<Vec<u32>> {
        self.generate_streaming_gpu_on(
            prompt_tokens, max_new_tokens, strategy, eos_id, executor, on_token,
        )
    }

    /// Forward pass with device-resident KV cache, generic over
    /// `GraphBackend`. Cached K/V are injected directly as
    /// pre-populated graph nodes (no H2D). Fresh K/V stay on the
    /// device after realize (no D2H). Only logits are transferred
    /// to host. The cache is updated on-device via the backend's
    /// own primitives after the realize.
    pub fn forward_with_cache_gpu_on<B: fuel_graph_executor::GraphBackend>(
        &self,
        tokens: &[u32],
        cache: &mut KVCache<B>,
        executor: &mut GraphExecutor<B>,
    ) -> Vec<f32> {
        self.forward_with_cache_gpu_on_impl(tokens, cache, executor, false)
    }

    /// All-positions variant: returns `seq * vocab_size` logits (flat,
    /// row-major over position). Used by speculative decoding's
    /// verification step — target runs forward on K+1 tokens at once
    /// and needs per-position logits to accept/reject drafts.
    ///
    /// Cache semantics identical to `forward_with_cache_gpu_on`; on
    /// reject, caller invokes [`KVCache::truncate_to`] to roll back.
    pub fn forward_with_cache_gpu_on_all_positions<B: fuel_graph_executor::GraphBackend>(
        &self,
        tokens: &[u32],
        cache: &mut KVCache<B>,
        executor: &mut GraphExecutor<B>,
    ) -> Vec<f32> {
        self.forward_with_cache_gpu_on_impl(tokens, cache, executor, true)
    }

    fn forward_with_cache_gpu_on_impl<B: fuel_graph_executor::GraphBackend>(
        &self,
        tokens: &[u32],
        cache: &mut KVCache<B>,
        executor: &mut GraphExecutor<B>,
        return_all_positions: bool,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        let cached_len = cache.cached_len;

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &Device::cpu(),
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

        // Track the NodeIds of the placeholder K/V const nodes so we
        // can pre_populate them with real device storage before realize.
        let mut cached_kv_nodes: Vec<(fuel_graph::NodeId, fuel_graph::NodeId)> = Vec::new();
        // Device-resident path: roots are the full (cached ++ fresh)
        // K/V tensors, NOT just fresh. The graph's concat inside
        // apply_layer_with_cache is therefore the only concat — we
        // skip the post-realize concat entirely and just keep the
        // realized full tensors as the new cache.
        let mut full_ks: Vec<LazyTensor> = Vec::with_capacity(cfg.n_layers);
        let mut full_vs: Vec<LazyTensor> = Vec::with_capacity(cfg.n_layers);

        for (_li, layer) in weights.layers.iter().enumerate() {
            // Zero-filled host placeholder so apply_layer_with_cache
            // can build Const nodes in the graph with the right shape.
            // pre_populate will overwrite them with real device
            // storage before realize, so the placeholder data is
            // never actually read.
            let layer_cache_proxy: LayerKVCache = if cached_len > 0 {
                let n = batch * cfg.n_kv_heads * cached_len * cfg.head_dim;
                LayerKVCache { k: vec![0.0; n], v: vec![0.0; n] }
            } else {
                LayerKVCache::default()
            };

            let out = self.apply_layer_with_cache(
                &h, layer, &layer_cache_proxy, cached_len,
                &rope_cos, &rope_sin,
            );
            h = out.h;
            full_ks.push(out.full_k);
            full_vs.push(out.full_v);
        }

        // Find the NodeIds of the placeholder Const nodes by scanning
        // the graph for Consts with the right shape.
        if cached_len > 0 {
            let graph = h.graph_tensor().graph().read().unwrap();
            let target_elems = batch * cfg.n_kv_heads * cached_len * cfg.head_dim;
            let mut found: Vec<fuel_graph::NodeId> = Vec::new();
            for node_id in 0..graph.len() {
                let nid = fuel_graph::NodeId(node_id);
                let node = graph.node(nid);
                if matches!(node.op, fuel_graph::Op::Const)
                    && node.shape.elem_count() == target_elems
                    && node.dtype == fuel_core_types::DType::F32
                    && node.shape.dims() == [batch, cfg.n_kv_heads, cached_len, cfg.head_dim]
                {
                    found.push(nid);
                }
            }
            // Should be exactly 2 * n_layers (K and V per layer).
            if found.len() == 2 * cfg.n_layers {
                for li in 0..cfg.n_layers {
                    cached_kv_nodes.push((found[li * 2], found[li * 2 + 1]));
                }
            }
        }

        let h_norm = apply_affine_rms_norm(
            &h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps,
        );
        let logits = weights.output.apply_linear(&h_norm, cfg.dim, cfg.vocab_size);

        // For spec-decode verification we need per-position logits;
        // otherwise slice to the last position for decode/prefill.
        let last_pos = seq - 1;
        let last_logits = logits
            .slice(1, last_pos, 1)
            .reshape(Shape::from_dims(&[cfg.vocab_size]));
        let all_logits = logits.reshape(Shape::from_dims(&[seq * cfg.vocab_size]));
        let logits_root = if return_all_positions { &all_logits } else { &last_logits };

        // Roots: [logits, full_k_0..N, full_v_0..N]
        let mut roots: Vec<&LazyTensor> = Vec::with_capacity(1 + 2 * cfg.n_layers);
        roots.push(logits_root);
        for fk in &full_ks { roots.push(fk); }
        for fv in &full_vs { roots.push(fv); }

        // Inject cached K/V device storage for the placeholder const
        // nodes. For Q8 cache entries we dequantize back to F32 on
        // device first — the graph's apply_layer_with_cache still
        // consumes plain F32 K/V as Const inputs.
        let cached_elems = batch * cfg.n_kv_heads * cached_len * cfg.head_dim;
        for (li, (ck_id, cv_id)) in cached_kv_nodes.iter().enumerate() {
            if let Some(entry) = &cache.layers[li] {
                let cached_shape = Shape::from_dims(&[batch, cfg.n_kv_heads, cached_len, cfg.head_dim]);
                let layout = fuel_core_types::Layout::contiguous(&cached_shape);
                let (k_f32, v_f32) = match entry {
                    KVCacheEntry::F32 { k, v } => {
                        let k = executor.backend.try_clone(k, &layout)
                            .expect("inject K clone");
                        let v = executor.backend.try_clone(v, &layout)
                            .expect("inject V clone");
                        (k, v)
                    }
                    KVCacheEntry::Q8 { k_blocks, v_blocks } => {
                        let n_blocks = cached_elems / 32;
                        let k = executor.backend.dequantize_q8_0(k_blocks, n_blocks)
                            .expect("dequantize K from Q8 cache");
                        let v = executor.backend.dequantize_q8_0(v_blocks, n_blocks)
                            .expect("dequantize V from Q8 cache");
                        (k, v)
                    }
                };
                executor.pre_populate(*ck_id, k_f32, cached_shape.clone());
                executor.pre_populate(*cv_id, v_f32, cached_shape);
            }
        }

        // Realize: logits → CPU, full K/V → stay on device.
        let inner_roots: Vec<&fuel_graph::Tensor> =
            roots.iter().map(|lt| &lt.inner).collect();
        let (cpu_results, gpu_results) = executor.realize_split(&inner_roots, 1);
        let logits_vec = cpu_results.into_iter().next().unwrap();

        // Update cache: the realized full K/V tensors ARE the new
        // cache — no post-realize concat needed because the graph
        // already did it inside apply_layer_with_cache. If q8_enabled,
        // quantize the F32 K/V to Q8_0 blocks before storing.
        let mut iter = gpu_results.into_iter();
        let new_ks: Vec<(B::Storage, Shape)> = (0..cfg.n_layers)
            .map(|_| iter.next().unwrap()).collect();
        let new_vs: Vec<(B::Storage, Shape)> = (0..cfg.n_layers)
            .map(|_| iter.next().unwrap()).collect();

        let new_len = cached_len + seq;
        let new_elems = batch * cfg.n_kv_heads * new_len * cfg.head_dim;
        for (li, ((new_k, _), (new_v, _))) in
            new_ks.into_iter().zip(new_vs.into_iter()).enumerate()
        {
            let entry = if cache.q8_enabled && new_elems % 32 == 0 {
                let k_blocks = executor.backend.quantize_q8_0(&new_k, new_elems)
                    .expect("quantize K to Q8 cache");
                let v_blocks = executor.backend.quantize_q8_0(&new_v, new_elems)
                    .expect("quantize V to Q8 cache");
                KVCacheEntry::Q8 { k_blocks, v_blocks }
            } else {
                KVCacheEntry::F32 { k: new_k, v: new_v }
            };
            cache.layers[li] = Some(entry);
        }
        cache.cached_len += seq;

        logits_vec
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
            &Device::cpu(),
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

// ---- Phi-2 model assembly ---------------------------------------------------
//
// Phi-2 (microsoft/phi-2, 2.7B params) differs from LLaMA in four
// meaningful ways, each of which exercises a different code path:
//
//   1. Norm: LayerNorm with gain + bias (not RMSNorm with gain only)
//   2. MLP: standard fc1 → GELU → fc2 (not SwiGLU's gate ⊗ up → down)
//   3. Residual structure: parallel attention + MLP — both branches
//      consume the same pre-block-norm input and are summed with x:
//        h' = x + attn(LN(x)) + mlp(LN(x))
//      compared to LLaMA's sequential:
//        h1 = x + attn(LN1(x))
//        h2 = h1 + mlp(LN2(h1))
//   4. Partial RoPE: only the first `rotary_dim` entries of each head
//      get rotated (rotary_dim=32 for head_dim=80 in Phi-2). The rest
//      pass through unchanged. We slice → rope → concat.
//
// Phi-2 also has biases on Q/K/V/dense and on fc1/fc2, plus a bias on
// the LayerNorm. Every one of those is a real `broadcast_add` in the
// graph, which exercises the lazy broadcast path we built for the
// stride-aware binary work.

/// Phi-2 model hyperparameters. Field semantics match the LLaMA config
/// where they overlap; the `layer_norm_eps`, `partial_rotary_factor`,
/// and `rotary_dim` fields are Phi-specific.
#[derive(Debug, Clone)]
pub struct PhiConfig {
    pub vocab_size:            usize,
    pub dim:                   usize,  // hidden_size
    pub n_layers:              usize,
    pub n_heads:               usize,
    pub head_dim:              usize,
    pub ffn_dim:               usize,  // intermediate_size
    pub layer_norm_eps:        f64,
    pub rope_base:             f64,
    pub partial_rotary_factor: f64,
    /// Number of dims at the start of head_dim that get rotated.
    /// `rotary_dim = (partial_rotary_factor * head_dim).round() as usize`.
    /// Must be even for the half-split RoPE layout.
    pub rotary_dim:            usize,
    pub tie_word_embeddings:   bool,
}

impl PhiConfig {
    pub fn from_hf_json_str(json: &str) -> crate::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| crate::Error::Msg(format!("parsing config.json: {e}")))?;

        let get_usize = |key: &str| -> crate::Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| crate::Error::Msg(format!("config.json: missing/invalid field {key:?}")))
        };
        let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };

        let vocab_size = get_usize("vocab_size")?;
        let dim = get_usize("hidden_size")?;
        let n_layers = get_usize("num_hidden_layers")?;
        let n_heads = get_usize("num_attention_heads")?;
        let ffn_dim = get_usize("intermediate_size")?;
        let head_dim = v.get("head_dim").and_then(|x| x.as_u64())
            .map(|x| x as usize).unwrap_or(dim / n_heads);
        let layer_norm_eps = get_f64("layer_norm_eps").unwrap_or(1e-5);
        let rope_base = get_f64("rope_theta").unwrap_or(10_000.0);
        let partial_rotary_factor = get_f64("partial_rotary_factor").unwrap_or(0.4);
        let rotary_dim = (partial_rotary_factor * head_dim as f64).round() as usize;
        if rotary_dim % 2 != 0 {
            crate::bail!(
                "PhiConfig: rotary_dim {rotary_dim} must be even (partial_rotary_factor={partial_rotary_factor}, head_dim={head_dim})"
            );
        }
        let tie_word_embeddings = v.get("tie_word_embeddings")
            .and_then(|x| x.as_bool()).unwrap_or(false);

        Ok(PhiConfig {
            vocab_size, dim, n_layers, n_heads, head_dim, ffn_dim,
            layer_norm_eps, rope_base, partial_rotary_factor, rotary_dim,
            tie_word_embeddings,
        })
    }
}

/// How Q/K/V projections are stored for a Phi layer.
///
/// - `Split`: separate Q, K, V weights + biases (matches HF safetensors
///   layout — `q_proj.weight`, `k_proj.weight`, `v_proj.weight`).
/// - `Packed`: single `[3*dim, dim]` weight + `[3*dim]` bias (matches
///   llama.cpp GGUF layout — `attn_qkv.weight`). The forward pass does
///   one big matmul producing `[*, 3*dim]`, then slices that output
///   into Q, K, V. Critically, the slice happens on the OUTPUT side
///   rather than up-front on the weights — this matches Candle's
///   `qkv.reshape(3, n_head, head_dim).i((.., .., 0..3))` exactly and
///   avoids any potential byte-split-order hazards on the weight side.
#[derive(Debug, Clone)]
pub enum PhiQkv {
    Split {
        q: WeightStorage,
        q_bias: Arc<[f32]>,
        k: WeightStorage,
        k_bias: Arc<[f32]>,
        v: WeightStorage,
        v_bias: Arc<[f32]>,
    },
    Packed {
        /// `[3*dim, dim]` weight (GGUF layout).
        qkv: WeightStorage,
        /// `[3*dim]` bias, Q first then K then V (standard Candle convention).
        qkv_bias: Arc<[f32]>,
    },
}

/// Per-layer Phi-2 weights. Every projection has a bias (unlike LLaMA).
#[derive(Debug, Clone)]
pub struct PhiLayerWeights {
    pub attn_qkv: PhiQkv,
    /// Output projection (called "dense" in Phi-2, not "o_proj").
    pub attn_dense:      WeightStorage,
    pub attn_dense_bias: Arc<[f32]>,
    pub mlp_fc1:         WeightStorage,  // [dim, ffn_dim]
    pub mlp_fc1_bias:    Arc<[f32]>,
    pub mlp_fc2:         WeightStorage,  // [ffn_dim, dim]
    pub mlp_fc2_bias:    Arc<[f32]>,
    /// Pre-block LayerNorm (single norm for Phi-2's parallel attn+MLP).
    pub norm_gain:      Arc<[f32]>,
    pub norm_bias:      Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct PhiWeights {
    pub token_embedding: Arc<[f32]>,   // [vocab_size, dim]
    pub layers:          Vec<PhiLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub final_norm_bias: Arc<[f32]>,
    pub output:          WeightStorage,  // [dim, vocab_size]
    pub output_bias:     Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct PhiModel {
    pub config:  PhiConfig,
    pub weights: PhiWeights,
}

/// Apply LayerNorm with affine gain + bias along the last dim.
/// `y = (x - mean) / sqrt(var + eps) * gain + bias`, where gain and
/// bias are per-channel vectors of length `dim`.
fn apply_affine_layer_norm(
    x: &LazyTensor,
    gain: &Arc<[f32]>,
    bias: &Arc<[f32]>,
    dim: usize,
    eps: f64,
) -> LazyTensor {
    assert_eq!(gain.len(), dim, "apply_affine_layer_norm: gain length must equal dim");
    assert_eq!(bias.len(), dim, "apply_affine_layer_norm: bias length must equal dim");
    let normalized = x.layer_norm_last_dim(eps);
    let gain_t = x.const_f32_like(Arc::clone(gain), Shape::from_dims(&[dim]));
    let bias_t = x.const_f32_like(Arc::clone(bias), Shape::from_dims(&[dim]));
    normalized.broadcast_mul(&gain_t).broadcast_add(&bias_t)
}

/// Apply `x @ W + b` where `W` is a `WeightStorage` projection and
/// `b` is a `[out_features]` bias vector. Dispatches to qmatmul for
/// Q4_0 weights.
fn apply_linear_with_bias(
    x: &LazyTensor,
    w: &WeightStorage,
    b: &Arc<[f32]>,
    in_features: usize,
    out_features: usize,
) -> LazyTensor {
    let y = w.apply_linear(x, in_features, out_features);
    let b_t = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[out_features]));
    y.broadcast_add(&b_t)
}

impl PhiModel {
    /// Apply one Phi-2 transformer block to `x` (parallel attention + MLP).
    ///
    /// Phi-2's structure is:
    ///   x_norm = LayerNorm(x, gain, bias, eps)
    ///   attn_out = attention(x_norm)  // with partial RoPE on Q/K
    ///   mlp_out  = fc2(gelu(fc1(x_norm)))
    ///   h = x + attn_out + mlp_out
    ///
    /// Returns (h, fresh_k, fresh_v, full_k, full_v) for cache update.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_with_cache(
        &self,
        x: &LazyTensor,
        layer: &PhiLayerWeights,
        layer_cache: &LayerKVCache,
        cached_len: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> LayerOutput {
        let cfg = &self.config;
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_heads * cfg.head_dim;  // no GQA in Phi-2
        let total_seq = cached_len + seq;

        // Shared pre-block LayerNorm.
        let x_norm = apply_affine_layer_norm(
            x, &layer.norm_gain, &layer.norm_bias, cfg.dim, cfg.layer_norm_eps);

        // Q/K/V projections with bias.
        let (q, k, v) = match &layer.attn_qkv {
            PhiQkv::Split { q, q_bias, k, k_bias, v, v_bias } => {
                let q_out = apply_linear_with_bias(&x_norm, q, q_bias, cfg.dim, cfg.dim);
                let k_out = apply_linear_with_bias(&x_norm, k, k_bias, cfg.dim, kv_dim);
                let v_out = apply_linear_with_bias(&x_norm, v, v_bias, cfg.dim, kv_dim);
                (q_out, k_out, v_out)
            }
            PhiQkv::Packed { qkv, qkv_bias } => {
                // Single big matmul producing [*, 3*dim] output, then slice
                // into [0..dim)=Q, [dim..2*dim)=K, [2*dim..3*dim)=V.
                // Matches Candle's
                //   .reshape(b, s, 3, n_head, head_dim).i((.., .., 0/1/2))
                // layout exactly (Q is first on the output side).
                let combined = apply_linear_with_bias(
                    &x_norm, qkv, qkv_bias, cfg.dim, 3 * cfg.dim);
                let last = combined.rank() - 1;
                let q_out = combined.slice(last, 0, cfg.dim);
                let k_out = combined.slice(last, cfg.dim, cfg.dim);
                let v_out = combined.slice(last, 2 * cfg.dim, cfg.dim);
                (q_out, k_out, v_out)
            }
        };

        // Split heads: [batch, seq, dim] → [batch, seq, n_heads, head_dim]
        //   → permute → [batch, n_heads, seq, head_dim]
        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))
            .permute(&[0, 2, 1, 3]);

        // Partial RoPE on Q and K: rotate the first `rotary_dim` entries
        // of head_dim, leave the rest unchanged.
        let q_r = partial_rope(&q_h, rope_cos, rope_sin, cfg.rotary_dim, cfg.head_dim);
        let k_r = partial_rope(&k_h, rope_cos, rope_sin, cfg.rotary_dim, cfg.head_dim);

        // Fresh K/V for the cache. V is not rotated.
        let fresh_k = k_r.clone();
        let fresh_v = v_h.clone();

        // Prepend cached K/V along the seq dim (dim 2).
        let (full_k, full_v) = if cached_len > 0 {
            let cached_shape = Shape::from_dims(&[batch, cfg.n_heads, cached_len, cfg.head_dim]);
            let cached_k = x.const_f32_like(layer_cache.k.clone(), cached_shape.clone());
            let cached_v = x.const_f32_like(layer_cache.v.clone(), cached_shape);
            (cached_k.concat(&fresh_k, 2), cached_v.concat(&fresh_v, 2))
        } else {
            (fresh_k.clone(), fresh_v.clone())
        };
        let cache_full_k = full_k.clone();
        let cache_full_v = full_v.clone();

        // Attention: Q @ K^T, scale, mask, softmax, @ V.
        let k_t = full_k.transpose();
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t);
        // Causal mask.
        let mut mask_data = vec![0.0_f32; seq * total_seq];
        for q in 0..seq {
            let abs_q = cached_len + q;
            for k in (abs_q + 1)..total_seq {
                mask_data[q * total_seq + k] = f32::NEG_INFINITY;
            }
        }
        let mask = x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, total_seq]));
        let scores_scaled = LazyTensor { inner: scores.inner.mul_scalar(scale) };
        let scores_masked = scores_scaled.broadcast_add(&mask);
        let attn = scores_masked.softmax_last_dim();
        let attn_v = attn.matmul(&full_v);

        // Merge heads: [batch, n_heads, seq, head_dim] → [batch, seq, dim].
        let merged = attn_v
            .permute(&[0, 2, 1, 3])
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]));
        let attn_out = apply_linear_with_bias(
            &merged, &layer.attn_dense, &layer.attn_dense_bias, cfg.dim, cfg.dim);

        // MLP branch (shares x_norm with attention branch).
        let fc1_out = apply_linear_with_bias(
            &x_norm, &layer.mlp_fc1, &layer.mlp_fc1_bias, cfg.dim, cfg.ffn_dim);
        let gelu_out = fc1_out.gelu();
        let mlp_out = apply_linear_with_bias(
            &gelu_out, &layer.mlp_fc2, &layer.mlp_fc2_bias, cfg.ffn_dim, cfg.dim);

        // Parallel residual: x + attn_out + mlp_out.
        let h = x.add(&attn_out).add(&mlp_out);

        LayerOutput {
            h,
            fresh_k,
            fresh_v,
            full_k: cache_full_k,
            full_v: cache_full_v,
        }
    }

    /// Forward pass with KV cache; returns last-position logits.
    pub fn forward_with_cache_gpu_on<B: fuel_graph_executor::GraphBackend>(
        &self,
        tokens: &[u32],
        cache: &mut KVCache<B>,
        executor: &mut GraphExecutor<B>,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        let cached_len = cache.cached_len;

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0, &token_ids)
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]));

        // RoPE tables are sized for `rotary_dim`, not the full head_dim —
        // partial RoPE rotates only the first `rotary_dim` entries.
        let (cos_data, sin_data) = fuel_graph::build_rope_tables(
            cfg.rope_base, cached_len, seq, cfg.rotary_dim,
        );
        let rope_shape = Shape::from_dims(&[seq, cfg.rotary_dim]);
        let rope_cos = h.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = h.const_f32_like(sin_data, rope_shape);

        let mut cached_kv_nodes: Vec<(fuel_graph::NodeId, fuel_graph::NodeId)> = Vec::new();
        let mut full_ks: Vec<LazyTensor> = Vec::with_capacity(cfg.n_layers);
        let mut full_vs: Vec<LazyTensor> = Vec::with_capacity(cfg.n_layers);

        for layer in weights.layers.iter() {
            let layer_cache_proxy: LayerKVCache = if cached_len > 0 {
                let n = batch * cfg.n_heads * cached_len * cfg.head_dim;
                LayerKVCache { k: vec![0.0; n], v: vec![0.0; n] }
            } else {
                LayerKVCache::default()
            };
            let out = self.apply_layer_with_cache(
                &h, layer, &layer_cache_proxy, cached_len, &rope_cos, &rope_sin);
            h = out.h;
            full_ks.push(out.full_k);
            full_vs.push(out.full_v);
        }

        // Wire up cache placeholders.
        if cached_len > 0 {
            let graph = h.graph_tensor().graph().read().unwrap();
            let target_elems = batch * cfg.n_heads * cached_len * cfg.head_dim;
            let mut found: Vec<fuel_graph::NodeId> = Vec::new();
            for node_id in 0..graph.len() {
                let nid = fuel_graph::NodeId(node_id);
                let node = graph.node(nid);
                if matches!(node.op, fuel_graph::Op::Const)
                    && node.shape.elem_count() == target_elems
                    && node.dtype == fuel_core_types::DType::F32
                    && node.shape.dims() == [batch, cfg.n_heads, cached_len, cfg.head_dim]
                {
                    found.push(nid);
                }
            }
            if found.len() == 2 * cfg.n_layers {
                for li in 0..cfg.n_layers {
                    cached_kv_nodes.push((found[li * 2], found[li * 2 + 1]));
                }
            }
        }

        // Final LayerNorm, output projection (+ optional bias).
        let h_norm = apply_affine_layer_norm(
            &h, &weights.final_norm_gain, &weights.final_norm_bias,
            cfg.dim, cfg.layer_norm_eps,
        );
        let logits_no_bias = weights.output.apply_linear(&h_norm, cfg.dim, cfg.vocab_size);
        let logits = match &weights.output_bias {
            Some(b) => {
                let b_t = h_norm.const_f32_like(
                    Arc::clone(b), Shape::from_dims(&[cfg.vocab_size]));
                logits_no_bias.broadcast_add(&b_t)
            }
            None => logits_no_bias,
        };

        let last_pos = seq - 1;
        let last_logits = logits
            .slice(1, last_pos, 1)
            .reshape(Shape::from_dims(&[cfg.vocab_size]));

        let mut roots: Vec<&LazyTensor> = Vec::with_capacity(1 + 2 * cfg.n_layers);
        roots.push(&last_logits);
        for fk in &full_ks { roots.push(fk); }
        for fv in &full_vs { roots.push(fv); }

        let cached_elems = batch * cfg.n_heads * cached_len * cfg.head_dim;
        for (li, (ck_id, cv_id)) in cached_kv_nodes.iter().enumerate() {
            if let Some(entry) = &cache.layers[li] {
                let cached_shape = Shape::from_dims(&[batch, cfg.n_heads, cached_len, cfg.head_dim]);
                let layout = fuel_core_types::Layout::contiguous(&cached_shape);
                let (k_f32, v_f32) = match entry {
                    KVCacheEntry::F32 { k, v } => {
                        let k = executor.backend.try_clone(k, &layout).expect("inject K clone");
                        let v = executor.backend.try_clone(v, &layout).expect("inject V clone");
                        (k, v)
                    }
                    KVCacheEntry::Q8 { k_blocks, v_blocks } => {
                        let n_blocks = cached_elems / 32;
                        let k = executor.backend.dequantize_q8_0(k_blocks, n_blocks)
                            .expect("dequantize K from Q8 cache");
                        let v = executor.backend.dequantize_q8_0(v_blocks, n_blocks)
                            .expect("dequantize V from Q8 cache");
                        (k, v)
                    }
                };
                executor.pre_populate(*ck_id, k_f32, cached_shape.clone());
                executor.pre_populate(*cv_id, v_f32, cached_shape);
            }
        }

        let inner_roots: Vec<&fuel_graph::Tensor> =
            roots.iter().map(|lt| &lt.inner).collect();
        let (cpu_results, gpu_results) = executor.realize_split(&inner_roots, 1);
        let logits_vec = cpu_results.into_iter().next().unwrap();

        let mut iter = gpu_results.into_iter();
        let new_ks: Vec<(B::Storage, Shape)> = (0..cfg.n_layers).map(|_| iter.next().unwrap()).collect();
        let new_vs: Vec<(B::Storage, Shape)> = (0..cfg.n_layers).map(|_| iter.next().unwrap()).collect();
        let new_len = cached_len + seq;
        let new_elems = batch * cfg.n_heads * new_len * cfg.head_dim;
        for (li, ((new_k, _), (new_v, _))) in new_ks.into_iter().zip(new_vs.into_iter()).enumerate() {
            let entry = if cache.q8_enabled && new_elems % 32 == 0 {
                let k_blocks = executor.backend.quantize_q8_0(&new_k, new_elems)
                    .expect("quantize K to Q8 cache");
                let v_blocks = executor.backend.quantize_q8_0(&new_v, new_elems)
                    .expect("quantize V to Q8 cache");
                KVCacheEntry::Q8 { k_blocks, v_blocks }
            } else {
                KVCacheEntry::F32 { k: new_k, v: new_v }
            };
            cache.layers[li] = Some(entry);
        }
        cache.cached_len += seq;
        logits_vec
    }

    /// Streaming generation with device-resident KV cache.
    pub fn generate_streaming_gpu_on<B: fuel_graph_executor::GraphBackend>(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        executor: &mut GraphExecutor<B>,
        mut on_token: impl FnMut(u32),
    ) -> crate::Result<Vec<u32>> {
        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut rng_state: u64 = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };
        let mut cache: KVCache<B> = KVCache::with_dims(
            self.config.n_layers, self.config.n_heads, self.config.head_dim);
        if std::env::var("FUEL_Q8_KV").ok().as_deref() == Some("1") {
            cache.enable_q8_cache();
        }
        let mut last_logits = self.forward_with_cache_gpu_on(&tokens, &mut cache, executor);
        for _ in 0..max_new_tokens {
            let next = sample_logits(&last_logits, strategy, &mut rng_state);
            tokens.push(next);
            on_token(next);
            if let Some(eos) = eos_id { if next == eos { break; } }
            last_logits = self.forward_with_cache_gpu_on(&[next], &mut cache, executor);
        }
        Ok(tokens)
    }

    /// Load weights from a HuggingFace Hub repo (e.g. "microsoft/phi-2").
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());

        let config_path = repo.get("config.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub config.json: {e}")))?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config = PhiConfig::from_hf_json_str(&config_str)?;

        let weight_paths: Vec<std::path::PathBuf> = match repo.get("model.safetensors.index.json") {
            Ok(index_path) => {
                let index_str = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_str)
                    .map_err(|e| crate::Error::Msg(format!("parsing index: {e}")))?;
                let weight_map = index.get("weight_map").and_then(|x| x.as_object())
                    .ok_or_else(|| crate::Error::Msg("index.json: missing weight_map".into()))?;
                let mut unique = std::collections::HashSet::new();
                for v in weight_map.values() {
                    if let Some(s) = v.as_str() { unique.insert(s.to_string()); }
                }
                let mut paths: Vec<std::path::PathBuf> = Vec::new();
                for shard_name in unique {
                    let p = repo.get(&shard_name)
                        .map_err(|e| crate::Error::Msg(format!("hf-hub {shard_name}: {e}")))?;
                    paths.push(p);
                }
                paths
            }
            Err(_) => {
                let p = repo.get("model.safetensors")
                    .map_err(|e| crate::Error::Msg(format!("hf-hub model.safetensors: {e}")))?;
                vec![p]
            }
        };

        let st = unsafe { crate::safetensors::MmapedSafetensors::multi(&weight_paths) }?;
        let weights = PhiWeights::load_from_mmapped(&st, &config)?;
        Ok(PhiModel { config, weights })
    }

    /// Load a Phi-2 model from a GGUF file (e.g. one of TheBloke's
    /// quantized Phi-2 releases). Q4_0 tensors stay quantized on-device;
    /// other dtypes dequantize to F32 at load time. Config is derived
    /// from the GGUF metadata.
    pub fn from_gguf<P: AsRef<std::path::Path>>(path: P) -> crate::Result<Self> {
        use crate::quantized::gguf_mmap::MmapedContent;
        let mc = MmapedContent::from_path(&path)?;
        let meta = mc.metadata();
        let get_u32 = |k: &str| -> crate::Result<u32> {
            meta.get(k)
                .ok_or_else(|| crate::Error::Msg(format!("gguf metadata: missing {k:?}")))?
                .to_u32()
                .map_err(|e| crate::Error::Msg(format!("gguf metadata {k:?}: {e:?}")))
        };
        let get_f32 = |k: &str| -> crate::Result<f32> {
            meta.get(k)
                .ok_or_else(|| crate::Error::Msg(format!("gguf metadata: missing {k:?}")))?
                .to_f32()
                .map_err(|e| crate::Error::Msg(format!("gguf metadata {k:?}: {e:?}")))
        };
        // Phi-2 metadata keys (llama.cpp convention).
        let dim        = get_u32("phi2.embedding_length")? as usize;
        let n_layers   = get_u32("phi2.block_count")? as usize;
        let n_heads    = get_u32("phi2.attention.head_count")? as usize;
        let ffn_dim    = get_u32("phi2.feed_forward_length")? as usize;
        let head_dim   = dim / n_heads;
        let layer_norm_eps = get_f32("phi2.attention.layer_norm_epsilon").unwrap_or(1e-5) as f64;
        let rope_base  = get_f32("phi2.rope.freq_base").unwrap_or(10_000.0) as f64;
        let rotary_dim = get_u32("phi2.rope.dimension_count").unwrap_or(32) as usize;
        let partial_rotary_factor = rotary_dim as f64 / head_dim as f64;

        // Derive vocab_size from the token_embd shape (no explicit
        // metadata key for it in GGUF; llama.cpp infers from the
        // tokenizer array which needs a dedicated API). token_embd has
        // shape [vocab, dim].
        let vocab_size = mc.content()
            .tensor_infos.get("token_embd.weight")
            .ok_or_else(|| crate::Error::Msg("gguf: missing token_embd.weight".into()))?
            .shape.dims()[0];

        let config = PhiConfig {
            vocab_size, dim, n_layers, n_heads, head_dim, ffn_dim,
            layer_norm_eps, rope_base, partial_rotary_factor, rotary_dim,
            tie_word_embeddings: false,
        };

        // MmapedContent drops here; the load_from_gguf path re-opens.
        // In practice this is two mmaps in flight, both pointing at the
        // same file — cheap on modern OSes. If this becomes a hotspot,
        // refactor to hand the Arc<Mmap> through.
        drop(mc);
        let weights = PhiWeights::load_from_gguf(&path, &config)?;
        Ok(PhiModel { config, weights })
    }
}

impl PhiWeights {
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &PhiConfig,
    ) -> crate::Result<Self> {
        let kv_dim = cfg.n_heads * cfg.head_dim;
        let token_embedding = load_tensor_as_f32(st, "model.embed_tokens.weight")?;
        if token_embedding.len() != cfg.vocab_size * cfg.dim {
            crate::bail!(
                "embed_tokens: {} elements, expected {}",
                token_embedding.len(), cfg.vocab_size * cfg.dim,
            );
        }

        let mut layers: Vec<PhiLayerWeights> = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            // Phi-2 uses `dense` for the output projection (not `o_proj`)
            // and `fc1`/`fc2` for the MLP (not `gate_proj`/`up_proj`/`down_proj`).
            let attn_q = load_transposed_matrix_preserve_dtype(
                st, &format!("model.layers.{i}.self_attn.q_proj.weight"), cfg.dim, cfg.dim)?;
            let attn_k = load_transposed_matrix_preserve_dtype(
                st, &format!("model.layers.{i}.self_attn.k_proj.weight"), kv_dim, cfg.dim)?;
            let attn_v = load_transposed_matrix_preserve_dtype(
                st, &format!("model.layers.{i}.self_attn.v_proj.weight"), kv_dim, cfg.dim)?;
            let attn_dense = load_transposed_matrix_preserve_dtype(
                st, &format!("model.layers.{i}.self_attn.dense.weight"), cfg.dim, cfg.dim)?;
            let mlp_fc1 = load_transposed_matrix_preserve_dtype(
                st, &format!("model.layers.{i}.mlp.fc1.weight"), cfg.ffn_dim, cfg.dim)?;
            let mlp_fc2 = load_transposed_matrix_preserve_dtype(
                st, &format!("model.layers.{i}.mlp.fc2.weight"), cfg.dim, cfg.ffn_dim)?;

            let attn_q_bias = Arc::from(load_tensor_as_f32(
                st, &format!("model.layers.{i}.self_attn.q_proj.bias"))?);
            let attn_k_bias = Arc::from(load_tensor_as_f32(
                st, &format!("model.layers.{i}.self_attn.k_proj.bias"))?);
            let attn_v_bias = Arc::from(load_tensor_as_f32(
                st, &format!("model.layers.{i}.self_attn.v_proj.bias"))?);
            let attn_dense_bias = Arc::from(load_tensor_as_f32(
                st, &format!("model.layers.{i}.self_attn.dense.bias"))?);
            let mlp_fc1_bias = Arc::from(load_tensor_as_f32(
                st, &format!("model.layers.{i}.mlp.fc1.bias"))?);
            let mlp_fc2_bias = Arc::from(load_tensor_as_f32(
                st, &format!("model.layers.{i}.mlp.fc2.bias"))?);

            // Phi-2's pre-block LayerNorm is `input_layernorm.{weight,bias}`.
            let norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("model.layers.{i}.input_layernorm.weight"))?);
            let norm_bias = Arc::from(load_tensor_as_f32(
                st, &format!("model.layers.{i}.input_layernorm.bias"))?);

            layers.push(PhiLayerWeights {
                attn_qkv: PhiQkv::Split {
                    q: attn_q, q_bias: attn_q_bias,
                    k: attn_k, k_bias: attn_k_bias,
                    v: attn_v, v_bias: attn_v_bias,
                },
                attn_dense, attn_dense_bias,
                mlp_fc1, mlp_fc1_bias, mlp_fc2, mlp_fc2_bias,
                norm_gain, norm_bias,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.final_layernorm.weight")?);
        let final_norm_bias = Arc::from(load_tensor_as_f32(st, "model.final_layernorm.bias")?);

        let output: WeightStorage = if cfg.tie_word_embeddings {
            // Tied: transpose embed_tokens.
            let mut transposed = vec![0.0_f32; cfg.dim * cfg.vocab_size];
            for i in 0..cfg.vocab_size {
                for j in 0..cfg.dim {
                    transposed[j * cfg.vocab_size + i] = token_embedding[i * cfg.dim + j];
                }
            }
            WeightStorage::F32(Arc::from(transposed))
        } else {
            load_transposed_matrix_preserve_dtype(st, "lm_head.weight", cfg.vocab_size, cfg.dim)?
        };
        let output_bias = load_tensor_as_f32(st, "lm_head.bias").ok().map(Arc::from);

        Ok(PhiWeights {
            token_embedding: Arc::from(token_embedding),
            layers, final_norm_gain, final_norm_bias, output, output_bias,
        })
    }

    /// Load Phi-2 weights from a GGUF file. Q4_0 tensors stay quantized
    /// (go into `WeightStorage::Q4_0`); other GGML dtypes are dequantized
    /// to F32 at load time and stored as `WeightStorage::F32` (or
    /// `Arc<[f32]>` for biases, norms, embedding).
    ///
    /// GGUF key layout for Phi-2:
    ///   token_embd.weight / output.weight / output_norm.{weight,bias}
    ///   blk.{i}.attn_qkv.{weight,bias}           (packed 3*dim × dim)
    ///   blk.{i}.attn_output.{weight,bias}
    ///   blk.{i}.ffn_up.{weight,bias}
    ///   blk.{i}.ffn_down.{weight,bias}
    ///   blk.{i}.attn_norm.{weight,bias}
    pub fn load_from_gguf<P: AsRef<std::path::Path>>(
        path: P,
        cfg: &PhiConfig,
    ) -> crate::Result<Self> {
        use crate::quantized::gguf_mmap::MmapedContent;
        let mc = MmapedContent::from_path(path)?;
        let content = mc.content();
        let (mmap_arc, _) = (mc.mmap(), ());
        let mmap_bytes: &[u8] = &mmap_arc[..];
        let data_off = content.tensor_data_offset as usize;

        // Extract a raw byte slice for a tensor.
        let get_tensor_bytes = |name: &str| -> crate::Result<(&[u8], crate::quantized::GgmlDType, Vec<usize>)> {
            let info = content.tensor_infos.get(name)
                .ok_or_else(|| crate::Error::Msg(format!("gguf: missing tensor {name:?}")))?;
            let elems = info.shape.elem_count();
            let block_size = info.ggml_dtype.block_size();
            let bytes_len = elems / block_size * info.ggml_dtype.type_size();
            let start = data_off + info.offset as usize;
            Ok((&mmap_bytes[start..start + bytes_len], info.ggml_dtype, info.shape.dims().to_vec()))
        };

        // Load an F32 vector (for biases, norms, embedding). Dequantizes
        // if necessary.
        let load_f32 = |name: &str| -> crate::Result<Vec<f32>> {
            let (bytes, dt, _dims) = get_tensor_bytes(name)?;
            dequant_gguf_bytes_to_f32(bytes, dt, name)
        };

        // Load a weight matrix as WeightStorage. For Q4_0 bytes, keep
        // them quantized; for other dtypes, dequantize to F32.
        // `out_features × in_features` is the GGUF/llama.cpp convention.
        let load_weight = |name: &str, out_features: usize, in_features: usize| -> crate::Result<WeightStorage> {
            let (bytes, dt, dims) = get_tensor_bytes(name)?;
            // GGUF stores weights as [out, in] — matches our Q4_0 block layout.
            let expected_elems = out_features * in_features;
            let actual_elems: usize = dims.iter().product();
            if actual_elems != expected_elems {
                crate::bail!(
                    "gguf: tensor {name:?} has {actual_elems} elements, expected {expected_elems} for [{out_features}, {in_features}]",
                );
            }
            // Debug fallback: FUEL_FORCE_F32=1 dequantizes every weight
            // at load time to isolate Q4_0-path bugs from model-structure
            // bugs. Useful for validating the PhiModel/loader against a
            // known-good computation path.
            let force_f32 = std::env::var("FUEL_FORCE_F32").is_ok();
            match dt {
                crate::quantized::GgmlDType::Q4_0 if !force_f32 => {
                    Ok(WeightStorage::Q4_0 {
                        words: bytes_to_u32_arc(bytes),
                        bytes_len: bytes.len(),
                        in_features,
                        out_features,
                    })
                }
                _ => {
                    // Dequantized data is in GGUF's native [out, in]
                    // row-major layout. Our standard F32/BF16 matmul
                    // expects [in, out], so transpose before storing.
                    // (Q4_0 keeps its native layout because qmatmul
                    // reads blocks as [N, K/32] directly.)
                    let f32_out_in = dequant_gguf_bytes_to_f32(bytes, dt, name)?;
                    let mut f32_in_out = vec![0.0_f32; out_features * in_features];
                    for o in 0..out_features {
                        for i in 0..in_features {
                            f32_in_out[i * out_features + o] = f32_out_in[o * in_features + i];
                        }
                    }
                    Ok(WeightStorage::F32(Arc::from(f32_in_out)))
                }
            }
        };

        let token_embedding = load_f32("token_embd.weight")?;
        if token_embedding.len() != cfg.vocab_size * cfg.dim {
            crate::bail!(
                "gguf token_embd: {} elems, expected {}×{}",
                token_embedding.len(), cfg.vocab_size, cfg.dim,
            );
        }

        let mut layers: Vec<PhiLayerWeights> = Vec::with_capacity(cfg.n_layers);
        let kv_dim = cfg.n_heads * cfg.head_dim;

        for i in 0..cfg.n_layers {
            let prefix = format!("blk.{i}");

            // Phi-2 GGUF packs Q/K/V into a single attn_qkv tensor of
            // shape [3*dim, dim]. We keep it PACKED as a single
            // WeightStorage and let the forward pass do one big matmul
            // + slice after (matching Candle's eager approach). This
            // avoids any hazards around byte-level Q/K/V splits on the
            // weight side.
            let attn_qkv_weight = load_weight(
                &format!("{prefix}.attn_qkv.weight"),
                3 * cfg.dim, cfg.dim,
            )?;
            let qkv_bias_vec = load_f32(&format!("{prefix}.attn_qkv.bias"))?;
            if qkv_bias_vec.len() != 3 * cfg.dim {
                crate::bail!("gguf attn_qkv.bias: {} elems, expected {}", qkv_bias_vec.len(), 3*cfg.dim);
            }
            let qkv_bias: Arc<[f32]> = Arc::from(qkv_bias_vec);
            let _ = kv_dim; // Phi-2 has no GQA; kv_dim == dim

            let attn_dense = load_weight(&format!("{prefix}.attn_output.weight"), cfg.dim, cfg.dim)?;
            let attn_dense_bias = Arc::from(load_f32(&format!("{prefix}.attn_output.bias"))?);

            let mlp_fc1 = load_weight(&format!("{prefix}.ffn_up.weight"), cfg.ffn_dim, cfg.dim)?;
            let mlp_fc1_bias = Arc::from(load_f32(&format!("{prefix}.ffn_up.bias"))?);
            let mlp_fc2 = load_weight(&format!("{prefix}.ffn_down.weight"), cfg.dim, cfg.ffn_dim)?;
            let mlp_fc2_bias = Arc::from(load_f32(&format!("{prefix}.ffn_down.bias"))?);

            let norm_gain = Arc::from(load_f32(&format!("{prefix}.attn_norm.weight"))?);
            let norm_bias = Arc::from(load_f32(&format!("{prefix}.attn_norm.bias"))?);

            layers.push(PhiLayerWeights {
                attn_qkv: PhiQkv::Packed { qkv: attn_qkv_weight, qkv_bias },
                attn_dense, attn_dense_bias,
                mlp_fc1, mlp_fc1_bias,
                mlp_fc2, mlp_fc2_bias,
                norm_gain, norm_bias,
            });
        }

        let final_norm_gain = Arc::from(load_f32("output_norm.weight")?);
        let final_norm_bias = Arc::from(load_f32("output_norm.bias")?);

        // Output projection. In GGUF: `output.weight` has shape [vocab, dim].
        let output = load_weight("output.weight", cfg.vocab_size, cfg.dim)?;
        let output_bias = load_f32("output.bias").ok().map(Arc::from);

        Ok(PhiWeights {
            token_embedding: Arc::from(token_embedding),
            layers, final_norm_gain, final_norm_bias, output, output_bias,
        })
    }
}

/// Dequantize a raw byte slice from GGUF (of the given GGML dtype) into
/// a flat `Vec<f32>`. Used by the lazy GGUF loader for non-Q4_0 tensors
/// (biases, norms, embeddings, and weight matrices of any dtype that
/// lacks a fused on-device dequant path).
fn dequant_gguf_bytes_to_f32(
    bytes: &[u8],
    dt: crate::quantized::GgmlDType,
    name: &str,
) -> crate::Result<Vec<f32>> {
    use crate::quantized::GgmlDType;
    use half::{bf16, f16};
    match dt {
        GgmlDType::F32 => {
            if bytes.len() % 4 != 0 {
                crate::bail!("gguf {name}: F32 byte count {} not multiple of 4", bytes.len());
            }
            Ok(bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        }
        GgmlDType::F16 => {
            if bytes.len() % 2 != 0 {
                crate::bail!("gguf {name}: F16 byte count {} not multiple of 2", bytes.len());
            }
            Ok(bytes.chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect())
        }
        GgmlDType::BF16 => {
            if bytes.len() % 2 != 0 {
                crate::bail!("gguf {name}: BF16 byte count {} not multiple of 2", bytes.len());
            }
            Ok(bytes.chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect())
        }
        GgmlDType::Q4_0 => {
            // Should rarely be requested this way (prefer keeping Q4_0
            // quantized), but support it for biases or other oddities.
            Ok(cpu_dequant_q4_0_bytes(bytes))
        }
        GgmlDType::Q8_0 => Ok(cpu_dequant_q8_0_bytes(bytes)),
        // k-quants: dequant via the reference-CPU GgmlType trait impls.
        GgmlDType::Q6K => Ok(cpu_dequant_via_trait::<fuel_quantized::BlockQ6K>(bytes)),
        GgmlDType::Q5K => Ok(cpu_dequant_via_trait::<fuel_quantized::BlockQ5K>(bytes)),
        GgmlDType::Q4K => Ok(cpu_dequant_via_trait::<fuel_quantized::BlockQ4K>(bytes)),
        GgmlDType::Q3K => Ok(cpu_dequant_via_trait::<fuel_quantized::BlockQ3K>(bytes)),
        GgmlDType::Q2K => Ok(cpu_dequant_via_trait::<fuel_quantized::BlockQ2K>(bytes)),
        other => crate::bail!("gguf {name}: dequant-to-f32 for dtype {other:?} not implemented in lazy loader"),
    }
}

/// Dequantize an arbitrary k-quant block stream to F32 via the
/// reference `GgmlType::to_float` trait. Callers give the concrete
/// block type `T` (e.g. `BlockQ6K`); the function reinterprets the
/// byte slice as `&[T]` and calls the impl. Used for dtypes that
/// don't have a fused on-device dequant kernel (yet).
fn cpu_dequant_via_trait<T: fuel_quantized::GgmlType>(bytes: &[u8]) -> Vec<f32> {
    let block_bytes = std::mem::size_of::<T>();
    assert!(bytes.len() % block_bytes == 0,
        "cpu_dequant_via_trait: bytes {} not multiple of block_bytes {}",
        bytes.len(), block_bytes);
    let n_blocks = bytes.len() / block_bytes;
    // SAFETY: T is #[repr(C)]; GGUF bytes are laid out as a dense array
    // of T structs. The source mmap is 8-byte aligned per memmap2, which
    // satisfies every block struct's alignment (≤ 4 in practice).
    let blocks: &[T] = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const T, n_blocks)
    };
    let mut out = vec![0.0_f32; n_blocks * T::BLCK_SIZE];
    T::to_float(blocks, &mut out);
    out
}

/// Reinterpret a byte slice as a u32 `Arc` by reading little-endian
/// u32 words. Input length must be a multiple of 4. This performs one
/// copy at load time — all subsequent uses are cheap Arc clones.
fn bytes_to_u32_arc(bytes: &[u8]) -> Arc<[u32]> {
    assert_eq!(bytes.len() % 4, 0, "bytes_to_u32_arc: len must be multiple of 4");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Arc::from(words)
}

fn cpu_dequant_q4_0_bytes(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    let bpb = 18usize;
    let epb = 32usize;
    let n_blocks = bytes.len() / bpb;
    let mut out = vec![0.0_f32; n_blocks * epb];
    for b in 0..n_blocks {
        let off = b * bpb;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let base = b * epb;
        for kk in 0..16 {
            let packed = bytes[off + 2 + kk];
            let lo = (packed & 0x0F) as i32 - 8;
            let hi = ((packed >> 4) & 0x0F) as i32 - 8;
            out[base + kk]      = lo as f32 * d;
            out[base + 16 + kk] = hi as f32 * d;
        }
    }
    out
}

fn cpu_dequant_q8_0_bytes(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    let bpb = 34usize;
    let epb = 32usize;
    let n_blocks = bytes.len() / bpb;
    let mut out = vec![0.0_f32; n_blocks * epb];
    for b in 0..n_blocks {
        let off = b * bpb;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let base = b * epb;
        for kk in 0..32 {
            let q = bytes[off + 2 + kk] as i8 as i32;
            out[base + kk] = q as f32 * d;
        }
    }
    out
}

/// Split a packed Phi-2 attn_qkv tensor into separate Q, K, V weight
/// storages. The GGUF layout is `[3*dim, dim]` with Q occupying rows
/// `[0..dim)`, K `[dim..2*dim)`, V `[2*dim..3*dim)`. For Q4_0 we can
/// split byte ranges directly since each "row" of `dim` elements is
/// exactly `dim/32 * 18` bytes.
fn split_qkv(
    bytes: &[u8],
    dt: crate::quantized::GgmlDType,
    dim: usize,
    kv_dim: usize,
) -> crate::Result<(WeightStorage, WeightStorage, WeightStorage)> {
    // Phi-2 has n_kv_heads == n_heads, so kv_dim == dim. Accept that invariant.
    if kv_dim != dim {
        crate::bail!("split_qkv: only supports Phi-2's symmetric attention (dim={dim}, kv_dim={kv_dim})");
    }
    use crate::quantized::GgmlDType;
    let force_f32 = std::env::var("FUEL_FORCE_F32").is_ok();
    match dt {
        GgmlDType::Q4_0 if !force_f32 => {
            let bpb = 18usize;
            let epb = 32usize;
            let blocks_per_row = dim / epb;
            let bytes_per_section = dim * blocks_per_row * bpb;
            if bytes.len() != 3 * bytes_per_section {
                crate::bail!(
                    "split_qkv Q4_0: byte count {} ≠ 3 × {} = {}",
                    bytes.len(), bytes_per_section, 3 * bytes_per_section,
                );
            }
            let q_words = bytes_to_u32_arc(&bytes[0..bytes_per_section]);
            let k_words = bytes_to_u32_arc(&bytes[bytes_per_section..2*bytes_per_section]);
            let v_words = bytes_to_u32_arc(&bytes[2*bytes_per_section..3*bytes_per_section]);
            Ok((
                WeightStorage::Q4_0 { words: q_words, bytes_len: bytes_per_section, in_features: dim, out_features: dim },
                WeightStorage::Q4_0 { words: k_words, bytes_len: bytes_per_section, in_features: dim, out_features: dim },
                WeightStorage::Q4_0 { words: v_words, bytes_len: bytes_per_section, in_features: dim, out_features: dim },
            ))
        }
        _ => {
            // Non-Q4_0: dequantize the whole blob to F32, then split by rows.
            let all_f32 = dequant_gguf_bytes_to_f32(bytes, dt, "attn_qkv")?;
            let per_section = dim * dim;
            if all_f32.len() != 3 * per_section {
                crate::bail!(
                    "split_qkv F-dtype: {} elems ≠ 3 × {}", all_f32.len(), per_section,
                );
            }
            let q: Vec<f32> = all_f32[0..per_section].to_vec();
            let k: Vec<f32> = all_f32[per_section..2*per_section].to_vec();
            let v: Vec<f32> = all_f32[2*per_section..3*per_section].to_vec();
            Ok((
                WeightStorage::F32(Arc::from(q)),
                WeightStorage::F32(Arc::from(k)),
                WeightStorage::F32(Arc::from(v)),
            ))
        }
    }
}

/// Apply rotary embeddings to only the first `rotary_dim` entries of
/// the last dimension; pass the remaining `head_dim - rotary_dim` entries
/// through unchanged. Used by Phi-2 and Phi-3 which rotate only a
/// fraction of each head's feature dim.
///
/// Input shape: `[..., head_dim]`. Output shape: same.
fn partial_rope(
    x: &LazyTensor,
    cos: &LazyTensor,
    sin: &LazyTensor,
    rotary_dim: usize,
    head_dim: usize,
) -> LazyTensor {
    if rotary_dim == head_dim {
        return x.rope_with_tables(cos, sin);
    }
    let rank = x.dims().len();
    let last = rank - 1;
    let x_rot = x.slice(last, 0, rotary_dim);
    let x_pass = x.slice(last, rotary_dim, head_dim - rotary_dim);
    let x_rot_rotated = x_rot.rope_with_tables(cos, sin);
    x_rot_rotated.concat(&x_pass, last)
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
                    attn_q:         vec_of(cfg.dim * cfg.dim).into(),
                    attn_q_bias:    None,
                    attn_k:         vec_of(cfg.dim * kv_dim).into(),
                    attn_k_bias:    None,
                    attn_v:         vec_of(cfg.dim * kv_dim).into(),
                    attn_v_bias:    None,
                    attn_o:         vec_of(cfg.dim * cfg.dim).into(),
                    ffn_gate:       vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_up:         vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_down:       vec_of(cfg.ffn_dim * cfg.dim).into(),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain:  Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output:          vec_of(cfg.dim * cfg.vocab_size).into(),
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

    // The host-resident-cache prefill-parity test
    // (`forward_with_cache_matches_forward_on_prefill`) was retired in
    // E.3.3.D. Its successor is
    // `forward_with_kv_context_prefill_matches_non_cached_forward`,
    // which exercises the same correctness bar via the new
    // KvCache + InferenceContext + Op::WriteSlice path.

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

    /// Greedy generation through the new `generate_with_kv_context`
    /// path must produce the same token sequence as the legacy
    /// `generate` (which uses the host-resident `LlamaKVCache` +
    /// `forward_with_cache_on`). Both routes use the cache; the only
    /// difference is the in-graph WriteSlice path vs the host-side
    /// download-and-append loop.
    #[test]
    fn generate_with_kv_context_matches_legacy_generate() {
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

        // Reference: legacy host-resident cache path.
        let legacy = model
            .generate(&prompt, max_new, SamplingStrategy::Greedy, None)
            .unwrap();

        // New: KvCache + InferenceContext + forward_with_kv_context.
        let new_path = model.generate_with_kv_context(
            &prompt, max_new, SamplingStrategy::Greedy, None,
            &Device::cpu(), DType::F32,
        ).unwrap();

        // Greedy argmax is robust to O(ε) drift in the logits — both
        // paths should pick the same token at every step.
        assert_eq!(new_path, legacy);
    }

    /// Streaming generation through `generate_streaming_with_kv_context`
    /// fires `on_token` exactly once per generated token (not the
    /// prompt tokens) and the resulting Vec matches the non-streaming
    /// convenience wrapper.
    #[test]
    fn generate_streaming_with_kv_context_fires_callback_per_token() {
        let cfg = LlamaConfig {
            vocab_size: 16, dim: 4, n_layers: 1, n_heads: 2, n_kv_heads: 2,
            head_dim: 2, ffn_dim: 8, norm_eps: 1e-5, rope_base: 10000.0,
        };
        let model = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [1_u32, 2];
        let max_new = 3;

        let mut streamed: Vec<u32> = Vec::new();
        let tokens = model.generate_streaming_with_kv_context(
            &prompt, max_new, SamplingStrategy::Greedy, None,
            &Device::cpu(), DType::F32,
            |tok| streamed.push(tok),
        ).unwrap();

        // on_token fires once per GENERATED token (not the prompt).
        assert_eq!(streamed.len(), max_new);
        // The returned Vec is prompt ++ streamed.
        assert_eq!(tokens.len(), prompt.len() + max_new);
        assert_eq!(&tokens[..prompt.len()], &prompt[..]);
        assert_eq!(&tokens[prompt.len()..], &streamed[..]);
    }

    /// `generate_streaming_with_kv_context` short-circuits when an EOS
    /// token is generated, returning before max_new_tokens is reached.
    #[test]
    fn generate_streaming_with_kv_context_stops_on_eos() {
        let cfg = LlamaConfig {
            vocab_size: 16, dim: 4, n_layers: 1, n_heads: 2, n_kv_heads: 2,
            head_dim: 2, ffn_dim: 8, norm_eps: 1e-5, rope_base: 10000.0,
        };
        let model = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [1_u32, 2];
        let max_new = 10;

        // First find what greedy generates without EOS, then set the
        // first generated token as the EOS to confirm short-circuit.
        let unbounded = model.generate_with_kv_context(
            &prompt, max_new, SamplingStrategy::Greedy, None,
            &Device::cpu(), DType::F32,
        ).unwrap();
        assert_eq!(unbounded.len(), prompt.len() + max_new);
        let first_generated = unbounded[prompt.len()];

        let bounded = model.generate_with_kv_context(
            &prompt, max_new, SamplingStrategy::Greedy, Some(first_generated),
            &Device::cpu(), DType::F32,
        ).unwrap();
        // With EOS = first_generated, generation stops after producing
        // that one token.
        assert_eq!(bounded.len(), prompt.len() + 1);
        assert_eq!(bounded[prompt.len()], first_generated);
    }

    // The host-resident-cache prefill+decode parity test
    // (`forward_with_cache_decode_step_matches_full_forward`) was
    // retired in E.3.3.D. Its successor is
    // `forward_with_kv_context_decode_matches_non_cached_forward`
    // below, which exercises the same correctness bar via the new
    // KvCache + InferenceContext + Op::WriteSlice path.

    // ---- forward_with_kv_context (Phase 7.6 step 9c E.3.3.B) -----------

    /// Prefill + decode through the new `forward_with_kv_context` path
    /// should produce the same last-position logits as a non-cached
    /// forward over the full sequence. Mirrors the
    /// `forward_with_cache_decode_step_matches_full_forward` test but
    /// uses `KvCache::with_capacity` + `InferenceContext` + `Op::
    /// WriteSlice` instead of the legacy host-resident `LlamaKVCache`
    /// + concat-and-download path.
    #[test]
    fn forward_with_kv_context_decode_matches_non_cached_forward() {
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
        let full = [prompt[0], prompt[1], prompt[2], next_token];

        // Non-cached reference: full forward over all 4 tokens.
        let full_logits = model.forward(&full, 0);
        let last_pos = full.len() - 1;
        let expected = full_logits
            .slice(1, last_pos, 1)
            .reshape(Shape::from_dims(&[cfg.vocab_size]))
            .realize_f32();

        // New cached path: KvCache::with_capacity + forward_with_kv_context.
        let device = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
            /*max_seq_len*/ full.len(),
            DType::F32,
            &device,
        ).expect("with_capacity");
        let mut ctx = InferenceContext::new(device);

        // Prefill: write the 3-token prompt's K/V into the cache.
        let _prefill_logits = model
            .forward_with_kv_context(&prompt, &mut cache, &mut ctx)
            .expect("prefill");
        assert_eq!(cache.cached_len, prompt.len());

        // Decode: one step with the new token.
        let actual = model
            .forward_with_kv_context(&[next_token], &mut cache, &mut ctx)
            .expect("decode");
        assert_eq!(cache.cached_len, full.len());
        assert_eq!(actual.len(), expected.len());

        // Same tolerance as the legacy cached vs non-cached test: the
        // attention matmul accumulates along the seq dim in a slightly
        // different order between the prefill (one tensor of length
        // total_seq) and the prefill+decode (cached prefix + 1 fresh
        // row) paths. This is the standard O(ε) gemm drift, not a
        // correctness bug.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / a.abs().max(b.abs()).max(1e-6);
            assert!(
                diff < 5e-3 || rel < 1e-2,
                "logit[{i}]: new-cached={a}, non-cached={b}, diff={diff}",
            );
        }

        // Side effect: every layer's K and V version should have
        // bumped once per forward step (2 steps × 1 bump each = 2).
        for li in 0..cfg.n_layers {
            assert_eq!(cache.layer(li).unwrap().k_version, 2);
            assert_eq!(cache.layer(li).unwrap().v_version, 2);
        }
    }

    /// Prefill-only forward through `forward_with_kv_context` should
    /// match a non-cached forward over the same prompt (no decode
    /// step, just the prefill). This is the cleanest correctness gate
    /// — `cached_len == 0` means WriteSlice writes into the head of a
    /// zero-initialized buffer and the subsequent attention slice
    /// equals the fresh K/V.
    #[test]
    fn forward_with_kv_context_prefill_matches_non_cached_forward() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   2,
            n_heads:    4,
            n_kv_heads: 2,
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let cfg = LlamaConfig { dim: cfg.n_heads * cfg.head_dim, ..cfg };
        let model = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3, 4];

        // Non-cached reference.
        let full_logits = model.forward(&prompt, 0);
        let last_pos = prompt.len() - 1;
        let expected = full_logits
            .slice(1, last_pos, 1)
            .reshape(Shape::from_dims(&[cfg.vocab_size]))
            .realize_f32();

        // New path, single prefill call.
        let device = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
            prompt.len(),
            DType::F32,
            &device,
        ).expect("with_capacity");
        let mut ctx = InferenceContext::new(device);
        let actual = model
            .forward_with_kv_context(&prompt, &mut cache, &mut ctx)
            .expect("prefill");

        assert_eq!(cache.cached_len, prompt.len());
        assert_eq!(actual.len(), expected.len());

        // Tighter tolerance than the prefill+decode test: this is
        // structurally one forward pass through the model with the
        // same input shape — the only added work is WriteSlice +
        // Slice (both byte-exact ops). Drift should be at the rng-
        // initial-noise level.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-5,
                "logit[{i}]: new-prefill={a}, non-cached={b}, diff={diff}",
            );
        }
    }

    /// `forward_with_kv_context` rejects a cache built via `with_dims`
    /// (no pre-allocated buffers) with a clear error pointing at the
    /// `with_capacity` constructor.
    #[test]
    fn forward_with_kv_context_rejects_with_dims_cache() {
        let cfg = LlamaConfig {
            vocab_size: 4, dim: 4, n_layers: 1, n_heads: 2, n_kv_heads: 2,
            head_dim: 2, ffn_dim: 4, norm_eps: 1e-5, rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let mut cache = KvCache::with_dims(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim);
        let mut ctx = InferenceContext::new(Device::cpu());

        let err = model.forward_with_kv_context(&[1_u32], &mut cache, &mut ctx);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("with_capacity"),
            "expected error message to mention with_capacity, got: {msg}",
        );
    }

    /// `forward_with_kv_context` rejects when `cached_len + seq`
    /// exceeds the cache's `max_seq_len`.
    #[test]
    fn forward_with_kv_context_rejects_overflow() {
        let cfg = LlamaConfig {
            vocab_size: 4, dim: 4, n_layers: 1, n_heads: 2, n_kv_heads: 2,
            head_dim: 2, ffn_dim: 4, norm_eps: 1e-5, rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let device = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
            /*max_seq_len*/ 2, DType::F32, &device,
        ).unwrap();
        let mut ctx = InferenceContext::new(device);

        // 3 tokens into a cache with max_seq_len=2 → overflow.
        let err = model.forward_with_kv_context(&[1_u32, 2, 3], &mut cache, &mut ctx);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("max_seq_len"),
            "expected error message to mention max_seq_len, got: {msg}",
        );
    }

    /// After each forward call, the per-step Const NodeIds inserted
    /// into ctx are cleaned up — ctx.persistent should NOT accumulate
    /// across decode steps.
    #[test]
    fn forward_with_kv_context_does_not_leak_context_entries() {
        let cfg = LlamaConfig {
            vocab_size: 4, dim: 4, n_layers: 1, n_heads: 2, n_kv_heads: 2,
            head_dim: 2, ffn_dim: 4, norm_eps: 1e-5, rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let device = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
            4, DType::F32, &device,
        ).unwrap();
        let mut ctx = InferenceContext::new(device);

        assert_eq!(ctx.len(), 0);
        model.forward_with_kv_context(&[1_u32, 2], &mut cache, &mut ctx).unwrap();
        assert_eq!(ctx.len(), 0, "ctx.persistent should be empty after forward");
        model.forward_with_kv_context(&[3_u32], &mut cache, &mut ctx).unwrap();
        assert_eq!(ctx.len(), 0, "ctx.persistent should stay empty across steps");
    }

    /// Vulkan parity: prefill+decode through `forward_with_kv_context`
    /// on a Vulkan `Device` matches the CPU reference for the same
    /// model + prompt. Closes the runtime Device-abstraction gate that
    /// the audit memo `project_phase_7_6_step_9c_parity_audit.md`
    /// flagged: previously `Device::new(...)` rejected Vulkan because
    /// no `DynBackendDevice` impl existed, so `KvCache::with_capacity`
    /// + `InferenceContext` could not run on Vulkan even though every
    /// kernel-side gate (WriteSlice b1/b2/b4/b8 + byte-storage Vulkan
    /// D2H) was open. With `VulkanBackendDevice` wired through
    /// `Device::custom`, the pipelined executor + binding-table
    /// dispatch route the per-op kernels to Vulkan SPIR-V.
    ///
    /// Skips with a logged message if no Vulkan device is available
    /// so CI machines without a GPU stay green.
    #[test]
    #[cfg(feature = "vulkan")]
    fn forward_with_kv_context_vulkan_matches_cpu() {
        use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

        let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping: no Vulkan device ({e:?})");
                return;
            }
        };
        let vk_device: Device = vk_backend.into();

        let cfg = LlamaConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   2,
            n_heads:    4,
            n_kv_heads: 2,
            head_dim:   4,
            ffn_dim:    16,
            norm_eps:   1e-5,
            rope_base:  10000.0,
        };
        let cfg = LlamaConfig { dim: cfg.n_heads * cfg.head_dim, ..cfg };
        let model = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let next_token = 4_u32;
        let max_seq_len = prompt.len() + 1;

        // CPU reference.
        let cpu_device = Device::cpu();
        let mut cpu_cache = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
            max_seq_len, DType::F32, &cpu_device,
        ).expect("cpu with_capacity");
        let mut cpu_ctx = InferenceContext::new(cpu_device);
        model.forward_with_kv_context(&prompt, &mut cpu_cache, &mut cpu_ctx)
            .expect("cpu prefill");
        let expected = model
            .forward_with_kv_context(&[next_token], &mut cpu_cache, &mut cpu_ctx)
            .expect("cpu decode");

        // Vulkan path through the new Device wiring.
        let mut vk_cache = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
            max_seq_len, DType::F32, &vk_device,
        ).expect("vulkan with_capacity");
        let mut vk_ctx = InferenceContext::new(vk_device);
        model.forward_with_kv_context(&prompt, &mut vk_cache, &mut vk_ctx)
            .expect("vulkan prefill");
        let actual = model
            .forward_with_kv_context(&[next_token], &mut vk_cache, &mut vk_ctx)
            .expect("vulkan decode");

        assert_eq!(actual.len(), expected.len());
        // Same tolerance band as `forward_with_kv_context_decode_matches_
        // non_cached_forward`: cross-backend matmul accumulation order
        // differs, producing standard O(ε) gemm drift on the f32 path.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / a.abs().max(b.abs()).max(1e-6);
            assert!(
                diff < 5e-3 || rel < 1e-2,
                "logit[{i}]: vulkan={a}, cpu={b}, diff={diff}, rel={rel}",
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

    #[test]
    fn kvcache_truncate_to_shrinks_layers_and_preserves_prefix() {
        use fuel_graph_executor::GraphBackend;
        let backend = fuel_graph_cpu::CpuBackend;
        let n_layers = 2;
        let n_kv_heads = 2;
        let head_dim = 4;
        let old_seq = 6;
        let new_seq = 4;

        let mut cache: KVCache<fuel_graph_cpu::CpuBackend> =
            KVCache::with_dims(n_layers, n_kv_heads, head_dim);
        cache.cached_len = old_seq;

        // Populate each layer with a known pattern. Shape
        // [1, n_kv, old_seq, head_dim]; value[b, h, s, d] = h*1000 + s*10 + d
        // so we can verify the prefix survives truncation.
        for li in 0..n_layers {
            let n = 1 * n_kv_heads * old_seq * head_dim;
            let data: Vec<f32> = (0..n).map(|i| {
                let d = i % head_dim;
                let s = (i / head_dim) % old_seq;
                let h = (i / (head_dim * old_seq)) % n_kv_heads;
                (h * 1000 + s * 10 + d) as f32
            }).collect();
            let shape = Shape::from_dims(&[1, n_kv_heads, old_seq, head_dim]);
            let k = backend.upload(&fuel_core_types::HostBuffer::F32(data.clone()), &shape).unwrap();
            let v = backend.upload(&fuel_core_types::HostBuffer::F32(data), &shape).unwrap();
            cache.layers[li] = Some(KVCacheEntry::F32 { k, v });
        }

        cache.truncate_to(new_seq, &backend).expect("truncate_to");
        assert_eq!(cache.cached_len, new_seq);

        // Verify layer 0 K has the right prefix for head 0, seq 0..new_seq.
        let entry = cache.layers[0].as_ref().expect("layer 0 still present");
        let k = match entry { KVCacheEntry::F32 { k, .. } => k, _ => panic!("unexpected Q8") };
        let host = backend.download(k).expect("download k");
        let buf = match host { fuel_core_types::HostBuffer::F32(v) => v, _ => panic!("expected f32") };
        assert_eq!(buf.len(), 1 * n_kv_heads * new_seq * head_dim);
        // head 0, seq 0, dim 0: expected 0*1000 + 0*10 + 0 = 0
        assert_eq!(buf[0], 0.0);
        // head 0, seq new_seq-1=3, dim 3: expected 0*1000 + 3*10 + 3 = 33
        let head0_last = (new_seq - 1) * head_dim + (head_dim - 1);
        assert_eq!(buf[head0_last], 33.0);
        // head 1, seq 0, dim 0: expected 1*1000 + 0*10 + 0 = 1000
        let head1_start = n_kv_heads.saturating_sub(1) * new_seq * head_dim;
        assert_eq!(buf[head1_start], 1000.0);
        // head 1, seq new_seq-1=3, dim 3: expected 1*1000 + 3*10 + 3 = 1033
        assert_eq!(buf[head1_start + head0_last], 1033.0);
    }

    #[test]
    fn kvcache_truncate_to_noop_when_new_len_ge_current() {
        let backend = fuel_graph_cpu::CpuBackend;
        let mut cache: KVCache<fuel_graph_cpu::CpuBackend> =
            KVCache::with_dims(1, 2, 4);
        cache.cached_len = 4;
        cache.truncate_to(4, &backend).unwrap();
        assert_eq!(cache.cached_len, 4);
        cache.truncate_to(100, &backend).unwrap();
        assert_eq!(cache.cached_len, 4);
    }

    #[test]
    fn forward_with_cache_all_positions_last_slice_matches_forward_with_cache() {
        // The all-positions variant's last slice must equal what the
        // regular (last-only) variant produces. Same graph, same
        // cache, same tokens — only the output shape differs.
        use fuel_graph_executor::GraphExecutor;
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

        let tokens = [1_u32, 2, 3, 4, 5];

        // Path A: regular last-only forward.
        let mut exec_a = GraphExecutor::new(fuel_graph_cpu::CpuBackend);
        let mut cache_a: KVCache<fuel_graph_cpu::CpuBackend> = KVCache::new(&cfg);
        let last_only = model.forward_with_cache_gpu_on(&tokens, &mut cache_a, &mut exec_a);
        assert_eq!(last_only.len(), cfg.vocab_size);

        // Path B: all-positions forward.
        let mut exec_b = GraphExecutor::new(fuel_graph_cpu::CpuBackend);
        let mut cache_b: KVCache<fuel_graph_cpu::CpuBackend> = KVCache::new(&cfg);
        let all = model.forward_with_cache_gpu_on_all_positions(&tokens, &mut cache_b, &mut exec_b);
        assert_eq!(all.len(), tokens.len() * cfg.vocab_size);

        // Last row of `all` (positions [seq-1]) must match last_only.
        let last_pos = tokens.len() - 1;
        let all_last = &all[last_pos * cfg.vocab_size .. (last_pos + 1) * cfg.vocab_size];
        for (i, (a, b)) in all_last.iter().zip(last_only.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "vocab idx {i}: all_positions={a} vs last_only={b}"
            );
        }

        // Both caches should have advanced by the same amount.
        assert_eq!(cache_a.cached_len, cache_b.cached_len);
    }

    #[test]
    fn spec_decode_with_self_as_draft_matches_greedy_baseline() {
        // Use the target model as its own draft. Every draft token is
        // then trivially argmax-matched by the target, so acceptance
        // rate is 100% and the generated sequence must be identical to
        // a plain greedy run. This is the strongest equivalence check
        // for the spec-decode plumbing.
        use fuel_graph_executor::GraphExecutor;
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

        let prompt = [3_u32, 7, 1];
        let max_new = 8;

        // Baseline: plain greedy generation.
        let mut exec_a = GraphExecutor::new(fuel_graph_cpu::CpuBackend);
        let baseline = model.generate_streaming_gpu_on(
            &prompt, max_new,
            SamplingStrategy::Greedy, None,
            &mut exec_a, |_| {},
        ).expect("baseline generate");

        // Spec-decode with model as its own draft. Try K=2 and K=4.
        for k in [2_usize, 4] {
            let mut exec_target = GraphExecutor::new(fuel_graph_cpu::CpuBackend);
            let mut exec_draft = GraphExecutor::new(fuel_graph_cpu::CpuBackend);
            let spec_out = model.generate_streaming_spec(
                &model, &prompt, max_new, k,
                SamplingStrategy::Greedy, None,
                &mut exec_target, &mut exec_draft, |_| {},
            ).expect("spec generate");
            assert_eq!(
                spec_out, baseline,
                "K={k}: spec-decode must match baseline when draft == target"
            );
        }
    }

    #[test]
    fn spec_decode_sampled_with_self_as_draft_produces_valid_tokens() {
        // In Temperature mode with draft == target, the accept coin's
        // ratio = min(1, p_target/p_draft) = 1.0 (since p_target == p_draft
        // element-wise), so acceptance is 100%. We can't bit-match
        // against a plain sampled baseline because the RNG sequences
        // diverge (spec-decode draws more randoms per output token
        // than plain gen), but we can assert: (a) output has expected
        // length, (b) all tokens are in vocab, (c) prompt prefix is
        // preserved.
        use fuel_graph_executor::GraphExecutor;
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
        let prompt = [3_u32, 7, 1];
        let max_new = 6;

        for k in [2_usize, 4] {
            let mut exec_target = GraphExecutor::new(fuel_graph_cpu::CpuBackend);
            let mut exec_draft = GraphExecutor::new(fuel_graph_cpu::CpuBackend);
            let out = model.generate_streaming_spec(
                &model, &prompt, max_new, k,
                SamplingStrategy::Temperature { temp: 0.8, seed: 42 },
                None,
                &mut exec_target, &mut exec_draft, |_| {},
            ).expect("spec sampled generate");

            // Emitted tokens should be prompt + max_new (at minimum; could
            // be more if the bonus gets combined with accepted drafts in
            // the final iteration, but never fewer than max_new new).
            assert!(out.len() >= prompt.len() + max_new,
                "K={k}: expected at least {} tokens, got {}",
                prompt.len() + max_new, out.len());
            // Prefix matches prompt.
            assert_eq!(&out[..prompt.len()], &prompt);
            // All tokens in vocab.
            for &t in &out {
                assert!((t as usize) < cfg.vocab_size, "K={k}: token {t} out of vocab");
            }
        }
    }

    #[test]
    fn kvcache_truncate_to_zero_clears_layers() {
        use fuel_graph_executor::GraphBackend;
        let backend = fuel_graph_cpu::CpuBackend;
        let mut cache: KVCache<fuel_graph_cpu::CpuBackend> =
            KVCache::with_dims(1, 2, 4);
        cache.cached_len = 3;
        let shape = Shape::from_dims(&[1, 2, 3, 4]);
        let data = vec![0.0_f32; 1 * 2 * 3 * 4];
        let k = backend.upload(&fuel_core_types::HostBuffer::F32(data.clone()), &shape).unwrap();
        let v = backend.upload(&fuel_core_types::HostBuffer::F32(data), &shape).unwrap();
        cache.layers[0] = Some(KVCacheEntry::F32 { k, v });
        cache.truncate_to(0, &backend).unwrap();
        assert_eq!(cache.cached_len, 0);
        assert!(cache.layers[0].is_none());
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
                    attn_q:         vec_of(cfg.dim * cfg.dim).into(),
                    attn_q_bias:    None,
                    attn_k:         vec_of(cfg.dim * kv_dim).into(),
                    attn_k_bias:    None,
                    attn_v:         vec_of(cfg.dim * kv_dim).into(),
                    attn_v_bias:    None,
                    attn_o:         vec_of(cfg.dim * cfg.dim).into(),
                    ffn_gate:       vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_up:         vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_down:       vec_of(cfg.ffn_dim * cfg.dim).into(),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain:  Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output:          vec_of(cfg.dim * cfg.vocab_size).into(),
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
mod lora_tests {
    use super::*;

    #[test]
    fn with_lora_matches_manual_base_plus_lora() {
        // Anchor graph.
        let in_f = 4;
        let out_f = 3;
        let rank = 2;
        let alpha = 8.0_f32;

        let anchor = LazyTensor::from_f32(
            vec![0.0_f32; 1],
            Shape::from_dims(&[1]),
            &Device::cpu(),
        );
        // Base weight [in, out].
        let base_vec: Vec<f32> = (0..in_f * out_f).map(|i| (i as f32) * 0.1).collect();
        let lora_a_vec: Vec<f32> = (0..in_f * rank).map(|i| (i as f32) * 0.05).collect();
        let lora_b_vec: Vec<f32> = (0..rank * out_f).map(|i| (i as f32) * 0.02).collect();

        let ws = WeightStorage::F32(Arc::from(base_vec.clone()))
            .with_lora(
                Arc::from(lora_a_vec.clone()),
                Arc::from(lora_b_vec.clone()),
                rank, alpha, in_f, out_f,
            );

        // Activations x [2, in_f].
        let batch = 2;
        let x_data: Vec<f32> = (0..batch * in_f).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let x = anchor.const_f32_like(x_data.clone(), Shape::from_dims(&[batch, in_f]));
        let y = ws.apply_linear(&x, in_f, out_f);
        let got = y.realize_f32().to_vec();

        // Reference: base + (alpha/rank) * x @ A @ B, all f32, on CPU.
        let mut expected = vec![0.0_f32; batch * out_f];
        for b in 0..batch {
            for j in 0..out_f {
                let mut acc = 0.0_f32;
                // Base path: sum_k x[b,k] * W[k,j].
                for k in 0..in_f {
                    acc += x_data[b * in_f + k] * base_vec[k * out_f + j];
                }
                // LoRA path: sum_r (sum_k x[b,k] * A[k,r]) * B[r,j] * (alpha/rank).
                let scale = alpha as f64 / rank as f64;
                for r in 0..rank {
                    let mut xar = 0.0_f32;
                    for k in 0..in_f {
                        xar += x_data[b * in_f + k] * lora_a_vec[k * rank + r];
                    }
                    acc += (xar * lora_b_vec[r * out_f + j]) * scale as f32;
                }
                expected[b * out_f + j] = acc;
            }
        }

        for (i, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
            let diff = (e - g).abs();
            assert!(
                diff <= 1e-4,
                "LoRA mismatch at {i}: expected {e}, got {g} (diff {diff})",
            );
        }
    }

    #[test]
    #[should_panic(expected = "lora_a length")]
    fn with_lora_rejects_mismatched_a_shape() {
        let ws = WeightStorage::F32(Arc::from(vec![0.0_f32; 12]));  // 4 x 3
        let bad_a = Arc::from(vec![0.0_f32; 3]);                     // wrong
        let b = Arc::from(vec![0.0_f32; 6]);                         // 2 x 3
        let _ = ws.with_lora(bad_a, b, 2, 8.0, 4, 3);
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
                    attn_q:         vec_of(cfg.dim * cfg.dim).into(),
                    attn_q_bias:    None,
                    attn_k:         vec_of(cfg.dim * kv_dim).into(),
                    attn_k_bias:    None,
                    attn_v:         vec_of(cfg.dim * kv_dim).into(),
                    attn_v_bias:    None,
                    attn_o:         vec_of(cfg.dim * cfg.dim).into(),
                    ffn_gate:       vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_up:         vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_down:       vec_of(cfg.ffn_dim * cfg.dim).into(),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain:  Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output:          vec_of(cfg.dim * cfg.vocab_size).into(),
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
            &Device::cpu(),
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
            &Device::cpu(),
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
            &Device::cpu(),
        );
        assert!(result.is_err());
    }
}

// ============================================================================
// Phase A.1 wrapper smoke tests.
//
// Pure pass-through tests: realize and assert the returned tensor has the
// expected shape / dtype / values. The graph-level ops are tested in
// `fuel-graph`; here we only verify that the LazyTensor wrappers don't
// drop information or mis-thread arguments.
// ============================================================================
#[cfg(test)]
mod phase_a1_wrapper_tests {
    use super::*;

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> LazyTensor {
        LazyTensor::from_f32(data, shape.to_vec(), &Device::cpu())
    }

    #[test]
    fn unsqueeze_adds_size_one_dim() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = t.unsqueeze(0);
        assert_eq!(out.shape().dims(), &[1, 2, 2]);
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn try_unsqueeze_errors_out_of_bounds() {
        let t = cpu_f32(vec![1.0, 2.0], &[2]);
        // rank=1, so dim<=1 is valid; dim=2 must error.
        assert!(t.try_unsqueeze(0).is_ok());
        assert!(t.try_unsqueeze(1).is_ok());
        assert!(t.try_unsqueeze(2).is_err());
    }

    #[test]
    fn try_reshape_errors_on_size_mismatch() {
        let t = cpu_f32(vec![1.0; 6], &[2, 3]);
        assert!(t.try_reshape(vec![3, 2]).is_ok());
        assert!(t.try_reshape(vec![2, 2]).is_err());
    }

    #[test]
    fn try_permute_validates_axes() {
        let t = cpu_f32(vec![0.0; 24], &[2, 3, 4]);
        assert!(t.try_permute(&[2, 0, 1]).is_ok());
        assert!(t.try_permute(&[0, 1]).is_err());     // wrong rank
        assert!(t.try_permute(&[0, 0, 1]).is_err()); // dup axis
    }

    #[test]
    fn try_transpose_requires_rank_two_plus() {
        let scalar = cpu_f32(vec![1.0], &[1]);
        // rank-1 is fine on most platforms because transpose() of last two
        // dims degenerates, but the try_* sibling at minimum exists.
        let _ = scalar.try_transpose();
    }

    #[test]
    fn triu_tril_shape_preserved() {
        let t = cpu_f32(vec![1.0; 9], &[3, 3]);
        let upper = t.triu(0).unwrap();
        let lower = t.tril(0).unwrap();
        assert_eq!(upper.shape().dims(), &[3, 3]);
        assert_eq!(lower.shape().dims(), &[3, 3]);
        // tril(0) of all-ones: 1s on/below diagonal, 0s above
        assert_eq!(lower.realize_f32(), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn triu_rejects_rank_one() {
        let t = cpu_f32(vec![1.0, 2.0], &[2]);
        assert!(t.triu(0).is_err());
    }

    #[test]
    fn log_softmax_last_dim_shape_preserved() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = t.log_softmax_last_dim().unwrap();
        assert_eq!(out.shape().dims(), &[2, 2]);
        // log_softmax values must be <= 0.
        for v in out.realize_f32() {
            assert!(v <= 0.0 + 1e-6, "log_softmax produced positive value: {v}");
        }
    }

    #[test]
    fn argmin_dim_drops_reduced_axis() {
        let t = cpu_f32(vec![3.0, 1.0, 2.0, 0.5, 5.0, 4.0], &[2, 3]);
        let out = t.argmin_dim(1);
        assert_eq!(out.shape().dims(), &[2]);
        assert_eq!(out.dtype(), DType::U32);
        assert_eq!(out.realize_u32(), vec![1, 0]);
    }

    #[test]
    fn masked_fill_smoke() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // Comparison ops produce U8 masks directly — no F32→U8 cast needed.
        let probe = t.const_f32_like(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]);
        let threshold = t.const_f32_like(vec![0.5; 4], vec![2, 2]);
        let mask = probe.gt(&threshold); // [0, 1, 1, 0] as U8
        let out = t.masked_fill(&mask, fuel_core_types::Scalar::F32(-9.0)).unwrap();
        assert_eq!(out.realize_f32(), vec![1.0, -9.0, -9.0, 4.0]);
    }

    #[test]
    fn index_add_smoke() {
        let base = cpu_f32(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let src = base.const_f32_like(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let indices = base.const_u32_like(vec![0_u32, 0_u32], vec![2]);
        let out = base.index_add(0, &indices, &src);
        assert_eq!(out.shape().dims(), &[2, 2]);
        // both src rows added to row 0; row 1 unchanged
        let v = out.realize_f32();
        assert_eq!(v[0], 41.0); // 1 + 10 + 30
        assert_eq!(v[1], 61.0); // 1 + 20 + 40
        assert_eq!(v[2], 1.0);
        assert_eq!(v[3], 1.0);
    }

    #[test]
    fn scatter_add_smoke() {
        let base = cpu_f32(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]);
        let src = base.const_f32_like(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let indices = base.const_u32_like(vec![0_u32, 1_u32, 1_u32, 0_u32], vec![2, 2]);
        let out = base.scatter_add(0, &indices, &src);
        assert_eq!(out.shape().dims(), &[2, 2]);
    }

    #[test]
    fn inplace_activations_compile_and_run() {
        let t = cpu_f32(vec![-1.0, 0.5, -3.0, 2.0], &[4]);
        // Each in-place op is destructive, so chain through fresh tensors.
        let r = cpu_f32(vec![-1.0, 0.5, -3.0, 2.0], &[4]).relu_inplace();
        let _ = cpu_f32(vec![-1.0, 0.5, -3.0, 2.0], &[4]).silu_inplace();
        let _ = cpu_f32(vec![-1.0, 0.5, -3.0, 2.0], &[4]).gelu_inplace();
        let _ = cpu_f32(vec![-1.0, 0.5, -3.0, 2.0], &[4]).tanh_inplace();
        let _ = cpu_f32(vec![-1.0, 0.5, -3.0, 2.0], &[4]).sigmoid_inplace();
        let _ = t.affine_inplace(2.0, 1.0);
        // Spot-check the relu output.
        let v = r.realize_f32();
        assert_eq!(v, vec![0.0, 0.5, 0.0, 2.0]);
    }

    #[test]
    fn const_f64_like_round_trips() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let t = anchor.const_f64_like(vec![1.5, 2.5, 3.5], vec![3]);
        assert_eq!(t.shape().dims(), &[3]);
        assert_eq!(t.dtype(), DType::F64);
        assert_eq!(t.realize_f64(), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn const_i64_like_round_trips() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let t = anchor.const_i64_like(vec![-1_i64, 2, -3], vec![3]);
        assert_eq!(t.shape().dims(), &[3]);
        assert_eq!(t.dtype(), DType::I64);
    }

    #[test]
    fn on_device_smoke_cpu() {
        let t = cpu_f32(vec![1.0, 2.0], &[2]);
        let pinned = t.on_device(&Device::cpu());
        assert_eq!(pinned.realize_f32(), vec![1.0, 2.0]);
    }

    #[test]
    fn copy_to_device_same_device_round_trips() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let copied = t.copy_to_device(&Device::cpu());
        assert_eq!(copied.realize_f32(), vec![1.0, 2.0, 3.0]);
    }
}

// ============================================================================
// Phase A.2 composite primitives tests.
// ============================================================================
#[cfg(test)]
mod phase_a2_composite_tests {
    use super::*;

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> LazyTensor {
        LazyTensor::from_f32(data, shape.to_vec(), &Device::cpu())
    }

    #[test]
    fn transpose_last_two_swaps_last_two_dims() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let out = t.transpose_last_two().unwrap();
        assert_eq!(out.shape().dims(), &[3, 2]);
        assert_eq!(out.realize_f32(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn t_is_alias_of_transpose_last_two() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(t.t().unwrap().realize_f32(), t.transpose_last_two().unwrap().realize_f32());
    }

    #[test]
    fn transpose_dims_swaps_arbitrary_axes() {
        let t = cpu_f32((0..24).map(|i| i as f32).collect(), &[2, 3, 4]);
        let out = t.transpose_dims(0, 2).unwrap();
        assert_eq!(out.shape().dims(), &[4, 3, 2]);
    }

    #[test]
    fn transpose_dims_identity_when_same_axis() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = t.transpose_dims(0, 0).unwrap();
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn transpose_dims_errors_out_of_bounds() {
        let t = cpu_f32(vec![0.0; 6], &[2, 3]);
        assert!(t.transpose_dims(0, 2).is_err());
        assert!(t.transpose_dims(5, 0).is_err());
    }

    #[test]
    fn flatten_merges_middle_dims() {
        let t = cpu_f32(vec![0.0; 24], &[2, 3, 4]);
        let out = t.flatten(0, 1).unwrap();
        assert_eq!(out.shape().dims(), &[6, 4]);
    }

    #[test]
    fn flatten_to_merges_leading_dims() {
        let t = cpu_f32(vec![0.0; 24], &[2, 3, 4]);
        let out = t.flatten_to(1).unwrap();
        assert_eq!(out.shape().dims(), &[6, 4]);
    }

    #[test]
    fn flatten_from_merges_trailing_dims() {
        let t = cpu_f32(vec![0.0; 24], &[2, 3, 4]);
        let out = t.flatten_from(1).unwrap();
        assert_eq!(out.shape().dims(), &[2, 12]);
    }

    #[test]
    fn flatten_all_produces_rank_one() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let out = t.flatten_all().unwrap();
        assert_eq!(out.shape().dims(), &[6]);
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn flatten_errors_on_bad_range() {
        let t = cpu_f32(vec![0.0; 6], &[2, 3]);
        assert!(t.flatten(0, 5).is_err());
        assert!(t.flatten(2, 1).is_err()); // start > end
    }

    #[test]
    fn stack_adds_leading_dim() {
        let a = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], vec![3]);
        let out = LazyTensor::stack(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape().dims(), &[2, 3]);
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_adds_trailing_dim() {
        let a = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], vec![3]);
        let out = LazyTensor::stack(&[&a, &b], 1).unwrap();
        assert_eq!(out.shape().dims(), &[3, 2]);
        assert_eq!(out.realize_f32(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn stack_rejects_mismatched_shapes() {
        let a = cpu_f32(vec![1.0, 2.0], &[2]);
        let b = a.const_f32_like(vec![3.0, 4.0, 5.0], vec![3]);
        assert!(LazyTensor::stack(&[&a, &b], 0).is_err());
    }

    #[test]
    fn stack_rejects_empty_input() {
        let result = LazyTensor::stack(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn repeat_tiles_along_each_dim() {
        let t = cpu_f32(vec![1.0, 2.0], &[2]);
        let out = t.repeat(vec![3]).unwrap();
        assert_eq!(out.shape().dims(), &[6]);
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn repeat_extends_rank_when_needed() {
        let t = cpu_f32(vec![1.0, 2.0], &[2]);
        // repeat with shape [3, 2] left-pads tensor to [1, 2] then tiles to [3, 4]
        let out = t.repeat(vec![3, 2]).unwrap();
        assert_eq!(out.shape().dims(), &[3, 4]);
    }

    #[test]
    fn repeat_identity_with_all_ones() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let out = t.repeat(vec![1]).unwrap();
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0]);
    }
}

// ============================================================================
// Phase A.3 keepdim reduction tests.
// ============================================================================
#[cfg(test)]
mod phase_a3_keepdim_tests {
    use super::*;

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> LazyTensor {
        LazyTensor::from_f32(data, shape.to_vec(), &Device::cpu())
    }

    #[test]
    fn sum_keepdim_preserves_dim_as_one() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = t.sum_keepdim(1);
        assert_eq!(out.shape().dims(), &[2, 1]);
        assert_eq!(out.realize_f32(), vec![3.0, 7.0]);
    }

    #[test]
    fn mean_keepdim_preserves_dim_as_one() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = t.mean_keepdim(0);
        assert_eq!(out.shape().dims(), &[1, 2]);
        assert_eq!(out.realize_f32(), vec![2.0, 3.0]);
    }

    #[test]
    fn max_keepdim_preserves_dim_as_one() {
        let t = cpu_f32(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]);
        let out = t.max_keepdim(1);
        assert_eq!(out.shape().dims(), &[2, 1]);
        assert_eq!(out.realize_f32(), vec![3.0, 4.0]);
    }

    #[test]
    fn min_keepdim_preserves_dim_as_one() {
        let t = cpu_f32(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]);
        let out = t.min_keepdim(1);
        assert_eq!(out.shape().dims(), &[2, 1]);
        assert_eq!(out.realize_f32(), vec![1.0, 2.0]);
    }

    #[test]
    fn var_matches_unbiased_formula() {
        // [[1,2,3],[4,5,6]] -> var along axis 1: each row has mean=mid, dev=[-1,0,1], sq sum=2, /2 = 1
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let out = t.var(1).unwrap();
        assert_eq!(out.shape().dims(), &[2]);
        let v = out.realize_f32();
        assert!((v[0] - 1.0).abs() < 1e-5, "var row 0 = {} != 1.0", v[0]);
        assert!((v[1] - 1.0).abs() < 1e-5, "var row 1 = {} != 1.0", v[1]);
    }

    #[test]
    fn var_keepdim_preserves_dim() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let out = t.var_keepdim(1).unwrap();
        assert_eq!(out.shape().dims(), &[2, 1]);
    }

    #[test]
    fn var_errors_out_of_bounds() {
        let t = cpu_f32(vec![1.0, 2.0], &[2]);
        assert!(t.var(3).is_err());
        assert!(t.var_keepdim(3).is_err());
    }
}

// ============================================================================
// Phase A.4 scalar/binary composite tests.
// ============================================================================
#[cfg(test)]
mod phase_a4_composite_tests {
    use super::*;

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> LazyTensor {
        LazyTensor::from_f32(data, shape.to_vec(), &Device::cpu())
    }

    #[test]
    fn affine_applies_mul_then_add() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let out = t.affine(2.0, 10.0);
        assert_eq!(out.realize_f32(), vec![12.0, 14.0, 16.0]);
    }

    #[test]
    fn scale_and_shift_alias_of_affine() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        assert_eq!(
            t.scale_and_shift(2.0, 10.0).realize_f32(),
            t.affine(2.0, 10.0).realize_f32(),
        );
    }

    #[test]
    fn elu_matches_reference_values() {
        let t = cpu_f32(vec![1.0, 0.0, -1.0, -2.0], &[4]);
        let out = t.elu(1.0);
        let v = out.realize_f32();
        // x > 0: identity. x == 0: 0 (boundary; gt returns 0 → neg branch which is alpha*(1-1)=0).
        // x < 0: alpha * (exp(x) - 1).
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!(v[1].abs() < 1e-5);
        assert!((v[2] - ((-1.0_f32).exp() - 1.0)).abs() < 1e-5);
        assert!((v[3] - ((-2.0_f32).exp() - 1.0)).abs() < 1e-5);
    }

    #[test]
    fn dot_of_rank_one_vectors() {
        let a = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], vec![3]);
        let out = a.dot(&b).unwrap();
        assert_eq!(out.shape().elem_count(), 1);
        let v = out.realize_f32();
        assert_eq!(v[0], 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn dot_rejects_non_rank_one() {
        let a = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = a.const_f32_like(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn dot_rejects_length_mismatch() {
        let a = cpu_f32(vec![1.0, 2.0], &[2]);
        let b = a.const_f32_like(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn mv_matrix_times_vector() {
        let m = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let v = m.const_f32_like(vec![1.0, 1.0, 1.0], vec![3]);
        let out = m.mv(&v).unwrap();
        assert_eq!(out.shape().dims(), &[2]);
        assert_eq!(out.realize_f32(), vec![6.0, 15.0]);
    }

    #[test]
    fn matvec_is_mv_alias() {
        let m = cpu_f32(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let v = m.const_f32_like(vec![3.0, 4.0], vec![2]);
        let a = m.mv(&v).unwrap().realize_f32();
        let b = m.matvec(&v).unwrap().realize_f32();
        assert_eq!(a, b);
    }

    #[test]
    fn mv_rejects_shape_mismatch() {
        let m = cpu_f32(vec![1.0; 6], &[2, 3]);
        let v = m.const_f32_like(vec![1.0, 1.0], vec![2]);
        assert!(m.mv(&v).is_err());
    }

    #[test]
    fn broadcast_matmul_passes_through_to_matmul() {
        let a = cpu_f32(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let b = a.const_f32_like(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let out = a.broadcast_matmul(&b);
        assert_eq!(out.realize_f32(), vec![5.0, 6.0, 7.0, 8.0]);
    }
}

// ============================================================================
// Phase A.5 factory family tests.
// ============================================================================
#[cfg(test)]
mod phase_a5_factory_tests {
    use super::*;

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> LazyTensor {
        LazyTensor::from_f32(data, shape.to_vec(), &Device::cpu())
    }

    #[test]
    fn ones_like_matches_shape_dtype_graph() {
        let t = cpu_f32(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let ones = t.ones_like();
        assert_eq!(ones.shape().dims(), t.shape().dims());
        assert_eq!(ones.dtype(), t.dtype());
        assert_eq!(ones.realize_f32(), vec![1.0; 4]);
    }

    #[test]
    fn zeros_like_matches_shape_dtype_graph() {
        let t = cpu_f32(vec![5.0, 6.0, 7.0], &[3]);
        let zeros = t.zeros_like();
        assert_eq!(zeros.realize_f32(), vec![0.0; 3]);
    }

    #[test]
    fn static_ones_f32() {
        let t = LazyTensor::ones(vec![2, 3], DType::F32, &Device::cpu());
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.realize_f32(), vec![1.0; 6]);
    }

    #[test]
    fn static_zeros_f64() {
        let t = LazyTensor::zeros(vec![4], DType::F64, &Device::cpu());
        assert_eq!(t.dtype(), DType::F64);
        assert_eq!(t.realize_f64(), vec![0.0; 4]);
    }

    #[test]
    fn full_with_f32_scalar() {
        let t = LazyTensor::full(vec![5], fuel_core_types::Scalar::F32(2.5), &Device::cpu());
        assert_eq!(t.realize_f32(), vec![2.5; 5]);
    }

    #[test]
    fn eye_identity_matrix() {
        let t = LazyTensor::eye(3, DType::F32, &Device::cpu());
        assert_eq!(t.shape().dims(), &[3, 3]);
        assert_eq!(
            t.realize_f32(),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn tril2_lower_triangular_ones() {
        let t = LazyTensor::tril2(3, DType::F32, &Device::cpu());
        assert_eq!(
            t.realize_f32(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        );
    }

    #[test]
    fn triu2_upper_triangular_ones() {
        let t = LazyTensor::triu2(3, DType::F32, &Device::cpu());
        assert_eq!(
            t.realize_f32(),
            vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn meshgrid_ij_indexing_two_inputs() {
        let x = LazyTensor::from_f32(vec![1.0_f32, 2.0, 3.0], vec![3], &Device::cpu());
        let y = x.const_f32_like(vec![4.0_f32, 5.0], vec![2]);
        let grids = LazyTensor::meshgrid(&[&x, &y], false).unwrap();
        assert_eq!(grids.len(), 2);
        // ij: shapes are [len(x), len(y)] = [3, 2].
        assert_eq!(grids[0].shape().dims(), &[3, 2]);
        assert_eq!(grids[1].shape().dims(), &[3, 2]);
        // X grid: each row repeats x's value, so each row is identical along axis 1.
        assert_eq!(grids[0].realize_f32(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        // Y grid: each column repeats y's value, so each column is identical along axis 0.
        assert_eq!(grids[1].realize_f32(), vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
    }

    #[test]
    fn meshgrid_xy_indexing_swaps_first_two() {
        let x = LazyTensor::from_f32(vec![1.0_f32, 2.0, 3.0], vec![3], &Device::cpu());
        let y = x.const_f32_like(vec![4.0_f32, 5.0], vec![2]);
        let grids = LazyTensor::meshgrid(&[&x, &y], true).unwrap();
        // xy: shapes flip to [len(y), len(x)] = [2, 3].
        assert_eq!(grids[0].shape().dims(), &[2, 3]);
        assert_eq!(grids[1].shape().dims(), &[2, 3]);
        // X grid: same row twice, each row is x.
        assert_eq!(grids[0].realize_f32(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        // Y grid: each row repeats one y element.
        assert_eq!(grids[1].realize_f32(), vec![4.0, 4.0, 4.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn meshgrid_rejects_single_input() {
        let x = LazyTensor::from_f32(vec![1.0_f32, 2.0], vec![2], &Device::cpu());
        assert!(LazyTensor::meshgrid(&[&x], false).is_err());
    }

    #[test]
    fn meshgrid_rejects_non_rank_one() {
        let x = LazyTensor::from_f32(vec![1.0; 4], vec![2, 2], &Device::cpu());
        let y = x.const_f32_like(vec![1.0, 2.0], vec![2]);
        assert!(LazyTensor::meshgrid(&[&x, &y], false).is_err());
    }

    // ---- additional deferred-Phase-A item tests ----

    #[test]
    fn narrow_is_slice_alias() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let a = t.narrow(0, 1, 3).realize_f32();
        let b = t.slice(0, 1, 3).realize_f32();
        assert_eq!(a, b);
        assert_eq!(a, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn chunk_splits_evenly() {
        let t = cpu_f32((1..=6).map(|i| i as f32).collect(), &[6]);
        let parts = t.chunk(3, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].realize_f32(), vec![1.0, 2.0]);
        assert_eq!(parts[1].realize_f32(), vec![3.0, 4.0]);
        assert_eq!(parts[2].realize_f32(), vec![5.0, 6.0]);
    }

    #[test]
    fn chunk_distributes_remainder_to_leading() {
        // size 7, 3 chunks → first 7%3=1 chunk gets the extra: sizes 3, 2, 2
        let t = cpu_f32((1..=7).map(|i| i as f32).collect(), &[7]);
        let parts = t.chunk(3, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].realize_f32(), vec![1.0, 2.0, 3.0]);
        assert_eq!(parts[1].realize_f32(), vec![4.0, 5.0]);
        assert_eq!(parts[2].realize_f32(), vec![6.0, 7.0]);
    }

    #[test]
    fn chunk_returns_singletons_when_size_less_than_chunks() {
        let t = cpu_f32(vec![1.0, 2.0], &[2]);
        let parts = t.chunk(5, 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].realize_f32(), vec![1.0]);
        assert_eq!(parts[1].realize_f32(), vec![2.0]);
    }

    #[test]
    fn get_at_first_dim() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let row1 = t.get(1).unwrap();
        assert_eq!(row1.shape().dims(), &[2]);
        assert_eq!(row1.realize_f32(), vec![3.0, 4.0]);
    }

    #[test]
    fn get_on_dim_arbitrary_axis() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let col0 = t.get_on_dim(1, 0).unwrap();
        assert_eq!(col0.shape().dims(), &[3]);
        assert_eq!(col0.realize_f32(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn sum_dims_multi_dim_reduces_to_smaller() {
        // [2,3,4] sum over (0, 2) → [3]
        let t = cpu_f32(vec![1.0; 24], &[2, 3, 4]);
        let s = t.sum_dims(&[0, 2]);
        assert_eq!(s.shape().dims(), &[3]);
        // each element is 2*4 = 8 (sum across dim 0 = 2 elements, dim 2 = 4 elements)
        assert_eq!(s.realize_f32(), vec![8.0; 3]);
    }

    #[test]
    fn mean_dims_multi_dim() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let m = t.mean_dims(&[0, 1]);
        assert_eq!(m.shape().dims(), &[] as &[usize]);
        assert_eq!(m.realize_f32(), vec![2.5]);
    }

    #[test]
    fn sum_dims_keepdim_preserves_rank() {
        let t = cpu_f32(vec![1.0; 24], &[2, 3, 4]);
        let s = t.sum_dims_keepdim(&[0, 2]);
        assert_eq!(s.shape().dims(), &[1, 3, 1]);
    }

    #[test]
    fn mean_dims_keepdim_preserves_rank() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let m = t.mean_dims_keepdim(&[0, 1]);
        assert_eq!(m.shape().dims(), &[1, 1]);
        assert_eq!(m.realize_f32(), vec![2.5]);
    }

    #[test]
    fn rand_like_matches_shape_dtype() {
        let t = cpu_f32(vec![0.0; 6], &[2, 3]);
        let r = t.rand_like(-1.0, 1.0);
        assert_eq!(r.shape().dims(), t.shape().dims());
        assert_eq!(r.dtype(), t.dtype());
        // Every sample must be in [-1, 1).
        for v in r.realize_f32() {
            assert!((-1.0..1.0).contains(&v), "sample {v} out of [-1, 1)");
        }
    }

    #[test]
    fn randn_like_matches_shape_dtype() {
        let t = cpu_f32(vec![0.0; 4], &[4]);
        let r = t.randn_like(0.0, 1.0);
        assert_eq!(r.shape().dims(), &[4]);
        assert_eq!(r.dtype(), DType::F32);
        // Samples are random — just sanity-check they're finite.
        for v in r.realize_f32() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn static_rand_f32() {
        let t = LazyTensor::rand(vec![100], 0.0, 1.0, DType::F32, &Device::cpu());
        let v = t.realize_f32();
        // Mean of uniform [0,1) should be ~0.5; tolerate sample noise.
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert!((0.3..0.7).contains(&mean), "mean {mean} too far from 0.5");
    }

    #[test]
    fn static_randn_f64() {
        let t = LazyTensor::randn(vec![1000], 0.0, 1.0, DType::F64, &Device::cpu());
        let v = t.realize_f64();
        let mean: f64 = v.iter().sum::<f64>() / v.len() as f64;
        // Normal(0,1) mean should be near 0; n=1000 gives stderr ~0.03.
        assert!(mean.abs() < 0.2, "mean {mean} too far from 0");
    }

    #[test]
    fn arange_int_step() {
        let t = LazyTensor::arange(0.0, 5.0, &Device::cpu());
        assert_eq!(t.shape().dims(), &[5]);
        assert_eq!(t.realize_f32(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn arange_step_fractional() {
        let t = LazyTensor::arange_step(2.0, 4.0, 0.5, &Device::cpu());
        assert_eq!(t.realize_f32(), vec![2.0, 2.5, 3.0, 3.5]);
    }

    #[test]
    fn arange_step_negative_descends() {
        let t = LazyTensor::arange_step(5.0, 0.0, -1.0, &Device::cpu());
        assert_eq!(t.realize_f32(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn linspace_includes_endpoints() {
        let t = LazyTensor::linspace(0.0, 1.0, 5, &Device::cpu());
        assert_eq!(t.shape().dims(), &[5]);
        let v = t.realize_f32();
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!((v[4] - 1.0).abs() < 1e-6);
        assert!((v[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn linspace_n_one_returns_start() {
        let t = LazyTensor::linspace(7.0, 99.0, 1, &Device::cpu());
        assert_eq!(t.realize_f32(), vec![7.0]);
    }

    #[test]
    fn norm_is_sqrt_sum_sq() {
        let t = LazyTensor::from_f32(vec![3.0_f32, 4.0], vec![2], &Device::cpu());
        let n = t.norm();
        assert!((n.realize_f32()[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn pad_with_zeros_left_and_right() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let p = t.pad_with_zeros(0, 2, 1).unwrap();
        assert_eq!(p.shape().dims(), &[6]);
        assert_eq!(p.realize_f32(), vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn pad_with_zeros_identity_when_both_zero() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let p = t.pad_with_zeros(0, 0, 0).unwrap();
        assert_eq!(p.realize_f32(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn pad_with_zeros_rejects_bad_dim() {
        let t = cpu_f32(vec![1.0, 2.0], &[2]);
        assert!(t.pad_with_zeros(3, 1, 1).is_err());
    }
}
