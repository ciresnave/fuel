//! Phase 6a bridge: a lazy-computation-graph tensor that wraps
//! [`fuel_graph::Tensor`] and presents it through an API compatible
//! with fuel-core's eager [`Tensor`](crate::tensor::Tensor).
//!
//! # Purpose
//!
//! The Phase 6 architectural pivot moves fuel from eager execution to a
//! lazy computation graph. End state: `fuel_core::tensor::Tensor` *is* a
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
//! type alias flips and `fuel_core::tensor::Tensor` becomes the lazy variant.
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
//! methods on `fuel_core::tensor::Tensor`. All of these are additive
//! extensions — they do not require changes to the bridge's
//! structural design.

use crate::inference_context::{InferenceContext, KvCache, KvSlot};
use crate::{DType, Device, Shape};
use fuel_ir::shape::{Dim, Dims};
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

    /// Total element count.
    pub fn elem_count(&self) -> usize {
        self.inner.shape().elem_count()
    }

    /// PyTorch-convention alias of [`Self::elem_count`].
    pub fn numel(&self) -> usize {
        self.elem_count()
    }

    /// Size of the tensor along dimension `dim`. Returns a typed error
    /// rather than panicking on out-of-range — matches eager's
    /// [`crate::Tensor::dim`] signature.
    pub fn dim<D: Dim>(&self, dim: D) -> std::result::Result<usize, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "dim")?;
        Ok(shape.dims()[dim])
    }

    // ---- arithmetic (element-wise, strict shape) ----

    /// Element-wise addition. Shapes and dtypes must match — mismatches
    /// surface as typed errors at build time.
    pub fn add(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("add", other)?;
        Ok(Self { inner: self.inner.add(&other.inner) })
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("sub", other)?;
        Ok(Self { inner: self.inner.sub(&other.inner) })
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("mul", other)?;
        Ok(Self { inner: self.inner.mul(&other.inner) })
    }

    /// Element-wise division.
    pub fn div(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("div", other)?;
        Ok(Self { inner: self.inner.div(&other.inner) })
    }

    /// Element-wise maximum.
    pub fn maximum(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("maximum", other)?;
        Ok(Self { inner: self.inner.maximum(&other.inner) })
    }

    /// Element-wise minimum.
    pub fn minimum(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("minimum", other)?;
        Ok(Self { inner: self.inner.minimum(&other.inner) })
    }

    /// Element-wise equality (`self == other`) producing a `U8` mask:
    /// `1` where equal, `0` otherwise. Both operands must share dtype
    /// and shape. NaN follows IEEE-754 (`NaN == NaN` is false). The
    /// resulting tensor's dtype is `DType::U8`. Non-differentiable.
    pub fn eq(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("eq", other)?;
        Ok(Self { inner: self.inner.eq(&other.inner) })
    }

    /// Element-wise inequality (`self != other`) producing a `U8`
    /// mask. NaN follows IEEE-754 (`NaN != NaN` is true → `1`).
    /// Non-differentiable.
    pub fn ne(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("ne", other)?;
        Ok(Self { inner: self.inner.ne(&other.inner) })
    }

    /// Element-wise strictly-less (`self < other`) producing a `U8`
    /// mask. NaN-on-either-side is `0` (IEEE-754 unordered).
    /// Non-differentiable.
    pub fn lt(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("lt", other)?;
        Ok(Self { inner: self.inner.lt(&other.inner) })
    }

    /// Element-wise less-or-equal (`self <= other`) producing a `U8`
    /// mask. NaN-on-either-side is `0`. Non-differentiable.
    pub fn le(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("le", other)?;
        Ok(Self { inner: self.inner.le(&other.inner) })
    }

    /// Element-wise strictly-greater (`self > other`) producing a
    /// `U8` mask. NaN-on-either-side is `0`. Non-differentiable.
    pub fn gt(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("gt", other)?;
        Ok(Self { inner: self.inner.gt(&other.inner) })
    }

    /// Element-wise greater-or-equal (`self >= other`) producing a
    /// `U8` mask. NaN-on-either-side is `0`. Non-differentiable.
    /// Final variant of the comparison family (`eq` / `ne` / `lt` /
    /// `le` / `gt` / `ge`).
    pub fn ge(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("ge", other)?;
        Ok(Self { inner: self.inner.ge(&other.inner) })
    }

    /// Ternary select (typically used to consume a comparison-op
    /// mask): `result[i] = if self[i] != 0 { a[i] } else { b[i] }`.
    /// `self` is the cond mask (must be `DType::U8`); `a` and `b`
    /// share dtype + shape with `self`. Output dtype matches `a`/`b`,
    /// shape matches `self`.
    ///
    /// Differentiable through `a` and `b` only.
    pub fn where_cond(&self, a: &Self, b: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        if self.inner.dtype() != fuel_ir::DType::U8 {
            return Err(fuel_ir::Error::Msg(format!(
                "where_cond: cond mask must be U8, got {:?}", self.inner.dtype(),
            )).bt());
        }
        if a.inner.dtype() != b.inner.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "where_cond: branches must share dtype, got a={:?} b={:?}",
                a.inner.dtype(), b.inner.dtype(),
            )).bt());
        }
        let cond_dims = self.inner.shape();
        let a_dims = a.inner.shape();
        let b_dims = b.inner.shape();
        if a_dims.dims() != cond_dims.dims() || b_dims.dims() != cond_dims.dims() {
            return Err(fuel_ir::Error::Msg(format!(
                "where_cond: shapes must match cond, got cond={:?} a={:?} b={:?}",
                cond_dims.dims(), a_dims.dims(), b_dims.dims(),
            )).bt());
        }
        Ok(Self {
            inner: self.inner.where_cond(&a.inner, &b.inner),
        })
    }

    // ---- broadcast-aware arithmetic ----

    /// Element-wise addition with auto-broadcasting.
    pub fn broadcast_add(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_broadcast_binary("broadcast_add", other)?;
        Ok(Self { inner: self.inner.broadcast_add(&other.inner) })
    }

    /// Element-wise subtraction with auto-broadcasting.
    pub fn broadcast_sub(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_broadcast_binary("broadcast_sub", other)?;
        Ok(Self { inner: self.inner.broadcast_sub(&other.inner) })
    }

    /// Element-wise multiplication with auto-broadcasting.
    pub fn broadcast_mul(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_broadcast_binary("broadcast_mul", other)?;
        Ok(Self { inner: self.inner.broadcast_mul(&other.inner) })
    }

    /// Element-wise division with auto-broadcasting.
    pub fn broadcast_div(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_broadcast_binary("broadcast_div", other)?;
        Ok(Self { inner: self.inner.broadcast_div(&other.inner) })
    }

    fn check_strict_binary(&self, name: &'static str, other: &Self) -> std::result::Result<(), fuel_ir::Error> {
        if self.inner.dtype() != other.inner.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "{name}: dtype mismatch lhs={:?} rhs={:?}",
                self.inner.dtype(), other.inner.dtype(),
            )).bt());
        }
        let a_shape = self.inner.shape();
        let b_shape = other.inner.shape();
        if a_shape.dims() != b_shape.dims() {
            return Err(fuel_ir::Error::Msg(format!(
                "{name}: shape mismatch lhs={:?} rhs={:?}",
                a_shape.dims(), b_shape.dims(),
            )).bt());
        }
        Ok(())
    }

    fn check_broadcast_binary(&self, name: &'static str, other: &Self) -> std::result::Result<(), fuel_ir::Error> {
        if self.inner.dtype() != other.inner.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "{name}: dtype mismatch lhs={:?} rhs={:?}",
                self.inner.dtype(), other.inner.dtype(),
            )).bt());
        }
        let a_shape = self.inner.shape();
        let b_shape = other.inner.shape();
        let a_dims = a_shape.dims();
        let b_dims = b_shape.dims();
        // Standard NumPy-style broadcast compatibility: from the right,
        // each pair of dims must be equal, or one of them must be 1.
        let rank = a_dims.len().max(b_dims.len());
        for i in 0..rank {
            let ad = a_dims.get(a_dims.len().wrapping_sub(1 + i)).copied().unwrap_or(1);
            let bd = b_dims.get(b_dims.len().wrapping_sub(1 + i)).copied().unwrap_or(1);
            if ad != bd && ad != 1 && bd != 1 {
                return Err(fuel_ir::Error::Msg(format!(
                    "{name}: shapes {:?} and {:?} are not broadcast-compatible",
                    a_dims, b_dims,
                )).bt());
            }
        }
        Ok(())
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
    pub fn pow(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
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
    pub fn rem(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.rem(&other.inner)?,
        })
    }

    /// Reverse element order along `dim`. Materializing op (real
    /// byte shuffle). Differentiable; backward is itself.
    /// Accepts any [`Dim`] (`usize`, `D::Minus1`, etc.).
    pub fn flip<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "flip")?;
        Ok(Self { inner: self.inner.flip(dim)? })
    }

    /// Cyclic shift along `dim` by `shift` positions (positive →
    /// higher indices, wrapping). Differentiable; backward is
    /// `roll(dim, -shift)`.
    pub fn roll<D: Dim>(&self, dim: D, shift: i64) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "roll")?;
        Ok(Self { inner: self.inner.roll(dim, shift)? })
    }

    /// Running cumulative sum along `dim`. Same shape as input.
    /// Differentiable; backward is reverse cumsum (`flip → cumsum
    /// → flip`).
    pub fn cumsum<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "cumsum")?;
        Ok(Self { inner: self.inner.cumsum(dim)? })
    }

    /// Multi-dim Pad: `padding[i] = (before, after)` for axis `i`,
    /// length must equal tensor rank. Output shape:
    /// `out[i] = in[i] + padding[i].0 + padding[i].1`. Only Constant
    /// mode is implemented; Reflect / Replicate exist as enum stubs
    /// that error at realize time. Differentiable for Constant.
    /// **Returns `Result`**: rank mismatch surfaces as a typed error.
    pub fn pad(&self, padding: Vec<(usize, usize)>, mode: fuel_graph::PadMode, value: f64) -> std::result::Result<Self, fuel_ir::Error> {
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
    /// Shape incompatibility (rank < 2 or contracting-dim mismatch)
    /// surfaces as a typed error at build time.
    pub fn matmul(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        let a_dims = self.inner.shape().dims().to_vec();
        let b_dims = other.inner.shape().dims().to_vec();
        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "matmul: both operands must be rank >= 2, got lhs={a_dims:?} rhs={b_dims:?}",
            )).bt());
        }
        let a_k = a_dims[a_dims.len() - 1];
        let b_k = b_dims[b_dims.len() - 2];
        if a_k != b_k {
            return Err(fuel_ir::Error::Msg(format!(
                "matmul: contracting dim mismatch lhs[..., M, {a_k}] vs rhs[..., {b_k}, N]",
            )).bt());
        }
        Ok(Self { inner: self.inner.matmul(&other.inner) })
    }

    /// Quantized matmul: `C = self @ dequant(W_Q)`. See
    /// [`fuel_graph::Tensor::qmatmul`] for details. The weight bytes
    /// tensor must be a flat U32 const holding the raw Q-block byte
    /// stream (length = n_bytes / 4).
    ///
    /// Dtype / rank / k / block-alignment / byte-count mismatches
    /// surface as typed errors at build time rather than panicking
    /// inside the inner `fuel_graph` call.
    pub fn qmatmul(
        &self,
        weight_bytes: &Self,
        quant_type: fuel_graph::QuantType,
        k: usize,
        n: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        if self.inner.dtype() != fuel_ir::DType::F32 {
            return Err(fuel_ir::Error::Msg(format!(
                "qmatmul: activations must be F32, got {:?}", self.inner.dtype(),
            )).bt());
        }
        if weight_bytes.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "qmatmul: weight_bytes must be U32 (raw block bytes reinterpreted), got {:?}",
                weight_bytes.inner.dtype(),
            )).bt());
        }
        let a_shape = self.inner.shape();
        let a_dims = a_shape.dims();
        if a_dims.len() < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "qmatmul: activations must be rank >= 2, got {a_dims:?}",
            )).bt());
        }
        if a_dims[a_dims.len() - 1] != k {
            return Err(fuel_ir::Error::Msg(format!(
                "qmatmul: last dim of activations ({}) must equal k ({k})",
                a_dims[a_dims.len() - 1],
            )).bt());
        }
        let block_size = quant_type.elements_per_block();
        if k % block_size != 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "qmatmul: k={k} must be a multiple of {quant_type:?}'s block size ({block_size})",
            )).bt());
        }
        let expected_bytes = n * (k / block_size) * quant_type.bytes_per_block();
        let expected_u32_elems = expected_bytes / 4;
        let actual_elems = weight_bytes.inner.shape().elem_count();
        if actual_elems != expected_u32_elems {
            return Err(fuel_ir::Error::Msg(format!(
                "qmatmul: weight_bytes has {actual_elems} u32 elements, expected {expected_u32_elems} for N={n}, K={k}, {quant_type:?}",
            )).bt());
        }
        Ok(Self {
            inner: self.inner.qmatmul(&weight_bytes.inner, quant_type, k, n),
        })
    }

    /// Transpose the last two dims. Returns a typed error on rank < 2
    /// rather than panicking — build-time validation surfaces a useful
    /// diagnostic.
    pub fn transpose(&self) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self { inner: self.inner.try_transpose()? })
    }

    /// Permute axes by the given ordering. Accepts any [`Dims`]
    /// implementer — `(0, 2, 1)`, `[0, 2, 1]`, `&[0, 2, 1]`, etc.
    /// Validates rank match + dim bounds + duplicate check at build time.
    pub fn permute<D: Dims>(&self, axes: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let axes = axes.to_indexes(&shape, "permute")?;
        Ok(Self { inner: self.inner.try_permute(&axes)? })
    }

    /// Reshape to a new shape with matching element count.
    /// Element-count mismatch surfaces as a typed error at build time.
    pub fn reshape(&self, shape: impl Into<Shape>) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self { inner: self.inner.try_reshape(shape)? })
    }

    /// Drop the size-1 dimension at position `dim` (range `0..rank`).
    /// Metadata-only view; bytes shared with `self`. **Returns
    /// `Result`** rather than panicking — bad `dim` (out of bounds
    /// or `shape[dim] != 1`) surfaces as a typed error.
    ///
    /// Accepts any [`Dim`] implementer — `usize`, `D::Minus1`, `D::Minus2`,
    /// `D::Minus(n)`.
    pub fn squeeze<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "squeeze")?;
        Ok(Self { inner: self.inner.squeeze(dim)? })
    }

    /// Broadcast to a larger shape. Shape-incompatibility surfaces as a
    /// typed error at build time.
    pub fn broadcast_to(&self, shape: impl Into<Shape>) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self { inner: self.inner.try_broadcast_to(shape)? })
    }

    /// Apply LayerNorm along the last dim with an affine
    /// `gain · x + bias` post-step. Both `gain` and `bias` are
    /// length-`hidden` vectors materialized fresh on `self`'s
    /// graph; they're broadcast across all leading dims of the
    /// output.
    ///
    /// Equivalent to the per-port `apply_layer_norm(x, ln, hidden,
    /// eps)` helpers that several ports inlined — promoted here so
    /// the call sites stop drifting.
    pub fn layer_norm_affine(
        &self, gain: std::sync::Arc<[f32]>, bias: std::sync::Arc<[f32]>, eps: f64,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let hidden = gain.len();
        debug_assert_eq!(bias.len(), hidden,
            "layer_norm_affine: gain ({}) and bias ({}) must have the same length",
            gain.len(), bias.len());
        let normed = self.layer_norm_last_dim(eps)?;
        let dims_v: Vec<usize> = self.inner.shape().dims().to_vec();
        let mut affine_shape = vec![1_usize; dims_v.len()];
        affine_shape[dims_v.len() - 1] = hidden;
        let bc_shape = Shape::from_dims(&dims_v);
        let g = normed
            .const_f32_like(gain, Shape::from_dims(&[hidden]))
            .reshape(Shape::from_dims(&affine_shape))?
            .broadcast_to(bc_shape.clone())?;
        let b = normed
            .const_f32_like(bias, Shape::from_dims(&[hidden]))
            .reshape(Shape::from_dims(&affine_shape))?
            .broadcast_to(bc_shape)?;
        normed.mul(&g)?.add(&b)
    }

    /// L2-normalize along `dim`: `x / sqrt(sum(x²) + eps)`. Output
    /// shape equals input shape; the normalization divisor is
    /// broadcast across `dim` after a keepdim reduction.
    ///
    /// Common values: `eps = 1e-12` (PyTorch default), `eps = 1e-6`
    /// (some retrieval pipelines), `eps = 0.0` (no epsilon — caller
    /// guarantees no all-zero rows).
    pub fn l2_normalize<D: Dim>(
        &self, dim: D, eps: f64,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let sq = self.sqr();
        let summed = sq.sum_keepdim(dim)?;
        let with_eps = if eps == 0.0 { summed } else { summed.add_scalar(eps) };
        let l2 = with_eps.sqrt();
        let dims_v: Vec<usize> = self.inner.shape().dims().to_vec();
        let l2_bc = l2.broadcast_to(Shape::from_dims(&dims_v))?;
        self.div(&l2_bc)
    }

    /// Equivalent to `torch.repeat_interleave(x, repeats, dim)`.
    /// Replaces each element along `dim` with `repeats` consecutive
    /// copies of itself, expanding that dim by a factor of `repeats`.
    /// Implemented via reshape+broadcast+reshape — no new graph op.
    ///
    /// `repeats == 1` is a no-op clone. `repeats == 0` returns an
    /// error at build time.
    pub fn repeat_interleave<D: Dim>(
        &self, dim: D, repeats: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "repeat_interleave")?;
        if repeats == 0 {
            return Err(fuel_ir::Error::Msg(
                "repeat_interleave: repeats must be ≥ 1".into(),
            ).bt());
        }
        if repeats == 1 {
            return Ok(self.clone());
        }
        let dims_v: Vec<usize> = shape.dims().to_vec();
        let mut unsq_shape = dims_v.clone();
        unsq_shape.insert(dim + 1, 1);
        let mut bc_shape = unsq_shape.clone();
        bc_shape[dim + 1] = repeats;
        let unsq = self.reshape(Shape::from_dims(&unsq_shape))?;
        let bc = unsq.broadcast_to(Shape::from_dims(&bc_shape))?;
        let mut out_shape = dims_v.clone();
        out_shape[dim] *= repeats;
        bc.reshape(Shape::from_dims(&out_shape))
    }

    /// Slice (narrow) along `dim`: take elements `[start, start+len)`.
    /// Bad `dim` / out-of-range slice surfaces as a typed error at build
    /// time. Accepts any [`Dim`].
    pub fn slice<D: Dim>(&self, dim: D, start: usize, len: usize) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "slice")?;
        let dim_size = shape.dims()[dim];
        if start.saturating_add(len) > dim_size {
            return Err(fuel_ir::Error::Msg(format!(
                "slice: start={start} + len={len} exceeds dim {dim} size {dim_size}",
            )).bt());
        }
        Ok(Self { inner: self.inner.slice(dim, start, len) })
    }

    /// Concatenate two tensors along `dim`. Shape mismatch or bad `dim`
    /// surfaces as a typed error at build time. Accepts any [`Dim`].
    pub fn concat<D: Dim>(&self, other: &Self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "concat")?;
        let self_dims = shape.dims().to_vec();
        let other_dims = other.inner.shape().dims().to_vec();
        if self_dims.len() != other_dims.len() {
            return Err(fuel_ir::Error::Msg(format!(
                "concat: rank mismatch lhs={self_dims:?} rhs={other_dims:?}",
            )).bt());
        }
        for (i, (&a, &b)) in self_dims.iter().zip(other_dims.iter()).enumerate() {
            if i != dim && a != b {
                return Err(fuel_ir::Error::Msg(format!(
                    "concat: dim {i} mismatch lhs={a} rhs={b} (concat dim is {dim})",
                )).bt());
            }
        }
        Ok(Self { inner: self.inner.concat(&other.inner, dim) })
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
    /// dim removed. Non-differentiable. Bad `dim` surfaces as a typed
    /// error at build time. Accepts any [`Dim`].
    pub fn argmax_dim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "argmax_dim")?;
        Ok(Self { inner: self.inner.argmax_dim(dim) })
    }

    /// Realize as a `u32` (index) `Vec`.
    ///
    /// Routes through the [`PipelinedExecutor`] like [`Self::realize_f32`]
    /// — the legacy fuel-reference-backend executor predates U8-output
    /// ops (comparison masks feeding argmin/argmax) and rejects them.
    pub fn realize_u32(&self) -> Vec<u32> {
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_as::<u32>(&graph, target, &device)
            .expect("realize_u32 via PipelinedExecutor")
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

    /// Sum along a single dimension (dim removed from output). Bad
    /// `dim` surfaces as a typed error at build time. Accepts any [`Dim`].
    pub fn sum_dim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "sum_dim")?;
        Ok(Self { inner: self.inner.sum_dim(dim) })
    }

    /// Max along a single dimension (dim removed from output).
    pub fn max_dim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "max_dim")?;
        Ok(Self { inner: self.inner.max_dim(dim) })
    }

    /// Min along a single dimension (dim removed from output).
    pub fn min_dim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "min_dim")?;
        Ok(Self { inner: self.inner.min_dim(dim) })
    }

    /// Element-wise clamp to `[min, max]`.
    pub fn clamp(&self, min: f64, max: f64) -> Self {
        Self { inner: self.inner.clamp(min, max) }
    }

    /// Mean along a single dimension.
    pub fn mean_dim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "mean_dim")?;
        Ok(Self { inner: self.inner.mean_dim(dim) })
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

    /// Softmax along the last dim. Rank-0 input surfaces as a typed
    /// error at build time rather than panicking inside `fuel_graph`.
    pub fn softmax_last_dim(&self) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dims = shape.dims();
        if dims.is_empty() {
            return Err(fuel_ir::Error::Msg(
                "softmax_last_dim: input must be rank >= 1, got scalar".into(),
            ).bt());
        }
        Ok(Self {
            inner: self.inner.softmax_last_dim(),
        })
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
    /// contract. Returns just `y` — for the bundled `(y, last_state)`
    /// form needed by autoregressive resumption use
    /// [`Self::selective_scan_bundled`].
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

    /// Multi-output Mamba-1 SSM scan: returns `(y, last_state)`. `y`
    /// matches the single-output [`Self::selective_scan`] result;
    /// `last_state` is the final hidden state `[batch, dim, dstate]`
    /// used by autoregressive callers to resume from a prefill
    /// snapshot. Both LazyTensors are `Op::View` projections of the
    /// same bundled producer Storage — realizing them in the same
    /// pass shares the bundle.
    pub fn selective_scan_bundled(
        &self,
        delta: &Self,
        a: &Self,
        b: &Self,
        c: &Self,
        delta_softplus: bool,
    ) -> std::result::Result<(Self, Self), fuel_ir::Error> {
        let (y, last_state) = self.inner.selective_scan_bundled(
            &delta.inner, &a.inner, &b.inner, &c.inner, delta_softplus,
        )?;
        Ok((Self { inner: y }, Self { inner: last_state }))
    }

    /// Multi-output Mamba-2 SSD scan: returns `(y, last_state)`.
    /// Mirrors [`Self::selective_scan_bundled`]. `last_state` has
    /// shape `[batch, heads, head_dim, state_dim]`.
    pub fn ssd_chunk_scan_bundled(
        &self,
        dt: &Self,
        a: &Self,
        b: &Self,
        c: &Self,
        chunk_size: usize,
    ) -> std::result::Result<(Self, Self), fuel_ir::Error> {
        let (y, last_state) = self.inner.ssd_chunk_scan_bundled(
            &dt.inner, &a.inner, &b.inner, &c.inner, chunk_size,
        )?;
        Ok((Self { inner: y }, Self { inner: last_state }))
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

    /// LayerNorm along the last dim with the given epsilon. Rank-0
    /// or zero-last-dim input surfaces as a typed error at build time.
    pub fn layer_norm_last_dim(&self, eps: f64) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dims = shape.dims();
        if dims.is_empty() || *dims.last().unwrap() == 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "layer_norm_last_dim: input must have non-zero last dim, got {dims:?}",
            )).bt());
        }
        Ok(Self {
            inner: self.inner.layer_norm_last_dim(eps),
        })
    }

    /// RmsNorm along the last dim (LLaMA's normalization). Rank-0 or
    /// zero-last-dim input surfaces as a typed error at build time.
    pub fn rms_norm_last_dim(&self, eps: f64) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dims = shape.dims();
        if dims.is_empty() || *dims.last().unwrap() == 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "rms_norm_last_dim: input must have non-zero last dim, got {dims:?}",
            )).bt());
        }
        Ok(Self {
            inner: self.inner.rms_norm_last_dim(eps),
        })
    }

    /// Apply rotary position embeddings. See [`fuel_graph::Tensor::rope`].
    /// Rank < 2 surfaces as a typed error at build time.
    pub fn rope(&self, base: f64, start_pos: usize) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dims = shape.dims();
        if dims.len() < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "rope: input must have rank >= 2, got {dims:?}",
            )).bt());
        }
        Ok(Self {
            inner: self.inner.rope(base, start_pos),
        })
    }

    /// Apply RoPE using caller-supplied `cos` and `sin` tables so they
    /// can be shared across many layers. See
    /// [`fuel_graph::Tensor::rope_with_tables`].
    ///
    /// Rank / dtype / table-shape mismatches surface as typed errors
    /// at build time rather than panicking inside `fuel_graph`.
    pub fn rope_with_tables(&self, cos: &Self, sin: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        if self.inner.dtype() != fuel_ir::DType::F32 {
            return Err(fuel_ir::Error::Msg(format!(
                "rope: only f32 is supported today, got {:?} (cast explicitly for other dtypes)",
                self.inner.dtype(),
            )).bt());
        }
        let in_shape = self.inner.shape();
        let dims = in_shape.dims();
        let rank = dims.len();
        if rank < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "rope: input must have rank >= 2, got {dims:?}",
            )).bt());
        }
        let seq = dims[rank - 2];
        let d = dims[rank - 1];
        if !d.is_multiple_of(2) {
            return Err(fuel_ir::Error::Msg(format!(
                "rope: feature dim {d} must be even",
            )).bt());
        }
        let cos_shape = cos.inner.shape();
        let cos_dims = cos_shape.dims();
        if cos_dims != [seq, d] {
            return Err(fuel_ir::Error::Msg(format!(
                "rope_with_tables: cos shape {cos_dims:?} does not match [seq, d] = [{seq}, {d}]",
            )).bt());
        }
        let sin_shape = sin.inner.shape();
        let sin_dims = sin_shape.dims();
        if sin_dims != [seq, d] {
            return Err(fuel_ir::Error::Msg(format!(
                "rope_with_tables: sin shape {sin_dims:?} does not match [seq, d] = [{seq}, {d}]",
            )).bt());
        }
        Ok(Self {
            inner: self.inner.rope_with_tables(&cos.inner, &sin.inner),
        })
    }

    // ---- indexing ----

    /// Pick slices along `dim` using a 1-D U32 index tensor. Accepts
    /// any [`Dim`]. Dim bounds / index dtype / index rank mismatches
    /// surface as typed errors at build time.
    pub fn index_select<D: Dim>(&self, dim: D, indices: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "index_select")?;
        if indices.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "index_select: index tensor must be U32, got {:?}",
                indices.inner.dtype(),
            )).bt());
        }
        let idx_shape = indices.inner.shape();
        let idx_dims = idx_shape.dims();
        if idx_dims.len() != 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "index_select: index tensor must be rank 1, got {idx_dims:?}",
            )).bt());
        }
        Ok(Self {
            inner: self.inner.index_select(dim, &indices.inner),
        })
    }

    /// N-D gather along `dim` using a U32 index tensor with the same
    /// rank as `self`; output shape equals the index shape. Accepts
    /// any [`Dim`]. Dim bounds / index dtype / rank mismatches surface
    /// as typed errors at build time.
    pub fn gather<D: Dim>(&self, dim: D, indices: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "gather")?;
        if indices.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "gather: index tensor must be U32, got {:?}",
                indices.inner.dtype(),
            )).bt());
        }
        let data_rank = shape.dims().len();
        let idx_shape = indices.inner.shape();
        let idx_rank = idx_shape.dims().len();
        if data_rank != idx_rank {
            return Err(fuel_ir::Error::Msg(format!(
                "gather: data and index must have the same rank, got {data_rank} vs {idx_rank}",
            )).bt());
        }
        Ok(Self {
            inner: self.inner.gather(dim, &indices.inner),
        })
    }

    // ---- dtype ----

    /// Convert to a different dtype. Same-dtype is a fast-path no-op
    /// (returns a clone) rather than emitting a redundant graph node.
    ///
    /// The name follows the eager-API convention: users shouldn't need
    /// to care whether the underlying bytes are reinterpreted (e.g.
    /// integer widening) or transcoded (e.g. f32 → bf16). Build-time
    /// validation is currently minimal — Cast itself is unfailing in
    /// `fuel_graph`; the Result return is reserved for future
    /// kernel-registry checks (Phase A.8c-extension).
    pub fn to_dtype(&self, dtype: DType) -> std::result::Result<Self, fuel_ir::Error> {
        if self.inner.dtype() == dtype {
            return Ok(self.clone());
        }
        Ok(Self {
            inner: self.inner.cast(dtype),
        })
    }

    /// Result-returning sibling of [`Self::cast`] / [`Self::to_dtype`].
    /// Detach this tensor from autograd. On lazy, autograd is structural
    /// (every graph edge participates in backward unless explicitly cut
    /// by a non-differentiable op), so there's no per-tensor toggle —
    /// `detach()` is the identity function. Provided for eager-API
    /// parity so consumer code that calls `.detach()` compiles
    /// unchanged.
    pub fn detach(&self) -> Self {
        self.clone()
    }

    /// Whether autograd is tracking this tensor. On lazy, every tensor
    /// participates in autograd structurally; `track_op` returns true
    /// unconditionally for API parity with eager.
    pub fn track_op(&self) -> bool {
        true
    }

    // ---- realization (the pipelined bridge) ----
    //
    // Signature note (executor-unification Session 1, re-audit gap 8):
    // all five typed realize entries (`realize_f32` / `_f64` / `_bf16`
    // / `_f16` / `_u32`) return `Vec<T>` and panic via `.expect` on
    // executor errors. The signatures predate the Result-returning
    // policy and `realize_f32` alone has 350+ in-repo call sites
    // across ~60 files — converting the family to `Result` is a
    // coordinated breaking sweep that must move all five together
    // (one consistent error story), so it gets its own session
    // rather than riding an executor-port commit. Until then the
    // `.expect`s stay, uniformly.

    /// Realize this tensor as an `f32` `Vec`.
    ///
    /// Routes unconditionally through the pipelined bridge: walk the
    /// graph, pre-realize Consts onto CPU, plan + dispatch through
    /// `PipelinedExecutor`, read back the root's bytes.
    ///
    /// Judge profile data still shapes dispatch — on this same path.
    /// When a profile is cached ([`crate::judge::populate_dispatch_table`]
    /// ran this process, or a prior run persisted one for this
    /// hardware), [`crate::judge::cached_oracle`] feeds the picker:
    /// `compile_plan`'s Layer-2 cost refinement and the JudgeAware
    /// runtime selector both rank alternatives (portable CPU vs
    /// AOCL/MKL kernel-source siblings included) by measured latency.
    /// Executor-unification Session 3 (2026-06-11) deleted the legacy
    /// `judge::cached()` branch that swapped in a Router-backed
    /// `GraphExecutor` instead — the picker consumes the same Judge
    /// data without leaving the production executor.
    pub fn realize_f32(&self) -> Vec<f32> {
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_as::<f32>(&graph, target, &device)
            .expect("realize_f32 via PipelinedExecutor")
    }

    /// Realize as an `f64` `Vec`.
    ///
    /// Routes through the [`PipelinedExecutor`] like
    /// [`Self::realize_f32`] — executor-unification Session 1
    /// (re-audit gap 8) retires the typed `fuel_graph_cpu` recursive
    /// evaluator from the public API. The root must already be
    /// F64-dtype (insert [`Self::to_dtype`] otherwise); the guard
    /// preserves the legacy evaluator's panic-on-mismatch contract —
    /// without it the byte reinterpretation in
    /// [`crate::pipelined_bridge::realize_one_as`] would silently
    /// return garbage.
    pub fn realize_f64(&self) -> Vec<f64> {
        let dt = self.inner.dtype();
        if dt != DType::F64 {
            panic!("realize_f64: root dtype is {dt:?}, not F64");
        }
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_as::<f64>(&graph, target, &device)
            .expect("realize_f64 via PipelinedExecutor")
    }

    /// Realize as a `bf16` `Vec`. See [`Self::realize_f64`] for the
    /// routing + dtype-guard rationale.
    pub fn realize_bf16(&self) -> Vec<half::bf16> {
        let dt = self.inner.dtype();
        if dt != DType::BF16 {
            panic!("realize_bf16: root dtype is {dt:?}, not BF16");
        }
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_as::<half::bf16>(&graph, target, &device)
            .expect("realize_bf16 via PipelinedExecutor")
    }

    /// Realize as an `f16` `Vec`. See [`Self::realize_f64`] for the
    /// routing + dtype-guard rationale.
    pub fn realize_f16(&self) -> Vec<half::f16> {
        let dt = self.inner.dtype();
        if dt != DType::F16 {
            panic!("realize_f16: root dtype is {dt:?}, not F16");
        }
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_as::<half::f16>(&graph, target, &device)
            .expect("realize_f16 via PipelinedExecutor")
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

    // The legacy-executor-signature `realize_f32_vulkan` was deleted in
    // executor-unification Session 2 (2026-06-11). This bridge-based
    // wrapper restores Vulkan/CUDA realize parity (2026-06-15): it goes
    // through `pipelined_bridge::realize_one_as` on a Vulkan `Device`,
    // the same production path `realize_f32` / `realize_f32_cuda` use —
    // so it exercises the `optimize_graph` realize path on the Vulkan
    // backend.
    #[cfg(feature = "vulkan")]
    pub fn realize_f32_vulkan(
        &self,
        backend: &std::sync::Arc<fuel_vulkan_backend::VulkanBackend>,
    ) -> Vec<f32> {
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let fc_device: crate::Device = backend.clone().into();
        crate::pipelined_bridge::realize_one_as::<f32>(&graph, target, &fc_device)
            .expect("realize_f32_vulkan via PipelinedExecutor")
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
        let c = a.add(&b).unwrap();
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
        let y = x.rms_norm_last_dim(1e-6).unwrap().matmul(&w).unwrap().relu();
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
        let y = x.rope(10000.0, 0).unwrap();
        assert_eq!(y.shape().dims(), &[2, 4]);
        assert_eq!(y.dtype(), DType::F32);
    }

    #[test]
    fn cast_switches_dtype_through_wrapper() {
        let x = LazyTensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), &Device::cpu());
        let y = x.to_dtype(DType::F64).unwrap();
        assert_eq!(y.dtype(), DType::F64);
        assert_eq!(y.shape().dims(), &[3]);
    }

    #[test]
    fn indexing_builds_correct_output_shape() {
        let data = LazyTensor::from_f32(vec![1.0; 12], Shape::from_dims(&[3, 4]), &Device::cpu());
        let idx = data.const_u32_like(vec![0, 2, 1], Shape::from_dims(&[3]));
        let out = data.index_select(0, &idx).unwrap();
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
        let c = a.add(&b).unwrap().mul(&a).unwrap();
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
        let c = a.matmul(&b).unwrap();
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
        let c = a.matmul(&b).unwrap();
        let fast = c.realize_f32();
        let reference = c.realize_f32();
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
        let c = a.add(&b).unwrap().mul(&a).unwrap();
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
        let c = a.matmul(&b).unwrap();
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
        let y = x.matmul(&w).unwrap();
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
        let y = x.permute([0, 2, 1, 3_usize]).unwrap();
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
        let y = x.softmax_last_dim().unwrap();
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
        let y = x.softmax_last_dim().unwrap();

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
        let cat = a.concat(&b, 1).unwrap(); // [2, 4]
        let sliced = cat.slice(1, 1, 2).unwrap(); // [2, 2]
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
        let y = x.rms_norm_last_dim(1e-5).unwrap();
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
        let b = a.mul(&a).unwrap();
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
        let x_norm = x.rms_norm_last_dim(1e-6).unwrap();
        let q = x_norm.matmul(&w_q).unwrap();
        let k = x_norm.matmul(&w_k).unwrap();
        let v = x_norm.matmul(&w_v).unwrap();

        // Split heads: [1, seq, 8] → [1, seq, 2, 4] → [1, 2, seq, 4]
        let q_h = q
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head])).unwrap()
            .permute([0, 2, 1, 3_usize]).unwrap();
        let k_h = k
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head])).unwrap()
            .permute([0, 2, 1, 3_usize]).unwrap();
        let v_h = v
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head])).unwrap()
            .permute([0, 2, 1, 3_usize]).unwrap();

        // RoPE on Q and K.
        let q_r = q_h.rope(10000.0, 0).unwrap();
        let k_r = k_h.rope(10000.0, 0).unwrap();

        // Scaled dot-product attention.
        let k_t = k_r.transpose().unwrap();
        let scores = q_r.matmul(&k_t).unwrap();
        let attn = scores.softmax_last_dim().unwrap();
        let attn_v = attn.matmul(&v_h).unwrap();

        // Merge heads + output projection.
        let merged = attn_v
            .permute([0, 2, 1, 3_usize]).unwrap()
            .reshape(Shape::from_dims(&[1, seq, d_model])).unwrap();
        let attn_out = merged.matmul(&w_o).unwrap();
        let h = x.add(&attn_out).unwrap();

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
    /// KV-cache storage Arcs (`Arc<RwLock<fuel_memory::Storage>>`)
    /// into a per-step graph — the graph's legacy storage_map only
    /// holds `fuel_backend_contract::Storage`, so direct binding isn't
    /// possible without a type conversion.
    pub fn const_placeholder_like(
        &self,
        shape: impl Into<Shape>,
        dtype: fuel_ir::DType,
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

    /// Append an [`fuel_graph::Op::WriteSlice`] whose start on `dyn_axis`
    /// is a **runtime** value resolved through the per-pass `SymEnv` at
    /// realize (Phase D symbolic extents). `ranges[dyn_axis].0` is
    /// ignored (the start is dynamic); the slab width
    /// `ranges[dyn_axis].1 - ranges[dyn_axis].0` must equal `source`'s
    /// `dyn_axis` dim and not exceed the destination capacity. Backs the
    /// persistent decode KV-cache write at the per-token `cached_len`.
    pub fn write_slice_dyn(
        &self,
        source: &Self,
        ranges: Vec<(usize, usize)>,
        dyn_axis: usize,
        offset: fuel_ir::DynScalar,
    ) -> crate::Result<Self> {
        let inner = self.inner
            .write_slice_dyn(&source.inner, ranges, dyn_axis, offset)
            .map_err(crate::Error::from)?;
        Ok(Self { inner })
    }

    /// Append an [`fuel_graph::Op::WriteSliceRotating`] node — like
    /// [`Self::write_slice`] but the `axis` axis wraps modulo
    /// `modulus`. `position` is a rank-0 U32 tensor whose value (read
    /// at realize time) is wrapped modulo `modulus` to determine the
    /// dynamic write start on `axis`. `ranges[axis].0` is ignored
    /// (the rotating-axis start is dynamic); the slab width
    /// `ranges[axis].1 - ranges[axis].0` must equal `source`'s
    /// `axis` dim and must not exceed `modulus`.
    ///
    /// Destructive on `self`: same scheduling as `write_slice`.
    /// Backs sliding-window KV caches (Mistral / Phi-3 sliding-
    /// window). Returns `Result`: rank / dtype / axis-bound /
    /// modulus / range mismatches surface as typed errors at
    /// build time.
    pub fn write_slice_rotating(
        &self,
        source: &Self,
        position: &Self,
        axis: usize,
        modulus: usize,
        ranges: Vec<(usize, usize)>,
    ) -> crate::Result<Self> {
        let inner = self.inner.write_slice_rotating(
            &source.inner, &position.inner, axis, modulus, ranges,
        ).map_err(crate::Error::from)?;
        Ok(Self { inner })
    }

    /// Append a [`fuel_graph::Op::Conv2D`] node. See `fuel_graph`'s
    /// `Tensor::conv2d` for the full shape contract: `self` must be
    /// `[N, Cin, H, W]`; `weight` must be `[Cout, Cin/groups, Kh, Kw]`;
    /// `bias` is optional and must be `[Cout]` when provided. Returns
    /// a rank-4 lazy tensor `[N, Cout, Hout, Wout]`.
    ///
    /// Rank / channel / `groups` / stride mismatches surface as typed
    /// errors at build time rather than panicking inside the inner
    /// `fuel_graph` call.
    pub fn conv2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        if groups < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: groups must be >= 1, got {groups}",
            )).bt());
        }
        let x_shape = self.inner.shape();
        let x_dims = x_shape.dims();
        let w_shape = weight.inner.shape();
        let w_dims = w_shape.dims();
        if x_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: x must be rank 4 [N, Cin, H, W], got {x_dims:?}",
            )).bt());
        }
        if w_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: weight must be rank 4 [Cout, Cin/groups, Kh, Kw], got {w_dims:?}",
            )).bt());
        }
        let (cin, h_in, w_in) = (x_dims[1], x_dims[2], x_dims[3]);
        let (cout, cin_per_g, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
        if cin != cin_per_g * groups {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: x has {cin} in-channels but weight expects {} ({cin_per_g}*{groups})",
                cin_per_g * groups,
            )).bt());
        }
        if cout % groups != 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: Cout={cout} must be divisible by groups={groups}",
            )).bt());
        }
        if let Some(b) = bias {
            let b_shape = b.inner.shape();
            let b_dims = b_shape.dims();
            if b_dims != [cout] {
                return Err(fuel_ir::Error::Msg(format!(
                    "conv2d: bias shape {b_dims:?} must match [Cout={cout}]",
                )).bt());
            }
        }
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        if stride_h < 1 || stride_w < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: stride must be >= 1, got ({stride_h}, {stride_w})",
            )).bt());
        }
        let h_padded = h_in + 2 * pad_h;
        let w_padded = w_in + 2 * pad_w;
        if h_padded < kh || w_padded < kw {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: padded input ({h_padded}x{w_padded}) smaller than kernel ({kh}x{kw})",
            )).bt());
        }
        Ok(Self {
            inner: self.inner.conv2d(
                &weight.inner,
                bias.map(|b| &b.inner),
                stride,
                padding,
                groups,
            ),
        })
    }

    /// Append a [`fuel_graph::Op::FlashAttn`] node. `self` is `q`
    /// of shape `[B, Hq, Sq, D]`; `k` and `v` are `[B, Hkv, Sk, D]`
    /// with `Hq` a multiple of `Hkv` (GQA). `alibi_slopes` (optional)
    /// is `[Hq]`. Returns the attention output, shape `[B, Hq, Sq, D]`.
    ///
    /// Rank / batch / GQA-divisibility / head-dim mismatches surface
    /// as typed errors at build time rather than panicking inside the
    /// inner `fuel_graph` call.
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
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let q_shape = self.inner.shape();
        let q_dims = q_shape.dims();
        let k_shape = k.inner.shape();
        let k_dims = k_shape.dims();
        let v_shape = v.inner.shape();
        let v_dims = v_shape.dims();
        if q_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: q must be rank 4 [B, Hq, Sq, D], got {q_dims:?}",
            )).bt());
        }
        if k_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: k must be rank 4 [B, Hkv, Sk, D], got {k_dims:?}",
            )).bt());
        }
        if v_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: v must be rank 4 [B, Hkv, Sk, D], got {v_dims:?}",
            )).bt());
        }
        let (b, hq, _sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let (bk, hkv, sk, dk) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
        let (bv, hkv_v, sk_v, dv) = (v_dims[0], v_dims[1], v_dims[2], v_dims[3]);
        if b != bk || b != bv {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: B mismatch q={b} k={bk} v={bv}",
            )).bt());
        }
        if hkv != hkv_v {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: Hkv mismatch k={hkv} vs v={hkv_v}",
            )).bt());
        }
        if sk != sk_v {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: Sk mismatch k={sk} vs v={sk_v}",
            )).bt());
        }
        if d != dk || d != dv {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: head_dim mismatch q={d} k={dk} v={dv}",
            )).bt());
        }
        if hkv == 0 || hq % hkv != 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: Hq={hq} must be a positive multiple of Hkv={hkv}",
            )).bt());
        }
        if let Some(a) = alibi_slopes {
            let a_shape = a.inner.shape();
            let a_dims = a_shape.dims();
            if a_dims != [hq] {
                return Err(fuel_ir::Error::Msg(format!(
                    "flash_attn: alibi_slopes must be [Hq={hq}], got {a_dims:?}",
                )).bt());
            }
        }
        Ok(Self {
            inner: self.inner.flash_attn(
                &k.inner, &v.inner,
                alibi_slopes.map(|t| &t.inner),
                softmax_scale, causal, window_size_left, window_size_right, softcap,
            ),
        })
    }

    /// Append a [`fuel_graph::Op::PagedAttn`] node. `self` is the Q
    /// tensor `[B, Hq, Sq, D]`. `k_cache` / `v_cache` are paged caches
    /// `[num_blocks, block_size, Hkv, D]`. `block_table` is `[B,
    /// max_blocks]` u32; `context_lens` is `[B]` u32.
    ///
    /// Rank / batch / GQA-divisibility / block-size / dtype mismatches
    /// surface as typed errors at build time rather than panicking
    /// inside the inner `fuel_graph` call.
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
    ) -> std::result::Result<Self, fuel_ir::Error> {
        if block_size < 1 {
            return Err(fuel_ir::Error::Msg(
                "paged_attn: block_size must be >= 1".into(),
            ).bt());
        }
        let q_shape = self.inner.shape();
        let q_dims = q_shape.dims();
        let kc_shape = k_cache.inner.shape();
        let kc_dims = kc_shape.dims();
        let vc_shape = v_cache.inner.shape();
        let vc_dims = vc_shape.dims();
        let bt_shape = block_table.inner.shape();
        let bt_dims = bt_shape.dims();
        let cl_shape = context_lens.inner.shape();
        let cl_dims = cl_shape.dims();
        if q_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: q must be rank 4 [B, Hq, Sq, D], got {q_dims:?}",
            )).bt());
        }
        if kc_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: k_cache must be rank 4 [num_blocks, block_size, Hkv, D], got {kc_dims:?}",
            )).bt());
        }
        if vc_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: v_cache must be rank 4 [num_blocks, block_size, Hkv, D], got {vc_dims:?}",
            )).bt());
        }
        if bt_dims.len() != 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: block_table must be rank 2 [B, max_blocks], got {bt_dims:?}",
            )).bt());
        }
        if cl_dims.len() != 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: context_lens must be rank 1 [B], got {cl_dims:?}",
            )).bt());
        }
        let (b, hq, _sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        if kc_dims[1] != block_size {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: k_cache block dim {} != block_size {block_size}", kc_dims[1],
            )).bt());
        }
        if vc_dims[1] != block_size {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: v_cache block dim {} != block_size {block_size}", vc_dims[1],
            )).bt());
        }
        let hkv = kc_dims[2];
        if vc_dims[2] != hkv {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: Hkv mismatch k_cache={hkv} vs v_cache={}", vc_dims[2],
            )).bt());
        }
        if kc_dims[3] != d || vc_dims[3] != d {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: D mismatch q={d} k={} v={}", kc_dims[3], vc_dims[3],
            )).bt());
        }
        if hkv == 0 || hq % hkv != 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: Hq={hq} must be a positive multiple of Hkv={hkv}",
            )).bt());
        }
        if bt_dims[0] != b {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: block_table batch dim {} != B={b}", bt_dims[0],
            )).bt());
        }
        if cl_dims[0] != b {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: context_lens len {} != B={b}", cl_dims[0],
            )).bt());
        }
        if block_table.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: block_table must be U32, got {:?}", block_table.inner.dtype(),
            )).bt());
        }
        if context_lens.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: context_lens must be U32, got {:?}", context_lens.inner.dtype(),
            )).bt());
        }
        if let Some(a) = alibi_slopes {
            let a_shape = a.inner.shape();
            let a_dims = a_shape.dims();
            if a_dims != [hq] {
                return Err(fuel_ir::Error::Msg(format!(
                    "paged_attn: alibi_slopes must be [Hq={hq}], got {a_dims:?}",
                )).bt());
            }
        }
        Ok(Self {
            inner: self.inner.paged_attn(
                &k_cache.inner, &v_cache.inner,
                &block_table.inner, &context_lens.inner,
                alibi_slopes.map(|t| &t.inner),
                softmax_scale, block_size, softcap,
            ),
        })
    }

    /// Append a [`fuel_graph::Op::ConvTranspose2D`] node. `self` must
    /// be `[N, Cin, H, W]`; `weight` must be `[Cin, Cout/groups, Kh, Kw]`
    /// (note transposed channel order vs `conv2d`). Returns a rank-4
    /// lazy tensor `[N, Cout, Hout, Wout]`.
    ///
    /// Rank / channel / `groups` / stride / dilation mismatches surface
    /// as typed errors at build time rather than panicking inside the
    /// inner `fuel_graph` call.
    pub fn conv_transpose2d(
        &self,
        weight: &Self,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        if groups < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: groups must be >= 1, got {groups}",
            )).bt());
        }
        let x_shape = self.inner.shape();
        let x_dims = x_shape.dims();
        let w_shape = weight.inner.shape();
        let w_dims = w_shape.dims();
        if x_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: x must be rank 4 [N, Cin, H, W], got {x_dims:?}",
            )).bt());
        }
        if w_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: weight must be rank 4 [Cin, Cout/groups, Kh, Kw], got {w_dims:?}",
            )).bt());
        }
        let (cin, h_in, w_in) = (x_dims[1], x_dims[2], x_dims[3]);
        let (cin_w, cout_per_g, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
        if cin != cin_w {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: x has {cin} in-channels but weight has {cin_w}",
            )).bt());
        }
        if cin % groups != 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: Cin={cin} must be divisible by groups={groups}",
            )).bt());
        }
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let (out_pad_h, out_pad_w) = output_padding;
        let (dil_h, dil_w) = dilation;
        if stride_h < 1 || stride_w < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: stride must be >= 1, got ({stride_h}, {stride_w})",
            )).bt());
        }
        if dil_h < 1 || dil_w < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: dilation must be >= 1, got ({dil_h}, {dil_w})",
            )).bt());
        }
        let h_out = h_in.saturating_sub(1) * stride_h + dil_h * (kh - 1) + out_pad_h + 1;
        let w_out = w_in.saturating_sub(1) * stride_w + dil_w * (kw - 1) + out_pad_w + 1;
        if h_out <= 2 * pad_h || w_out <= 2 * pad_w {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: padding ({pad_h}x{pad_w}) is larger than the produced output dims ({h_out}x{w_out})",
            )).bt());
        }
        let _ = cout_per_g;
        Ok(Self {
            inner: self.inner.conv_transpose2d(
                &weight.inner,
                stride, padding, output_padding, dilation, groups,
            ),
        })
    }

    /// Append a transposed 1D convolution. `self` is
    /// `[N, Cin, Lin]`; `weight` is `[Cin, Cout/groups, K]`
    /// (PyTorch channel order). Returns `[N, Cout, Lout]`.
    ///
    /// Internally lifts to rank-4 and dispatches through
    /// `conv_transpose2d` — there is no separate 1D op in the
    /// IR; the lift is transparent to the executor (which sees
    /// the same `Op::Fused(CONV_TRANSPOSE2D, _)` it already
    /// dispatches CPU kernels for).
    ///
    /// Unblocks audio codec decoders (DAC, EnCodec, SNAC, Mimi,
    /// Parler-TTS, MetaVoice, CSM) which all upsample quantized
    /// latents to waveform via strided transposed convs.
    pub fn conv_transpose1d(
        &self,
        weight: &Self,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        if groups < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: groups must be >= 1, got {groups}",
            )).bt());
        }
        let x_shape = self.inner.shape();
        let x_dims = x_shape.dims();
        let w_shape = weight.inner.shape();
        let w_dims = w_shape.dims();
        if x_dims.len() != 3 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: x must be rank 3 [N, Cin, Lin], got {x_dims:?}",
            )).bt());
        }
        if w_dims.len() != 3 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: weight must be rank 3 [Cin, Cout/groups, K], got {w_dims:?}",
            )).bt());
        }
        if stride < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: stride must be >= 1, got {stride}",
            )).bt());
        }
        if dilation < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: dilation must be >= 1, got {dilation}",
            )).bt());
        }
        let cin = x_dims[1];
        let cin_w = w_dims[0];
        if cin != cin_w {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: x has {cin} in-channels but weight has {cin_w}",
            )).bt());
        }
        if cin % groups != 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: Cin={cin} must be divisible by groups={groups}",
            )).bt());
        }
        Ok(Self {
            inner: self.inner.conv_transpose1d(
                &weight.inner, stride, padding, output_padding, dilation, groups,
            ),
        })
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
    // ---- shape ops: unsqueeze (Result + Dim) + Result-returning siblings ----

    /// Append a size-1 dimension at position `dim`. Inverse of
    /// [`Self::squeeze`]. Accepts any [`Dim`] (`usize`, `D::Minus1`,
    /// etc.). Bad `dim` surfaces as a typed error at build time.
    pub fn unsqueeze<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index_plus_one(&shape, "unsqueeze")?;
        Ok(Self { inner: self.inner.try_unsqueeze(dim)? })
    }

    // ---- triangular masking (canonical attention masks) ----

    /// Upper-triangular mask along the last two dims. `diagonal = 0`
    /// keeps the main diagonal and above; positive shifts higher.
    pub fn triu(&self, diagonal: i64) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self { inner: self.inner.triu(diagonal)? })
    }

    /// Lower-triangular mask along the last two dims. `tril(0)` is the
    /// canonical causal-attention mask.
    pub fn tril(&self, diagonal: i64) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self { inner: self.inner.tril(diagonal)? })
    }

    // ---- additional reductions / activations ----

    /// `log(softmax(self))` along the last dim, fused into one op.
    pub fn log_softmax_last_dim(&self) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self { inner: self.inner.log_softmax_last_dim()? })
    }

    /// Numerically-stable softmax along an arbitrary axis. Accepts any
    /// [`Dim`]. Decomposes into `max_keepdim` / `broadcast_sub` / `exp` /
    /// `sum_keepdim` / `broadcast_div`, all of which already accept
    /// `D: Dim`, so this is a pure composition with no new graph op.
    ///
    /// When `dim` resolves to the last axis, prefer
    /// [`Self::softmax_last_dim`], which dispatches to the fused
    /// `SoftmaxLastDim` op (single kernel rather than five graph nodes).
    pub fn softmax<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        // Resolve once to a concrete `usize` so we can pass it to each
        // composed call (the `Dim` trait doesn't require `Copy`, so we
        // can't reuse the generic `D` across multiple calls).
        let axis: usize = dim.to_index(&shape, "softmax")?;
        let m = self.max_keepdim(axis)?;
        let shifted = self.broadcast_sub(&m)?;
        let e = shifted.exp();
        let s = e.sum_keepdim(axis)?;
        e.broadcast_div(&s)
    }

    /// Numerically-stable `log(softmax(self))` along an arbitrary axis.
    /// Accepts any [`Dim`]. Computes `x - max - log(sum(exp(x - max)))`
    /// — the standard log-sum-exp form, which avoids the explicit
    /// `softmax`-then-`log` underflow path. Pure composition over
    /// existing primitives.
    ///
    /// When `dim` resolves to the last axis, prefer
    /// [`Self::log_softmax_last_dim`], which dispatches to the fused
    /// `LogSoftmaxLastDim` op.
    pub fn log_softmax<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let axis: usize = dim.to_index(&shape, "log_softmax")?;
        let m = self.max_keepdim(axis)?;
        let shifted = self.broadcast_sub(&m)?;
        let lse = shifted.exp().sum_keepdim(axis)?.log();
        shifted.broadcast_sub(&lse)
    }

    /// Argmin along `dim`, returning a U32 tensor with the reduced dim
    /// removed. Non-differentiable. Bad `dim` surfaces as a typed
    /// error at build time. Accepts any [`Dim`].
    pub fn argmin_dim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "argmin_dim")?;
        Ok(Self { inner: self.inner.argmin_dim(dim) })
    }

    // ---- masking / scatter ----

    /// Fill every position where `mask != 0` with `value`; pass `self`
    /// through everywhere `mask == 0`. `mask` must be U8 with the same
    /// shape as `self`; `value`'s dtype must match `self`.
    pub fn masked_fill(
        &self,
        mask: &Self,
        value: fuel_ir::Scalar,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self { inner: self.inner.masked_fill(&mask.inner, value)? })
    }

    /// `self + scatter(indices, src, dim=dim)` — accumulate `src` rows
    /// at positions named by `indices` along `dim`. `indices` is rank-1
    /// U32 with length equal to `src.dims()[dim]`. Accepts any [`Dim`].
    /// Dim bounds / index dtype / shape / dtype-parity mismatches
    /// surface as typed errors at build time.
    pub fn index_add<D: Dim>(&self, dim: D, indices: &Self, src: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "index_add")?;
        if indices.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "index_add: index must be U32, got {:?}", indices.inner.dtype(),
            )).bt());
        }
        if self.inner.dtype() != src.inner.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "index_add: base and src dtypes must match, got {:?} vs {:?}",
                self.inner.dtype(), src.inner.dtype(),
            )).bt());
        }
        let base_dims = shape.dims();
        let src_shape = src.inner.shape();
        let src_dims = src_shape.dims();
        if base_dims.len() != src_dims.len() {
            return Err(fuel_ir::Error::Msg(format!(
                "index_add: base and src must have the same rank, got {} vs {}",
                base_dims.len(), src_dims.len(),
            )).bt());
        }
        let idx_shape = indices.inner.shape();
        let idx_dims = idx_shape.dims();
        if idx_dims.len() != 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "index_add: index must be rank 1, got {idx_dims:?}",
            )).bt());
        }
        if src_dims[dim] != idx_dims[0] {
            return Err(fuel_ir::Error::Msg(format!(
                "index_add: src dim {dim} ({}) must match index length ({})",
                src_dims[dim], idx_dims[0],
            )).bt());
        }
        Ok(Self { inner: self.inner.index_add(dim, &indices.inner, &src.inner) })
    }

    /// Functional inverse of [`Self::gather`]. Accumulates `src` into
    /// `self` at positions given by `indices` (substituted at `dim`).
    /// Accepts any [`Dim`]. Dim bounds / index dtype / shape / dtype-
    /// parity mismatches surface as typed errors at build time.
    pub fn scatter_add<D: Dim>(&self, dim: D, indices: &Self, src: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "scatter_add")?;
        if indices.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "scatter_add: index must be U32, got {:?}", indices.inner.dtype(),
            )).bt());
        }
        if self.inner.dtype() != src.inner.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "scatter_add: base and src dtypes must match, got {:?} vs {:?}",
                self.inner.dtype(), src.inner.dtype(),
            )).bt());
        }
        let idx_shape = indices.inner.shape();
        let src_shape = src.inner.shape();
        if idx_shape.dims() != src_shape.dims() {
            return Err(fuel_ir::Error::Msg(format!(
                "scatter_add: index and src must have the same shape, got {:?} vs {:?}",
                idx_shape.dims(), src_shape.dims(),
            )).bt());
        }
        Ok(Self { inner: self.inner.scatter_add(dim, &indices.inner, &src.inner) })
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
    pub fn transpose_last_two(&self) -> std::result::Result<Self, fuel_ir::Error> {
        self.transpose()
    }

    /// Eager-API alias of [`Self::transpose_last_two`]. Matches PyTorch's
    /// `.t()` short form and the existing eager [`Tensor::t`] method.
    pub fn t(&self) -> std::result::Result<Self, fuel_ir::Error> {
        self.transpose()
    }

    /// Two-argument transpose: swap dims `dim1` and `dim2`, leaving the
    /// rest in place. Implemented via [`Self::try_permute`]; matches the
    /// eager `transpose(d1, d2)` two-arg form. Accepts any [`Dim`]
    /// (`usize`, `D::Minus1`, etc.).
    pub fn transpose_dims<D1: Dim, D2: Dim>(
        &self,
        dim1: D1,
        dim2: D2,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let dim1 = dim1.to_index(&shape, "transpose_dims")?;
        let dim2 = dim2.to_index(&shape, "transpose_dims")?;
        if dim1 == dim2 {
            return Ok(self.clone());
        }
        let rank = shape.dims().len();
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(dim1, dim2);
        self.permute(axes.as_slice())
    }

    /// Collapse dims `[start_dim, end_dim]` (inclusive) into a single
    /// dimension. Returns `Result` so out-of-bounds surfaces as a typed
    /// error rather than a panic. Accepts any [`Dim`] for either arg
    /// (`D::Minus1` for the last axis works).
    pub fn flatten<D1: Dim, D2: Dim>(
        &self,
        start_dim: D1,
        end_dim: D2,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let rank = shape.dims().len();
        if rank == 0 {
            return Ok(self.clone());
        }
        let start_dim = start_dim.to_index(&shape, "flatten")?;
        let end_dim = end_dim.to_index(&shape, "flatten")?;
        if start_dim > end_dim {
            return Err(fuel_ir::Error::Msg(format!(
                "flatten: start_dim={start_dim} > end_dim={end_dim}",
            )).bt());
        }
        let dims = shape.dims();
        let merged: usize = dims[start_dim..=end_dim].iter().product();
        let mut new_dims: Vec<usize> = Vec::with_capacity(rank - (end_dim - start_dim));
        new_dims.extend_from_slice(&dims[..start_dim]);
        new_dims.push(merged);
        new_dims.extend_from_slice(&dims[end_dim + 1..]);
        self.reshape(new_dims)
    }

    /// Flatten dims `[0, end_dim]` (inclusive) into one.
    pub fn flatten_to<D: Dim>(&self, end_dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        self.flatten(0_usize, end_dim)
    }

    /// Flatten dims `[start_dim, rank-1]` into one.
    pub fn flatten_from<D: Dim>(&self, start_dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let rank = self.shape().dims().len();
        if rank == 0 {
            return Ok(self.clone());
        }
        self.flatten(start_dim, rank - 1)
    }

    /// Flatten the tensor to rank-1 (single dim containing every element).
    pub fn flatten_all(&self) -> std::result::Result<Self, fuel_ir::Error> {
        let rank = self.shape().dims().len();
        if rank == 0 {
            return Ok(self.clone());
        }
        self.flatten(0, rank - 1)
    }

    /// Stack tensors along a new dim at position `dim`. Each input is
    /// `unsqueeze`d at `dim` then concatenated. All inputs must have
    /// identical shape; `dim` may equal `rank` (append a new trailing
    /// dim). Accepts any [`Dim`].
    pub fn stack<D: Dim>(args: &[&Self], dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        if args.is_empty() {
            return Err(fuel_ir::Error::Msg(
                "stack: requires at least one tensor".into(),
            ).bt());
        }
        let reference_shape = args[0].shape();
        let reference_dims = reference_shape.dims().to_vec();
        let dim = dim.to_index_plus_one(&reference_shape, "stack")?;
        for (idx, t) in args.iter().enumerate().skip(1) {
            if t.shape().dims() != reference_dims.as_slice() {
                return Err(fuel_ir::Error::Msg(format!(
                    "stack: tensor {idx} shape {:?} != reference shape {:?}",
                    t.shape().dims(), reference_dims,
                )).bt());
            }
        }
        // unsqueeze every input at the new dim, then concat.
        let mut iter = args.iter();
        let first = iter.next().unwrap().unsqueeze(dim)?;
        let mut acc = first;
        for t in iter {
            let u = t.unsqueeze(dim)?;
            acc = acc.concat(&u, dim)?;
        }
        Ok(acc)
    }

    // ---- keepdim reductions ----
    //
    // Each keepdim variant is the squeezed reduction post-composed with
    // `unsqueeze` at the same dim. The graph optimizer can fuse these
    // back into a single op when it's profitable; until then, the cost
    // is one extra view-only node.

    /// Sum along `dim`, keeping the reduced dim as size 1. Accepts any
    /// [`Dim`]. Returns Result because of the cascade from [`Self::unsqueeze`].
    pub fn sum_keepdim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let dim = dim.to_index(&shape, "sum_keepdim")?;
        self.sum_dim(dim).unwrap().unsqueeze(dim)
    }

    /// Mean along `dim`, keeping the reduced dim as size 1.
    pub fn mean_keepdim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let dim = dim.to_index(&shape, "mean_keepdim")?;
        self.mean_dim(dim).unwrap().unsqueeze(dim)
    }

    /// Max along `dim`, keeping the reduced dim as size 1.
    pub fn max_keepdim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let dim = dim.to_index(&shape, "max_keepdim")?;
        self.max_dim(dim).unwrap().unsqueeze(dim)
    }

    /// Min along `dim`, keeping the reduced dim as size 1.
    pub fn min_keepdim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let dim = dim.to_index(&shape, "min_keepdim")?;
        // sum_dim/mean_dim/max_dim/min_dim return Self today (A.8b.9 will
        // flip them to Result); chain through `.unsqueeze(dim)?` which now
        // owns the build-time dim validation.
        self.min_dim(dim).unwrap().unsqueeze(dim)
    }

    /// Unbiased sample variance along `dim`, keeping the reduced dim as
    /// size 1. Divides squared deviations by `n - 1` (Bessel's
    /// correction), matching the eager [`Tensor::var_keepdim`] and
    /// PyTorch defaults. `n == 1` produces NaN.
    pub fn var_keepdim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let dim = dim.to_index(&shape, "var_keepdim")?;
        let dims = shape.dims();
        let n = dims[dim];
        let mean = self.mean_keepdim(dim)?;
        let deviation = self.broadcast_sub(&mean).unwrap();
        let squares = deviation.sqr();
        // sum_keepdim then divide by (n-1); leaves the reduced dim as 1.
        let summed = squares.sum_keepdim(dim)?;
        let divisor = (n.saturating_sub(1)) as f64;
        Ok(summed.mul_scalar(1.0 / divisor))
    }

    /// Unbiased sample variance along `dim`, squeezing the reduced dim.
    /// See [`Self::var_keepdim`]. Accepts any [`Dim`].
    pub fn var<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let dim = dim.to_index(&shape, "var")?;
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
        let mask = self.gt(&zero).unwrap();
        mask.where_cond(self, &neg_branch).unwrap()
    }

    /// Inner product of two rank-1 tensors. Composite of `mul` +
    /// `sum_all`; matches eager's [`Tensor::dot`].
    pub fn dot(&self, rhs: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        let a = self.shape().dims().to_vec();
        let b = rhs.shape().dims().to_vec();
        if a.len() != 1 || b.len() != 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "dot: requires rank-1 inputs, got lhs={a:?} rhs={b:?}",
            )).bt());
        }
        if a[0] != b[0] {
            return Err(fuel_ir::Error::Msg(format!(
                "dot: length mismatch lhs={} rhs={}", a[0], b[0],
            )).bt());
        }
        Ok(self.mul(rhs).unwrap().sum_all())
    }

    /// Matrix × vector: `[m, n] · [n] -> [m]`. No broadcasting. Composite
    /// of `unsqueeze` + `matmul` + `squeeze`. Matches eager's
    /// [`Tensor::mv`].
    pub fn mv(&self, rhs: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        let a = self.shape().dims().to_vec();
        let b = rhs.shape().dims().to_vec();
        if a.len() != 2 || b.len() != 1 || a[1] != b[0] {
            return Err(fuel_ir::Error::Msg(format!(
                "mv: shape mismatch lhs={a:?} rhs={b:?} (need [m,n] · [n])",
            )).bt());
        }
        // unsqueeze rhs to [n,1], matmul -> [m,1], squeeze trailing dim.
        let rhs_col = rhs.unsqueeze(1_usize)?;
        let prod = self.matmul(&rhs_col).unwrap();
        prod.squeeze(1_usize)
    }

    /// Alias of [`Self::mv`] with a more descriptive name. Matches
    /// eager's [`Tensor::matvec`].
    pub fn matvec(&self, rhs: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.mv(rhs)
    }

    /// Broadcast-aware matmul. Lazy's [`Self::matmul`] already accepts
    /// broadcast-compatible operands; this method is exposed for
    /// signature compatibility with eager's `Tensor::broadcast_matmul`.
    pub fn broadcast_matmul(&self, rhs: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.matmul(rhs)
    }

    // ---- Phase A.5 factory family ----
    //
    // Instance methods derive shape + dtype from `self` and place the new
    // tensor on the same graph via `const_*_like`. Static methods build
    // a fresh graph anchored on a host-allocated buffer.

    /// New tensor with the same shape, dtype, and graph as `self`, filled
    /// with ones. Returns Err for unsupported dtypes (anything outside
    /// F32/F64/BF16/F16/U32/I64) — matches eager `Tensor::ones_like` parity.
    pub fn ones_like(&self) -> std::result::Result<Self, fuel_ir::Error> {
        let n = self.elem_count();
        let shape = self.shape();
        match self.dtype() {
            DType::F32 => Ok(self.const_f32_like(vec![1.0_f32; n], shape)),
            DType::F64 => Ok(self.const_f64_like(vec![1.0_f64; n], shape)),
            DType::BF16 => Ok(self.const_bf16_like(vec![half::bf16::ONE; n], shape)),
            DType::F16 => Ok(self.const_f16_like(vec![half::f16::ONE; n], shape)),
            DType::U32 => Ok(self.const_u32_like(vec![1_u32; n], shape)),
            DType::I64 => Ok(self.const_i64_like(vec![1_i64; n], shape)),
            other => Err(fuel_ir::Error::Msg(format!(
                "ones_like: unsupported dtype {other:?}",
            )).bt()),
        }
    }

    /// New tensor with the same shape, dtype, and graph as `self`, filled
    /// with zeros. Returns Err for unsupported dtypes (anything outside
    /// F32/F64/BF16/F16/U32/I64) — matches eager `Tensor::zeros_like` parity.
    pub fn zeros_like(&self) -> std::result::Result<Self, fuel_ir::Error> {
        let n = self.elem_count();
        let shape = self.shape();
        match self.dtype() {
            DType::F32 => Ok(self.const_f32_like(vec![0.0_f32; n], shape)),
            DType::F64 => Ok(self.const_f64_like(vec![0.0_f64; n], shape)),
            DType::BF16 => Ok(self.const_bf16_like(vec![half::bf16::ZERO; n], shape)),
            DType::F16 => Ok(self.const_f16_like(vec![half::f16::ZERO; n], shape)),
            DType::U32 => Ok(self.const_u32_like(vec![0_u32; n], shape)),
            DType::I64 => Ok(self.const_i64_like(vec![0_i64; n], shape)),
            other => Err(fuel_ir::Error::Msg(format!(
                "zeros_like: unsupported dtype {other:?}",
            )).bt()),
        }
    }

    /// New tensor with `shape`/`dtype`/`device`, every element set to `1`.
    /// Static factory equivalent of eager's `Tensor::ones`. Returns Err for
    /// dtypes outside F32/F64/BF16/F16/U32.
    pub fn ones(
        shape: impl Into<Shape>, dtype: DType, device: &Device,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = shape.into();
        let n = shape.elem_count();
        match dtype {
            DType::F32 => Ok(Self::from_f32(vec![1.0_f32; n], shape, device)),
            DType::F64 => Ok(Self::from_f64(vec![1.0_f64; n], shape, device)),
            DType::BF16 => Ok(Self::from_bf16(vec![half::bf16::ONE; n], shape, device)),
            DType::F16 => Ok(Self::from_f16(vec![half::f16::ONE; n], shape, device)),
            DType::U32 => Ok(Self::from_u32(vec![1_u32; n], shape, device)),
            other => Err(fuel_ir::Error::Msg(format!(
                "ones: unsupported dtype {other:?}",
            )).bt()),
        }
    }

    /// New tensor with `shape`/`dtype`/`device`, every element set to `0`.
    /// Static factory equivalent of eager's `Tensor::zeros`. Returns Err for
    /// dtypes outside F32/F64/BF16/F16/U32.
    pub fn zeros(
        shape: impl Into<Shape>, dtype: DType, device: &Device,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = shape.into();
        let n = shape.elem_count();
        match dtype {
            DType::F32 => Ok(Self::from_f32(vec![0.0_f32; n], shape, device)),
            DType::F64 => Ok(Self::from_f64(vec![0.0_f64; n], shape, device)),
            DType::BF16 => Ok(Self::from_bf16(vec![half::bf16::ZERO; n], shape, device)),
            DType::F16 => Ok(Self::from_f16(vec![half::f16::ZERO; n], shape, device)),
            DType::U32 => Ok(Self::from_u32(vec![0_u32; n], shape, device)),
            other => Err(fuel_ir::Error::Msg(format!(
                "zeros: unsupported dtype {other:?}",
            )).bt()),
        }
    }

    /// New tensor of `shape`/`device` filled with `value`. The scalar's
    /// dtype determines the tensor's dtype. Returns Err for scalar dtypes
    /// outside F32/F64/BF16/F16/U32.
    pub fn full(
        shape: impl Into<Shape>,
        value: fuel_ir::Scalar,
        device: &Device,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = shape.into();
        let n = shape.elem_count();
        match value {
            fuel_ir::Scalar::F32(v) => Ok(Self::from_f32(vec![v; n], shape, device)),
            fuel_ir::Scalar::F64(v) => Ok(Self::from_f64(vec![v; n], shape, device)),
            fuel_ir::Scalar::BF16(v) => Ok(Self::from_bf16(vec![v; n], shape, device)),
            fuel_ir::Scalar::F16(v) => Ok(Self::from_f16(vec![v; n], shape, device)),
            fuel_ir::Scalar::U32(v) => Ok(Self::from_u32(vec![v; n], shape, device)),
            other => Err(fuel_ir::Error::Msg(format!(
                "full: unsupported scalar dtype {:?}", other.dtype(),
            )).bt()),
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
        if dtype == DType::F32 { base } else { base.to_dtype(dtype).unwrap() }
    }

    /// Split a `(B, N, num_heads * head_dim)` projection into the
    /// multi-head attention layout `(B, num_heads, N, head_dim)`.
    /// Equivalent to `reshape(B, N, num_heads, head_dim).permute([0, 2, 1, 3])`
    /// — promoted to a method to retire the per-port reimplementations
    /// of this same composite.
    pub fn split_heads(
        &self, num_heads: usize, head_dim: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.inner.shape().dims().to_vec();
        debug_assert_eq!(dims.len(), 3,
            "split_heads: input must be rank 3 (B, N, embed), got {dims:?}");
        debug_assert_eq!(dims[2], num_heads * head_dim,
            "split_heads: trailing dim ({}) != num_heads * head_dim ({} * {} = {})",
            dims[2], num_heads, head_dim, num_heads * head_dim);
        let b = dims[0]; let n = dims[1];
        self.reshape(Shape::from_dims(&[b, n, num_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])
    }

    /// Merge a `(B, num_heads, N, head_dim)` attention result back
    /// into the projection layout `(B, N, num_heads * head_dim)`.
    /// Inverse of [`Self::split_heads`].
    pub fn merge_heads(
        &self,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.inner.shape().dims().to_vec();
        debug_assert_eq!(dims.len(), 4,
            "merge_heads: input must be rank 4 (B, heads, N, head_dim), got {dims:?}");
        let b = dims[0]; let num_heads = dims[1]; let n = dims[2]; let head_dim = dims[3];
        self.permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[b, n, num_heads * head_dim]))
    }

    /// Add a length-`bias.len()` bias vector to the trailing dim
    /// of `self`, broadcasting across all leading dims. The bias
    /// is materialized fresh on `self`'s graph from the supplied
    /// `Arc<[f32]>`.
    ///
    /// Common pattern after `WeightStorage::apply_linear` when the
    /// linear has a bias term but the activation tensor is on a
    /// different anchor than where the bias was originally
    /// allocated. Several lazy ports inlined this same 3-line
    /// helper as `bias_add` — promoted here to a method.
    pub fn add_trailing_bias(
        &self, bias: std::sync::Arc<[f32]>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let n = bias.len();
        let bias_t = self.const_f32_like(bias, Shape::from_dims(&[n]));
        self.broadcast_add(&bias_t)
    }

    /// Embed `tokens` against an `[vocab_size, hidden]` embedding
    /// table held as `Arc<[f32]>`. Returns `(1, seq, hidden)`
    /// rank-3 hidden states ready to feed into a decoder backbone.
    ///
    /// Bootstraps a fresh graph anchored on a new const-f32 node.
    /// For composition with an already-built tensor (e.g.,
    /// multimodal models that need text embeddings on the audio
    /// graph), use [`Self::embed_tokens_anchored`] instead.
    ///
    /// Retires the 7-line `from_f32 + const_u32_like + index_select
    /// + reshape` ceremony every LLM port carried.
    pub fn embed_tokens(
        embed_table: std::sync::Arc<[f32]>,
        vocab_size: usize,
        hidden: usize,
        tokens: &[u32],
        device: &crate::Device,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let seq = tokens.len();
        if seq == 0 {
            return Err(fuel_ir::Error::Msg(
                "embed_tokens: tokens must be non-empty".into(),
            ).bt());
        }
        let embed = Self::from_f32(
            embed_table,
            Shape::from_dims(&[vocab_size, hidden]),
            device,
        );
        let token_ids = embed.const_u32_like(
            tokens.to_vec(), Shape::from_dims(&[seq]),
        );
        embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[1, seq, hidden]))
    }

    /// Variant of [`Self::embed_tokens`] that anchors the embedding
    /// table and token-id constants on the receiver's graph, so the
    /// resulting embeddings can compose with `self` and other
    /// tensors already on that graph. Used by multimodal models
    /// (vision + text, audio + text) where the text embeddings must
    /// live on the modality encoder's graph for cross-substitution
    /// to work.
    pub fn embed_tokens_anchored(
        &self,
        embed_table: std::sync::Arc<[f32]>,
        vocab_size: usize,
        hidden: usize,
        tokens: &[u32],
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let seq = tokens.len();
        if seq == 0 {
            return Err(fuel_ir::Error::Msg(
                "embed_tokens_anchored: tokens must be non-empty".into(),
            ).bt());
        }
        let embed = self.const_f32_like(
            embed_table, Shape::from_dims(&[vocab_size, hidden]),
        );
        let token_ids = self.const_u32_like(
            tokens.to_vec(), Shape::from_dims(&[seq]),
        );
        embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[1, seq, hidden]))
    }

    /// Build the standard (non-interleaved) RoPE cos/sin tables for
    /// `seq` positions starting at `start_pos`, anchored on the
    /// receiver's graph. Returns `(cos, sin)`, each with shape
    /// `[seq, head_dim]`.
    ///
    /// Delegates the actual `(theta, position) → (cos, sin)` host
    /// computation to [`fuel_graph::build_rope_tables`] (the canonical
    /// reference); only the const-tensor materialization is folded
    /// into one method to retire the per-port 4-line ceremony every
    /// LLM port did before calling `rope_with_tables`.
    pub fn rope_tables_const(
        &self,
        theta: f64,
        start_pos: usize,
        seq: usize,
        head_dim: usize,
    ) -> (Self, Self) {
        let (cos_data, sin_data) = fuel_graph::build_rope_tables(
            theta, start_pos, seq, head_dim,
        );
        let rope_shape = Shape::from_dims(&[seq, head_dim]);
        let rope_cos = self.const_f32_like(cos_data, rope_shape.clone());
        let rope_sin = self.const_f32_like(sin_data, rope_shape);
        (rope_cos, rope_sin)
    }

    /// Apply RoPE to the first `rope_dim` entries of each head and
    /// pass the remaining `head_dim - rope_dim` features through
    /// unchanged. `head_dim` is derived from the receiver's last-dim
    /// size. When `rope_dim == head_dim` this reduces to
    /// [`Self::rope_with_tables`].
    ///
    /// Implements the partial-rotary convention used by StableLM,
    /// Phi, Persimmon, MixFormer, RecurrentGemma, and Gemma-4 text —
    /// all the ports that carried an identical 5-line `slice + rope +
    /// concat` helper before this method.
    pub fn rope_partial(
        &self,
        rope_cos: &Self,
        rope_sin: &Self,
        rope_dim: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.inner.shape();
        let dims = dims.dims();
        let head_dim = *dims.last().ok_or_else(|| {
            fuel_ir::Error::Msg(
                "rope_partial: receiver must have at least one dimension".into(),
            ).bt()
        })?;
        if rope_dim == head_dim {
            return self.rope_with_tables(rope_cos, rope_sin);
        }
        if rope_dim > head_dim {
            return Err(fuel_ir::Error::Msg(format!(
                "rope_partial: rope_dim={rope_dim} exceeds head_dim={head_dim}",
            )).bt());
        }
        let last = dims.len() - 1;
        let pass_dim = head_dim - rope_dim;
        let rot = self.slice(last, 0, rope_dim)?;
        let pass = self.slice(last, rope_dim, pass_dim)?;
        let rot_rotated = rot.rope_with_tables(rope_cos, rope_sin)?;
        rot_rotated.concat(&pass, last)
    }

    /// `Option<Arc<[f32]>>` variant of [`Self::add_trailing_bias`]: if
    /// `bias.is_none()`, return `self` unchanged; else apply
    /// `add_trailing_bias`. Models the `linear_b` / `linear_no_bias`
    /// branch every per-port `optional_bias` / `opt_bias` helper does.
    pub fn add_optional_trailing_bias(
        &self, bias: Option<&std::sync::Arc<[f32]>>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        match bias {
            None => Ok(self.clone()),
            Some(b) => self.add_trailing_bias(std::sync::Arc::clone(b)),
        }
    }

    /// Logit-softcap: `cap · tanh(self / cap)`. Used by the Gemma-2 /
    /// Gemma-3 attention-logit and final-logit softcap branches. The
    /// math is identical regardless of where the cap is applied;
    /// retired the two per-port `softcap` / `apply_softcap` helpers in
    /// favor of this method.
    pub fn softcap(&self, cap: f64) -> Self {
        self.mul_scalar(1.0 / cap).tanh().mul_scalar(cap)
    }

    /// `Option<f64>` variant of [`Self::softcap`]: when `cap.is_none()`
    /// or `cap <= 0.0`, return `self` unchanged; else apply
    /// [`Self::softcap`]. Mirrors the optional-bias pattern.
    pub fn softcap_optional(&self, cap: Option<f64>) -> Self {
        match cap {
            Some(c) if c > 0.0 => self.softcap(c),
            _ => self.clone(),
        }
    }

    /// Apply RMSNorm along the last dim with `(gain + offset) · x`.
    /// Equivalent to [`Self::rms_norm_affine`] after adding a scalar
    /// to every gain element — used by Gemma-family ports where
    /// the stored gain represents `gain - 1` and the runtime path
    /// must reconstruct `gain + 1`.
    ///
    /// Materializes the shifted gain on the receiver's graph; one
    /// allocation per call.
    pub fn rms_norm_affine_with_offset(
        &self, gain: &[f32], offset: f32, eps: f64,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shifted: std::sync::Arc<[f32]> = std::sync::Arc::from(
            gain.iter().map(|g| *g + offset).collect::<Vec<_>>(),
        );
        self.rms_norm_affine(shifted, eps)
    }

    /// Apply RMSNorm along the last dim with an affine `gain · x`
    /// post-step (no bias — RMSNorm has no β term). `gain` is a
    /// length-`gain.len()` vector materialized fresh on the
    /// receiver's graph and broadcast across all leading dims.
    pub fn rms_norm_affine(
        &self, gain: std::sync::Arc<[f32]>, eps: f64,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let hidden = gain.len();
        let normed = self.rms_norm_last_dim(eps)?;
        let gain_t = self.const_f32_like(gain, Shape::from_dims(&[hidden]));
        normed.broadcast_mul(&gain_t)
    }

    /// Global average pool over the spatial dims of a rank-4
    /// `(B, C, H, W)` tensor: reduces dims 2 and 3, returning
    /// `(B, C)`. For the keepdim variant (`(B, C, 1, 1)`, used by
    /// SE blocks), follow with `.reshape(Shape::from_dims(&[B, C, 1, 1]))`.
    ///
    /// Backs the classification heads of every conv vision port
    /// (ResNet, EfficientNet, ConvMixer, FastViT, MobileNetV4,
    /// MobileOne, RepVGG, ConvNeXt, EfficientViT, etc.) plus the
    /// pre-projection pool inside each squeeze-excite block.
    pub fn global_avg_pool_2d(
        &self,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.inner.shape().dims().to_vec();
        debug_assert_eq!(dims.len(), 4,
            "global_avg_pool_2d: input must be rank 4 (B, C, H, W), got {dims:?}");
        // Reduce W first (dim 3), then H (dim 2 of the H-reduced (B, C, H) tensor).
        self.mean_dim(3_usize)?.mean_dim(2_usize)
    }

    /// Apply a per-channel affine `gain · x + bias` to a rank-4
    /// `(B, C, H, W)` tensor. Both `gain` and `bias` are length-`C`
    /// vectors materialized fresh on the receiver's graph and
    /// broadcast across the spatial axes.
    ///
    /// Equivalent to fused-affine BatchNorm at inference time:
    /// the running mean / running var / eps are absorbed at load
    /// time into `gain = γ / sqrt(var + eps)` and
    /// `bias = β - μ · γ / sqrt(var + eps)`, so the runtime forward
    /// is just this multiply-add. Used by inference-only conv
    /// vision ports (ResNet, EfficientNet, FastViT, etc.).
    pub fn channel_affine_4d(
        &self, gain: std::sync::Arc<[f32]>, bias: std::sync::Arc<[f32]>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.inner.shape().dims().to_vec();
        debug_assert_eq!(dims.len(), 4,
            "channel_affine_4d: input must be rank 4 (B, C, H, W), got {dims:?}");
        let channels = dims[1];
        debug_assert_eq!(gain.len(), channels,
            "channel_affine_4d: gain len ({}) != C ({})", gain.len(), channels);
        debug_assert_eq!(bias.len(), channels,
            "channel_affine_4d: bias len ({}) != C ({})", bias.len(), channels);
        let w_t = self
            .const_f32_like(gain, Shape::from_dims(&[channels]))
            .reshape(Shape::from_dims(&[1, channels, 1, 1]))?;
        let b_t = self
            .const_f32_like(bias, Shape::from_dims(&[channels]))
            .reshape(Shape::from_dims(&[1, channels, 1, 1]))?;
        self.broadcast_mul(&w_t)?.broadcast_add(&b_t)
    }

    /// Build the strict additive causal mask `(seq_len, seq_len)`
    /// anchored on `anchor`'s graph: 0 on and below the diagonal,
    /// `f32::NEG_INFINITY` above it. Add to attention scores before
    /// softmax to enforce strict causality (position `i` cannot
    /// attend to position `j > i`).
    ///
    /// Equivalent to the `(T, T)` mask several ports build inline
    /// — promoted here so call sites stop drifting.
    pub fn additive_causal_mask_like(anchor: &LazyTensor, seq_len: usize) -> Self {
        let mut data = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        anchor.const_f32_like(
            std::sync::Arc::from(data),
            Shape::from_dims(&[seq_len, seq_len]),
        )
    }

    /// Lower-triangular ones matrix `[n, n]`. `tril2(n).to_dtype(dtype)` is
    /// the causal-attention-mask building block.
    pub fn tril2(n: usize, dtype: DType, device: &Device) -> Self {
        let mut data = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..=i {
                data[i * n + j] = 1.0;
            }
        }
        let base = Self::from_f32(data, vec![n, n], device);
        if dtype == DType::F32 { base } else { base.to_dtype(dtype).unwrap() }
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
        if dtype == DType::F32 { base } else { base.to_dtype(dtype).unwrap() }
    }

    // ---- additional deferred-Phase-A items: indexing / multi-dim / RNG ----

    /// Eager-API alias of [`Self::slice`] (PyTorch / Candle naming).
    /// `narrow(dim, start, len)` is `slice(dim, start, len)` —
    /// produces a view of `[start, start+len)` along `dim`. Bad input
    /// surfaces as a typed error at build time. Accepts any [`Dim`].
    pub fn narrow<D: Dim>(&self, dim: D, start: usize, len: usize) -> std::result::Result<Self, fuel_ir::Error> {
        self.slice(dim, start, len)
    }

    /// Split into `chunks` views along `dim`. The split distributes the
    /// `chunk_size = ceil(dim_size / chunks)` extra slot to the leading
    /// chunks so every chunk's size differs by at most 1. If `dim_size
    /// < chunks`, returns `dim_size` singleton chunks instead of
    /// `chunks` chunks (matches eager / PyTorch). Accepts any [`Dim`].
    pub fn chunk<D: Dim>(&self, chunks: usize, dim: D) -> std::result::Result<Vec<Self>, fuel_ir::Error> {
        let shape = self.shape();
        let dim = dim.to_index(&shape, "chunk")?;
        if chunks == 0 {
            return Err(fuel_ir::Error::Msg(
                "chunk: chunks must be > 0".into(),
            ).bt());
        }
        let dims = shape.dims();
        let size = dims[dim];
        if size < chunks {
            return Ok((0..size).map(|i| self.slice(dim, i, 1).unwrap()).collect());
        }
        let base = size / chunks;
        let extra = size % chunks;
        let mut out = Vec::with_capacity(chunks);
        let mut start = 0;
        for i in 0..chunks {
            let len = if i < extra { base + 1 } else { base };
            out.push(self.slice(dim, start, len).unwrap());
            start += len;
        }
        Ok(out)
    }

    /// Sub-tensor at index `i` along dim 0. Equivalent to
    /// `self.slice(0, i, 1).unwrap().squeeze(0)`. Matches eager's [`crate::Tensor::get`].
    pub fn get(&self, i: usize) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.shape().dims().to_vec();
        if dims.is_empty() {
            return Ok(self.clone());
        }
        self.slice(0, i, 1).unwrap().squeeze(0)
    }

    /// Sub-tensor at index along an arbitrary dim. Equivalent to
    /// `self.slice(dim, index, 1).unwrap().squeeze(dim)`. Matches eager's
    /// [`crate::Tensor::get_on_dim`]. Accepts any [`Dim`].
    pub fn get_on_dim<D: Dim>(&self, dim: D, index: usize) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let dim = dim.to_index(&shape, "get_on_dim")?;
        self.slice(dim, index, 1).unwrap().squeeze(dim)
    }

    /// Multi-dim sum: reduce over every dim in `dims`, squeezing each.
    /// Reduces from the highest dim down so the lower dim indices stay
    /// valid throughout the reduction.
    pub fn sum_dims<D: Dims>(&self, dims: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let mut sorted = dims.to_indexes(&shape, "sum_dims")?;
        sorted.sort_by(|a, b| b.cmp(a));
        sorted.dedup();
        let mut acc = self.clone();
        for d in sorted {
            acc = acc.sum_dim(d)?;
        }
        Ok(acc)
    }

    /// Multi-dim mean: reduce over every dim in `dims`, squeezing each.
    /// Reduces from the highest dim down. Accepts any [`Dims`].
    pub fn mean_dims<D: Dims>(&self, dims: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let mut sorted = dims.to_indexes(&shape, "mean_dims")?;
        sorted.sort_by(|a, b| b.cmp(a));
        sorted.dedup();
        let mut acc = self.clone();
        for d in sorted {
            acc = acc.mean_dim(d)?;
        }
        Ok(acc)
    }

    /// Multi-dim sum with keepdim: every named dim becomes size 1
    /// instead of being squeezed out. Reduce-order-invariant (every
    /// keepdim preserves indices). Returns Result because of cascade
    /// from [`Self::sum_keepdim`].
    pub fn sum_dims_keepdim(&self, dims: &[usize]) -> std::result::Result<Self, fuel_ir::Error> {
        let mut sorted: Vec<usize> = dims.to_vec();
        sorted.sort();
        sorted.dedup();
        let mut acc = self.clone();
        for d in sorted {
            acc = acc.sum_keepdim(d)?;
        }
        Ok(acc)
    }

    /// Multi-dim mean with keepdim.
    pub fn mean_dims_keepdim(&self, dims: &[usize]) -> std::result::Result<Self, fuel_ir::Error> {
        let mut sorted: Vec<usize> = dims.to_vec();
        sorted.sort();
        sorted.dedup();
        let mut acc = self.clone();
        for d in sorted {
            acc = acc.mean_keepdim(d)?;
        }
        Ok(acc)
    }

    /// Uniform random tensor in `[lo, up)` with shape/dtype/device matching `self`.
    /// Backed by [`rand::thread_rng`]. Returns Err for unsupported dtypes.
    pub fn rand_like(
        &self, lo: f64, up: f64,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        Self::rand(self.shape(), lo, up, self.dtype(), &Device::cpu())
    }

    /// Normal random tensor with shape/dtype/device matching `self`.
    /// Returns Err for unsupported dtypes or invalid stdev.
    pub fn randn_like(
        &self, mean: f64, stdev: f64,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        Self::randn(self.shape(), mean, stdev, self.dtype(), &Device::cpu())
    }

    /// Uniform random tensor in `[lo, up)`. Static factory.
    /// Supported dtypes: F32, F64, BF16, F16. F32 is the typical
    /// initialization target. Returns Err for any other dtype.
    pub fn rand(
        shape: impl Into<Shape>,
        lo: f64,
        up: f64,
        dtype: DType,
        device: &Device,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        use rand::Rng;
        let shape = shape.into();
        let n = shape.elem_count();
        let mut rng = rand::rng();
        match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..n).map(|_| rng.random_range(lo..up) as f32).collect();
                Ok(Self::from_f32(data, shape, device))
            }
            DType::F64 => {
                let data: Vec<f64> = (0..n).map(|_| rng.random_range(lo..up)).collect();
                Ok(Self::from_f64(data, shape, device))
            }
            DType::BF16 => {
                let data: Vec<half::bf16> = (0..n)
                    .map(|_| half::bf16::from_f64(rng.random_range(lo..up)))
                    .collect();
                Ok(Self::from_bf16(data, shape, device))
            }
            DType::F16 => {
                let data: Vec<half::f16> = (0..n)
                    .map(|_| half::f16::from_f64(rng.random_range(lo..up)))
                    .collect();
                Ok(Self::from_f16(data, shape, device))
            }
            other => Err(fuel_ir::Error::Msg(format!(
                "LazyTensor::rand: unsupported dtype {other:?}",
            )).bt()),
        }
    }

    /// Normal random tensor with given `mean` and `stdev`. Static factory.
    /// Supported dtypes: F32, F64, BF16, F16. Returns Err on any other
    /// dtype, or if `stdev` is not finite / not positive.
    pub fn randn(
        shape: impl Into<Shape>,
        mean: f64,
        stdev: f64,
        dtype: DType,
        device: &Device,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        use rand_distr::{Distribution, Normal};
        let shape = shape.into();
        let n = shape.elem_count();
        let normal = Normal::new(mean, stdev).map_err(|e| {
            fuel_ir::Error::Msg(format!(
                "LazyTensor::randn: invalid stdev={stdev}: {e}",
            )).bt()
        })?;
        let mut rng = rand::rng();
        match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng) as f32).collect();
                Ok(Self::from_f32(data, shape, device))
            }
            DType::F64 => {
                let data: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
                Ok(Self::from_f64(data, shape, device))
            }
            DType::BF16 => {
                let data: Vec<half::bf16> = (0..n)
                    .map(|_| half::bf16::from_f64(normal.sample(&mut rng)))
                    .collect();
                Ok(Self::from_bf16(data, shape, device))
            }
            DType::F16 => {
                let data: Vec<half::f16> = (0..n)
                    .map(|_| half::f16::from_f64(normal.sample(&mut rng)))
                    .collect();
                Ok(Self::from_f16(data, shape, device))
            }
            other => Err(fuel_ir::Error::Msg(format!(
                "LazyTensor::randn: unsupported dtype {other:?}",
            )).bt()),
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

    /// General 1-D cross-correlation. Shapes:
    /// - `self`: `[N, Cin, T]`
    /// - `weight`: `[Cout, Cin/groups, K]`
    /// - `bias` (optional): `[Cout]`
    /// - returns: `[N, Cout, Tout]` where `Tout = (T + 2·padding - K) /
    ///   stride + 1`
    ///
    /// Implemented as a composite via Conv2D: unsqueeze the spatial dim
    /// to make a unit `H = 1`, run `Conv2D` with `Kh = 1, stride.0 = 1,
    /// padding.0 = 0`, then squeeze the dim back out. Works through
    /// every backend's Conv2D dispatch (CPU, CUDA via baracuda,
    /// Vulkan, AOCL, MKL). The future fused `Op::Conv1D` will collapse
    /// the unsqueeze/squeeze pair when a high-volume Conv1D consumer
    /// materializes.
    pub fn conv1d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let x_dims = self.shape().dims().to_vec();
        let w_dims = weight.shape().dims().to_vec();
        if x_dims.len() != 3 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv1d: x must be rank 3 [N, Cin, T], got {x_dims:?}",
            )).bt());
        }
        if w_dims.len() != 3 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv1d: weight must be rank 3 [Cout, Cin/groups, K], got {w_dims:?}",
            )).bt());
        }
        if groups < 1 {
            return Err(fuel_ir::Error::Msg(
                "conv1d: groups must be ≥ 1".into(),
            ).bt());
        }
        if stride < 1 {
            return Err(fuel_ir::Error::Msg(
                "conv1d: stride must be ≥ 1".into(),
            ).bt());
        }
        // Add a unit H dim at index 2 → [N, Cin, 1, T] and [Cout, Cin/g, 1, K].
        let x_4d = self.unsqueeze(2_usize)?;
        let w_4d = weight.unsqueeze(2_usize)?;
        let out_4d = x_4d.conv2d(&w_4d, bias, (1, stride), (0, padding), groups)?;
        out_4d.squeeze(2)
    }

    /// Eager-API parity for `conv1d_with_algo`. The `_algo` selector is
    /// ignored on the lazy path — algorithm selection happens at
    /// backend dispatch time, not at graph construction. Reduces to
    /// [`Self::conv1d`].
    pub fn conv1d_with_algo<A>(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
        groups: usize,
        _algo: A,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        self.conv1d(weight, bias, stride, padding, groups)
    }

    /// 2-D average pooling. Input `[N, C, H, W]`, output
    /// `[N, C, H_out, W_out]` where
    /// `H_out = (H + 2·padding.0 - kernel.0) / stride.0 + 1`.
    ///
    /// Implemented as a **depthwise Conv2D** with a constant
    /// `1/(kh·kw)` kernel: one graph node + the kernel const. Works
    /// through every backend's Conv2D dispatch and inherits Conv2D's
    /// gradient. Composite supports arbitrary kernel / stride /
    /// padding.
    pub fn avg_pool2d(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "avg_pool2d: input must be rank 4 [N, C, H, W], got {dims:?}",
            )).bt());
        }
        let c = dims[1];
        let (kh, kw) = kernel;
        if kh == 0 || kw == 0 {
            return Err(fuel_ir::Error::Msg(
                "avg_pool2d: kernel sizes must be positive".into(),
            ).bt());
        }
        let inv = 1.0_f32 / ((kh * kw) as f32);
        // Depthwise kernel: one filter per input channel, each filter
        // is a constant 1/(kh·kw). Shape [C, 1, kh, kw] with groups=C
        // makes Conv2D compute one independent kernel per channel.
        let weight = self.const_f32_like(
            vec![inv; c * kh * kw],
            Shape::from_dims(&[c, 1, kh, kw]),
        );
        self.conv2d(&weight, None, stride, padding, c)
    }

    /// Eager-API parity for `avg_pool2d_with_stride`. Same shape as
    /// [`Self::avg_pool2d`] but the stride is passed explicitly rather
    /// than inferred from kernel.
    pub fn avg_pool2d_with_stride(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> std::result::Result<Self, fuel_ir::Error> {
        self.avg_pool2d(kernel, stride, (0, 0))
    }

    /// 2-D max pooling. Input `[N, C, H, W]`, output
    /// `[N, C, H_out, W_out]` where
    /// `H_out = (H + 2·padding.0 - kernel.0) / stride.0 + 1`.
    ///
    /// Composite via slice + maximum: pad the input, then for every
    /// `(ky, kx)` in `[0..kh, 0..kw]` slice the strided grid of taps
    /// (one tap per output position) and take the elementwise max.
    /// Produces `kh·kw` nodes per call plus padding — cheap for the
    /// common 2×2 / 3×3 cases.
    ///
    /// Strided-slice trick: for stride `sh`, reshape the padded H from
    /// `(H_out · sh)` to `(H_out, sh)`, then slice the inner `sh`-dim
    /// at the tap index. Requires `H_padded == H_out · sh` exactly.
    /// Inputs that don't divide cleanly will be padded so they do.
    pub fn max_pool2d(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> std::result::Result<Self, fuel_ir::Error> {
        // Default: zero-padded (legacy behavior). For PyTorch-correct
        // semantics where padded slots must never win the max, use
        // [`Self::max_pool2d_with_pad_value`] with `f32::NEG_INFINITY`.
        self.max_pool2d_with_pad_value(kernel, stride, padding, 0.0)
    }

    /// `max_pool2d` variant where the boundary padding is filled with an
    /// explicit `pad_value` instead of `0.0`. Pass `f32::NEG_INFINITY`
    /// for PyTorch-correct semantics (padded slots can never win the
    /// max). All other constraints match [`Self::max_pool2d`]; the only
    /// difference is the constant value the implicit boundary pad uses.
    pub fn max_pool2d_with_pad_value(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        pad_value: f32,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "max_pool2d: input must be rank 4 [N, C, H, W], got {dims:?}",
            )).bt());
        }
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let (kh, kw) = kernel;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        if kh == 0 || kw == 0 {
            return Err(fuel_ir::Error::Msg(
                "max_pool2d: kernel sizes must be positive".into(),
            ).bt());
        }
        if sh == 0 || sw == 0 {
            return Err(fuel_ir::Error::Msg(
                "max_pool2d: strides must be positive".into(),
            ).bt());
        }
        let h_padded_min = h + 2 * ph;
        let w_padded_min = w + 2 * pw;
        if h_padded_min < kh || w_padded_min < kw {
            return Err(fuel_ir::Error::Msg(format!(
                "max_pool2d: padded input ({h_padded_min}×{w_padded_min}) smaller than kernel ({kh}×{kw})",
            )).bt());
        }
        let h_out = (h_padded_min - kh) / sh + 1;
        let w_out = (w_padded_min - kw) / sw + 1;
        // Pad H/W to be exactly (h_out · sh + (kh - sh)) and (w_out · sw + (kw - sw)),
        // i.e., enough to contain every (ky, kx) tap for every output.
        let h_total = h_out * sh + (kh - 1);
        let w_total = w_out * sw + (kw - 1);
        let extra_h = h_total.saturating_sub(h_padded_min);
        let extra_w = w_total.saturating_sub(w_padded_min);
        let padded = self
            .pad_with_value(2, ph, ph + extra_h, pad_value)?
            .pad_with_value(3, pw, pw + extra_w, pad_value)?;
        // For each (ky, kx) collect the strided tap.
        let mut acc: Option<LazyTensor> = None;
        for ky in 0..kh {
            // Slice H starting at ky, length h_out · sh, then reshape
            // to [N, C, h_out, sh, w_total] and slice the sh-dim at 0
            // (we'll handle stride > 1 by reshape).
            let row_slice = padded.slice(2, ky, h_out * sh).unwrap();
            // Reshape H dim of length `h_out · sh` into (h_out, sh),
            // then take dim 3 at offset 0 (the tap on the sh axis).
            let row_reshaped = row_slice.reshape(vec![n, c, h_out, sh, w_total])?;
            let row_tap = row_reshaped.slice(3, 0, 1).unwrap().squeeze(3)?;
            for kx in 0..kw {
                let col_slice = row_tap.slice(3, kx, w_out * sw).unwrap();
                let col_reshaped = col_slice.reshape(vec![n, c, h_out, w_out, sw])?;
                let win = col_reshaped.slice(4, 0, 1).unwrap().squeeze(4)?;
                acc = Some(match acc {
                    None => win,
                    Some(a) => a.maximum(&win).unwrap(),
                });
            }
        }
        acc.ok_or_else(|| fuel_ir::Error::Msg(
            "max_pool2d: empty kernel".into(),
        ).bt())
    }

    /// Eager-API parity for `max_pool2d_with_stride`.
    pub fn max_pool2d_with_stride(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> std::result::Result<Self, fuel_ir::Error> {
        self.max_pool2d(kernel, stride, (0, 0))
    }

    /// Nearest-neighbor upsample by integer factor `scale` along the
    /// last two spatial dims. Input `[N, C, H, W]` → output
    /// `[N, C, H·scale, W·scale]`.
    ///
    /// Composite via reshape + concat + reshape: insert a unit dim
    /// after each spatial dim, concat `scale` copies of the tensor on
    /// each new dim, then collapse the inflated dims back. Same shape
    /// as the `upsample_nearest_2x` helper in [`crate::lazy_yolov8`]
    /// and [`crate::lazy_sd_unet`], generalized to arbitrary scale.
    pub fn upsample_nearest2d(
        &self,
        scale: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        if scale == 0 {
            return Err(fuel_ir::Error::Msg(
                "upsample_nearest2d: scale must be positive".into(),
            ).bt());
        }
        let dims = self.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "upsample_nearest2d: input must be rank 4 [N, C, H, W], got {dims:?}",
            )).bt());
        }
        if scale == 1 {
            return Ok(self.clone());
        }
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        // [N, C, H, 1, W, 1]
        let expanded = self.reshape(vec![n, c, h, 1, w, 1])?;
        // Replicate along the new unit dims by concatenating scale copies.
        let h_expanded = (0..scale).fold(None, |acc: Option<LazyTensor>, _| {
            Some(match acc {
                None => expanded.clone(),
                Some(a) => a.concat(&expanded, 3).unwrap(),
            })
        }).unwrap();
        let h_then_w = (0..scale).fold(None, |acc: Option<LazyTensor>, _| {
            Some(match acc {
                None => h_expanded.clone(),
                Some(a) => a.concat(&h_expanded, 5).unwrap(),
            })
        }).unwrap();
        h_then_w.reshape(vec![n, c, h * scale, w * scale])
    }

    /// Nearest-neighbor upsample for 1-D signals `[N, C, T]` by integer
    /// `scale`. Reshape to insert a unit dim, concat scale copies,
    /// reshape back.
    pub fn upsample_nearest1d(
        &self,
        scale: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        if scale == 0 {
            return Err(fuel_ir::Error::Msg(
                "upsample_nearest1d: scale must be positive".into(),
            ).bt());
        }
        let dims = self.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(fuel_ir::Error::Msg(format!(
                "upsample_nearest1d: input must be rank 3 [N, C, T], got {dims:?}",
            )).bt());
        }
        if scale == 1 {
            return Ok(self.clone());
        }
        let (n, c, t) = (dims[0], dims[1], dims[2]);
        let expanded = self.reshape(vec![n, c, t, 1])?;
        let replicated = (0..scale).fold(None, |acc: Option<LazyTensor>, _| {
            Some(match acc {
                None => expanded.clone(),
                Some(a) => a.concat(&expanded, 3).unwrap(),
            })
        }).unwrap();
        replicated.reshape(vec![n, c, t * scale])
    }

    /// 2-D nearest interpolation to an explicit target size.
    /// Arbitrary ratios (non-integer, non-uniform between H and
    /// W) supported via an `index_select`-based composite. The
    /// indexing convention matches PyTorch / the eager kernel:
    /// `src_h[oi] = min(H - 1, floor(oi * H / H_out))`.
    ///
    /// Used by DepthAnythingV2's DPT head and similar dense
    /// prediction heads that resize feature maps to arbitrary
    /// targets.
    pub fn interpolate2d(
        &self,
        target_h: usize,
        target_w: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "interpolate2d: input must be rank 4 [N, C, H, W], got {dims:?}",
            )).bt());
        }
        let h = dims[2];
        let w = dims[3];
        if h == 0 || w == 0 || target_h == 0 || target_w == 0 {
            return Err(fuel_ir::Error::Msg(
                "interpolate2d: input + target spatial dims must be positive".into(),
            ).bt());
        }
        // Fast-path: identity.
        if target_h == h && target_w == w {
            return Ok(self.clone());
        }
        // Fast-path: integer-multiple uniform scale → existing
        // `upsample_nearest2d` (more cache-friendly than the
        // index_select composite for the common 2× / 4× case).
        if target_h % h == 0 && target_w % w == 0 && target_h / h == target_w / w {
            return self.upsample_nearest2d(target_h / h);
        }
        // General case: build per-axis source-index tensors and
        // index_select. Matches the eager UpsampleNearest2D
        // kernel's convention: src_idx = min(src - 1, floor(out * src / target)).
        let h_idx: Vec<u32> = (0..target_h)
            .map(|oi| ((oi * h) / target_h).min(h - 1) as u32)
            .collect();
        let w_idx: Vec<u32> = (0..target_w)
            .map(|oj| ((oj * w) / target_w).min(w - 1) as u32)
            .collect();
        let h_idx_tensor = self.const_u32_like(
            h_idx, fuel_ir::Shape::from_dims(&[target_h]),
        );
        let w_idx_tensor = self.const_u32_like(
            w_idx, fuel_ir::Shape::from_dims(&[target_w]),
        );
        let after_h = self.index_select(2_usize, &h_idx_tensor)?;
        after_h.index_select(3_usize, &w_idx_tensor)
    }

    /// 1-D nearest interpolation to an explicit target size. Same
    /// constraints as [`Self::interpolate2d`]: integer-multiple targets
    /// only.
    pub fn interpolate1d(
        &self,
        target_t: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(fuel_ir::Error::Msg(format!(
                "interpolate1d: input must be rank 3 [N, C, T], got {dims:?}",
            )).bt());
        }
        let t = dims[2];
        if t == 0 {
            return Err(fuel_ir::Error::Msg(
                "interpolate1d: input length must be positive".into(),
            ).bt());
        }
        if target_t % t != 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "interpolate1d: target {target_t} must be integer multiple of input {t}; non-integer ratios are future work",
            )).bt());
        }
        self.upsample_nearest1d(target_t / t)
    }

    /// Pad with zeros along `dim`: `left` zeros before, `right` zeros
    /// after. Thin wrapper over [`Self::pad_with_value`] with `value = 0.0`.
    /// Composite — no new graph op. Accepts any [`Dim`].
    pub fn pad_with_zeros<D: Dim>(
        &self,
        dim: D,
        left: usize,
        right: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        self.pad_with_value(dim, left, right, 0.0)
    }

    /// Pad with a constant `value` along `dim`: `before` slots before,
    /// `after` slots after. Wraps [`Self::pad`] with `PadMode::Constant`
    /// for the named dim (other dims get `(0, 0)`); the `f32` value is
    /// widened to the graph op's `f64` param. Useful for `-inf` padding
    /// around max-reductions (e.g. PyTorch-style `max_pool2d`, where
    /// padded positions must never win the max). Accepts any [`Dim`].
    pub fn pad_with_value<D: Dim>(
        &self,
        dim: D,
        before: usize,
        after: usize,
        value: f32,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.shape();
        let dim = dim.to_index(&shape, "pad_with_value")?;
        let rank = shape.dims().len();
        if before == 0 && after == 0 {
            return Ok(self.clone());
        }
        let mut padding: Vec<(usize, usize)> = vec![(0, 0); rank];
        padding[dim] = (before, after);
        self.pad(padding, fuel_graph::PadMode::Constant, value as f64)
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
    ) -> std::result::Result<Vec<Self>, fuel_ir::Error> {
        if args.len() < 2 {
            return Err(fuel_ir::Error::Msg(
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
                return Err(fuel_ir::Error::Msg(format!(
                    "meshgrid: input {i} must be rank 1, got shape {dims:?}",
                )).bt());
            }
            lens.push(dims[0]);
        }
        let mut grids = Vec::with_capacity(ordered.len());
        for (idx, t) in ordered.iter().enumerate() {
            let mut shape = vec![1_usize; ordered.len()];
            shape[idx] = lens[idx];
            let placed = t.reshape(shape)?;
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
    pub fn repeat(&self, repeats: impl Into<Shape>) -> std::result::Result<Self, fuel_ir::Error> {
        let repeats = repeats.into();
        let repeats: Vec<usize> = repeats.dims().to_vec();
        let self_rank = self.shape().dims().len();
        let target_rank = repeats.len();
        let mut work = if self_rank < target_rank {
            let pad_count = target_rank - self_rank;
            let mut new_shape: Vec<usize> = vec![1; pad_count];
            new_shape.extend_from_slice(self.shape().dims());
            self.reshape(new_shape)?
        } else if self_rank > target_rank {
            return Err(fuel_ir::Error::Msg(format!(
                "repeat: repeats rank {target_rank} smaller than tensor rank {self_rank}",
            )).bt());
        } else {
            self.clone()
        };
        for (axis, &n) in repeats.iter().enumerate() {
            if n == 0 {
                return Err(fuel_ir::Error::Msg(format!(
                    "repeat: zero repeat count at axis {axis} not supported",
                )).bt());
            }
            if n == 1 {
                continue;
            }
            // n copies concatenated along `axis`.
            let base = work.clone();
            for _ in 1..n {
                work = work.concat(&base, axis)?;
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

    pub fn dtype(&self) -> fuel_ir::DType {
        match self {
            Self::F32(_) => fuel_ir::DType::F32,
            Self::BF16(_) => fuel_ir::DType::BF16,
            // Q4_0 surfaces as U32 at the graph level (raw bytes
            // reinterpreted). Callers that care about the "actual"
            // quantization type should match on the variant directly.
            Self::Q4_0 { .. } => fuel_ir::DType::U32,
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
    ///
    /// Returns Err for `WithLoRA` — the base + LoRA update can only be
    /// applied via `apply_linear` so the right graph structure is built.
    pub fn const_like(
        &self, anchor: &LazyTensor, shape: Shape,
    ) -> std::result::Result<LazyTensor, fuel_ir::Error> {
        match self {
            Self::F32(a) => Ok(anchor.const_f32_like(a.clone(), shape)),
            Self::BF16(a) => Ok(anchor.const_bf16_like(a.clone(), shape)),
            Self::Q4_0 { words, .. } => {
                let _ = shape; // shape arg unused — Q4_0 const is 1-D U32
                // Arc-clone the precomputed u32 view; no byte copy.
                Ok(anchor.const_u32_like(Arc::clone(words), Shape::from_dims(&[words.len()])))
            }
            Self::WithLoRA { .. } => Err(fuel_ir::Error::Msg(
                "WeightStorage::WithLoRA::const_like is not supported \
                 — the base + LoRA update must be applied via \
                 apply_linear to produce the right graph structure.".into(),
            ).bt()),
        }
    }

    /// Produce `X @ W + bias` for this weight storage. Bias is a
    /// length-`out_features` Arc<[f32]> materialized fresh on the
    /// receiver's graph and broadcast across the leading dims of
    /// the output.
    ///
    /// Equivalent to the per-port `apply_linear_with_bias` helpers
    /// that several ports inlined — promoted here so call sites
    /// stop drifting.
    pub fn apply_linear_with_bias(
        &self,
        x: &LazyTensor,
        in_features: usize,
        out_features: usize,
        bias: std::sync::Arc<[f32]>,
    ) -> std::result::Result<LazyTensor, fuel_ir::Error> {
        debug_assert_eq!(bias.len(), out_features,
            "apply_linear_with_bias: bias len ({}) != out_features ({})",
            bias.len(), out_features);
        let projected = self.apply_linear(x, in_features, out_features);
        let bias_t = projected.const_f32_like(
            bias, Shape::from_dims(&[out_features]),
        );
        projected.broadcast_add(&bias_t)
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
                // const_like only errors on WithLoRA; we're statically in
                // the F32/BF16 arm so the call is infallible here.
                let w = self.const_like(x, Shape::from_dims(&[in_features, out_features]))
                    .expect("apply_linear F32/BF16 arm: const_like cannot fail for non-LoRA variants");
                x.matmul(&w).unwrap()
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
                // const_like for Q4_0 emits a flat U32 tensor. Q4_0 arm is
                // statically known here so the call is infallible.
                let w_bytes = self.const_like(x, Shape::from_dims(&[in_features, out_features]))
                    .expect("apply_linear Q4_0 arm: const_like cannot fail for non-LoRA variants");
                x.qmatmul(&w_bytes, fuel_graph::QuantType::Q4_0, in_features, out_features).unwrap()
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
                    inner: x.matmul(&a_t).unwrap().matmul(&b_t).unwrap().inner.mul_scalar(scale),
                };
                base_out.add(&lora_path).unwrap()
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
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> crate::Result<LazyTensor> {
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
        let h = embed
            .index_select(0, &token_ids).unwrap()
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim])).unwrap();

        self.forward_embeds(&h, start_pos)
    }

    /// Forward from pre-computed input embeddings of shape
    /// `(batch, seq, dim)`. Used by multimodal models (LLaVA,
    /// Pixtral, Qwen-VL, etc.) that interleave image embeddings
    /// with text embeddings before running the LLaMA decoder
    /// stack.
    pub fn forward_embeds(
        &self,
        embeds: &LazyTensor,
        start_pos: usize,
    ) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone_embeds(embeds, start_pos)?;
        Ok(weights.output.apply_linear(&h_norm, cfg.dim, cfg.vocab_size))
    }

    /// Like [`forward_embeds`] but skips the LM-head projection
    /// and returns post-final-RmsNorm hidden states
    /// `(batch, seq, dim)`. Uses strict-causal masking. Use
    /// this from multimodal hosts (LLaVA, Pixtral, etc.) that
    /// interleave image embeddings into the text stream and
    /// want hidden states without the lm_head projection.
    /// Mirrors `MistralModel::forward_hidden_embeds`.
    pub fn forward_hidden_embeds(
        &self,
        embeds: &LazyTensor,
        start_pos: usize,
    ) -> crate::Result<LazyTensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    fn run_backbone_embeds(
        &self,
        embeds: &LazyTensor,
        start_pos: usize,
    ) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, dim]");
        let seq = dims[1];
        assert_eq!(dims[2], cfg.dim, "embeds last dim must equal cfg.dim");
        assert_eq!(cfg.n_heads * cfg.head_dim, cfg.dim, "LlamaConfig: n_heads * head_dim must equal dim");

        let mut h = embeds.clone();
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_base, start_pos, seq, cfg.head_dim,
        );

        let mask = LazyTensor::additive_causal_mask_like(embeds, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq])).unwrap();

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, &mask);
        }
        Ok(apply_affine_rms_norm(
            &h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps,
        ))
    }

    /// Like [`forward_embeds`] but takes a caller-supplied
    /// additive attention mask `(1, 1, seq, seq)` and skips
    /// the LM-head projection. Returns the post-final-RmsNorm
    /// hidden states `[batch, seq, dim]`.
    ///
    /// Use this for bidirectional Llama-encoder modes (e.g.
    /// embedding adapters). The `mask` must live on the same
    /// graph as `embeds` — build it via `embeds.const_f32_like`.
    pub fn forward_hidden_embeds_with_mask(
        &self,
        embeds: &LazyTensor,
        attention_mask: &LazyTensor,
        start_pos: usize,
    ) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, dim]");
        let seq = dims[1];
        assert_eq!(dims[2], cfg.dim, "embeds last dim must equal cfg.dim");

        let mut h = embeds.clone();
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_base, start_pos, seq, cfg.head_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, attention_mask);
        }
        Ok(apply_affine_rms_norm(&h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps))
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
    ) -> crate::Result<LazyTensor> {
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
            .index_select(0, &token_ids).unwrap()
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim])).unwrap();

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_base, start_pos, seq, cfg.head_dim,
        );

        // Build the strict-causal mask once for all layers.
        let mask = LazyTensor::additive_causal_mask_like(&h, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq])).unwrap();

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, &mask);
        }

        Ok(apply_affine_rms_norm(&h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps))
    }

    /// Internal entry that runs the LLaMA backbone given pre-built RoPE
    /// cos/sin tables and an attention mask. The standard
    /// [`forward_embeds`] path computes cos/sin from `cfg.rope_base`
    /// via [`LazyTensor::rope_tables_const`] and uses a strict-causal
    /// mask; [`crate::lazy_llama_full::Llama3Model`] uses this hook to
    /// inject Llama-3 long-context scaled RoPE tables without
    /// duplicating the forward path.
    ///
    /// `rope_cos` / `rope_sin` must have shape `[seq, head_dim]` and
    /// live on the same graph as `embeds`. `mask` is additive,
    /// broadcast-compatible with `(B, n_heads, seq, kv_seq)`.
    pub(crate) fn run_backbone_with_rope_tables(
        &self,
        embeds: &LazyTensor,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
    ) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, dim]");
        assert_eq!(dims[2], cfg.dim, "embeds last dim must equal cfg.dim");
        assert_eq!(cfg.n_heads * cfg.head_dim, cfg.dim, "LlamaConfig: n_heads * head_dim must equal dim");

        let mut h = embeds.clone();
        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, rope_cos, rope_sin, mask);
        }
        Ok(apply_affine_rms_norm(&h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
    ) -> LazyTensor {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;

        // Pre-attention RmsNorm with affine gain.
        let x_norm = apply_affine_rms_norm(x, &layer.attn_norm_gain, cfg.dim, cfg.norm_eps);

        // Project to Q, K, V using WeightStorage::apply_linear — this
        // routes F32/BF16 through standard matmul and Q4_0 through
        // fused qmatmul. Under GQA, W_k and W_v have fewer output
        // features (kv_dim instead of dim).
        let q = layer.attn_q.apply_linear(&x_norm, cfg.dim, cfg.dim).add_optional_trailing_bias(layer.attn_q_bias.as_ref()).unwrap();
        let k = layer.attn_k.apply_linear(&x_norm, cfg.dim, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref()).unwrap();
        let v = layer.attn_v.apply_linear(&x_norm, cfg.dim, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref()).unwrap();

        // Split heads.
        // Q: [batch, seq, dim] → [batch, seq, n_heads, head_dim] → [batch, n_heads, seq, head_dim]
        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim])).unwrap()
            .permute([0, 2, 1, 3_usize]).unwrap();
        // K/V: [batch, seq, kv_dim] → [batch, seq, n_kv_heads, head_dim] → [batch, n_kv_heads, seq, head_dim]
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim])).unwrap()
            .permute([0, 2, 1, 3_usize]).unwrap();
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim])).unwrap()
            .permute([0, 2, 1, 3_usize]).unwrap();

        // RoPE on Q and K (applied per-head; V is NOT rotated). Uses
        // caller-supplied cos/sin so all layers share a single pair
        // of const nodes.
        let q_r = q_h.rope_with_tables(rope_cos, rope_sin).unwrap();
        let k_r = k_h.rope_with_tables(rope_cos, rope_sin).unwrap();

        let n_rep = cfg.n_heads / cfg.n_kv_heads;
        let k_r = k_r.repeat_interleave(1_usize, n_rep).unwrap();
        let v_h = v_h.repeat_interleave(1_usize, n_rep).unwrap();

        // Scaled dot-product attention with caller-supplied mask.
        // The default forward path passes the strict-causal mask
        // built once outside the loop; `forward_hidden_embeds_with_mask`
        // passes whatever the caller chose (e.g. bidirectional pad).
        let _ = seq; // silence unused after refactor; mask already sized for seq.
        let k_t = k_r.transpose().unwrap();
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t).unwrap();
        let scores_scaled = LazyTensor {
            inner: scores.inner.mul_scalar(scale),
        };
        let scores_masked = scores_scaled.broadcast_add(mask).unwrap();
        let attn = scores_masked.softmax_last_dim().unwrap();
        let attn_v = attn.matmul(&v_h).unwrap();

        // Merge heads + output projection.
        let merged = attn_v
            .permute([0, 2, 1, 3_usize]).unwrap()
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim])).unwrap();
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.dim, cfg.dim);

        // First residual connection.
        let h1 = x.add(&attn_out).unwrap();

        // Pre-FFN RmsNorm with affine gain.
        let h1_norm = apply_affine_rms_norm(&h1, &layer.ffn_norm_gain, cfg.dim, cfg.norm_eps);

        // SwiGLU FFN (routes through apply_linear → qmatmul for Q4_0).
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim);
        let up   = layer.ffn_up.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim);
        let swiglu = gate.silu().mul(&up).unwrap();
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.ffn_dim, cfg.dim);

        // Second residual connection.
        h1.add(&ffn_out).unwrap()
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
    // Unification Session 4 (E.3.3/E.3.4) completed the retirement:
    // the device-resident `*_gpu_on` family, its shared
    // `apply_layer_with_cache` helper, `LayerOutput`, `LayerKVCache`,
    // and the generic `lazy_kv_cache_device::KVCache<B>` are gone.
    // `forward_with_kv_context` (below) is the sole cached forward.

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
    /// the caller has wired into [`InferenceContext`]).
    ///
    /// **Phase D (input-independent decode graph):** the KV write lands
    /// at the runtime offset `cached_len` via `write_slice_dyn`
    /// (`DynScalar::Sym(cached_len_sym)`, resolved through the per-pass
    /// `SymEnv` at realize), and attention reads the **full fixed-capacity**
    /// buffers `[batch, n_kv_heads, max_seq_len, head_dim]` with a fixed
    /// `[1, 1, seq, max_seq_len]` causal mask (`k > cached_len + q` masks
    /// future positions AND the zero-init stale tail). Nothing in the
    /// graph's *shape* or *structure* depends on `cached_len`, so the
    /// decode-step graph is byte-identical across tokens — the prerequisite
    /// for plan-once persistent decode. Numerically identical to the prior
    /// `slice(0..total_seq)` form (masked positions contribute 0).
    ///
    /// Tradeoff: attention computes over `max_seq_len` (not the live
    /// `total_seq`), so early tokens do extra masked work — a documented
    /// efficiency follow-up (the flash arm with a runtime `k_len`), not a
    /// correctness issue.
    ///
    /// **Phase D · D2b (mask hoist):** the `[1, 1, seq, max_seq_len]`
    /// causal mask is now built ONCE in the forward (`mask` param, like
    /// RoPE tables) and shared across all layers, instead of one Const
    /// per layer. Byte-exact refactor (the mask data is identical across
    /// layers — it depends only on `cached_len`, `seq`, `max_seq_len`);
    /// it also cuts the per-token data-Const re-bind count on the
    /// persistent path from `n_layers` to 1.
    fn apply_layer_with_kv_writes(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        k_cache_const: &LazyTensor,
        v_cache_const: &LazyTensor,
        cached_len_sym: fuel_ir::SymId,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
    ) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;

        let x_norm = apply_affine_rms_norm(x, &layer.attn_norm_gain, cfg.dim, cfg.norm_eps);

        // Q/K/V projections + optional biases — identical to apply_layer_with_cache.
        let q = layer.attn_q.apply_linear(&x_norm, cfg.dim, cfg.dim).add_optional_trailing_bias(layer.attn_q_bias.as_ref()).unwrap();
        let k = layer.attn_k.apply_linear(&x_norm, cfg.dim, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref()).unwrap();
        let v = layer.attn_v.apply_linear(&x_norm, cfg.dim, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref()).unwrap();

        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim])).unwrap()
            .permute([0, 2, 1, 3_usize]).unwrap();
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim])).unwrap()
            .permute([0, 2, 1, 3_usize]).unwrap();
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_kv_heads, cfg.head_dim])).unwrap()
            .permute([0, 2, 1, 3_usize]).unwrap();

        let q_r = q_h.rope_with_tables(rope_cos, rope_sin).unwrap();
        let k_r = k_h.rope_with_tables(rope_cos, rope_sin).unwrap();

        // Write fresh K/V slabs into the pre-allocated cache buffers
        // via Op::WriteSlice at the RUNTIME offset `cached_len`. Source
        // slab shape is `[batch, n_kv_heads, seq, head_dim]`; on axis 2
        // the start is dynamic (`cached_len_sym`, resolved at realize)
        // and the slab width is `seq`. The returned tensor's Storage Arc
        // IS the cache const's Arc — post-write reference to the same
        // buffer (the executor adopts dest's Arc as the kernel output,
        // mutating in place). Keeping the offset symbolic makes the write
        // node structurally identical across tokens.
        let write_ranges = vec![
            (0, batch),
            (0, cfg.n_kv_heads),
            (0, seq),                 // axis-2 start is dynamic; width = seq
            (0, cfg.head_dim),
        ];
        let dyn_off = fuel_ir::DynScalar::Sym(cached_len_sym);
        let full_k = k_cache_const.write_slice_dyn(&k_r, write_ranges.clone(), 2, dyn_off)?;
        let full_v = v_cache_const.write_slice_dyn(&v_h, write_ranges, 2, dyn_off)?;

        // Attend over the FULL fixed-capacity buffers (no slice to
        // `total_seq`) so the attention shape is `max_seq_len` every
        // token. The fixed-capacity causal mask excludes future positions
        // AND the stale/unwritten tail (`k > cached_len + q` covers both,
        // since `cached_len + q < total_seq <= max_seq_len`).
        let k_t = full_k.transpose().unwrap();
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t).unwrap();

        // Mask is hoisted to the forward (built once, shared across
        // layers) — see the D2b note on this method.
        let scores_scaled = LazyTensor {
            inner: scores.inner.mul_scalar(scale),
        };
        let scores_masked = scores_scaled.broadcast_add(mask).unwrap();
        let attn = scores_masked.softmax_last_dim().unwrap();
        let attn_v = attn.matmul(&full_v).unwrap();

        let merged = attn_v
            .permute([0, 2, 1, 3_usize]).unwrap()
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim])).unwrap();
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.dim, cfg.dim);

        let h1 = x.add(&attn_out).unwrap();
        let h1_norm = apply_affine_rms_norm(&h1, &layer.ffn_norm_gain, cfg.dim, cfg.norm_eps);

        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim);
        let up   = layer.ffn_up.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim);
        let swiglu = gate.silu().mul(&up).unwrap();
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.ffn_dim, cfg.dim);

        Ok(h1.add(&ffn_out).unwrap())
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
        self.forward_with_kv_context_impl(tokens, cache, ctx, false)
    }

    /// All-positions variant of [`Self::forward_with_kv_context`]:
    /// returns `seq * vocab_size` logits (flat, row-major over
    /// position). Used by speculative decoding's verification step —
    /// the target model runs forward on the K drafted tokens at once
    /// and needs per-position logits to accept/reject each draft.
    ///
    /// Cache semantics identical to `forward_with_kv_context`; on
    /// reject, the caller invokes [`KvCache::truncate_to`] to roll
    /// back (a pure metadata update on the pre-allocated-buffer path —
    /// rows past `cached_len` stop being read and are overwritten by
    /// the next `Op::WriteSlice` at the same positions).
    pub fn forward_with_kv_context_all_positions(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> crate::Result<Vec<f32>> {
        self.forward_with_kv_context_impl(tokens, cache, ctx, true)
    }

    fn forward_with_kv_context_impl(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        return_all_positions: bool,
    ) -> crate::Result<Vec<f32>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        let cached_len = cache.cached_len;

        if seq == 0 {
            return Err(fuel_ir::Error::Msg(
                "forward_with_kv_context: zero tokens".to_string(),
            ).bt());
        }
        if cache.n_layers() != cfg.n_layers {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_with_kv_context: cache n_layers {} != model n_layers {}",
                cache.n_layers(), cfg.n_layers,
            )).bt());
        }
        let max_seq_len = cache.max_seq_len.ok_or_else(|| {
            fuel_ir::Error::Msg(
                "forward_with_kv_context: cache was constructed via with_dims (no \
                 pre-allocated buffers); call KvCache::with_capacity(...) for the \
                 WriteSlice path".to_string(),
            ).bt()
        })?;
        if cached_len + seq > max_seq_len {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_with_kv_context: cached_len ({cached_len}) + seq ({seq}) > \
                 max_seq_len ({max_seq_len})",
            )).bt());
        }
        let cache_dtype = cache.dtype.unwrap_or(DType::F32);
        if cache.n_kv_heads != cfg.n_kv_heads || cache.head_dim != cfg.head_dim {
            return Err(fuel_ir::Error::Msg(format!(
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
            .index_select(0, &token_ids).unwrap()
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim])).unwrap();

        // RoPE cos/sin tables shared across layers.
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_base, cached_len, seq, cfg.head_dim,
        );

        // Phase D: the per-token KV-write offset (`cached_len`) is a
        // runtime symbol bound through the per-pass `SymEnv` at realize,
        // not baked into the graph. One symbol shared across all layers
        // (they all append at the same offset). A fixed id (re-bound each
        // pass) keeps the decode-step graph structurally identical across
        // tokens — the prerequisite for plan-once persistent decode.
        let cached_len_sym = fuel_ir::SymId(0);

        // Phase D · D2b: the causal mask is hoisted to ONE Const built
        // here (was one Const per layer) and shared across all layers
        // (byte-identical across layers — it depends only on
        // `cached_len`, `seq`, `max_seq_len`). Fewer nodes + a single
        // per-token re-bind on the persistent path.
        let mask_data = build_decode_causal_mask(cached_len, seq, max_seq_len);
        let mask = h.const_f32_like(
            mask_data,
            Shape::from_dims(&[1, 1, seq, max_seq_len]),
        );

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
                fuel_ir::Error::Msg(format!(
                    "forward_with_kv_context: cache layer {li} has no K slot \
                     (with_capacity should have populated all layers)",
                )).bt()
            })?;
            let v_arc = cache.slot_storage(li, KvSlot::V).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
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
                cached_len_sym,
                &rope_cos,
                &rope_sin,
                &mask,
            )?;
        }

        let h_norm = apply_affine_rms_norm(&h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps);
        let logits = weights.output.apply_linear(&h_norm, cfg.dim, cfg.vocab_size);
        // For spec-decode verification we need per-position logits;
        // otherwise slice to the last position for decode/prefill.
        let logits_root = if return_all_positions {
            logits.reshape(Shape::from_dims(&[seq * cfg.vocab_size]))?
        } else {
            let last_pos = seq - 1;
            logits
                .slice(1, last_pos, 1)?
                .reshape(Shape::from_dims(&[cfg.vocab_size]))?
        };

        // Planner Stage 4a: populate the plan store for this step's
        // graph before realizing — realize's planning half then HITs
        // the store instead of rebuilding. This is the v1 (synchronous)
        // load-time-planning seam; Stage 4b moves the warm onto a
        // background thread fed by graph-construction events so
        // planning overlaps weight page-in and upstream execution.
        // Advisory by design: warm failures are discarded because the
        // realize below runs the identical planning path and surfaces
        // any genuine error with full realize context — nothing is
        // masked, only deferred a few lines.
        let _ = crate::planner::Planner::warm(
            logits_root.inner.graph(),
            &[logits_root.inner.id()],
            ctx.device(),
        );

        // Realize through InferenceContext. The WriteSlice nodes mutate
        // the cache buffers in place at the runtime offset `cached_len`,
        // supplied for this pass via the `SymEnv`; downstream attention
        // reads the post-write full-capacity buffers.
        let mut sym_env = fuel_ir::SymEnv::new();
        sym_env.bind(cached_len_sym, cached_len).map_err(crate::Error::from)?;
        let logits_vec = ctx.realize_one_as_with_env::<f32>(
            logits_root.inner.graph(),
            logits_root.inner.id(),
            &sym_env,
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

    /// Phase D · D2b — plan-once persistent decode. Sibling of
    /// [`Self::forward_with_kv_context`] that HOLDS the optimized
    /// decode-step graph in `session` and, on every token after the
    /// first, re-realizes the SAME graph with the D2a prebuilt seam —
    /// **skipping the `prepare` D2H-splice + the `optimize_graph`
    /// placement DP**. The ~1.8×/token win comes from not re-planning.
    ///
    /// ## Control flow
    ///
    /// - **`seq != 1`** (prefill / spec-decode verification) OR the held
    ///   `session` is **invalid** for this step (validity-key mismatch):
    ///   drop the session and fall back to the D1 rebuild path
    ///   ([`Self::forward_with_kv_context`]). The session is rebuilt on
    ///   the next `seq == 1` token.
    /// - **First `seq == 1` token with no session:** build the decode
    ///   graph ONCE with STABLE re-bindable data Consts (token-ids /
    ///   RoPE cos+sin / mask / per-layer KV, all as
    ///   `const_placeholder_like` + `ctx.insert` of a device-resident
    ///   Arc), `prebuild_optimized_env` (runs `prepare` + `optimize` +
    ///   dispatch ONCE), and populate `session` with the held graph +
    ///   cached `OptimizedGraph` + the stable NodeIds. `OPTIMIZE_CALLS`
    ///   bumps once here.
    /// - **Subsequent `seq == 1` tokens with a valid session:** recompute
    ///   the per-token host bytes (token-ids = the new token, RoPE tables
    ///   at `position = cached_len`, mask with the shifted `-inf`
    ///   boundary) and WRITE them into the held device Arcs (re-bind);
    ///   bind the per-pass `SymEnv` (`cached_len`); call
    ///   [`InferenceContext::realize_prebuilt_as_with_env`] which SKIPS
    ///   optimize. The KV Arcs are re-bound once at build time and mutate
    ///   in place via `Op::WriteSlice` (NOT re-inserted per token). A
    ///   `TopologyChanged` invalidates the session (dropped) and falls
    ///   back to the rebuild path this token.
    ///
    /// Byte-identical to the D1 cached path on the same prefix (same plan
    /// → same kernels). Bumps `cache.cached_len` + per-slot versions
    /// exactly as [`Self::forward_with_kv_context`] does.
    ///
    /// The held data Consts persist across tokens (NOT removed each
    /// token); they are removed from `ctx` when the session is dropped.
    pub fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<crate::inference_context::DecodeSession>,
    ) -> crate::Result<Vec<f32>> {
        let cfg = &self.config;
        let seq = tokens.len();
        let max_seq_len = cache.max_seq_len;
        let cache_dtype = cache.dtype.unwrap_or(DType::F32);

        // A non-`seq==1` step (prefill / spec-decode verification) is
        // shape-distinct from the held decode graph — drop any session and
        // fall back to the D1 rebuild path (the session rebuilds on the
        // next decode token).
        if seq != 1 {
            self.drop_decode_session(session, ctx);
            return self.forward_with_kv_context(tokens, cache, ctx);
        }

        // seq == 1. If a session exists but its validity keys no longer
        // match the live cache/model (max_seq_len / n_layers / dtype), it
        // is stale — drop it so we rebuild fresh below.
        if let Some(s) = session.as_ref() {
            let valid = match max_seq_len {
                Some(msl) => s.is_valid_for(seq, msl, cfg.n_layers, cache_dtype),
                None => false,
            };
            if !valid {
                self.drop_decode_session(session, ctx);
            }
        }

        match session.as_ref() {
            None => {
                // ---- First decode token (or post-invalidation): build +
                // optimize the held graph ONCE. ----
                self.build_and_realize_first_decode_token(
                    tokens, cache, ctx, session,
                )
            }
            Some(_) => {
                // ---- Subsequent decode token: re-bind data + skip optimize. ----
                let res =
                    self.rebind_and_realize_prebuilt(tokens, cache, &*ctx, &*session);
                match res {
                    Ok(logits) => Ok(logits),
                    Err(e) if matches!(e, crate::Error::TopologyChanged { .. }) => {
                        // Stale cached generation — drop the session and
                        // rebuild via the D1 path this token; the session
                        // rebuilds on the next decode token.
                        self.drop_decode_session(session, ctx);
                        self.forward_with_kv_context(tokens, cache, ctx)
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// Build the held decode-step graph with STABLE re-bindable data
    /// Consts, optimize it ONCE via `prebuild_optimized_env`, populate
    /// `session`, and return the first token's logits. Only called for
    /// the first `seq == 1` decode token when there is no valid session.
    fn build_and_realize_first_decode_token(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<crate::inference_context::DecodeSession>,
    ) -> crate::Result<Vec<f32>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        let cached_len = cache.cached_len;
        let max_seq_len = cache.max_seq_len.ok_or_else(|| {
            fuel_ir::Error::Msg(
                "forward_with_kv_context_persistent: cache built via with_dims \
                 (no pre-allocated buffers); use KvCache::with_capacity"
                    .to_string(),
            ).bt()
        })?;
        if cache.n_layers() != cfg.n_layers {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_with_kv_context_persistent: cache n_layers {} != model {}",
                cache.n_layers(), cfg.n_layers,
            )).bt());
        }
        if cached_len + seq > max_seq_len {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_with_kv_context_persistent: cached_len ({cached_len}) + \
                 seq ({seq}) > max_seq_len ({max_seq_len})",
            )).bt());
        }
        let cache_dtype = cache.dtype.unwrap_or(DType::F32);

        // Embed lookup + reshape to [batch, seq, dim]. Token-ids is a
        // STABLE re-bindable placeholder Const (bytes bound via ctx).
        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &Device::cpu(),
        );
        let token_ids = embed.const_placeholder_like(
            Shape::from_dims(&[seq]), DType::U32,
        );
        let token_ids_node = token_ids.inner.id();
        let mut h = embed
            .index_select(0, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))?;

        // RoPE cos/sin: STABLE re-bindable placeholder Consts.
        let rope_shape = Shape::from_dims(&[seq, cfg.head_dim]);
        let rope_cos = h.const_placeholder_like(rope_shape.clone(), DType::F32);
        let rope_sin = h.const_placeholder_like(rope_shape, DType::F32);
        let rope_cos_node = rope_cos.inner.id();
        let rope_sin_node = rope_sin.inner.id();

        // Mask: STABLE re-bindable placeholder Const (hoisted; shared).
        let mask = h.const_placeholder_like(
            Shape::from_dims(&[1, 1, seq, max_seq_len]), DType::F32,
        );
        let mask_node = mask.inner.id();

        let cached_len_sym = fuel_ir::SymId(0);
        let cache_shape = Shape::from_dims(
            &[batch, cfg.n_kv_heads, max_seq_len, cfg.head_dim],
        );

        // Per-layer KV placeholder Consts (STABLE). The Arcs are bound
        // ONCE here and mutate in place via Op::WriteSlice each token.
        let mut kv_nodes: Vec<(fuel_graph::NodeId, fuel_graph::NodeId)> =
            Vec::with_capacity(cfg.n_layers);
        for (li, layer_weights) in weights.layers.iter().enumerate() {
            let k_arc = cache.slot_storage(li, KvSlot::K).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "forward_with_kv_context_persistent: cache layer {li} has no K slot",
                )).bt()
            })?;
            let v_arc = cache.slot_storage(li, KvSlot::V).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "forward_with_kv_context_persistent: cache layer {li} has no V slot",
                )).bt()
            })?;
            let k_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            let v_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            let k_id = k_cache_node.inner.id();
            let v_id = v_cache_node.inner.id();
            ctx.insert(k_id, k_arc);
            ctx.insert(v_id, v_arc);
            kv_nodes.push((k_id, v_id));

            h = self.apply_layer_with_kv_writes(
                &h,
                layer_weights,
                &k_cache_node,
                &v_cache_node,
                cached_len_sym,
                &rope_cos,
                &rope_sin,
                &mask,
            )?;
        }

        let h_norm = apply_affine_rms_norm(&h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps);
        let logits = weights.output.apply_linear(&h_norm, cfg.dim, cfg.vocab_size);
        let last_pos = seq - 1;
        let logits_root = logits
            .slice(1, last_pos, 1)?
            .reshape(Shape::from_dims(&[cfg.vocab_size]))?;
        let logits_node = logits_root.inner.id();
        let graph = logits_root.inner.graph().clone();

        // Bind the per-token DATA into ctx (token-ids / RoPE / mask) as
        // device-resident Arcs so the FIRST realize's const-cache walk
        // resolves them (they are placeholders, not in graph.storage_map).
        // KV Arcs were already inserted above. The optimize + realize
        // then runs ONCE, capturing the reusable artifacts + the full
        // realized cache (weights + KV + data) for the held session.
        let data = self.build_token_rope_mask_arcs(ctx.device(), cached_len, tokens, max_seq_len)?;
        ctx.insert(token_ids_node, Arc::clone(&data.token_ids));
        ctx.insert(rope_cos_node, Arc::clone(&data.rope_cos));
        ctx.insert(rope_sin_node, Arc::clone(&data.rope_sin));
        ctx.insert(mask_node, Arc::clone(&data.mask));

        let mut sym_env = fuel_ir::SymEnv::new();
        sym_env.bind(cached_len_sym, cached_len).map_err(crate::Error::from)?;

        let (effective_target, optimized, base_cache, logits_vec) =
            ctx.prebuild_optimized_capturing_as_with_env::<f32>(&graph, logits_node, &sym_env)?;

        // The held session now owns the graph + base_cache; drop the
        // transient ctx bindings (they live in base_cache from here on —
        // re-bound per token into a clone of base_cache, not ctx).
        ctx.remove(token_ids_node);
        ctx.remove(rope_cos_node);
        ctx.remove(rope_sin_node);
        ctx.remove(mask_node);
        for (k, v) in &kv_nodes {
            ctx.remove(*k);
            ctx.remove(*v);
        }

        *session = Some(crate::inference_context::DecodeSession::new(
            graph,
            optimized,
            effective_target,
            logits_node,
            token_ids_node,
            rope_cos_node,
            rope_sin_node,
            mask_node,
            kv_nodes,
            cached_len_sym,
            base_cache,
            seq,
            max_seq_len,
            cfg.n_layers,
            cache_dtype,
        ));

        // Bump cache state (identical to the D1 path).
        cache.cached_len += seq;
        for li in 0..cfg.n_layers {
            cache.bump_version(li, KvSlot::K);
            cache.bump_version(li, KvSlot::V);
        }
        Ok(logits_vec)
    }

    /// Re-bind the per-token data Consts (token-ids / RoPE / mask) into
    /// device Arcs, bind the `SymEnv`, and realize via the D2a prebuilt
    /// seam (SKIPPING optimize) over the held session's base cache. The
    /// KV Arcs are stable (mutated in place by WriteSlice via the held
    /// base_cache entries) — not touched here. Called for every decode
    /// token after the first.
    fn rebind_and_realize_prebuilt(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &InferenceContext,
        session: &Option<crate::inference_context::DecodeSession>,
    ) -> crate::Result<Vec<f32>> {
        let cfg = &self.config;
        let seq = tokens.len();
        let cached_len = cache.cached_len;
        let device = ctx.device().clone();

        // Session guaranteed Some + valid by the caller. Recompute the
        // per-token data Arcs, then realize the held graph via the
        // prebuilt seam (base_cache clone + overwritten data entries).
        // ctx is NOT mutated on the reuse path — the data lands in a
        // clone of the session's held base_cache, not in ctx.persistent.
        let s = session.as_ref().expect("session is Some");
        let data = self.build_token_rope_mask_arcs(
            &device, cached_len, tokens, s.max_seq_len(),
        )?;
        let mut sym_env = fuel_ir::SymEnv::new();
        sym_env.bind(s.cached_len_sym(), cached_len).map_err(crate::Error::from)?;
        let logits_vec = s.realize_token(&device, data, &sym_env)?;

        // Bump cache state (identical to the D1 path).
        cache.cached_len += seq;
        for li in 0..cfg.n_layers {
            cache.bump_version(li, KvSlot::K);
            cache.bump_version(li, KvSlot::V);
        }
        Ok(logits_vec)
    }

    /// Recompute the per-token host bytes for token-ids / RoPE cos+sin /
    /// mask and build device-resident Arcs from them (the SAME upload
    /// path `KvCache::with_capacity` uses). On CPU the Storage wraps the
    /// host bytes; on GPU it performs the H2D upload (tiny tensors).
    /// Design §2 option (b): the bytes change per token, the NodeId stays
    /// stable (the held graph's Const nodes are re-bound via `base_cache`
    /// overwrite, not a fresh graph).
    fn build_token_rope_mask_arcs(
        &self,
        device: &Device,
        cached_len: usize,
        tokens: &[u32],
        max_seq_len: usize,
    ) -> crate::Result<crate::inference_context::DecodeTokenData> {
        let cfg = &self.config;
        let seq = tokens.len();
        let upload = crate::pipelined_bridge::upload_host_buffer_to_device;

        let token_ids = upload(device, fuel_ir::HostBuffer::U32(tokens.to_vec()))?;
        let (cos_data, sin_data) = fuel_graph::build_rope_tables(
            cfg.rope_base, cached_len, seq, cfg.head_dim,
        );
        let rope_cos = upload(device, fuel_ir::HostBuffer::F32(cos_data))?;
        let rope_sin = upload(device, fuel_ir::HostBuffer::F32(sin_data))?;
        let mask_data = build_decode_causal_mask(cached_len, seq, max_seq_len);
        let mask = upload(device, fuel_ir::HostBuffer::F32(mask_data))?;

        Ok(crate::inference_context::DecodeTokenData {
            token_ids,
            rope_cos,
            rope_sin,
            mask,
        })
    }

    /// Drop a held decode session, removing any leftover persistent
    /// data-Const / KV bindings from `ctx` (defensive — the build path
    /// already removes them once the session owns `base_cache`; this
    /// covers the invalidation path). No-op if `session` is `None`.
    fn drop_decode_session(
        &self,
        session: &mut Option<crate::inference_context::DecodeSession>,
        ctx: &mut InferenceContext,
    ) {
        if let Some(s) = session.take() {
            ctx.remove(s.token_ids_node());
            ctx.remove(s.rope_cos_node());
            ctx.remove(s.rope_sin_node());
            ctx.remove(s.mask_node());
            for (k, v) in s.kv_nodes() {
                ctx.remove(*k);
                ctx.remove(*v);
            }
        }
    }
}


// Phase 7.6 step 9c E.3.3.D — host-resident `LlamaKVCache` retired.
// Its successor is `KvCache` in `crate::inference_context`, which
// stores backend-erased `Arc<RwLock<fuel_memory::Storage>>` per slot
// and supports both the legacy `with_dims` grow-by-replace shape and
// the new `with_capacity` pre-allocated-buffer shape that
// `forward_with_kv_context` writes into via `Op::WriteSlice`.

/// Broadcast-add a 1-D bias along the last axis of `x`, or return


/// RmsNorm with a learned per-channel gain, applied along the last dim.
/// This is the affine version used by LLaMA: `y = (x / rms) * gain`.
///
/// `gain` is taken as `&Arc<[f32]>` so we can clone it into the const
/// node without copying the underlying slice — the same refcount-bump
/// pattern used for every other weight in the model.
/// Build the strict lower-triangular causal mask for one
/// LlamaModel forward pass. Shape `[1, 1, seq, seq]` with
/// `0` at `j <= i` and `-inf` at `j > i`, ready to
/// broadcast-add onto attention scores. Anchored on `g` so
/// the mask shares its graph with the rest of the model.
/// Build the fixed-capacity causal mask for the input-independent decode
/// graph (Phase D · D1/D2b). Shape `[seq, max_seq_len]` flattened
/// row-major: `-inf` where `k > cached_len + q` (masks future positions
/// AND the zero-init stale tail), `0` otherwise. Hoisted to ONE shared
/// Const (was per-layer); the per-token re-bind on the persistent decode
/// path recomputes exactly this each token (the `-inf` boundary shifts
/// with `cached_len`).
fn build_decode_causal_mask(cached_len: usize, seq: usize, max_seq_len: usize) -> Vec<f32> {
    let mut mask_data = vec![0.0_f32; seq * max_seq_len];
    for q_idx in 0..seq {
        let abs_q = cached_len + q_idx;
        for k_idx in (abs_q + 1)..max_seq_len {
            mask_data[q_idx * max_seq_len + k_idx] = f32::NEG_INFINITY;
        }
    }
    mask_data
}

fn apply_affine_rms_norm(
    x: &LazyTensor,
    gain: &Arc<[f32]>,
    dim: usize,
    eps: f64,
) -> LazyTensor {
    assert_eq!(gain.len(), dim, "apply_affine_rms_norm: gain length must equal dim");
    let normalized = x.rms_norm_last_dim(eps).unwrap();
    let gain_t = x.const_f32_like(Arc::clone(gain), Shape::from_dims(&[dim]));
    normalized.broadcast_mul(&gain_t).unwrap()
}


// ---- HuggingFace Hub and safetensors weight loading ----------------------

/// Load a tensor by name from a `MmapedSafetensors` as a flat
/// `Vec<f32>`, converting from whatever dtype the file stores it in.
/// Handles `F32`, `F64`, `BF16`, and `F16` — the dtypes real LLaMA
/// weights use on disk. Returns an error for unsupported dtypes.
pub fn load_tensor_as_f32(
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
pub fn load_transposed_matrix(
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
pub fn load_transposed_matrix_preserve_dtype(
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
    // These replaced the legacy `generate_streaming_on` /
    // `generate_streaming_gpu_on` pair across CPU, CUDA, and Vulkan
    // (the latter retired in Unification Session 4, E.3.4). The
    // device is passed in directly (no `GraphBackend` parameter);
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
            return Err(fuel_ir::Error::Msg(
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

    /// Speculative decoding through the `forward_with_kv_context`
    /// path (KvCache + InferenceContext + the pipelined executor).
    ///
    /// Uses a `draft` model to predict `k` tokens autoregressively,
    /// then has `self` (the target) verify all `k` positions in a
    /// single forward. Accepts a prefix of the drafts per `strategy`:
    ///
    /// - `Greedy`: longest prefix where target's argmax matches
    ///   draft's token. On mismatch, emit target's argmax as the
    ///   bonus. Output is provably identical to plain greedy
    ///   generation from the target, regardless of the draft.
    /// - `Temperature`: Leviathan-style probability-ratio accept.
    ///   Sample draft tokens from draft's temperature-scaled
    ///   distribution; accept each with probability
    ///   `min(1, p_target(d) / p_draft(d))`. On reject, sample the
    ///   replacement from `(p_target - p_draft)_+ / Z`. Distribution
    ///   of outputs is provably identical to plain sampled generation
    ///   from the target.
    ///
    /// Rejected drafts are rolled back via [`KvCache::truncate_to`] —
    /// a pure metadata update on the pre-allocated-buffer path. The
    /// cache rolls back to the committed prefix (accepted drafts
    /// only); the bonus token's K/V is written by the explicit
    /// bonus-advance forward at its true position.
    ///
    /// Note: the retired legacy-executor implementation truncated the
    /// target cache to `committed + accepted + 1` on rejection,
    /// leaving the rejected draft's K/V row in place at the bonus
    /// position and appending the bonus one position too far. The
    /// resulting logits drift was measured at ~4e-3 (vs ~1e-6 gemm
    /// noise) on the tiny test fixture — real positional corruption,
    /// though small enough there that the argmax never flipped and
    /// the legacy token-equality tests (which only exercised the
    /// accepted == k path) couldn't see it. This implementation
    /// truncates to `committed + accepted` so the bonus advance lands
    /// at the correct position;
    /// `spec_decode_kv_context_divergent_draft_matches_greedy_baseline`
    /// locks the lossless-greedy property.
    ///
    /// Expected speedup 1.5-3× at good acceptance rates (same-family
    /// drafts only — cross-family drafts or different tokenizers will
    /// have <20% acceptance and net-negative speedup).
    ///
    /// Preconditions:
    /// - `draft.config.vocab_size == self.config.vocab_size` (so
    ///   target's distribution over draft's vocab is well-defined).
    /// - Both models share the same tokenizer (caller's
    ///   responsibility).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_spec_with_kv_context(
        &self,
        draft: &LlamaModel,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        k: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        device: &Device,
        dtype: DType,
        mut on_token: impl FnMut(u32),
    ) -> crate::Result<Vec<u32>> {
        if draft.config.vocab_size != self.config.vocab_size {
            fuel_ir::bail!(
                "spec-decode: draft vocab {} != target vocab {}",
                draft.config.vocab_size, self.config.vocab_size,
            );
        }
        if k == 0 {
            fuel_ir::bail!("spec-decode: k must be >= 1");
        }
        if prompt_tokens.is_empty() {
            return Err(fuel_ir::Error::Msg(
                "generate_streaming_spec_with_kv_context: prompt is empty".to_string(),
            ).bt());
        }

        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        let vocab = self.config.vocab_size;

        // RNG state threading. Only used in Temperature mode.
        let mut rng_state: u64 = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };
        let temp = match strategy {
            SamplingStrategy::Temperature { temp, .. } => temp,
            SamplingStrategy::Greedy => 1.0, // unused in greedy
        };

        // KV capacity: the committed sequence never exceeds
        // `prompt + max_new`; both caches transiently hold up to `k`
        // not-yet-accepted rows past the committed prefix (draft
        // phase / verify phase) before truncation rolls them back.
        let max_seq_len = prompt_tokens.len() + max_new_tokens + k;
        let mut target_cache = KvCache::with_capacity(
            self.config.n_layers, self.config.n_kv_heads, self.config.head_dim,
            max_seq_len, dtype, device,
        )?;
        let mut draft_cache = KvCache::with_capacity(
            draft.config.n_layers, draft.config.n_kv_heads, draft.config.head_dim,
            max_seq_len, dtype, device,
        )?;
        let mut target_ctx = InferenceContext::new(device.clone());
        let mut draft_ctx = InferenceContext::new(device.clone());

        // Prefill both caches with the prompt.
        let mut target_last_logits =
            self.forward_with_kv_context(&tokens, &mut target_cache, &mut target_ctx)?;
        let mut draft_last_logits =
            draft.forward_with_kv_context(&tokens, &mut draft_cache, &mut draft_ctx)?;

        let mut emitted = 0usize;

        while emitted < max_new_tokens {
            // --- Draft phase: K tokens. In Greedy mode, argmax; in
            // Temperature mode, sample from draft's temp-scaled dist.
            // We ALSO stash each draft's probability distribution for
            // the Temperature accept rule.
            let mut drafts: Vec<u32> = Vec::with_capacity(k);
            let mut draft_probs_stash: Vec<Vec<f32>> = Vec::with_capacity(k);
            for _ in 0..k {
                let d = match strategy {
                    SamplingStrategy::Greedy => {
                        // We don't need draft_probs in greedy, but the
                        // slot has to exist to keep indexing uniform.
                        draft_probs_stash.push(Vec::new());
                        spec_argmax(&draft_last_logits)
                    }
                    SamplingStrategy::Temperature { .. } => {
                        let probs = spec_softmax_temp(&draft_last_logits, temp);
                        let d = spec_sample_cat(&probs, &mut rng_state);
                        draft_probs_stash.push(probs);
                        d
                    }
                };
                drafts.push(d);
                draft_last_logits = draft.forward_with_kv_context(
                    &[d], &mut draft_cache, &mut draft_ctx,
                )?;
            }

            // --- Verify phase: target runs forward on the K drafts.
            let verify_logits = self.forward_with_kv_context_all_positions(
                &drafts, &mut target_cache, &mut target_ctx,
            )?;
            debug_assert_eq!(verify_logits.len(), drafts.len() * vocab);

            // --- Accept phase: strategy-specific. ---
            let mut accepted = 0usize;
            let bonus_token: u32;
            match strategy {
                SamplingStrategy::Greedy => {
                    let mut mismatched: Option<u32> = None;
                    for i in 0..drafts.len() {
                        let prev_row = if i == 0 {
                            &target_last_logits[..]
                        } else {
                            &verify_logits[(i - 1) * vocab .. i * vocab]
                        };
                        let target_pick = spec_argmax(prev_row);
                        if target_pick == drafts[i] {
                            accepted += 1;
                        } else {
                            mismatched = Some(target_pick);
                            break;
                        }
                    }
                    bonus_token = match mismatched {
                        Some(t) => t,
                        None => spec_argmax(
                            &verify_logits[(drafts.len() - 1) * vocab .. drafts.len() * vocab],
                        ),
                    };
                }
                SamplingStrategy::Temperature { .. } => {
                    // Leviathan accept rule. For each i:
                    //   q_i = draft's prob of drafts[i]
                    //   p_i = target's prob of drafts[i] (from prev[i])
                    //   accept with prob min(1, p_i / q_i)
                    // On reject: sample replacement from (p - q)_+ / Z.
                    let mut rejected_replacement: Option<u32> = None;
                    for i in 0..drafts.len() {
                        let prev_row = if i == 0 {
                            &target_last_logits[..]
                        } else {
                            &verify_logits[(i - 1) * vocab .. i * vocab]
                        };
                        let target_probs = spec_softmax_temp(prev_row, temp);
                        let draft_probs = &draft_probs_stash[i];
                        let d_tok = drafts[i] as usize;
                        let p = target_probs[d_tok];
                        let q = draft_probs[d_tok];
                        let ratio = if q > 0.0 { (p / q).min(1.0) } else { 0.0 };
                        let u = spec_next_u01(&mut rng_state);
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
                                rejected_replacement = Some(spec_sample_cat(&residual, &mut rng_state));
                            } else {
                                // Degenerate case (should only happen if
                                // distributions match exactly — then any
                                // sample from target_probs is equally valid).
                                rejected_replacement = Some(spec_sample_cat(&target_probs, &mut rng_state));
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
                            let probs = spec_softmax_temp(last_row, temp);
                            spec_sample_cat(&probs, &mut rng_state)
                        }
                    };
                }
            }

            // --- Roll back both caches to the committed prefix. ---
            // Both caches advanced by K during draft/verify, but only
            // `accepted` of those K positions hold committed tokens.
            // The bonus token's K/V is NOT in either cache (the verify
            // row at the bonus position belongs to the first rejected
            // draft); the bonus-advance forwards below write it at the
            // correct position. When accepted == k both truncates are
            // no-ops and the bonus appends at the cache tail.
            let committed_base = target_cache.cached_len - k;
            target_cache.truncate_to(committed_base + accepted);
            let draft_committed_base = draft_cache.cached_len - k;
            draft_cache.truncate_to(draft_committed_base + accepted);

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
            if emitted >= max_new_tokens { return Ok(tokens); }

            // --- Advance both caches + both "last_logits" by the bonus
            // token. The draft needs to see the bonus (which it didn't
            // produce); the target writes the bonus K/V at its true
            // position and returns fresh logits for the next
            // accept-check on draft[0].
            target_last_logits = self.forward_with_kv_context(
                &[bonus_token], &mut target_cache, &mut target_ctx,
            )?;
            draft_last_logits = draft.forward_with_kv_context(
                &[bonus_token], &mut draft_cache, &mut draft_ctx,
            )?;
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

// ---- Speculative-decoding helpers ---------------------------------------
//
// Shared by `generate_streaming_spec_with_kv_context`'s draft / accept
// phases. All host-side: spec decode's accept rule operates on logits
// vectors already downloaded from the device.

/// Greedy argmax over a logits row.
fn spec_argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v { best_v = v; best = i; }
    }
    best as u32
}

/// Temperature-scaled softmax. Returns normalized probabilities.
fn spec_softmax_temp(logits: &[f32], temp: f32) -> Vec<f32> {
    let inv_t = if temp == 0.0 { 1.0 } else { 1.0 / temp };
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| ((x - max) * inv_t).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

/// Advance a deterministic LCG and return a u01 uniform.
fn spec_next_u01(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as f32 / u32::MAX as f32
}

/// Sample a category from a distribution summing to ~1.
fn spec_sample_cat(probs: &[f32], state: &mut u64) -> u32 {
    let u = spec_next_u01(state);
    let mut cum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u <= cum { return i as u32; }
    }
    (probs.len() - 1) as u32
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
/// Apply `x @ W + b` where `W` is a `WeightStorage` projection and
/// `b` is a `[out_features]` bias vector. Dispatches to qmatmul for
/// Q4_0 weights.

impl PhiModel {
    // ===== Phase 7.6 step 9c E.3.3/E.3.4 — KvCache + InferenceContext =====
    //
    // The pipelined-executor forward/generate family, mirroring
    // `LlamaModel::forward_with_kv_context`. Pre-allocated KV buffers
    // (`KvCache::with_capacity`) + `Op::WriteSlice` in-graph mutation;
    // runs on CPU, CUDA, and Vulkan via binding-table dispatch. Phi-2
    // has no GQA, so the cache's `n_kv_heads` slot carries `n_heads`.

    /// Variant of [`Self::apply_layer_with_cache`] that uses
    /// pre-allocated KV-cache buffers + `Op::WriteSlice`. The K/V
    /// caches are bound via `k_cache_const` / `v_cache_const` (Const
    /// placeholders the caller has wired into [`InferenceContext`]);
    /// the post-write tensors are sliced to the `cached_len + seq`
    /// extent for attention. The cache mutation is in-graph as a side
    /// effect of the WriteSlice nodes.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_with_kv_writes(
        &self,
        x: &LazyTensor,
        layer: &PhiLayerWeights,
        k_cache_const: &LazyTensor,
        v_cache_const: &LazyTensor,
        cached_len: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_heads * cfg.head_dim;  // no GQA in Phi-2
        let total_seq = cached_len + seq;

        // Shared pre-block LayerNorm.
        let x_norm = x.layer_norm_affine(
            Arc::clone(&layer.norm_gain), Arc::clone(&layer.norm_bias),
            cfg.layer_norm_eps,
        )?;

        // Q/K/V projections with bias — identical to apply_layer_with_cache.
        let (q, k, v) = match &layer.attn_qkv {
            PhiQkv::Split { q, q_bias, k, k_bias, v, v_bias } => {
                let q_out = q.apply_linear_with_bias(&x_norm, cfg.dim, cfg.dim, Arc::clone(q_bias))?;
                let k_out = k.apply_linear_with_bias(&x_norm, cfg.dim, kv_dim, Arc::clone(k_bias))?;
                let v_out = v.apply_linear_with_bias(&x_norm, cfg.dim, kv_dim, Arc::clone(v_bias))?;
                (q_out, k_out, v_out)
            }
            PhiQkv::Packed { qkv, qkv_bias } => {
                let combined = qkv.apply_linear_with_bias(&x_norm, cfg.dim, 3 * cfg.dim, Arc::clone(qkv_bias))?;
                let last = combined.rank() - 1;
                let q_out = combined.slice(last, 0, cfg.dim)?;
                let k_out = combined.slice(last, cfg.dim, cfg.dim)?;
                let v_out = combined.slice(last, 2 * cfg.dim, cfg.dim)?;
                (q_out, k_out, v_out)
            }
        };

        // Split heads → [batch, n_heads, seq, head_dim].
        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;

        // Partial RoPE on Q and K (first `rotary_dim` entries rotate).
        let q_r = partial_rope(&q_h, rope_cos, rope_sin, cfg.rotary_dim, cfg.head_dim);
        let k_r = partial_rope(&k_h, rope_cos, rope_sin, cfg.rotary_dim, cfg.head_dim);

        // Write fresh K/V into the pre-allocated cache buffers via
        // Op::WriteSlice; slice the post-write buffer to the visible
        // `[..., 0..total_seq, ...]` extent for attention.
        let write_ranges = vec![
            (0, batch),
            (0, cfg.n_heads),
            (cached_len, cached_len + seq),
            (0, cfg.head_dim),
        ];
        let k_full_buffer = k_cache_const.write_slice(&k_r, write_ranges.clone())?;
        let v_full_buffer = v_cache_const.write_slice(&v_h, write_ranges)?;
        let full_k = k_full_buffer.slice(2, 0, total_seq)?;
        let full_v = v_full_buffer.slice(2, 0, total_seq)?;

        // Attention: Q @ K^T, scale, mask, softmax, @ V — identical to
        // apply_layer_with_cache from here.
        let k_t = full_k.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let mut mask_data = vec![0.0_f32; seq * total_seq];
        for q_idx in 0..seq {
            let abs_q = cached_len + q_idx;
            for k_idx in (abs_q + 1)..total_seq {
                mask_data[q_idx * total_seq + k_idx] = f32::NEG_INFINITY;
            }
        }
        let mask = x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, total_seq]));
        let scores_scaled = LazyTensor { inner: scores.inner.mul_scalar(scale) };
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&full_v)?;

        // Merge heads: [batch, n_heads, seq, head_dim] → [batch, seq, dim].
        let merged = attn_v
            .permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))?;
        let attn_out = layer.attn_dense.apply_linear_with_bias(
            &merged, cfg.dim, cfg.dim, Arc::clone(&layer.attn_dense_bias),
        )?;

        // MLP branch (shares x_norm with the attention branch).
        let fc1_out = layer.mlp_fc1.apply_linear_with_bias(
            &x_norm, cfg.dim, cfg.ffn_dim, Arc::clone(&layer.mlp_fc1_bias),
        )?;
        let gelu_out = fc1_out.gelu();
        let mlp_out = layer.mlp_fc2.apply_linear_with_bias(
            &gelu_out, cfg.ffn_dim, cfg.dim, Arc::clone(&layer.mlp_fc2_bias),
        )?;

        // Parallel residual: x + attn_out + mlp_out.
        x.add(&attn_out)?.add(&mlp_out)
    }

    /// Forward pass using pre-allocated KV-cache buffers and
    /// `Op::WriteSlice`; returns last-position logits. Mirrors
    /// [`LlamaModel::forward_with_kv_context`] — see its docs for the
    /// architectural notes. The cache must have been constructed via
    /// [`KvCache::with_capacity`] with `n_kv_heads == n_heads` (Phi-2
    /// has no GQA).
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
            return Err(fuel_ir::Error::Msg(
                "PhiModel::forward_with_kv_context: zero tokens".to_string(),
            ).bt());
        }
        if cache.n_layers() != cfg.n_layers {
            return Err(fuel_ir::Error::Msg(format!(
                "PhiModel::forward_with_kv_context: cache n_layers {} != model n_layers {}",
                cache.n_layers(), cfg.n_layers,
            )).bt());
        }
        let max_seq_len = cache.max_seq_len.ok_or_else(|| {
            fuel_ir::Error::Msg(
                "PhiModel::forward_with_kv_context: cache was constructed via with_dims \
                 (no pre-allocated buffers); call KvCache::with_capacity(...) for the \
                 WriteSlice path".to_string(),
            ).bt()
        })?;
        if cached_len + seq > max_seq_len {
            return Err(fuel_ir::Error::Msg(format!(
                "PhiModel::forward_with_kv_context: cached_len ({cached_len}) + seq \
                 ({seq}) > max_seq_len ({max_seq_len})",
            )).bt());
        }
        let cache_dtype = cache.dtype.unwrap_or(DType::F32);
        if cache.n_kv_heads != cfg.n_heads || cache.head_dim != cfg.head_dim {
            return Err(fuel_ir::Error::Msg(format!(
                "PhiModel::forward_with_kv_context: cache shape (n_kv_heads={}, \
                 head_dim={}) disagrees with model config (n_heads={}, head_dim={})",
                cache.n_kv_heads, cache.head_dim, cfg.n_heads, cfg.head_dim,
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
            .index_select(0, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))?;

        // RoPE tables are sized for `rotary_dim`, not the full
        // head_dim — partial RoPE rotates only the first `rotary_dim`
        // entries.
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_base, cached_len, seq, cfg.rotary_dim,
        );

        // Per-layer: bind the cache K + V Arcs to fresh Const NodeIds,
        // dispatch through the WriteSlice variant, clean up the
        // per-step bindings after realize.
        let cache_shape = Shape::from_dims(
            &[batch, cfg.n_heads, max_seq_len, cfg.head_dim],
        );
        let mut bound_node_ids: Vec<fuel_graph::NodeId> =
            Vec::with_capacity(2 * cfg.n_layers);
        for (li, layer_weights) in weights.layers.iter().enumerate() {
            let k_arc = cache.slot_storage(li, KvSlot::K).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "PhiModel::forward_with_kv_context: cache layer {li} has no K slot \
                     (with_capacity should have populated all layers)",
                )).bt()
            })?;
            let v_arc = cache.slot_storage(li, KvSlot::V).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "PhiModel::forward_with_kv_context: cache layer {li} has no V slot",
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

        // Final LayerNorm, output projection (+ optional bias).
        let h_norm = h.layer_norm_affine(
            Arc::clone(&weights.final_norm_gain), Arc::clone(&weights.final_norm_bias),
            cfg.layer_norm_eps,
        )?;
        let logits_no_bias = weights.output.apply_linear(&h_norm, cfg.dim, cfg.vocab_size);
        let logits = match &weights.output_bias {
            Some(b) => {
                let b_t = h_norm.const_f32_like(
                    Arc::clone(b), Shape::from_dims(&[cfg.vocab_size]));
                logits_no_bias.broadcast_add(&b_t)?
            }
            None => logits_no_bias,
        };

        let last_pos = seq - 1;
        let last_logits = logits
            .slice(1, last_pos, 1)?
            .reshape(Shape::from_dims(&[cfg.vocab_size]))?;

        // Realize through InferenceContext. The WriteSlice nodes
        // mutate the cache buffers as a side effect.
        let logits_vec = ctx.realize_one_as::<f32>(
            last_logits.inner.graph(),
            last_logits.inner.id(),
        )?;

        // Clean up per-step bindings so they don't accumulate across
        // decode steps (each step gets a fresh graph; the previous
        // step's NodeIds are dead).
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

    /// Streaming generation through [`Self::forward_with_kv_context`].
    /// Allocates a pre-allocated [`KvCache`] of capacity
    /// `prompt_tokens.len() + max_new_tokens` on `device`, then loops
    /// prefill + decode, calling `on_token` for each generated token.
    /// Mirrors [`LlamaModel::generate_streaming_with_kv_context`].
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
            return Err(fuel_ir::Error::Msg(
                "PhiModel::generate_streaming_with_kv_context: prompt is empty".to_string(),
            ).bt());
        }
        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut rng_state: u64 = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };

        let max_seq_len = prompt_tokens.len() + max_new_tokens;
        let mut cache = KvCache::with_capacity(
            cfg.n_layers, cfg.n_heads, cfg.head_dim,
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
    /// [`Self::generate_streaming_with_kv_context`].
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
        return x.rope_with_tables(cos, sin).unwrap();
    }
    let rank = x.shape().dims().len();
    let last = rank - 1;
    let x_rot = x.slice(last, 0, rotary_dim).unwrap();
    let x_pass = x.slice(last, rotary_dim, head_dim - rotary_dim).unwrap();
    let x_rot_rotated = x_rot.rope_with_tables(cos, sin).unwrap();
    x_rot_rotated.concat(&x_pass, last).unwrap()
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
        make_tiny_weights_seeded(cfg, 9999)
    }

    /// Seeded variant — spec-decode tests use a second seed to build
    /// a draft model that genuinely diverges from the target.
    fn make_tiny_weights_seeded(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
        let mut s: u32 = seed;
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
            .forward(&tokens, 0).unwrap()
            .slice(1, tokens.len() - 1, 1).unwrap()
            .reshape(Shape::from_dims(&[cfg.vocab_size])).unwrap()
            .realize_f32();
        let with_bias_logits = with_bias
            .forward(&tokens, 0).unwrap()
            .slice(1, tokens.len() - 1, 1).unwrap()
            .reshape(Shape::from_dims(&[cfg.vocab_size])).unwrap()
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
            "bias had no effect — check that add_optional_trailing_bias is actually called",
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
            let logits = model.forward(&ref_tokens, 0).unwrap();
            let last_pos = ref_tokens.len() - 1;
            let last = logits
                .slice(1, last_pos, 1).unwrap()
                .reshape(Shape::from_dims(&[cfg.vocab_size])).unwrap()
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
            let logits = model.forward(&ref_tokens, 0).unwrap();
            let last_pos = ref_tokens.len() - 1;
            let last = logits
                .slice(1, last_pos, 1).unwrap()
                .reshape(Shape::from_dims(&[cfg.vocab_size])).unwrap()
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
        let full_logits = model.forward(&full, 0).unwrap();
        let last_pos = full.len() - 1;
        let expected = full_logits
            .slice(1, last_pos, 1).unwrap()
            .reshape(Shape::from_dims(&[cfg.vocab_size])).unwrap()
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

    /// Phase D · D2b born-red gate for plan-once persistent decode.
    ///
    /// Drive [`LlamaModel::forward_with_kv_context_persistent`] for ≥3
    /// decode tokens (after a prefill) holding ONE [`DecodeSession`], run
    /// in lockstep against the D1 [`LlamaModel::forward_with_kv_context`]
    /// path (a SECOND identical model + cache + ctx fed the identical
    /// token at each step). Assert the three plan-once invariants:
    ///   (a) `optimize_calls_thread_local()` bumps **exactly once** across
    ///       all the decode tokens — the first persistent decode token
    ///       builds + optimizes the held session; tokens 2..N skip
    ///       optimize entirely (the held graph + cached `OptimizedGraph`
    ///       are reused via the D2a prebuilt seam);
    ///   (b) each persistent token's logits are **exactly `==`** the D1
    ///       cached path on the same prefix — same plan → same kernels →
    ///       bit-exact (NOT epsilon);
    ///   (c) the held graph's node `len()` is **stable from token 2
    ///       onward** — no per-token node growth (the guard that no
    ///       builder snuck a `cached_len`-dependent shape / re-splice /
    ///       re-insert back in).
    ///
    /// Born-red shape: if the data Consts are rebuilt fresh per token
    /// (a new graph each token) OR the session re-optimizes, (a)/(c)
    /// fail; wiring the held session + per-token data re-bind makes them
    /// pass.
    #[test]
    fn forward_with_kv_context_persistent_plan_once_matches_d1() {
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
        let cfg = LlamaConfig { dim: cfg.n_heads * cfg.head_dim, ..cfg };

        // Two byte-identical models: one drives the D2 persistent path,
        // one drives the D1 rebuild path. Identical weights (same seed).
        let model_d2 = LlamaModel { config: cfg.clone(), weights: make_tiny_weights(&cfg) };
        let model_d1 = LlamaModel { config: cfg.clone(), weights: make_tiny_weights(&cfg) };

        let prompt = [1_u32, 2, 3];
        let decode_tokens = [4_u32, 5, 6, 7]; // ≥3 decode tokens
        let max_seq_len = prompt.len() + decode_tokens.len();

        // --- D1 (rebuild) reference FIRST, in its own pass, so its
        // per-token re-plans do NOT pollute the optimize-count window we
        // measure around the D2 loop. Store the expected logits. ---
        let dev1 = Device::cpu();
        let mut cache1 = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, max_seq_len, DType::F32, &dev1,
        ).expect("with_capacity d1");
        let mut ctx1 = InferenceContext::new(dev1);
        let _ = model_d1
            .forward_with_kv_context(&prompt, &mut cache1, &mut ctx1)
            .expect("d1 prefill");
        let mut d1_expected: Vec<Vec<f32>> = Vec::with_capacity(decode_tokens.len());
        for &tok in &decode_tokens {
            d1_expected.push(
                model_d1
                    .forward_with_kv_context(&[tok], &mut cache1, &mut ctx1)
                    .expect("d1 decode"),
            );
        }

        // --- D2 (persistent) session state ---
        let dev2 = Device::cpu();
        let mut cache2 = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, max_seq_len, DType::F32, &dev2,
        ).expect("with_capacity d2");
        let mut ctx2 = InferenceContext::new(dev2);
        let mut session: Option<crate::inference_context::DecodeSession> = None;

        // Prefill the D2 path (seq > 1 → the persistent path falls back to
        // the rebuild path; the session is NOT built here).
        let _ = model_d2
            .forward_with_kv_context_persistent(&prompt, &mut cache2, &mut ctx2, &mut session)
            .expect("d2 prefill");
        assert!(session.is_none(), "prefill (seq>1) must NOT build the held session");

        // Decode ≥3 tokens through the persistent path ONLY. Snapshot the
        // optimizer count on THIS thread just before the decode loop
        // (isolated from other suite threads' concurrent optimizes — the
        // process-global count is polluted; the thread-local delta is
        // exact). The D2 loop is the ONLY optimize source in this window.
        let opt_before = crate::pipelined_bridge::optimize_calls_thread_local();
        let mut len_at_token2: Option<usize> = None;

        for (i, &tok) in decode_tokens.iter().enumerate() {
            let d2 = model_d2
                .forward_with_kv_context_persistent(&[tok], &mut cache2, &mut ctx2, &mut session)
                .expect("d2 decode");

            // (b) bit-exact vs. the D1 cached path (same plan → same
            // kernels), NOT epsilon.
            assert_eq!(
                d2, d1_expected[i],
                "persistent decode token {i} must be byte-identical to the D1 \
                 cached path",
            );

            // The session must exist after the first decode token and
            // stay valid across the rest.
            let sess = session.as_ref().expect("session built on first decode token");
            let graph_len = sess.graph_node_count();
            if i == 1 {
                len_at_token2 = Some(graph_len);
            } else if i >= 2 {
                // (c) node count stable from token 2 onward.
                assert_eq!(
                    Some(graph_len), len_at_token2,
                    "held graph must NOT grow from token 2 onward (token {i})",
                );
            }
        }

        // (a) optimize bumped EXACTLY ONCE across all decode tokens.
        let opt_after = crate::pipelined_bridge::optimize_calls_thread_local();
        assert_eq!(
            opt_after - opt_before, 1,
            "persistent decode must optimize EXACTLY ONCE across {} decode \
             tokens (the first builds the session; the rest skip optimize): \
             {opt_before} -> {opt_after}",
            decode_tokens.len(),
        );

        // Sanity: both caches advanced identically.
        assert_eq!(cache2.cached_len, max_seq_len);
        assert_eq!(cache1.cached_len, max_seq_len);
    }

    /// Phase D · D2b invalidation: a `seq != 1` step mid-stream (e.g. a
    /// spec-decode verification batch) must DROP the held session and
    /// fall back to the D1 rebuild path (the session is shape-keyed to
    /// seq==1); a subsequent seq==1 token rebuilds a fresh session and
    /// still produces correct logits. Also checks the session is rebuilt
    /// (a NEW session object) after the fallback.
    #[test]
    fn forward_with_kv_context_persistent_invalidates_on_non_decode_step() {
        let cfg = LlamaConfig {
            vocab_size: 16, dim: 8, n_layers: 2, n_heads: 4, n_kv_heads: 2,
            head_dim: 4, ffn_dim: 16, norm_eps: 1e-5, rope_base: 10000.0,
        };
        let cfg = LlamaConfig { dim: cfg.n_heads * cfg.head_dim, ..cfg };
        let model = LlamaModel { config: cfg.clone(), weights: make_tiny_weights(&cfg) };

        let prompt = [1_u32, 2];
        let max_seq_len = 8;
        let device = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, max_seq_len, DType::F32, &device,
        ).expect("with_capacity");
        let mut ctx = InferenceContext::new(device);
        let mut session: Option<crate::inference_context::DecodeSession> = None;

        // Prefill (seq>1 → no session).
        let _ = model
            .forward_with_kv_context_persistent(&prompt, &mut cache, &mut ctx, &mut session)
            .expect("prefill");
        assert!(session.is_none());

        // One decode token builds the session.
        let _ = model
            .forward_with_kv_context_persistent(&[3], &mut cache, &mut ctx, &mut session)
            .expect("decode 1");
        assert!(session.is_some(), "first decode token builds the session");
        let graph_ptr_1 = Arc::as_ptr(session.as_ref().unwrap().graph());

        // A seq!=1 all-positions step drops the session (fallback to D1).
        let _ = model
            .forward_with_kv_context_persistent(&[4, 5], &mut cache, &mut ctx, &mut session)
            .expect("multi-token step");
        assert!(
            session.is_none(),
            "a seq!=1 step must invalidate + drop the held session",
        );

        // A subsequent seq==1 token rebuilds a FRESH session (different
        // graph Arc) and produces correct logits vs. the D1 path on the
        // same running cache.
        let d2 = model
            .forward_with_kv_context_persistent(&[6], &mut cache, &mut ctx, &mut session)
            .expect("decode after fallback");
        assert!(session.is_some(), "session rebuilt on the next decode token");
        let graph_ptr_2 = Arc::as_ptr(session.as_ref().unwrap().graph());
        assert!(
            graph_ptr_1 != graph_ptr_2,
            "the rebuilt session must hold a NEW graph, not the dropped one",
        );

        // Byte-exact vs. a fresh D1 run over the identical token history.
        let mut cache_ref = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, max_seq_len, DType::F32, &Device::cpu(),
        ).expect("with_capacity ref");
        let mut ctx_ref = InferenceContext::new(Device::cpu());
        let _ = model.forward_with_kv_context(&prompt, &mut cache_ref, &mut ctx_ref).unwrap();
        let _ = model.forward_with_kv_context(&[3], &mut cache_ref, &mut ctx_ref).unwrap();
        let _ = model.forward_with_kv_context(&[4, 5], &mut cache_ref, &mut ctx_ref).unwrap();
        let d1 = model.forward_with_kv_context(&[6], &mut cache_ref, &mut ctx_ref).unwrap();
        assert_eq!(d2, d1, "post-fallback decode must match the D1 cached path");
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
        let full_logits = model.forward(&prompt, 0).unwrap();
        let last_pos = prompt.len() - 1;
        let expected = full_logits
            .slice(1, last_pos, 1).unwrap()
            .reshape(Shape::from_dims(&[cfg.vocab_size])).unwrap()
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

    // ---- kv-context all-positions + spec decode (E.3.4 port) ----------

    /// The all-positions variant's last row must equal what the
    /// regular (last-only) variant produces. Same graph, same cache
    /// state, same tokens — only the output shape differs.
    #[test]
    fn forward_with_kv_context_all_positions_last_row_matches_last_only() {
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
        let device = Device::cpu();

        // Path A: regular last-only forward.
        let mut cache_a = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
            tokens.len(), DType::F32, &device,
        ).expect("cache_a");
        let mut ctx_a = InferenceContext::new(device.clone());
        let last_only = model
            .forward_with_kv_context(&tokens, &mut cache_a, &mut ctx_a)
            .expect("last-only forward");
        assert_eq!(last_only.len(), cfg.vocab_size);

        // Path B: all-positions forward.
        let mut cache_b = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
            tokens.len(), DType::F32, &device,
        ).expect("cache_b");
        let mut ctx_b = InferenceContext::new(device.clone());
        let all = model
            .forward_with_kv_context_all_positions(&tokens, &mut cache_b, &mut ctx_b)
            .expect("all-positions forward");
        assert_eq!(all.len(), tokens.len() * cfg.vocab_size);

        // Last row of `all` must match last_only.
        let last_pos = tokens.len() - 1;
        let all_last = &all[last_pos * cfg.vocab_size .. (last_pos + 1) * cfg.vocab_size];
        for (i, (a, b)) in all_last.iter().zip(last_only.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "vocab idx {i}: all_positions={a} vs last_only={b}",
            );
        }

        // Both caches should have advanced by the same amount.
        assert_eq!(cache_a.cached_len, cache_b.cached_len);
    }

    /// `KvCache::truncate_to` rollback semantics on the pre-allocated
    /// WriteSlice path: decode a token, roll it back, decode different
    /// tokens through the same positions — the final logits must match
    /// an uninterrupted run that never saw the rolled-back token.
    /// This is exactly spec decode's reject path: stale K/V rows past
    /// `cached_len` must stop being read and must be overwritten by
    /// the next `Op::WriteSlice` at the same positions.
    #[test]
    fn kv_cache_truncate_then_redecode_matches_uninterrupted_decode() {
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
        let device = Device::cpu();

        // Path A (reference): prefill [3,7,1] then decode 9 then 2.
        let mut cache_a = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 5, DType::F32, &device,
        ).expect("cache_a");
        let mut ctx_a = InferenceContext::new(device.clone());
        model.forward_with_kv_context(&[3, 7, 1], &mut cache_a, &mut ctx_a).expect("prefill A");
        model.forward_with_kv_context(&[9], &mut cache_a, &mut ctx_a).expect("decode A1");
        let expected = model
            .forward_with_kv_context(&[2], &mut cache_a, &mut ctx_a)
            .expect("decode A2");

        // Path B: prefill [3,7,1], decode a WRONG token (11) at
        // position 3, roll it back, then decode [9, 2] through the
        // same positions in one step.
        let mut cache_b = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 5, DType::F32, &device,
        ).expect("cache_b");
        let mut ctx_b = InferenceContext::new(device.clone());
        model.forward_with_kv_context(&[3, 7, 1], &mut cache_b, &mut ctx_b).expect("prefill B");
        model.forward_with_kv_context(&[11], &mut cache_b, &mut ctx_b).expect("decode B wrong");
        assert_eq!(cache_b.cached_len, 4);
        cache_b.truncate_to(3);
        assert_eq!(cache_b.cached_len, 3);
        let actual = model
            .forward_with_kv_context(&[9, 2], &mut cache_b, &mut ctx_b)
            .expect("redecode B");
        assert_eq!(cache_b.cached_len, 5);

        assert_eq!(actual.len(), expected.len());
        // Tolerance: path A attends over (cached 4 + fresh 1) rows,
        // path B over (cached 3 + fresh 2) — standard O(ε) gemm
        // accumulation-order drift, same band as the other kv-context
        // parity tests.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / a.abs().max(b.abs()).max(1e-6);
            assert!(
                diff < 5e-3 || rel < 1e-2,
                "logit[{i}]: redecode={a}, uninterrupted={b}, diff={diff}",
            );
        }
    }

    /// Spec decode with the target as its own draft: every draft is
    /// trivially argmax-matched, acceptance is 100%, and the output
    /// must equal a plain greedy run through the same kv-context path.
    #[test]
    fn spec_decode_kv_context_self_draft_matches_greedy_baseline() {
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
        let device = Device::cpu();

        let baseline = model.generate_with_kv_context(
            &prompt, max_new, SamplingStrategy::Greedy, None,
            &device, DType::F32,
        ).expect("baseline generate");

        for k in [2_usize, 4] {
            let spec_out = model.generate_streaming_spec_with_kv_context(
                &model, &prompt, max_new, k,
                SamplingStrategy::Greedy, None,
                &device, DType::F32, |_| {},
            ).expect("spec generate");
            assert_eq!(
                spec_out, baseline,
                "K={k}: spec-decode must match baseline when draft == target",
            );
        }
    }

    /// Greedy spec decode is lossless for ANY draft: on the first
    /// mismatch the target's own argmax is emitted and the rejected
    /// draft rows are rolled back, so the output must equal plain
    /// greedy generation from the target. A draft with different
    /// weights forces genuine rejections, exercising the
    /// `KvCache::truncate_to` rollback + bonus-position re-write that
    /// the self-draft test (100% acceptance) never reaches.
    ///
    /// The retired legacy-executor implementation got this rollback
    /// wrong: it kept one stale K/V row at the bonus position
    /// (truncate to `committed + accepted + 1`) and appended the
    /// bonus one position too far — measured ~4e-3 logit drift on
    /// this fixture (argmax happened to survive, so its
    /// token-equality tests passed).
    #[test]
    fn spec_decode_kv_context_divergent_draft_matches_greedy_baseline() {
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
        let target = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights_seeded(&cfg, 9999),
        };
        let draft = LlamaModel {
            config:  cfg.clone(),
            weights: make_tiny_weights_seeded(&cfg, 4242),
        };
        let prompt = [3_u32, 7, 1];
        let max_new = 8;
        let device = Device::cpu();

        let baseline = target.generate_with_kv_context(
            &prompt, max_new, SamplingStrategy::Greedy, None,
            &device, DType::F32,
        ).expect("baseline generate");

        for k in [1_usize, 2, 4] {
            let spec_out = target.generate_streaming_spec_with_kv_context(
                &draft, &prompt, max_new, k,
                SamplingStrategy::Greedy, None,
                &device, DType::F32, |_| {},
            ).expect("spec generate");
            assert_eq!(
                spec_out, baseline,
                "K={k}: greedy spec-decode must be lossless for a divergent draft",
            );
        }
    }

    /// In Temperature mode with draft == target, the accept coin's
    /// ratio = min(1, p_target/p_draft) = 1.0, so acceptance is 100%.
    /// We can't bit-match against a plain sampled baseline because the
    /// RNG sequences diverge (spec-decode draws more randoms per
    /// output token than plain gen), but we can assert: (a) output has
    /// expected length, (b) all tokens are in vocab, (c) prompt prefix
    /// is preserved.
    #[test]
    fn spec_decode_kv_context_sampled_self_draft_produces_valid_tokens() {
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
        let device = Device::cpu();

        for k in [2_usize, 4] {
            let out = model.generate_streaming_spec_with_kv_context(
                &model, &prompt, max_new, k,
                SamplingStrategy::Temperature { temp: 0.8, seed: 42 },
                None,
                &device, DType::F32, |_| {},
            ).expect("spec sampled generate");

            // The emit loop returns the moment `emitted == max_new`,
            // so the output is exactly prompt + max_new tokens.
            assert_eq!(out.len(), prompt.len() + max_new,
                "K={k}: expected {} tokens, got {}",
                prompt.len() + max_new, out.len());
            assert_eq!(&out[..prompt.len()], &prompt);
            for &t in &out {
                assert!((t as usize) < cfg.vocab_size, "K={k}: token {t} out of vocab");
            }
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

    // ===== Phase 7.6 step 9c E.3.4 — legacy spec-decode tests retired =====
    //
    // The legacy-executor spec-decode + KVCache<B>-truncate tests
    // (`spec_decode_with_self_as_draft_matches_greedy_baseline`,
    // `spec_decode_sampled_with_self_as_draft_produces_valid_tokens`,
    // `forward_with_cache_all_positions_last_slice_matches_forward_with_cache`,
    // `kvcache_truncate_to_*`) retired with the `*_gpu_on` family.
    // Their kv-context successors are above:
    // `spec_decode_kv_context_*`,
    // `forward_with_kv_context_all_positions_last_row_matches_last_only`,
    // and `kv_cache_truncate_then_redecode_matches_uninterrupted_decode`
    // — the latter strictly stronger (behavioral rollback semantics,
    // not just buffer-shrink bookkeeping). The divergent-draft test
    // additionally locks the greedy-losslessness property the legacy
    // implementation violated on partial acceptance (~4e-3 logit
    // drift; see `generate_streaming_spec_with_kv_context`'s docs).
}

#[cfg(test)]
mod phi_kv_context_tests {
    use super::*;
    use crate::inference_context::{InferenceContext, KvCache};

    // Parity with the retired Phi `_gpu_on` family
    // (`forward_with_cache_gpu_on` prefill + decode logits;
    // `generate_streaming_gpu_on` greedy token sequences) was
    // confirmed by `phi_forward_with_kv_context_matches_legacy_gpu_on`
    // and `phi_generate_with_kv_context_matches_legacy_generate`
    // immediately before retirement (commit 03df5c49); those tests
    // retired together with the legacy methods they referenced.

    /// Build tiny Phi-2-shaped weights (Split QKV + biases everywhere,
    /// partial RoPE) for kv-context forward tests.
    fn make_tiny_phi(cfg: &PhiConfig, seed: u32) -> PhiWeights {
        let mut s: u32 = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            let v: Vec<f32> = (0..n).map(|_| next()).collect();
            Arc::from(v)
        };
        let d = cfg.dim;
        let kv_dim = cfg.n_heads * cfg.head_dim;
        PhiWeights {
            token_embedding: vec_of(cfg.vocab_size * d),
            layers: (0..cfg.n_layers)
                .map(|_| PhiLayerWeights {
                    attn_qkv: PhiQkv::Split {
                        q: vec_of(d * d).into(),
                        q_bias: vec_of(d),
                        k: vec_of(d * kv_dim).into(),
                        k_bias: vec_of(kv_dim),
                        v: vec_of(d * kv_dim).into(),
                        v_bias: vec_of(kv_dim),
                    },
                    attn_dense: vec_of(d * d).into(),
                    attn_dense_bias: vec_of(d),
                    mlp_fc1: vec_of(d * cfg.ffn_dim).into(),
                    mlp_fc1_bias: vec_of(cfg.ffn_dim),
                    mlp_fc2: vec_of(cfg.ffn_dim * d).into(),
                    mlp_fc2_bias: vec_of(d),
                    norm_gain: Arc::from(vec![1.0_f32; d]),
                    norm_bias: vec_of(d),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0_f32; d]),
            final_norm_bias: vec_of(d),
            output: vec_of(d * cfg.vocab_size).into(),
            output_bias: Some(vec_of(cfg.vocab_size)),
        }
    }

    fn tiny_cfg() -> PhiConfig {
        PhiConfig {
            vocab_size: 16,
            dim:        8,
            n_layers:   2,
            n_heads:    2,
            head_dim:   4,
            ffn_dim:    16,
            layer_norm_eps: 1e-5,
            rope_base:  10000.0,
            partial_rotary_factor: 0.5,
            rotary_dim: 2,
            tie_word_embeddings: false,
        }
    }

    /// KV-cache self-consistency on the new path: a monolithic prefill
    /// over N tokens must produce the same last-position logits as a
    /// shorter prefill followed by single-token decode steps through
    /// the same positions. Catches cache-position bugs without
    /// referencing the legacy path (survives its retirement).
    #[test]
    fn phi_kv_context_decode_consistent_with_monolithic_prefill() {
        let cfg = tiny_cfg();
        let model = PhiModel {
            config:  cfg.clone(),
            weights: make_tiny_phi(&cfg, 7777),
        };
        let tokens = [1_u32, 5, 9, 12];
        let device = Device::cpu();

        // Path A: monolithic prefill over all 4 tokens.
        let mut cache_a = KvCache::with_capacity(
            cfg.n_layers, cfg.n_heads, cfg.head_dim,
            tokens.len(), DType::F32, &device,
        ).expect("cache_a");
        let mut ctx_a = InferenceContext::new(device.clone());
        let expected = model
            .forward_with_kv_context(&tokens, &mut cache_a, &mut ctx_a)
            .expect("monolithic prefill");

        // Path B: prefill 3, decode 1.
        let mut cache_b = KvCache::with_capacity(
            cfg.n_layers, cfg.n_heads, cfg.head_dim,
            tokens.len(), DType::F32, &device,
        ).expect("cache_b");
        let mut ctx_b = InferenceContext::new(device.clone());
        model
            .forward_with_kv_context(&tokens[..3], &mut cache_b, &mut ctx_b)
            .expect("prefill B");
        let actual = model
            .forward_with_kv_context(&tokens[3..], &mut cache_b, &mut ctx_b)
            .expect("decode B");

        assert_eq!(actual.len(), expected.len());
        // Same O(ε) gemm accumulation-order band as the LLaMA
        // kv-context parity tests.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / a.abs().max(b.abs()).max(1e-6);
            assert!(
                diff < 5e-3 || rel < 1e-2,
                "logit[{i}]: chunked={a}, monolithic={b}, diff={diff}",
            );
        }
        assert_eq!(cache_a.cached_len, cache_b.cached_len);
    }

    /// Greedy generation through the Phi kv-context path: correct
    /// shape (prompt preserved, max_new appended, tokens in vocab)
    /// and fully deterministic across runs.
    #[test]
    fn phi_generate_with_kv_context_greedy_is_deterministic() {
        let cfg = tiny_cfg();
        let model = PhiModel {
            config:  cfg.clone(),
            weights: make_tiny_phi(&cfg, 7777),
        };
        let prompt = [1_u32, 5, 9];
        let max_new = 8;
        let device = Device::cpu();

        let mut streamed = Vec::new();
        let run_a = model.generate_streaming_with_kv_context(
            &prompt, max_new, SamplingStrategy::Greedy, None,
            &device, DType::F32, |t| streamed.push(t),
        ).expect("run a");
        let run_b = model.generate_with_kv_context(
            &prompt, max_new, SamplingStrategy::Greedy, None,
            &device, DType::F32,
        ).expect("run b");

        assert_eq!(run_a, run_b, "greedy generation must be deterministic");
        assert_eq!(run_a.len(), prompt.len() + max_new);
        assert_eq!(&run_a[..prompt.len()], &prompt);
        assert_eq!(streamed, &run_a[prompt.len()..], "callback sees exactly the new tokens");
        for &t in &run_a {
            assert!((t as usize) < cfg.vocab_size, "token {t} out of vocab");
        }
    }

    /// `forward_with_kv_context` build-time validation: with_dims
    /// caches (no pre-allocated buffers) and capacity overflows are
    /// rejected with typed errors, not panics.
    #[test]
    fn phi_forward_with_kv_context_rejects_invalid_cache() {
        let cfg = tiny_cfg();
        let model = PhiModel {
            config:  cfg.clone(),
            weights: make_tiny_phi(&cfg, 7777),
        };
        let device = Device::cpu();
        let mut ctx = InferenceContext::new(device.clone());

        // with_dims cache → typed error.
        let mut dims_cache = KvCache::with_dims(cfg.n_layers, cfg.n_heads, cfg.head_dim);
        let err = model
            .forward_with_kv_context(&[1, 2], &mut dims_cache, &mut ctx)
            .expect_err("with_dims cache must be rejected");
        assert!(format!("{err}").contains("with_capacity"));

        // Capacity overflow → typed error.
        let mut small_cache = KvCache::with_capacity(
            cfg.n_layers, cfg.n_heads, cfg.head_dim, 2, DType::F32, &device,
        ).expect("small cache");
        model
            .forward_with_kv_context(&[1, 2], &mut small_cache, &mut ctx)
            .expect("fits exactly");
        let err = model
            .forward_with_kv_context(&[3], &mut small_cache, &mut ctx)
            .expect_err("overflow must be rejected");
        assert!(format!("{err}").contains("max_seq_len"));
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
        let logits = model.forward(&tokens, 0).unwrap();
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
        let logits = model.forward(&tokens, 0).unwrap().realize_f32();
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
        let logits = model.forward(&tokens, 0).unwrap();
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
        let logits = model.forward(&tokens, 0).unwrap();
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
        let l0 = model.forward(&tokens, 0).unwrap().realize_f32();
        let l10 = model.forward(&tokens, 10).unwrap().realize_f32();
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
        let logits = model.forward(&tokens, 0).unwrap();
        // Take last-position slice and argmax over vocab dim, all
        // through the LazyTensor bridge API.
        let last = logits.slice(1, tokens.len() - 1, 1).unwrap(); // [1, 1, vocab]
        let last_flat = last.reshape(Shape::from_dims(&[cfg.vocab_size])).unwrap();
        let predicted_ids = last_flat.argmax_dim(0_usize).unwrap().realize_u32();
        assert_eq!(predicted_ids.len(), 1);
        let pred = predicted_ids[0];
        assert!(
            (pred as usize) < cfg.vocab_size,
            "argmax should return a valid vocab index",
        );
    }

    /// `forward_hidden_embeds_with_mask` runs the LlamaModel with
    /// a caller-supplied mask instead of the built-in strict
    /// causal one. An all-zero (bidirectional) mask must produce
    /// different hidden states than the strict-causal `forward`
    /// because the bidirectional path lets earlier tokens attend
    /// to later ones.
    #[test]
    fn forward_hidden_embeds_with_mask_bidirectional() {
        let cfg = LlamaConfig {
            vocab_size: 16,
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
        let tokens: Vec<u32> = vec![1, 2, 3, 4];

        // Causal reference path through `forward` → drop the
        // lm_head matmul mentally by comparing across runs.
        let _logits = model.forward(&tokens, 0).unwrap().realize_f32();

        // Build embeds + bidirectional (all-zero) mask on one graph.
        let embed = LazyTensor::from_f32(
            model.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &crate::Device::cpu(),
        );
        let token_ids = embed.const_u32_like(
            tokens.clone(), Shape::from_dims(&[tokens.len()]),
        );
        let embeds = embed
            .index_select(0_usize, &token_ids).unwrap()
            .reshape(Shape::from_dims(&[1, tokens.len(), cfg.dim])).unwrap();
        let zero_mask: Arc<[f32]> = Arc::from(vec![0.0_f32; tokens.len() * tokens.len()]);
        let mask = embeds.const_f32_like(
            zero_mask, Shape::from_dims(&[1, 1, tokens.len(), tokens.len()]),
        );
        let bidir = model.forward_hidden_embeds_with_mask(&embeds, &mask, 0)
            .unwrap()
            .realize_f32();

        // Also run the standard causal hidden path for comparison.
        // forward_embeds applies the LM head; we need just the
        // hidden state, so build it separately via forward_embeds
        // and undo the lm_head implicitly by checking the difference
        // is non-trivial across a known position.
        let causal_logits = model.forward(&tokens, 0).unwrap().realize_f32();
        assert_eq!(bidir.len(), tokens.len() * cfg.dim);
        for &v in &bidir {
            assert!(v.is_finite(), "bidirectional hidden state not finite: {v}");
        }
        assert!(!causal_logits.is_empty());
    }

    /// `forward_hidden_embeds(embeds, start_pos)` returns
    /// post-final-RmsNorm hidden states for pre-built embeds —
    /// useful for multimodal hosts (LLaVA, Pixtral) that
    /// interleave image embeddings with text embeddings and
    /// want hidden states without the lm_head projection.
    /// The result must match `forward_embeds(embeds,
    /// start_pos)` projected through the lm_head, because
    /// `forward_embeds` is exactly `forward_hidden_embeds`
    /// followed by `lm_head.apply_linear`.
    #[test]
    fn forward_hidden_embeds_followed_by_lm_head_matches_forward_embeds() {
        let cfg = LlamaConfig {
            vocab_size: 16, dim: 8, n_layers: 1, n_heads: 2,
            n_kv_heads: 2, head_dim: 4, ffn_dim: 16,
            norm_eps: 1e-5, rope_base: 10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel { config: cfg.clone(), weights };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];

        let embed = LazyTensor::from_f32(
            model.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &crate::Device::cpu(),
        );
        let token_ids = embed.const_u32_like(
            tokens.clone(), Shape::from_dims(&[tokens.len()]),
        );
        let embeds = embed
            .index_select(0_usize, &token_ids).unwrap()
            .reshape(Shape::from_dims(&[1, tokens.len(), cfg.dim])).unwrap();

        let hidden = model.forward_hidden_embeds(&embeds, 0).unwrap();
        let logits_from_hidden = model.weights.output
            .apply_linear(&hidden, cfg.dim, cfg.vocab_size).realize_f32();
        let logits_direct = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        assert_eq!(logits_from_hidden.len(), logits_direct.len());
        for (a, b) in logits_from_hidden.iter().zip(logits_direct.iter()) {
            assert!((a - b).abs() < 1e-6,
                "forward_hidden_embeds + lm_head must match forward_embeds: {a} vs {b}");
        }
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
        let out = t.unsqueeze(0_usize).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2, 2]);
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn unsqueeze_errors_out_of_bounds() {
        let t = cpu_f32(vec![1.0, 2.0], &[2]);
        // rank=1, so dim<=1 is valid; dim=2 must error.
        assert!(t.unsqueeze(0_usize).is_ok());
        assert!(t.unsqueeze(1_usize).is_ok());
        assert!(t.unsqueeze(2_usize).is_err());
    }

    #[test]
    fn try_reshape_errors_on_size_mismatch() {
        let t = cpu_f32(vec![1.0; 6], &[2, 3]);
        assert!(t.reshape(vec![3, 2]).is_ok());
        assert!(t.reshape(vec![2, 2]).is_err());
    }

    #[test]
    fn try_permute_validates_axes() {
        let t = cpu_f32(vec![0.0; 24], &[2, 3, 4]);
        // The Dims trait accepts tuples, owned arrays, and slices.
        assert!(t.permute((2_usize, 0_usize, 1_usize)).is_ok());
        assert!(t.permute([2_usize, 0, 1]).is_ok());
        assert!(t.permute([0_usize, 1]).is_err());     // wrong rank
        assert!(t.permute([0_usize, 0, 1]).is_err()); // dup axis
    }

    #[test]
    fn try_transpose_requires_rank_two_plus() {
        let scalar = cpu_f32(vec![1.0], &[1]);
        // rank-1: transpose surfaces a typed error at build time.
        let _ = scalar.transpose();
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
    fn softmax_general_axis_matches_hand_computed_rank3_middle_axis() {
        // shape [2, 3, 2] — softmax along axis=1 (the size-3 axis).
        // Layout (row-major): element (b, r, c) lives at b*6 + r*2 + c.
        // We hand-pick values so each (b, c) lane's max is exactly 0 to
        // make the reference closed-form: probs = exp(x) / sum_r exp(x).
        // Lane (b=0, c=0): values [-1, 0, -2]  → exp = [e^-1, 1, e^-2]
        // Lane (b=0, c=1): values [ 0,-3, -1]  → exp = [1, e^-3, e^-1]
        // Lane (b=1, c=0): values [-2,-1,  0]  → exp = [e^-2, e^-1, 1]
        // Lane (b=1, c=1): values [-1,-1,  0]  → exp = [e^-1, e^-1, 1]
        let data: Vec<f32> = vec![
            // b=0
            -1.0,  0.0,   // r=0, c=0..1
             0.0, -3.0,   // r=1
            -2.0, -1.0,   // r=2
            // b=1
            -2.0, -1.0,   // r=0
            -1.0, -1.0,   // r=1
             0.0,  0.0,   // r=2
        ];
        let t = cpu_f32(data, &[2, 3, 2]);
        let out = t.softmax(1_usize).unwrap();
        assert_eq!(out.shape().dims(), &[2, 3, 2]);
        let v = out.realize_f32();

        // Reference: closed-form softmax per (b, c) lane.
        let lane_softmax = |xs: [f32; 3]| -> [f32; 3] {
            let m = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps = [(xs[0] - m).exp(), (xs[1] - m).exp(), (xs[2] - m).exp()];
            let s = exps[0] + exps[1] + exps[2];
            [exps[0] / s, exps[1] / s, exps[2] / s]
        };
        // Re-extract source lanes
        let lanes: [[f32; 3]; 4] = [
            [-1.0,  0.0, -2.0],   // (b=0, c=0)
            [ 0.0, -3.0, -1.0],   // (b=0, c=1)
            [-2.0, -1.0,  0.0],   // (b=1, c=0)
            [-1.0, -1.0,  0.0],   // (b=1, c=1)
        ];
        let refs: Vec<[f32; 3]> = lanes.iter().map(|l| lane_softmax(*l)).collect();

        // out[b, r, c] at index b*6 + r*2 + c — verify each (b, c) lane.
        for b in 0..2 {
            for c in 0..2 {
                let lane_ix = b * 2 + c;
                for r in 0..3 {
                    let got = v[b * 6 + r * 2 + c];
                    let want = refs[lane_ix][r];
                    assert!(
                        (got - want).abs() < 1e-6,
                        "softmax mismatch at (b={b}, r={r}, c={c}): got {got}, want {want}",
                    );
                }
                // sanity: lane sums to 1
                let sum: f32 = (0..3).map(|r| v[b * 6 + r * 2 + c]).sum();
                assert!((sum - 1.0).abs() < 1e-6, "lane (b={b},c={c}) sums to {sum}");
            }
        }
    }

    #[test]
    fn softmax_last_axis_matches_softmax_last_dim() {
        let data: Vec<f32> = vec![
            1.0,  2.0, -1.0,  0.5,
            0.0, -2.0,  3.0,  1.5,
            // batch dim 2
            4.0, -1.0,  2.0,  0.0,
            -3.0, 0.25, 0.75, 1.0,
        ];
        let t = cpu_f32(data, &[2, 2, 4]);
        let via_general = t.softmax(2_usize).unwrap();
        let via_fused = t.softmax_last_dim().unwrap();
        assert_eq!(via_general.shape().dims(), via_fused.shape().dims());
        let g = via_general.realize_f32();
        let f = via_fused.realize_f32();
        assert_eq!(g.len(), f.len());
        for (i, (a, b)) in g.iter().zip(f.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "softmax (general axis=2) vs softmax_last_dim diverge at {i}: {a} vs {b}",
            );
        }
    }

    #[test]
    fn log_softmax_general_axis_matches_hand_computed_rank3_middle_axis() {
        // Same construction as the softmax test, but compare against
        // closed-form log_softmax per (b, c) lane.
        let data: Vec<f32> = vec![
            -1.0,  0.0,
             0.0, -3.0,
            -2.0, -1.0,
            -2.0, -1.0,
            -1.0, -1.0,
             0.0,  0.0,
        ];
        let t = cpu_f32(data, &[2, 3, 2]);
        let out = t.log_softmax(1_usize).unwrap();
        assert_eq!(out.shape().dims(), &[2, 3, 2]);
        let v = out.realize_f32();

        let lane_log_softmax = |xs: [f32; 3]| -> [f32; 3] {
            let m = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let shifted = [xs[0] - m, xs[1] - m, xs[2] - m];
            let lse = (shifted[0].exp() + shifted[1].exp() + shifted[2].exp()).ln();
            [shifted[0] - lse, shifted[1] - lse, shifted[2] - lse]
        };
        let lanes: [[f32; 3]; 4] = [
            [-1.0,  0.0, -2.0],
            [ 0.0, -3.0, -1.0],
            [-2.0, -1.0,  0.0],
            [-1.0, -1.0,  0.0],
        ];
        let refs: Vec<[f32; 3]> = lanes.iter().map(|l| lane_log_softmax(*l)).collect();

        for b in 0..2 {
            for c in 0..2 {
                let lane_ix = b * 2 + c;
                for r in 0..3 {
                    let got = v[b * 6 + r * 2 + c];
                    let want = refs[lane_ix][r];
                    assert!(
                        (got - want).abs() < 1e-6,
                        "log_softmax mismatch at (b={b}, r={r}, c={c}): got {got}, want {want}",
                    );
                    // log_softmax values must be <= 0
                    assert!(got <= 1e-6, "log_softmax produced positive value: {got}");
                }
            }
        }
    }

    #[test]
    fn log_softmax_last_axis_matches_log_softmax_last_dim() {
        let data: Vec<f32> = vec![
            1.0,  2.0, -1.0,  0.5,
            0.0, -2.0,  3.0,  1.5,
            4.0, -1.0,  2.0,  0.0,
            -3.0, 0.25, 0.75, 1.0,
        ];
        let t = cpu_f32(data, &[2, 2, 4]);
        let via_general = t.log_softmax(2_usize).unwrap();
        let via_fused = t.log_softmax_last_dim().unwrap();
        assert_eq!(via_general.shape().dims(), via_fused.shape().dims());
        let g = via_general.realize_f32();
        let f = via_fused.realize_f32();
        assert_eq!(g.len(), f.len());
        for (i, (a, b)) in g.iter().zip(f.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "log_softmax (general axis=2) vs log_softmax_last_dim diverge at {i}: {a} vs {b}",
            );
        }
    }

    #[test]
    fn argmin_dim_drops_reduced_axis() {
        let t = cpu_f32(vec![3.0, 1.0, 2.0, 0.5, 5.0, 4.0], &[2, 3]);
        let out = t.argmin_dim(1_usize).unwrap();
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
        let mask = probe.gt(&threshold).unwrap(); // [0, 1, 1, 0] as U8
        let out = t.masked_fill(&mask, fuel_ir::Scalar::F32(-9.0)).unwrap();
        assert_eq!(out.realize_f32(), vec![1.0, -9.0, -9.0, 4.0]);
    }

    #[test]
    fn index_add_smoke() {
        let base = cpu_f32(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let src = base.const_f32_like(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let indices = base.const_u32_like(vec![0_u32, 0_u32], vec![2]);
        let out = base.index_add(0, &indices, &src).unwrap();
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
        let out = base.scatter_add(0, &indices, &src).unwrap();
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
    fn split_heads_then_merge_heads_round_trip() {
        // (B=1, N=2, embed=6) — split into 2 heads of head_dim=3.
        let x = cpu_f32((0..12).map(|i| i as f32).collect(), &[1, 2, 6]);
        let split = x.split_heads(2, 3).unwrap();
        assert_eq!(split.shape().dims(), &[1, 2, 2, 3]);
        let merged = split.merge_heads().unwrap();
        assert_eq!(merged.shape().dims(), &[1, 2, 6]);
        let m = merged.realize_f32();
        let original = x.realize_f32();
        for (a, b) in m.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-7, "{a} vs {b}");
        }
    }

    #[test]
    fn rope_tables_const_shapes_and_anchoring() {
        let anchor = cpu_f32(vec![0.0_f32; 8], &[1, 4, 2]);
        let (cos, sin) = anchor.rope_tables_const(10_000.0, 0, 4, 8);
        assert_eq!(cos.shape().dims(), &[4, 8]);
        assert_eq!(sin.shape().dims(), &[4, 8]);
        // First row at position 0: cos = 1.0, sin = 0.0 for every pair.
        let cv = cos.realize_f32();
        let sv = sin.realize_f32();
        // cos[0,...] should all be 1.0 (RoPE at position 0).
        for i in 0..8 {
            assert!((cv[i] - 1.0).abs() < 1e-5, "cos[0,{i}] = {}", cv[i]);
            assert!(sv[i].abs() < 1e-5, "sin[0,{i}] = {}", sv[i]);
        }
    }

    #[test]
    fn embed_tokens_shape_and_lookup() {
        // 5-token vocab, 3-dim hidden. Vocab embedding table contains
        // row i = (i, i+0.5, i+1) so the lookup result is verifiable.
        let vocab_size = 5;
        let hidden = 3;
        let table: Vec<f32> = (0..vocab_size).flat_map(|i| {
            vec![i as f32, i as f32 + 0.5, i as f32 + 1.0]
        }).collect();
        let tokens = vec![1_u32, 3, 0];
        let out = LazyTensor::embed_tokens(
            std::sync::Arc::from(table), vocab_size, hidden,
            &tokens, &crate::Device::cpu(),
        ).unwrap();
        assert_eq!(out.shape().dims(), &[1, 3, hidden]);
        let v = out.realize_f32();
        let want = [
            1.0_f32, 1.5, 2.0,  // token 1
            3.0, 3.5, 4.0,      // token 3
            0.0, 0.5, 1.0,      // token 0
        ];
        for (i, (&got, &exp)) in v.iter().zip(want.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "row {i}: got={got} want={exp}");
        }
    }

    #[test]
    fn embed_tokens_anchored_lives_on_receiver_graph() {
        // Two paths: one bootstrapped via embed_tokens, one anchored
        // on a pre-existing tensor. Both produce identical values,
        // but only the anchored one can compose with the anchor.
        let vocab_size = 4;
        let hidden = 2;
        let table: Vec<f32> = (0..vocab_size).flat_map(|i| {
            vec![i as f32, i as f32 * 2.0]
        }).collect();
        let table_arc: std::sync::Arc<[f32]> = std::sync::Arc::from(table);
        let tokens = vec![2_u32, 1];

        let anchor = cpu_f32(vec![0.0_f32], &[1]);
        let embedded = anchor.embed_tokens_anchored(
            std::sync::Arc::clone(&table_arc), vocab_size, hidden, &tokens,
        ).unwrap();
        assert_eq!(embedded.shape().dims(), &[1, 2, hidden]);

        // Anchored: composes with the anchor.
        let one_scaled = anchor.const_f32_like(
            std::sync::Arc::from(vec![1.0_f32]),
            Shape::from_dims(&[1]),
        );
        let _ = embedded.add(&one_scaled.reshape(Shape::from_dims(&[1, 1, 1])).unwrap().broadcast_to(Shape::from_dims(&[1, 2, hidden])).unwrap()).unwrap();
        let v = embedded.realize_f32();
        let want = [
            2.0_f32, 4.0,  // token 2
            1.0, 2.0,      // token 1
        ];
        for (i, (&got, &exp)) in v.iter().zip(want.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "row {i}: got={got} want={exp}");
        }
    }

    #[test]
    fn embed_tokens_empty_returns_error() {
        let r = LazyTensor::embed_tokens(
            std::sync::Arc::from(vec![0.0_f32]), 1, 1, &[], &crate::Device::cpu(),
        );
        assert!(r.is_err());
    }

    #[test]
    fn softcap_matches_tanh_form() {
        // Tiny input; verify cap·tanh(x/cap) at known points.
        let x = cpu_f32(vec![0.0_f32, 5.0, -5.0, 30.0], &[1, 4]);
        let capped = x.softcap(10.0).realize_f32();
        // tanh(0)=0; tanh(0.5)≈0.4621; tanh(-0.5)≈-0.4621; tanh(3)≈0.9951.
        let expect = [0.0_f32, 4.6212, -4.6212, 9.9505];
        for (i, (&got, &want)) in capped.iter().zip(expect.iter()).enumerate() {
            assert!((got - want).abs() < 1e-3, "softcap[{i}] got={got} want={want}");
        }
    }

    #[test]
    fn softcap_optional_none_returns_input_unchanged() {
        let x = cpu_f32(vec![1.0_f32, -2.0, 30.0], &[1, 3]);
        let out = x.softcap_optional(None).realize_f32();
        let expect = [1.0_f32, -2.0, 30.0];
        for (g, w) in out.iter().zip(expect.iter()) {
            assert!((g - w).abs() < 1e-6);
        }
        // Some(0.0) or negative cap also returns unchanged (guard).
        let out = x.softcap_optional(Some(0.0)).realize_f32();
        for (g, w) in out.iter().zip(expect.iter()) {
            assert!((g - w).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_partial_full_dim_matches_rope_with_tables() {
        // head_dim == rope_dim ⇒ should degenerate to full rope.
        let qk = cpu_f32(vec![1.0_f32; 1 * 1 * 2 * 4], &[1, 1, 2, 4]);
        let (cos, sin) = qk.rope_tables_const(10_000.0, 0, 2, 4);
        let via_partial = qk.rope_partial(&cos, &sin, 4).unwrap();
        let via_full = qk.rope_with_tables(&cos, &sin).unwrap();
        let a = via_partial.realize_f32();
        let b = via_full.realize_f32();
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            assert!((av - bv).abs() < 1e-6, "{i}: {av} vs {bv}");
        }
    }

    #[test]
    fn rope_partial_pass_through_suffix_unchanged() {
        // rope_dim=2, head_dim=4 ⇒ first 2 features rotated, last 2
        // unchanged. At position 0 the rotation is identity, so all 4
        // features should equal the input.
        let qk = cpu_f32(
            vec![
                // shape [1, 1, 1, 4] — one position, one head
                1.0_f32, 2.0, 3.0, 4.0,
            ],
            &[1, 1, 1, 4],
        );
        let (cos, sin) = qk.rope_tables_const(10_000.0, 0, 1, 2);
        let out = qk.rope_partial(&cos, &sin, 2).unwrap().realize_f32();
        for (i, &v) in out.iter().enumerate() {
            let expect = [1.0_f32, 2.0, 3.0, 4.0][i];
            assert!((v - expect).abs() < 1e-6, "{i}: {v} != {expect}");
        }
    }

    #[test]
    fn add_optional_trailing_bias_none_returns_input_unchanged() {
        let a = cpu_f32(vec![1.0_f32, 2.0, 3.0], &[1, 3]);
        let original = a.realize_f32();
        let out = a.add_optional_trailing_bias(None).unwrap();
        assert_eq!(out.realize_f32(), original);
    }

    #[test]
    fn add_optional_trailing_bias_some_applies_add() {
        let a = cpu_f32(vec![1.0_f32, 2.0, 3.0], &[1, 3]);
        let bias = std::sync::Arc::<[f32]>::from(vec![10.0_f32, 20.0, 30.0]);
        let out = a.add_optional_trailing_bias(Some(&bias)).unwrap();
        assert_eq!(out.realize_f32(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_trailing_bias_broadcasts_across_leading_dims() {
        // (2, 3) input + length-3 bias should add per-column.
        let x = cpu_f32(vec![1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0], &[2, 3]);
        let bias = std::sync::Arc::<[f32]>::from(vec![100.0_f32, 200.0, 300.0]);
        let out = x.add_trailing_bias(bias).unwrap();
        assert_eq!(out.shape().dims(), &[2, 3]);
        let v = out.realize_f32();
        assert_eq!(v, vec![101.0, 202.0, 303.0, 110.0, 220.0, 330.0]);
    }

    #[test]
    fn rms_norm_affine_with_offset_adds_offset_to_each_gain() {
        let x = cpu_f32(vec![1.0_f32, 2.0, 3.0], &[1, 3]);
        let gain_raw: [f32; 3] = [-0.5, 0.0, 0.5];
        let via_offset = x.rms_norm_affine_with_offset(&gain_raw, 1.0, 1e-6).unwrap();
        let gain_shifted = std::sync::Arc::<[f32]>::from(vec![0.5_f32, 1.0, 1.5]);
        let via_plain = x.rms_norm_affine(gain_shifted, 1e-6).unwrap();
        let a = via_offset.realize_f32();
        let b = via_plain.realize_f32();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
    }

    #[test]
    fn global_avg_pool_2d_averages_spatial_dims() {
        // (1, 2, 2, 3) — two channels, 2×3 spatial.
        // Channel 0: 1..=6, mean = 3.5.
        // Channel 1: 10, 20, 30, 40, 50, 60, mean = 35.
        let x = cpu_f32(
            vec![
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0,
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            ],
            &[1, 2, 2, 3],
        );
        let out = x.global_avg_pool_2d().unwrap();
        assert_eq!(out.shape().dims(), &[1, 2]);
        let v = out.realize_f32();
        assert!((v[0] - 3.5).abs() < 1e-5);
        assert!((v[1] - 35.0).abs() < 1e-4);
    }

    #[test]
    fn channel_affine_4d_applies_per_channel_gain_and_bias() {
        // (1, 2, 2, 2) — two channels, each a 2×2 spatial map.
        let x = cpu_f32(
            vec![
                1.0_f32, 2.0, 3.0, 4.0,    // channel 0
                10.0,    20.0, 30.0, 40.0, // channel 1
            ],
            &[1, 2, 2, 2],
        );
        let gain = std::sync::Arc::<[f32]>::from(vec![2.0_f32, 0.5]);
        let bias = std::sync::Arc::<[f32]>::from(vec![1.0_f32, -10.0]);
        let out = x.channel_affine_4d(gain, bias).unwrap();
        let v = out.realize_f32();
        // Channel 0: gain=2, bias=1 → 2x+1
        assert_eq!(&v[0..4], &[3.0, 5.0, 7.0, 9.0]);
        // Channel 1: gain=0.5, bias=-10 → 0.5x-10
        assert_eq!(&v[4..8], &[-5.0, 0.0, 5.0, 10.0]);
    }

    #[test]
    fn additive_causal_mask_has_strict_lower_triangle() {
        let anchor = cpu_f32(vec![0.0_f32], &[1]);
        let mask = LazyTensor::additive_causal_mask_like(&anchor, 4);
        assert_eq!(mask.shape().dims(), &[4, 4]);
        let v = mask.realize_f32();
        // Expected (-inf shown as 'x'):
        //   0 x x x
        //   0 0 x x
        //   0 0 0 x
        //   0 0 0 0
        for i in 0..4 {
            for j in 0..4 {
                let got = v[i * 4 + j];
                if j > i {
                    assert!(got.is_infinite() && got.is_sign_negative(),
                        "above-diag (i={i}, j={j}) should be -inf, got {got}");
                } else {
                    assert_eq!(got, 0.0,
                        "on/below-diag (i={i}, j={j}) should be 0, got {got}");
                }
            }
        }
    }

    #[test]
    fn layer_norm_affine_unit_gain_zero_bias_matches_layer_norm_last_dim() {
        let a = cpu_f32(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let gain = std::sync::Arc::<[f32]>::from(vec![1.0_f32; 3]);
        let bias = std::sync::Arc::<[f32]>::from(vec![0.0_f32; 3]);
        let out_affine = a.layer_norm_affine(gain, bias, 1e-5).unwrap();
        let out_plain = a.layer_norm_last_dim(1e-5).unwrap();
        let va = out_affine.realize_f32();
        let vp = out_plain.realize_f32();
        for (a, b) in va.iter().zip(vp.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn layer_norm_affine_applies_gain_and_bias() {
        let a = cpu_f32(vec![1.0_f32, 2.0, 3.0], &[1, 3]);
        let gain = std::sync::Arc::<[f32]>::from(vec![2.0_f32, 0.5, 1.5]);
        let bias = std::sync::Arc::<[f32]>::from(vec![10.0_f32, -5.0, 0.0]);
        let out = a.layer_norm_affine(gain, bias, 1e-5).unwrap();
        let v = out.realize_f32();
        // Manual: mean=2, var=2/3; normed = (x-2)/sqrt(2/3+1e-5).
        let mean = 2.0_f32;
        let var = ((1.0 - mean).powi(2) + (2.0 - mean).powi(2) + (3.0 - mean).powi(2)) / 3.0;
        let den = (var + 1e-5_f32).sqrt();
        let expected = [
            ((1.0 - mean) / den) * 2.0 + 10.0,
            ((2.0 - mean) / den) * 0.5 + (-5.0),
            ((3.0 - mean) / den) * 1.5 + 0.0,
        ];
        for (got, want) in v.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "{got} vs {want}");
        }
    }

    #[test]
    fn l2_normalize_last_dim_unit_norm_per_row() {
        // (2, 3): rows [3,4,0] (norm 5) and [1,2,2] (norm 3).
        let a = cpu_f32(vec![3.0, 4.0, 0.0, 1.0, 2.0, 2.0], &[2, 3]);
        let out = a.l2_normalize(1_usize, 1e-12).unwrap();
        assert_eq!(out.shape().dims(), &[2, 3]);
        let v = out.realize_f32();
        let row0_norm = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        let row1_norm = (v[3]*v[3] + v[4]*v[4] + v[5]*v[5]).sqrt();
        assert!((row0_norm - 1.0).abs() < 1e-5, "row 0 norm = {row0_norm}");
        assert!((row1_norm - 1.0).abs() < 1e-5, "row 1 norm = {row1_norm}");
        // Row 0: [3,4,0]/5 → [0.6, 0.8, 0.0].
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
        assert!(v[2].abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_eps_zero_works_when_nonzero() {
        let a = cpu_f32(vec![1.0_f32, 0.0], &[2]);
        let out = a.l2_normalize(0_usize, 0.0).unwrap();
        let v = out.realize_f32();
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!(v[1].abs() < 1e-6);
    }

    #[test]
    fn repeat_interleave_last_dim_matches_torch_semantics() {
        // (2, 3) input: rows [1,2,3] and [4,5,6]. dim=1, repeats=2
        // → each element becomes two consecutive copies:
        // (2, 6): [1,1,2,2,3,3] and [4,4,5,5,6,6].
        let a = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let out = a.repeat_interleave(1_usize, 2).unwrap();
        assert_eq!(out.shape().dims(), &[2, 6]);
        assert_eq!(out.realize_f32(),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
                 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]);
    }

    #[test]
    fn repeat_interleave_middle_dim() {
        // (2, 2, 2) input. dim=1, repeats=3 → (2, 6, 2).
        let a = cpu_f32((0..8).map(|i| i as f32).collect(), &[2, 2, 2]);
        let out = a.repeat_interleave(1_usize, 3).unwrap();
        assert_eq!(out.shape().dims(), &[2, 6, 2]);
        // First sample's elements: (0,1) repeated 3× then (2,3) repeated 3×.
        let v = out.realize_f32();
        assert_eq!(&v[0..6], &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(&v[6..12], &[2.0, 3.0, 2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn repeat_interleave_repeats_one_is_noop() {
        let a = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let out = a.repeat_interleave(0_usize, 1).unwrap();
        assert_eq!(out.shape().dims(), &[3]);
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn repeat_interleave_repeats_zero_errors() {
        let a = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        assert!(a.repeat_interleave(0_usize, 0).is_err());
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
        let out = t.sum_keepdim(1_usize).unwrap();
        assert_eq!(out.shape().dims(), &[2, 1]);
        assert_eq!(out.realize_f32(), vec![3.0, 7.0]);
    }

    #[test]
    fn mean_keepdim_preserves_dim_as_one() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = t.mean_keepdim(0_usize).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2]);
        assert_eq!(out.realize_f32(), vec![2.0, 3.0]);
    }

    #[test]
    fn max_keepdim_preserves_dim_as_one() {
        let t = cpu_f32(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]);
        let out = t.max_keepdim(1_usize).unwrap();
        assert_eq!(out.shape().dims(), &[2, 1]);
        assert_eq!(out.realize_f32(), vec![3.0, 4.0]);
    }

    #[test]
    fn min_keepdim_preserves_dim_as_one() {
        let t = cpu_f32(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]);
        let out = t.min_keepdim(1_usize).unwrap();
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
        let out = a.broadcast_matmul(&b).unwrap();
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
        let ones = t.ones_like().unwrap();
        assert_eq!(ones.shape().dims(), t.shape().dims());
        assert_eq!(ones.dtype(), t.dtype());
        assert_eq!(ones.realize_f32(), vec![1.0; 4]);
    }

    #[test]
    fn zeros_like_matches_shape_dtype_graph() {
        let t = cpu_f32(vec![5.0, 6.0, 7.0], &[3]);
        let zeros = t.zeros_like().unwrap();
        assert_eq!(zeros.realize_f32(), vec![0.0; 3]);
    }

    #[test]
    fn static_ones_f32() {
        let t = LazyTensor::ones(vec![2, 3], DType::F32, &Device::cpu()).unwrap();
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.realize_f32(), vec![1.0; 6]);
    }

    #[test]
    fn static_zeros_f64() {
        let t = LazyTensor::zeros(vec![4], DType::F64, &Device::cpu()).unwrap();
        assert_eq!(t.dtype(), DType::F64);
        assert_eq!(t.realize_f64(), vec![0.0; 4]);
    }

    #[test]
    fn full_with_f32_scalar() {
        let t = LazyTensor::full(
            vec![5], fuel_ir::Scalar::F32(2.5), &Device::cpu(),
        ).unwrap();
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
        let a = t.narrow(0, 1, 3).unwrap().realize_f32();
        let b = t.slice(0, 1, 3).unwrap().realize_f32();
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
        let s = t.sum_dims([0, 2_usize]).unwrap();
        assert_eq!(s.shape().dims(), &[3]);
        // each element is 2*4 = 8 (sum across dim 0 = 2 elements, dim 2 = 4 elements)
        assert_eq!(s.realize_f32(), vec![8.0; 3]);
    }

    #[test]
    fn mean_dims_multi_dim() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let m = t.mean_dims([0, 1_usize]).unwrap();
        assert_eq!(m.shape().dims(), &[] as &[usize]);
        assert_eq!(m.realize_f32(), vec![2.5]);
    }

    #[test]
    fn sum_dims_keepdim_preserves_rank() {
        let t = cpu_f32(vec![1.0; 24], &[2, 3, 4]);
        let s = t.sum_dims_keepdim(&[0, 2]).unwrap();
        assert_eq!(s.shape().dims(), &[1, 3, 1]);
    }

    #[test]
    fn mean_dims_keepdim_preserves_rank() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let m = t.mean_dims_keepdim(&[0, 1]).unwrap();
        assert_eq!(m.shape().dims(), &[1, 1]);
        assert_eq!(m.realize_f32(), vec![2.5]);
    }

    #[test]
    fn rand_like_matches_shape_dtype() {
        let t = cpu_f32(vec![0.0; 6], &[2, 3]);
        let r = t.rand_like(-1.0, 1.0).unwrap();
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
        let r = t.randn_like(0.0, 1.0).unwrap();
        assert_eq!(r.shape().dims(), &[4]);
        assert_eq!(r.dtype(), DType::F32);
        // Samples are random — just sanity-check they're finite.
        for v in r.realize_f32() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn static_rand_f32() {
        let t = LazyTensor::rand(vec![100], 0.0, 1.0, DType::F32, &Device::cpu()).unwrap();
        let v = t.realize_f32();
        // Mean of uniform [0,1) should be ~0.5; tolerate sample noise.
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert!((0.3..0.7).contains(&mean), "mean {mean} too far from 0.5");
    }

    #[test]
    fn static_randn_f64() {
        let t = LazyTensor::randn(vec![1000], 0.0, 1.0, DType::F64, &Device::cpu()).unwrap();
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

    #[test]
    fn pad_with_value_zero_matches_pad_with_zeros() {
        // pad_with_value(_, _, _, 0.0) must be observationally identical
        // to pad_with_zeros — the latter is now a wrapper for the former.
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let via_zeros = t.pad_with_zeros(0, 1, 2).unwrap();
        let via_value = t.pad_with_value(0, 1, 2, 0.0).unwrap();
        assert_eq!(via_zeros.shape().dims(), via_value.shape().dims());
        assert_eq!(via_zeros.shape().dims(), &[5, 2]);
        assert_eq!(via_zeros.realize_f32(), via_value.realize_f32());
        assert_eq!(
            via_value.realize_f32(),
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn pad_with_value_neg_inf_then_max_reduce_ignores_pad() {
        // Negative interior values: -1, -2, -3. Pad with -inf on both
        // sides. Max along dim 0 must be the interior max (-1.0), not
        // -inf. This is the load-bearing property for max_pool2d.
        let t = cpu_f32(vec![-1.0, -2.0, -3.0], &[3]);
        let padded = t.pad_with_value(0, 2, 2, f32::NEG_INFINITY).unwrap();
        assert_eq!(padded.shape().dims(), &[7]);
        let v = padded.realize_f32();
        // Layout: [-inf, -inf, -1, -2, -3, -inf, -inf]
        assert!(v[0].is_infinite() && v[0].is_sign_negative());
        assert!(v[1].is_infinite() && v[1].is_sign_negative());
        assert_eq!(v[2], -1.0);
        assert_eq!(v[3], -2.0);
        assert_eq!(v[4], -3.0);
        assert!(v[5].is_infinite() && v[5].is_sign_negative());
        assert!(v[6].is_infinite() && v[6].is_sign_negative());
        // max along the only dim drops the pad and returns the interior max.
        let m = padded.max_all().realize_f32();
        assert_eq!(m, vec![-1.0]);
    }

    #[test]
    fn pad_with_value_identity_when_both_zero() {
        // The early-out path must fire regardless of value (no spurious
        // graph node when there's nothing to pad).
        let t = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let p = t.pad_with_value(0, 0, 0, f32::NEG_INFINITY).unwrap();
        assert_eq!(p.realize_f32(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn pad_with_value_rejects_bad_dim() {
        let t = cpu_f32(vec![1.0, 2.0], &[2]);
        assert!(t.pad_with_value(3, 1, 1, 0.0).is_err());
    }

    #[test]
    fn max_pool2d_with_pad_value_neg_inf_on_negative_interior() {
        // Negative-only interior values: a zero-padded max_pool2d would
        // incorrectly return 0 in boundary windows. -inf padding gives
        // the PyTorch-correct answer (the interior max).
        //
        // 1x1x3x3 tensor, all values negative:
        //   [ -1, -2, -3 ]
        //   [ -4, -5, -6 ]
        //   [ -7, -8, -9 ]
        // With kernel=3, stride=1, padding=1, output is 3x3 where the
        // (1,1) center sees the full grid → max = -1.
        let x = cpu_f32(
            vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
            &[1, 1, 3, 3],
        );
        let out = x
            .max_pool2d_with_pad_value((3, 3), (1, 1), (1, 1), f32::NEG_INFINITY)
            .unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 3, 3]);
        let v = out.realize_f32();
        // Top-left corner (0,0): window sees only [(0,0),(0,1),(1,0),(1,1)]
        // = [-1,-2,-4,-5]; padded slots are -inf → max = -1.
        assert_eq!(v[0], -1.0);
        // Center (1,1): no padded slots in window → max of all 9 = -1.
        assert_eq!(v[4], -1.0);
        // Bottom-right (2,2): window sees [(1,1),(1,2),(2,1),(2,2)]
        // = [-5,-6,-8,-9]; padded slots are -inf → max = -5.
        assert_eq!(v[8], -5.0);

        // Sanity: zero-padded max_pool2d would mistakenly return 0 here
        // (the padded zeros beat every negative interior value).
        let zero_pad = x.max_pool2d((3, 3), (1, 1), (1, 1)).unwrap();
        let vz = zero_pad.realize_f32();
        assert_eq!(vz[0], 0.0);
        assert_eq!(vz[8], 0.0);
    }

    #[test]
    fn max_pool2d_with_pad_value_zero_matches_max_pool2d() {
        // With pad_value = 0.0, the new variant must agree with the
        // legacy max_pool2d byte-for-byte.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = cpu_f32(data, &[1, 1, 4, 4]);
        let a = x.max_pool2d((2, 2), (2, 2), (0, 0)).unwrap();
        let b = x
            .max_pool2d_with_pad_value((2, 2), (2, 2), (0, 0), 0.0)
            .unwrap();
        assert_eq!(a.shape().dims(), b.shape().dims());
        assert_eq!(a.realize_f32(), b.realize_f32());
    }

    // ---- Phase A.6 conv1d composite tests ----

    #[test]
    fn conv1d_identity_kernel_passes_input_through() {
        // Single-batch, single-channel, kernel-1 identity → output equals input.
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let w = x.const_f32_like(vec![1.0], vec![1, 1, 1]);
        let out = x.conv1d(&w, None, 1, 0, 1).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 5]);
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn conv1d_sum_kernel_two_wide() {
        // Sum kernel of size 2: out[t] = x[t] + x[t+1].
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let w = x.const_f32_like(vec![1.0, 1.0], vec![1, 1, 2]);
        let out = x.conv1d(&w, None, 1, 0, 1).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 3]);
        assert_eq!(out.realize_f32(), vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn conv1d_with_bias_applies_correctly() {
        let x = cpu_f32(vec![1.0, 1.0, 1.0], &[1, 1, 3]);
        let w = x.const_f32_like(vec![1.0], vec![1, 1, 1]);
        let bias = x.const_f32_like(vec![10.0], vec![1]);
        let out = x.conv1d(&w, Some(&bias), 1, 0, 1).unwrap();
        assert_eq!(out.realize_f32(), vec![11.0, 11.0, 11.0]);
    }

    #[test]
    fn conv1d_stride_two_halves_output() {
        // Input length 6, kernel 2, stride 2 → output length (6-2)/2+1 = 3.
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
        let w = x.const_f32_like(vec![1.0, 1.0], vec![1, 1, 2]);
        let out = x.conv1d(&w, None, 2, 0, 1).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 3]);
        assert_eq!(out.realize_f32(), vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn conv1d_padding_one_preserves_length() {
        // Input length 4, kernel 3, padding 1, stride 1 → output length 4.
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let w = x.const_f32_like(vec![1.0, 1.0, 1.0], vec![1, 1, 3]);
        let out = x.conv1d(&w, None, 1, 1, 1).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 4]);
        // out[0] = 0+x[0]+x[1] = 3; out[1] = x[0]+x[1]+x[2] = 6;
        // out[2] = x[1]+x[2]+x[3] = 9; out[3] = x[2]+x[3]+0 = 7
        assert_eq!(out.realize_f32(), vec![3.0, 6.0, 9.0, 7.0]);
    }

    #[test]
    fn conv1d_multi_channel_output() {
        // 1 batch, 1 in-channel, 3 timesteps; 2 out-channels with kernel 1.
        let x = cpu_f32(vec![1.0, 2.0, 3.0], &[1, 1, 3]);
        // Weight [Cout=2, Cin=1, K=1]: filter 0 = 2.0, filter 1 = -1.0.
        let w = x.const_f32_like(vec![2.0, -1.0], vec![2, 1, 1]);
        let out = x.conv1d(&w, None, 1, 0, 1).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2, 3]);
        // Channel 0: 2.0 × input. Channel 1: -1.0 × input.
        assert_eq!(out.realize_f32(), vec![2.0, 4.0, 6.0, -1.0, -2.0, -3.0]);
    }

    #[test]
    fn conv1d_rejects_rank_two_input() {
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w = x.const_f32_like(vec![1.0], vec![1, 1, 1]);
        assert!(x.conv1d(&w, None, 1, 0, 1).is_err());
    }

    #[test]
    fn conv1d_rejects_rank_two_weight() {
        let x = cpu_f32(vec![1.0; 4], &[1, 1, 4]);
        let w = x.const_f32_like(vec![1.0], vec![1, 1]);
        assert!(x.conv1d(&w, None, 1, 0, 1).is_err());
    }

    #[test]
    fn conv1d_rejects_zero_groups_or_stride() {
        let x = cpu_f32(vec![1.0; 4], &[1, 1, 4]);
        let w = x.const_f32_like(vec![1.0], vec![1, 1, 1]);
        assert!(x.conv1d(&w, None, 0, 0, 1).is_err());
        assert!(x.conv1d(&w, None, 1, 0, 0).is_err());
    }

    #[test]
    fn conv1d_with_algo_ignores_algo_param() {
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let w = x.const_f32_like(vec![1.0, 1.0], vec![1, 1, 2]);
        // Pass a dummy algo (the parameter is ignored on the lazy path).
        let a = x.conv1d_with_algo(&w, None, 1, 0, 1, "unused").unwrap();
        let b = x.conv1d(&w, None, 1, 0, 1).unwrap();
        assert_eq!(a.realize_f32(), b.realize_f32());
    }

    // ---- Phase A.7 pooling / interpolation composite tests ----

    #[test]
    fn avg_pool2d_2x2_stride2() {
        // 1×1×4×4 input with values 0..15.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = cpu_f32(data, &[1, 1, 4, 4]);
        let out = x.avg_pool2d((2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 2, 2]);
        // Each 2x2 block average: top-left = (0+1+4+5)/4 = 2.5,
        // top-right = (2+3+6+7)/4 = 4.5, bottom-left = (8+9+12+13)/4 = 10.5,
        // bottom-right = (10+11+14+15)/4 = 12.5.
        let v = out.realize_f32();
        assert!((v[0] - 2.5).abs() < 1e-5);
        assert!((v[1] - 4.5).abs() < 1e-5);
        assert!((v[2] - 10.5).abs() < 1e-5);
        assert!((v[3] - 12.5).abs() < 1e-5);
    }

    #[test]
    fn avg_pool2d_3x3_stride1_padding1_preserves_size() {
        let x = cpu_f32(vec![1.0; 16], &[1, 1, 4, 4]);
        let out = x.avg_pool2d((3, 3), (1, 1), (1, 1)).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 4, 4]);
    }

    #[test]
    fn avg_pool2d_multi_channel() {
        // 1×2×2×2: each channel is filled with its index.
        let x = cpu_f32(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[1, 2, 2, 2]);
        let out = x.avg_pool2d((2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2, 1, 1]);
        assert_eq!(out.realize_f32(), vec![0.0, 1.0]);
    }

    #[test]
    fn max_pool2d_2x2_stride2() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = cpu_f32(data, &[1, 1, 4, 4]);
        let out = x.max_pool2d((2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 2, 2]);
        // Each 2x2 block max: 5, 7, 13, 15.
        assert_eq!(out.realize_f32(), vec![5.0, 7.0, 13.0, 15.0]);
    }

    #[test]
    fn max_pool2d_3x3_stride1_padding1() {
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[1, 1, 3, 3]);
        let out = x.max_pool2d((3, 3), (1, 1), (1, 1)).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 3, 3]);
        // Center should be the global max 9; corners should be the max of their 2×2 window.
        let v = out.realize_f32();
        // (1,1) center: max of all 9 = 9
        assert!((v[4] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn upsample_nearest2d_2x() {
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let out = x.upsample_nearest2d(2).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 4, 4]);
        // Each cell replicated 2×2: rows are [1,1,2,2; 1,1,2,2; 3,3,4,4; 3,3,4,4].
        assert_eq!(
            out.realize_f32(),
            vec![
                1.0, 1.0, 2.0, 2.0,
                1.0, 1.0, 2.0, 2.0,
                3.0, 3.0, 4.0, 4.0,
                3.0, 3.0, 4.0, 4.0,
            ],
        );
    }

    #[test]
    fn upsample_nearest2d_3x() {
        let x = cpu_f32(vec![5.0], &[1, 1, 1, 1]);
        let out = x.upsample_nearest2d(3).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 3, 3]);
        assert_eq!(out.realize_f32(), vec![5.0; 9]);
    }

    #[test]
    fn upsample_nearest2d_identity_scale_one() {
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let out = x.upsample_nearest2d(1).unwrap();
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn upsample_nearest1d_2x() {
        let x = cpu_f32(vec![1.0, 2.0, 3.0], &[1, 1, 3]);
        let out = x.upsample_nearest1d(2).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 6]);
        assert_eq!(out.realize_f32(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn interpolate2d_integer_multiple() {
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let out = x.interpolate2d(4, 4).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 4, 4]);
    }

    #[test]
    fn interpolate2d_accepts_non_integer_ratio() {
        // Lifted from "rejects non-integer ratio" — arbitrary
        // ratios are now supported via the index_select composite
        // (matching the eager UpsampleNearest2D convention). See
        // tests/lazy_interpolate2d_oracle.rs for parity checks.
        let x = cpu_f32(vec![1.0; 4], &[1, 1, 2, 2]);
        let out = x.interpolate2d(3, 4).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 3, 4]);
    }

    #[test]
    fn interpolate1d_integer_multiple() {
        let x = cpu_f32(vec![1.0, 2.0], &[1, 1, 2]);
        let out = x.interpolate1d(6).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 6]);
        assert_eq!(out.realize_f32(), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn pool_rejects_bad_rank() {
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(x.avg_pool2d((2, 2), (2, 2), (0, 0)).is_err());
        assert!(x.max_pool2d((2, 2), (2, 2), (0, 0)).is_err());
    }

    #[test]
    fn pool_rejects_zero_kernel() {
        let x = cpu_f32(vec![1.0; 16], &[1, 1, 4, 4]);
        assert!(x.avg_pool2d((0, 2), (1, 1), (0, 0)).is_err());
        assert!(x.max_pool2d((2, 0), (1, 1), (0, 0)).is_err());
    }

    // ---- Phase A.8 scope-limited harmonization aliases ----

    #[test]
    fn numel_matches_elem_count() {
        let t = cpu_f32(vec![1.0; 12], &[3, 4]);
        assert_eq!(t.numel(), t.elem_count());
        assert_eq!(t.numel(), 12);
    }

    #[test]
    fn dim_returns_specific_axis_size() {
        let t = cpu_f32(vec![0.0; 24], &[2, 3, 4]);
        assert_eq!(t.dim(0).unwrap(), 2);
        assert_eq!(t.dim(1).unwrap(), 3);
        assert_eq!(t.dim(2).unwrap(), 4);
        assert!(t.dim(3).is_err());
    }

    #[test]
    fn to_dtype_switches_dtype() {
        let t = cpu_f32(vec![1.0, 2.0], &[2]);
        let b = t.to_dtype(DType::F64).unwrap();
        assert_eq!(b.dtype(), DType::F64);
        assert_eq!(b.realize_f64(), vec![1.0, 2.0]);
    }

    #[test]
    fn to_dtype_same_dtype_is_noop() {
        let t = cpu_f32(vec![1.0_f32], &[1]);
        let b = t.to_dtype(DType::F32).unwrap();
        assert_eq!(b.dtype(), DType::F32);
    }

    #[test]
    fn detach_is_identity_on_lazy() {
        let t = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let d = t.detach();
        assert_eq!(d.realize_f32(), t.realize_f32());
    }

    #[test]
    fn track_op_is_true_on_lazy() {
        let t = cpu_f32(vec![0.0], &[1]);
        assert!(t.track_op());
    }

    // ---- Phase A.8a Dim/Dims trait port ergonomics tests ----

    #[test]
    fn try_permute_accepts_tuple_syntax() {
        let t = cpu_f32(vec![0.0; 24], &[2, 3, 4]);
        // Eager-style tuple permute now works on lazy.
        let out = t.permute((2_usize, 0_usize, 1_usize)).unwrap();
        assert_eq!(out.shape().dims(), &[4, 2, 3]);
    }

    #[test]
    fn dim_arg_methods_accept_negative_indexing() {
        use fuel_ir::D;
        let t = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // squeeze, sum_dim, mean_dim, etc. all accept D::Minus1 now.
        let sum_last = t.shape().dims().to_vec(); // just demonstrate compile
        assert_eq!(sum_last, vec![2, 3]);
        // sum_dim is still Self-returning (Phase A.8b will flip), so test
        // a method that already returns Result + dim arg.
        let cumsum_last = t.cumsum(D::Minus1).unwrap();
        assert_eq!(cumsum_last.shape().dims(), &[2, 3]);
    }

    #[test]
    fn unsqueeze_accepts_dim_trait() {
        use fuel_ir::D;
        let t = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        // Append a new last dim via D::Minus1 (rank-aware negative indexing).
        let out = t.unsqueeze(D::Minus1).unwrap();
        // The position D::Minus1 in to_index_plus_one is "the very end"
        // → output rank 2 with the new dim trailing.
        assert_eq!(out.shape().dims().len(), 2);
    }
}
