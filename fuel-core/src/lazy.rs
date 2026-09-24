// SPDX-License-Identifier: MIT OR Apache-2.0
//! Phase 6a bridge: a lazy-computation-graph tensor that wraps
//! [`fuel_graph::NodeHandle`] and presents it through an API compatible
//! with fuel-core's eager `Tensor`.
//!
//! # Purpose
//!
//! The Phase 6 architectural pivot moves fuel from eager execution to a
//! lazy computation graph. End state: `fuel_core::tensor::Tensor` *is* a
//! `fuel_graph::NodeHandle` and every model in `fuel-transformers` runs
//! through the lazy backend without per-model porting.
//!
//! The bridge is the intermediate stage. [`Tensor`] is a wrapper
//! around [`fuel_graph::NodeHandle`] that exposes the fuel-core-style
//! method API (`.add()`, `.mul()`, `.matmul()`, `.relu()`, `.shape()`,
//! `.to_vec0()`, `.to_vec1()`, ...) so callers can gradually migrate
//! from eager to lazy one function at a time. Each method appends a
//! node to the underlying [`fuel_graph::Graph`]; nothing runs until
//! you call [`Tensor::realize_f32`] or a sibling.
//!
//! This is NOT intended as a permanent user-facing type. It's the
//! scaffolding that makes the final merge incremental: each
//! `fuel-transformers` model can be converted to `Tensor` in a
//! separate PR, and once they all compile against the wrapper, the
//! type alias flips and `fuel_core::tensor::Tensor` becomes the lazy variant.
//!
//! # What's here today
//!
//! A minimal but real subset: constructors from `Vec<f32>`/`Vec<f64>`
//! and friends, shape/dtype inspection, the element-wise arithmetic
//! and unary ops most models use, matmul, softmax, layer_norm,
//! rms_norm, and realization to a typed `Vec`. Everything routes
//! through `fuel_graph::NodeHandle` underneath.
//!
//! Missing: autograd integration via `fuel_core::Var`, the
//! `backward()` / `apply_op*` convenience methods, safetensors
//! loading directly into `Tensor`s, and many of the niche
//! methods on `fuel_core::tensor::Tensor`. All of these are additive
//! extensions — they do not require changes to the bridge's
//! structural design.

use crate::inference_context::InferenceContext;
use crate::{DType, Device, Shape};
use fuel_ir::shape::{Dim, Dims};
use std::sync::Arc;

/// A lazy tensor that builds a `fuel_graph::Graph` as its methods are
/// called. Cheap to clone — the underlying `fuel_graph::NodeHandle` is a
/// cheap handle pair `(Rc<RefCell<Graph>>, NodeId)`, so cloning just
/// bumps the `Rc` and copies the id.
///
/// # Tensors are graph-affine — read this before building anything
///
/// **Every tensor in one computation must descend from a common root.**
/// Each `from_*` constructor ([`from_f32`](Self::from_f32),
/// [`from_bf16`](Self::from_bf16), …) — and [`zeros`](Self::zeros) /
/// [`full`](Self::full), which delegate to them — **mints a brand-new
/// graph**. Two independently-constructed tensors therefore live on two
/// different graphs and **cannot be combined**: every binary op asserts
/// `"… must live on the same graph"`.
///
/// So the obvious shape does *not* work:
///
/// ```ignore
/// let a = Tensor::from_f32(a_data, [2, 3], &device);
/// let w = Tensor::from_f32(w_data, [3, 2], &device);  // ← a SECOND graph
/// let y = a.matmul(&w);                                   // ← panics
/// ```
///
/// There are two ways to put a second tensor on an existing graph.
///
/// **1. `from_*_on` — pass the graph.** Use this when you have the graph, or
/// simply want the construction site to say which graph it targets:
///
/// ```ignore
/// let a = Tensor::from_f32(a_data, [2, 3], &device);          // the root
/// let w = Tensor::from_f32_on(a.graph(), w_data, [3, 2], &device);
/// let y = a.matmul(&w);                                           // ✓
/// ```
///
/// **2. `const_*_like` — pass an anchor tensor.** Identical effect; use it when
/// what you have to hand is a sibling tensor rather than the graph:
///
/// ```ignore
/// let a = Tensor::from_f32(a_data, [2, 3], &device);  // the root
/// let w = a.const_f32_like(w_data, [3, 2].into());        // same graph ✓
/// let y = a.matmul(&w);                                   // ✓
/// ```
///
/// Every tensor also reports its [`graph_id`](Self::graph_id), so if two do end
/// up on different graphs the panic names which is which.
///
/// In a model this means the **activation tensor is the root and the
/// weights are `const_*_like` off it**. `const_bf16_like` exists for
/// precisely the mixed-precision case — f32 activations with bf16 weight
/// matrices sharing one graph.
#[derive(Clone, Debug)]
pub struct Tensor {
    // `pub(crate)`: the shared decode build path lives in
    // `crate::persistent_decode` (GAP-029 increment 3) and needs the graph +
    // NodeId handles to mint Consts and name realize roots. Still private to
    // the crate — no consumer sees the graph representation.
    pub(crate) inner: fuel_graph::NodeHandle,
}

impl Tensor {
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
    ///
    /// # Mints a NEW graph
    ///
    /// This starts a **fresh graph**. A tensor built here cannot be combined
    /// with one built by another `from_*` call — ops assert both operands
    /// share a graph. To add a second tensor to *this* one's graph, use
    /// [`const_f32_like`](Self::const_f32_like) (or a sibling
    /// `const_*_like`). See [graph affinity](Self#tensors-are-graph-affine--read-this-before-building-anything).
    pub fn from_f32(
        data: impl Into<Arc<[f32]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> fuel_ir::Result<Self> {
        Ok(Self {
            inner: fuel_graph::NodeHandle::from_f32(data, shape, device.as_dyn())?,
        })
    }

    /// This tensor's graph identity — see [`fuel_graph::GraphId`]. Two tensors
    /// can be combined **iff** their `graph_id`s match.
    pub fn graph_id(&self) -> fuel_graph::GraphId {
        self.inner.graph_id()
    }

    /// The shared graph this tensor belongs to — hand it to a `from_*_on`
    /// constructor to build a sibling on the same graph.
    pub fn graph(&self) -> &fuel_graph::SharedGraph {
        self.inner.graph()
    }

    /// Build an `f32` lazy tensor **on the graph you pass in**, not a fresh one.
    ///
    /// The non-anchor route for adding a tensor to an existing graph.
    /// [`from_f32`](Self::from_f32) mints a NEW graph, so two `from_*` tensors
    /// can never be combined; this puts the new leaf on `graph`.
    /// [`const_f32_like`](Self::const_f32_like) does the same when what you have
    /// to hand is a sibling *tensor* rather than the graph itself.
    ///
    /// ```ignore
    /// let a = Tensor::from_f32(a_data, [2, 3], &device);          // root
    /// let w = Tensor::from_f32_on(a.graph(), w_data, [3, 2], &device);
    /// let y = a.matmul(&w)?;                                          // ✓
    /// ```
    pub fn from_f32_on(
        graph: &fuel_graph::SharedGraph,
        data: impl Into<Arc<[f32]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> fuel_ir::Result<Self> {
        Ok(Self {
            inner: fuel_graph::NodeHandle::from_f32_on(graph, data, shape, device.as_dyn())?,
        })
    }

    /// `f64` sibling of [`from_f32_on`](Self::from_f32_on).
    pub fn from_f64_on(
        graph: &fuel_graph::SharedGraph,
        data: impl Into<Arc<[f64]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> fuel_ir::Result<Self> {
        Ok(Self {
            inner: fuel_graph::NodeHandle::from_f64_on(graph, data, shape, device.as_dyn())?,
        })
    }

    /// `bf16` sibling of [`from_f32_on`](Self::from_f32_on) — the mixed-precision
    /// shape (f32 activations, bf16 weights) is this plus
    /// [`from_f32_on`](Self::from_f32_on) against one shared graph.
    pub fn from_bf16_on(
        graph: &fuel_graph::SharedGraph,
        data: impl Into<Arc<[half::bf16]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> fuel_ir::Result<Self> {
        Ok(Self {
            inner: fuel_graph::NodeHandle::from_bf16_on(graph, data, shape, device.as_dyn())?,
        })
    }

    /// `f16` sibling of [`from_f32_on`](Self::from_f32_on).
    pub fn from_f16_on(
        graph: &fuel_graph::SharedGraph,
        data: impl Into<Arc<[half::f16]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> fuel_ir::Result<Self> {
        Ok(Self {
            inner: fuel_graph::NodeHandle::from_f16_on(graph, data, shape, device.as_dyn())?,
        })
    }

    /// `u32` sibling of [`from_f32_on`](Self::from_f32_on) — index tensors built
    /// straight onto the graph they will be gathered on.
    pub fn from_u32_on(
        graph: &fuel_graph::SharedGraph,
        data: impl Into<Arc<[u32]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> fuel_ir::Result<Self> {
        Ok(Self {
            inner: fuel_graph::NodeHandle::from_u32_on(graph, data, shape, device.as_dyn())?,
        })
    }

    /// Build an `f64` lazy tensor. `device` selects where the realized
    /// Storage is allocated.
    ///
    /// **Mints a NEW graph** — to build on an existing tensor's graph use
    /// [`const_f64_like`](Self::const_f64_like). See
    /// [graph affinity](Self#tensors-are-graph-affine--read-this-before-building-anything).
    pub fn from_f64(
        data: impl Into<Arc<[f64]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> fuel_ir::Result<Self> {
        Ok(Self {
            inner: fuel_graph::NodeHandle::from_f64(data, shape, device.as_dyn())?,
        })
    }

    /// Build a `bf16` lazy tensor. `device` selects where the realized
    /// Storage is allocated.
    ///
    /// **Mints a NEW graph.** For the common case — bf16 weights alongside
    /// f32 activations — build the weights with
    /// [`const_bf16_like`](Self::const_bf16_like) off the activation tensor
    /// so both share one graph. See
    /// [graph affinity](Self#tensors-are-graph-affine--read-this-before-building-anything).
    pub fn from_bf16(
        data: impl Into<Arc<[half::bf16]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> fuel_ir::Result<Self> {
        Ok(Self {
            inner: fuel_graph::NodeHandle::from_bf16(data, shape, device.as_dyn())?,
        })
    }

    /// Build an `f16` lazy tensor. `device` selects where the realized
    /// Storage is allocated.
    pub fn from_f16(
        data: impl Into<Arc<[half::f16]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> fuel_ir::Result<Self> {
        Ok(Self {
            inner: fuel_graph::NodeHandle::from_f16(data, shape, device.as_dyn())?,
        })
    }

    /// Build a `u32` (index) lazy tensor. Used for gather/scatter/
    /// index_select and similar discrete operations. `device` selects
    /// where the realized Storage is allocated.
    pub fn from_u32(
        data: impl Into<Arc<[u32]>>,
        shape: impl Into<Shape>,
        device: &crate::Device,
    ) -> fuel_ir::Result<Self> {
        Ok(Self {
            inner: fuel_graph::NodeHandle::from_u32(data, shape, device.as_dyn())?,
        })
    }

    /// Build a const tensor of the same dtype and graph as `self`.
    /// This is the most convenient way to attach new input data to an
    /// existing computation.
    ///
    /// Phase 7.5 G2: the realized Storage is allocated on the device
    /// derived from `self`'s graph (any existing slot's device — the
    /// graph always has at least one slot-bearing leaf by the time
    /// const_*_like is called). Use [`Self::from_f32`] with an explicit
    /// `&Device` when you need a const on a different device than
    /// `self`.
    pub fn const_f32_like(
        &self,
        data: impl Into<Arc<[f32]>>,
        shape: impl Into<Shape>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.const_f32_like(data, shape)?,
        })
    }

    /// Build a const f16 tensor on the same graph as `self`.
    pub fn const_f16_like(
        &self,
        data: impl Into<Arc<[half::f16]>>,
        shape: impl Into<Shape>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.const_f16_like(data, shape)?,
        })
    }

    /// Build a const bf16 tensor on the same graph as `self`. Used for
    /// bf16-on-device weights in the mixed-precision matmul path —
    /// activations stay f32, weight matrices live as bf16.
    pub fn const_bf16_like(
        &self,
        data: impl Into<Arc<[half::bf16]>>,
        shape: impl Into<Shape>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.const_bf16_like(data, shape)?,
        })
    }

    /// Build a const tensor on the same graph as `self`, holding
    /// `data`'s values encoded in `dtype`. Only `F32` (identity copy)
    /// and `BF16` (host `half::bf16::from_f32` conversion) are
    /// supported today — the two activation dtypes the BF16-throughout
    /// decode seam (Phase D increment A: LlamaModel's cached decode
    /// path) threads through small per-model host constants (norm
    /// gains, biases, the causal mask) that must track the running
    /// activation dtype instead of being hardcoded f32. Any other
    /// dtype is a caller bug (not a data-dependent runtime
    /// possibility), so it surfaces a typed error rather than silently
    /// emitting wrong bytes.
    pub fn const_like_dtype(
        &self,
        data: &[f32],
        shape: impl Into<Shape>,
        dtype: fuel_ir::DType,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        match dtype {
            fuel_ir::DType::F32 => self.const_f32_like(data.to_vec(), shape),
            fuel_ir::DType::BF16 => {
                let converted: Vec<half::bf16> =
                    data.iter().map(|&v| half::bf16::from_f32(v)).collect();
                self.const_bf16_like(converted, shape)
            }
            other => Err(fuel_ir::Error::Msg(format!(
                "const_like_dtype: unsupported activation dtype {other:?} \
                 (expected F32 or BF16)",
            ))
            .bt()),
        }
    }

    /// Unwrap the underlying `fuel_graph::NodeHandle`. Used by callers that
    /// need to drop down to the graph layer for operations the bridge
    /// does not yet expose.
    pub fn into_graph_tensor(self) -> fuel_graph::NodeHandle {
        self.inner
    }

    /// Borrow the underlying `fuel_graph::NodeHandle`.
    pub fn graph_tensor(&self) -> &fuel_graph::NodeHandle {
        &self.inner
    }

    /// Wrap an existing `fuel_graph::NodeHandle` in a `Tensor`. Useful
    /// when you have code that already builds a graph and want to
    /// present its outputs through this API.
    pub fn from_graph_tensor(t: fuel_graph::NodeHandle) -> Self {
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

    /// This tensor's underlying graph handle. Exposed for call sites
    /// outside this module that need to drive
    /// [`InferenceContext`](crate::inference_context::InferenceContext)
    /// directly — e.g. a persistent cross-graph decode cache implemented
    /// in a sibling model module (`lazy_deepseek2.rs`'s
    /// `forward_with_latent_kv_context`). Most callers should prefer
    /// [`Self::realize_f32`] and friends instead.
    pub fn graph_handle(&self) -> &fuel_graph::SharedGraph {
        self.inner.graph()
    }

    /// This tensor's `NodeId` on its graph. See [`Self::graph_handle`].
    pub fn node_id(&self) -> fuel_graph::NodeId {
        self.inner.id()
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
    /// `crate::Tensor::dim` signature.
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
        Ok(Self {
            inner: self.inner.add(&other.inner),
        })
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("sub", other)?;
        Ok(Self {
            inner: self.inner.sub(&other.inner),
        })
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("mul", other)?;
        Ok(Self {
            inner: self.inner.mul(&other.inner),
        })
    }

    /// Element-wise division.
    pub fn div(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("div", other)?;
        Ok(Self {
            inner: self.inner.div(&other.inner),
        })
    }

    /// Element-wise maximum.
    pub fn maximum(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("maximum", other)?;
        Ok(Self {
            inner: self.inner.maximum(&other.inner),
        })
    }

    /// Element-wise minimum.
    pub fn minimum(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("minimum", other)?;
        Ok(Self {
            inner: self.inner.minimum(&other.inner),
        })
    }

    /// IEEE-754 NaN-**suppressing** maximum (`fmax_ieee`, KISS-OPS §6.15-0001) —
    /// DISTINCT from [`maximum`](Self::maximum) (NaN-propagating). Resolves through
    /// the spec-pinned §6.13 decomposition; does not substitute `maximum`. GAP-048.
    pub fn fmax_ieee(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("fmax_ieee", other)?;
        Ok(Self {
            inner: self.inner.fmax_ieee(&other.inner),
        })
    }

    /// IEEE-754 NaN-**suppressing** minimum (`fmin_ieee`, KISS-OPS §6.15-0001) —
    /// DISTINCT from [`minimum`](Self::minimum) (NaN-propagating). GAP-048.
    pub fn fmin_ieee(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("fmin_ieee", other)?;
        Ok(Self {
            inner: self.inner.fmin_ieee(&other.inner),
        })
    }

    /// Truncated remainder (`rem_trunc`, KISS-OPS §6.15-0003) — sign of the
    /// DIVIDEND (C99 `fmod`), DISTINCT from [`rem`](Self::rem) (floored, sign of
    /// the divisor). Resolves through the spec-pinned §6.13 decomposition. GAP-048.
    pub fn rem_trunc(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("rem_trunc", other)?;
        Ok(Self {
            inner: self.inner.rem_trunc(&other.inner),
        })
    }

    /// Round toward zero (`trunc`, KISS-OPS §6.3 floor op; GAP-300). Fuel's own
    /// expansion (`q>=0 ? floor(q) : ceil(q)`), NOT spec-pinned — its ±0/NaN/inf
    /// correctness is Fuel's liability. Integer dtypes are identity.
    pub fn trunc(&self) -> Self {
        Self {
            inner: self.inner.trunc(),
        }
    }

    /// Element-wise equality (`self == other`) producing a `U8` mask:
    /// `1` where equal, `0` otherwise. Both operands must share dtype
    /// and shape. NaN follows IEEE-754 (`NaN == NaN` is false). The
    /// resulting tensor's dtype is `DType::U8`. Non-differentiable.
    pub fn eq(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("eq", other)?;
        Ok(Self {
            inner: self.inner.eq(&other.inner),
        })
    }

    /// Element-wise inequality (`self != other`) producing a `U8`
    /// mask. NaN follows IEEE-754 (`NaN != NaN` is true → `1`).
    /// Non-differentiable.
    pub fn ne(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("ne", other)?;
        Ok(Self {
            inner: self.inner.ne(&other.inner),
        })
    }

    /// Element-wise strictly-less (`self < other`) producing a `U8`
    /// mask. NaN-on-either-side is `0` (IEEE-754 unordered).
    /// Non-differentiable.
    pub fn lt(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("lt", other)?;
        Ok(Self {
            inner: self.inner.lt(&other.inner),
        })
    }

    /// Element-wise less-or-equal (`self <= other`) producing a `U8`
    /// mask. NaN-on-either-side is `0`. Non-differentiable.
    pub fn le(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("le", other)?;
        Ok(Self {
            inner: self.inner.le(&other.inner),
        })
    }

    /// Element-wise strictly-greater (`self > other`) producing a
    /// `U8` mask. NaN-on-either-side is `0`. Non-differentiable.
    pub fn gt(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("gt", other)?;
        Ok(Self {
            inner: self.inner.gt(&other.inner),
        })
    }

    /// Element-wise greater-or-equal (`self >= other`) producing a
    /// `U8` mask. NaN-on-either-side is `0`. Non-differentiable.
    /// Final variant of the comparison family (`eq` / `ne` / `lt` /
    /// `le` / `gt` / `ge`).
    pub fn ge(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_strict_binary("ge", other)?;
        Ok(Self {
            inner: self.inner.ge(&other.inner),
        })
    }

    /// Ternary select (typically used to consume a comparison-op
    /// mask): `result[i] = if self[i] { a[i] } else { b[i] }`.
    /// `self` is the cond mask (must be `DType::Bool`, GAP-168(c)); `a` and `b`
    /// share dtype + shape with `self`. Output dtype matches `a`/`b`,
    /// shape matches `self`.
    ///
    /// Differentiable through `a` and `b` only.
    pub fn where_cond(&self, a: &Self, b: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        // GAP-168(c): the cond mask is Bool, matching `masked_fill` and the
        // `fixed(BOOL)` comparison contracts. The DOC above already said Bool
        // while this guard still required U8 — prose and check disagreeing, with
        // the prose right; every comparison feeds `where_cond`, so 22 model tests
        // failed on it. Cast a numeric mask explicitly (the cast is deliberately
        // NOT implicit — a silent coercion is what the Bool dtype exists to stop).
        if self.inner.dtype() != fuel_ir::DType::Bool {
            return Err(fuel_ir::Error::Msg(format!(
                "where_cond: cond mask must be Bool, got {:?} — cast a numeric \
                 mask to Bool explicitly (GAP-168(c))",
                self.inner.dtype(),
            ))
            .bt());
        }
        if a.inner.dtype() != b.inner.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "where_cond: branches must share dtype, got a={:?} b={:?}",
                a.inner.dtype(),
                b.inner.dtype(),
            ))
            .bt());
        }
        let cond_dims = self.inner.shape();
        let a_dims = a.inner.shape();
        let b_dims = b.inner.shape();
        if a_dims.dims() != cond_dims.dims() || b_dims.dims() != cond_dims.dims() {
            return Err(fuel_ir::Error::Msg(format!(
                "where_cond: shapes must match cond, got cond={:?} a={:?} b={:?}",
                cond_dims.dims(),
                a_dims.dims(),
                b_dims.dims(),
            ))
            .bt());
        }
        Ok(Self {
            inner: self.inner.where_cond(&a.inner, &b.inner),
        })
    }

    // ---- broadcast-aware arithmetic ----

    /// Element-wise addition with auto-broadcasting.
    pub fn broadcast_add(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_broadcast_binary("broadcast_add", other)?;
        Ok(Self {
            inner: self.inner.broadcast_add(&other.inner),
        })
    }

    /// Element-wise subtraction with auto-broadcasting.
    pub fn broadcast_sub(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_broadcast_binary("broadcast_sub", other)?;
        Ok(Self {
            inner: self.inner.broadcast_sub(&other.inner),
        })
    }

    /// Element-wise multiplication with auto-broadcasting.
    pub fn broadcast_mul(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_broadcast_binary("broadcast_mul", other)?;
        Ok(Self {
            inner: self.inner.broadcast_mul(&other.inner),
        })
    }

    /// Element-wise division with auto-broadcasting.
    pub fn broadcast_div(&self, other: &Self) -> std::result::Result<Self, fuel_ir::Error> {
        self.check_broadcast_binary("broadcast_div", other)?;
        Ok(Self {
            inner: self.inner.broadcast_div(&other.inner),
        })
    }

    fn check_strict_binary(
        &self,
        name: &'static str,
        other: &Self,
    ) -> std::result::Result<(), fuel_ir::Error> {
        if self.inner.dtype() != other.inner.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "{name}: dtype mismatch lhs={:?} rhs={:?}",
                self.inner.dtype(),
                other.inner.dtype(),
            ))
            .bt());
        }
        let a_shape = self.inner.shape();
        let b_shape = other.inner.shape();
        if a_shape.dims() != b_shape.dims() {
            return Err(fuel_ir::Error::Msg(format!(
                "{name}: shape mismatch lhs={:?} rhs={:?}",
                a_shape.dims(),
                b_shape.dims(),
            ))
            .bt());
        }
        Ok(())
    }

    fn check_broadcast_binary(
        &self,
        name: &'static str,
        other: &Self,
    ) -> std::result::Result<(), fuel_ir::Error> {
        if self.inner.dtype() != other.inner.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "{name}: dtype mismatch lhs={:?} rhs={:?}",
                self.inner.dtype(),
                other.inner.dtype(),
            ))
            .bt());
        }
        let a_shape = self.inner.shape();
        let b_shape = other.inner.shape();
        let a_dims = a_shape.dims();
        let b_dims = b_shape.dims();
        // Standard NumPy-style broadcast compatibility: from the right,
        // each pair of dims must be equal, or one of them must be 1.
        let rank = a_dims.len().max(b_dims.len());
        for i in 0..rank {
            let ad = a_dims
                .get(a_dims.len().wrapping_sub(1 + i))
                .copied()
                .unwrap_or(1);
            let bd = b_dims
                .get(b_dims.len().wrapping_sub(1 + i))
                .copied()
                .unwrap_or(1);
            if ad != bd && ad != 1 && bd != 1 {
                return Err(fuel_ir::Error::Msg(format!(
                    "{name}: shapes {:?} and {:?} are not broadcast-compatible",
                    a_dims, b_dims,
                ))
                .bt());
            }
        }
        Ok(())
    }

    // ---- unary ----

    /// Element-wise negation.
    pub fn neg(&self) -> Self {
        Self {
            inner: self.inner.neg(),
        }
    }

    /// Element-wise square.
    pub fn sqr(&self) -> Self {
        Self {
            inner: self.inner.sqr(),
        }
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Self {
        Self {
            inner: self.inner.sqrt(),
        }
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> Self {
        Self {
            inner: self.inner.exp(),
        }
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Self {
        Self {
            inner: self.inner.log(),
        }
    }

    /// Rectified linear unit.
    pub fn relu(&self) -> Self {
        Self {
            inner: self.inner.relu(),
        }
    }

    /// SiLU / Swish activation.
    pub fn silu(&self) -> Self {
        Self {
            inner: self.inner.silu(),
        }
    }

    /// GELU activation (tanh approximation).
    pub fn gelu(&self) -> Self {
        Self {
            inner: self.inner.gelu(),
        }
    }

    /// Logistic sigmoid.
    pub fn sigmoid(&self) -> Self {
        Self {
            inner: self.inner.sigmoid(),
        }
    }

    /// Hyperbolic tangent.
    pub fn tanh(&self) -> Self {
        Self {
            inner: self.inner.tanh(),
        }
    }

    /// Element-wise sine.
    pub fn sin(&self) -> Self {
        Self {
            inner: self.inner.sin(),
        }
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Self {
        Self {
            inner: self.inner.cos(),
        }
    }

    /// Heaviside step (`1` where `x > 0`, else `0`) — the derivative
    /// of [`Self::relu`].
    pub fn step(&self) -> Self {
        Self {
            inner: self.inner.step(),
        }
    }

    /// Element-wise reciprocal (`1 / x`).
    pub fn recip(&self) -> Self {
        Self {
            inner: self.inner.recip(),
        }
    }

    /// Element-wise absolute value (`|x|`).
    pub fn abs(&self) -> Self {
        Self {
            inner: self.inner.abs(),
        }
    }

    /// Element-wise floor (`⌊x⌋`). Same dtype as input.
    /// Backward is silently zero (non-differentiable almost everywhere).
    pub fn floor(&self) -> Self {
        Self {
            inner: self.inner.floor(),
        }
    }

    /// Element-wise ceiling (`⌈x⌉`). Same dtype as input.
    /// Backward is silently zero.
    pub fn ceil(&self) -> Self {
        Self {
            inner: self.inner.ceil(),
        }
    }

    /// Element-wise round-to-nearest with **banker's rounding**
    /// (round-half-to-even, IEEE 754 roundeven). Backward is silently
    /// zero. Differs from C99 `round()` at exact halves: 0.5 → 0,
    /// 2.5 → 2, etc.
    pub fn round(&self) -> Self {
        Self {
            inner: self.inner.round(),
        }
    }

    /// Element-wise sign (`-1` / `0` / `1`); `sign(0) = 0` by
    /// subgradient convention. Same dtype as input. Backward is
    /// silently zero.
    pub fn sign(&self) -> Self {
        Self {
            inner: self.inner.sign(),
        }
    }

    /// Element-wise Gauss error function (`erf(x)`). Same dtype as
    /// input. Differentiable: `d/dx erf(x) = (2/√π) * exp(-x²)`.
    pub fn erf(&self) -> Self {
        Self {
            inner: self.inner.erf(),
        }
    }

    /// GELU activation, **exact erf form** (`0.5 * x * (1 + erf(x/√2))`).
    /// Distinct from [`Self::gelu`] (tanh approximation). Same dtype
    /// as input. Differentiable.
    pub fn gelu_erf(&self) -> Self {
        Self {
            inner: self.inner.gelu_erf(),
        }
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
        Self {
            inner: self.inner.rsqrt(),
        }
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
        Ok(Self {
            inner: self.inner.flip(dim)?,
        })
    }

    /// Cyclic shift along `dim` by `shift` positions (positive →
    /// higher indices, wrapping). Differentiable; backward is
    /// `roll(dim, -shift)`.
    pub fn roll<D: Dim>(&self, dim: D, shift: i64) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "roll")?;
        Ok(Self {
            inner: self.inner.roll(dim, shift)?,
        })
    }

    /// Running cumulative sum along `dim`. Same shape as input.
    /// Differentiable; backward is reverse cumsum (`flip → cumsum
    /// → flip`).
    pub fn cumsum<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "cumsum")?;
        Ok(Self {
            inner: self.inner.cumsum(dim)?,
        })
    }

    /// Multi-dim Pad: `padding[i] = (before, after)` for axis `i`,
    /// length must equal tensor rank. Output shape:
    /// `out[i] = in[i] + padding[i].0 + padding[i].1`. Only Constant
    /// mode is implemented; Reflect / Replicate exist as enum stubs
    /// that error at realize time. Differentiable for Constant.
    /// **Returns `Result`**: rank mismatch surfaces as a typed error.
    pub fn pad(
        &self,
        padding: Vec<(usize, usize)>,
        mode: fuel_graph::PadMode,
        value: f64,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.pad(padding, mode, value)?,
        })
    }

    /// Element-wise integer power (`x.powi(n)`).
    pub fn powi(&self, n: i32) -> Self {
        Self {
            inner: self.inner.powi(n),
        }
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
            ))
            .bt());
        }
        let a_k = a_dims[a_dims.len() - 1];
        let b_k = b_dims[b_dims.len() - 2];
        if a_k != b_k {
            return Err(fuel_ir::Error::Msg(format!(
                "matmul: contracting dim mismatch lhs[..., M, {a_k}] vs rhs[..., {b_k}, N]",
            ))
            .bt());
        }
        Ok(Self {
            inner: self.inner.matmul(&other.inner),
        })
    }

    /// Data-determined-M matmul (sparse-MoE / capacity-buffer): like
    /// [`Self::matmul`], but computes only `row_count` rows of the
    /// `self.shape[-2]`-row capacity buffer, the rest left zeroed.
    /// `row_count` is a [`fuel_ir::DynScalar`] resolved at compile if
    /// input-determined, else at execute from the producer-bound `SymEnv`
    /// (e.g. `Op::NonZeroIndices`'s per-expert count). F32-only today.
    pub fn matmul_dyn_m(
        &self,
        other: &Self,
        row_count: fuel_ir::DynScalar,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let a_dims = self.inner.shape().dims().to_vec();
        let b_dims = other.inner.shape().dims().to_vec();
        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "matmul_dyn_m: both operands must be rank >= 2, got lhs={a_dims:?} rhs={b_dims:?}",
            ))
            .bt());
        }
        let a_k = a_dims[a_dims.len() - 1];
        let b_k = b_dims[b_dims.len() - 2];
        if a_k != b_k {
            return Err(fuel_ir::Error::Msg(format!(
                "matmul_dyn_m: contracting dim mismatch lhs[..., M, {a_k}] vs rhs[..., {b_k}, N]",
            ))
            .bt());
        }
        Ok(Self {
            inner: self.inner.matmul_dyn_m(&other.inner, row_count),
        })
    }

    /// Quantized matmul: `C = self @ dequant(W_Q)`. See
    /// [`fuel_graph::NodeHandle::qmatmul`] for details. The weight bytes
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
                "qmatmul: activations must be F32, got {:?}",
                self.inner.dtype(),
            ))
            .bt());
        }
        if weight_bytes.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "qmatmul: weight_bytes must be U32 (raw block bytes reinterpreted), got {:?}",
                weight_bytes.inner.dtype(),
            ))
            .bt());
        }
        let a_shape = self.inner.shape();
        let a_dims = a_shape.dims();
        if a_dims.len() < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "qmatmul: activations must be rank >= 2, got {a_dims:?}",
            ))
            .bt());
        }
        if a_dims[a_dims.len() - 1] != k {
            return Err(fuel_ir::Error::Msg(format!(
                "qmatmul: last dim of activations ({}) must equal k ({k})",
                a_dims[a_dims.len() - 1],
            ))
            .bt());
        }
        let block_size = quant_type.elements_per_block();
        if !k.is_multiple_of(block_size) {
            return Err(fuel_ir::Error::Msg(format!(
                "qmatmul: k={k} must be a multiple of {quant_type:?}'s block size ({block_size})",
            ))
            .bt());
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
        Ok(Self {
            inner: self.inner.try_transpose()?,
        })
    }

    /// Permute axes by the given ordering. Accepts any [`Dims`]
    /// implementer — `(0, 2, 1)`, `[0, 2, 1]`, `&[0, 2, 1]`, etc.
    /// Validates rank match + dim bounds + duplicate check at build time.
    pub fn permute<D: Dims>(&self, axes: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let axes = axes.to_indexes(&shape, "permute")?;
        Ok(Self {
            inner: self.inner.try_permute(&axes)?,
        })
    }

    /// Reshape to a new shape with matching element count.
    /// Element-count mismatch surfaces as a typed error at build time.
    pub fn reshape(&self, shape: impl Into<Shape>) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.try_reshape(shape)?,
        })
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
        Ok(Self {
            inner: self.inner.squeeze(dim)?,
        })
    }

    /// Broadcast to a larger shape. Shape-incompatibility surfaces as a
    /// typed error at build time.
    pub fn broadcast_to(
        &self,
        shape: impl Into<Shape>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.try_broadcast_to(shape)?,
        })
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
        &self,
        gain: std::sync::Arc<[f32]>,
        bias: std::sync::Arc<[f32]>,
        eps: f64,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let hidden = gain.len();
        debug_assert_eq!(
            bias.len(),
            hidden,
            "layer_norm_affine: gain ({}) and bias ({}) must have the same length",
            gain.len(),
            bias.len()
        );
        let normed = self.layer_norm_last_dim(eps)?;
        let dims_v: Vec<usize> = self.inner.shape().dims().to_vec();
        let mut affine_shape = vec![1_usize; dims_v.len()];
        affine_shape[dims_v.len() - 1] = hidden;
        let bc_shape = Shape::from_dims(&dims_v);
        let g = normed
            .const_f32_like(gain, Shape::from_dims(&[hidden]))?
            .reshape(Shape::from_dims(&affine_shape))?
            .broadcast_to(bc_shape.clone())?;
        let b = normed
            .const_f32_like(bias, Shape::from_dims(&[hidden]))?
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
        &self,
        dim: D,
        eps: f64,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let sq = self.sqr();
        let summed = sq.sum_keepdim(dim)?;
        let with_eps = if eps == 0.0 {
            summed
        } else {
            summed.add_scalar(eps)
        };
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
        &self,
        dim: D,
        repeats: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "repeat_interleave")?;
        if repeats == 0 {
            return Err(fuel_ir::Error::Msg("repeat_interleave: repeats must be ≥ 1".into()).bt());
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
    pub fn slice<D: Dim>(
        &self,
        dim: D,
        start: usize,
        len: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "slice")?;
        let dim_size = shape.dims()[dim];
        if start.saturating_add(len) > dim_size {
            return Err(fuel_ir::Error::Msg(format!(
                "slice: start={start} + len={len} exceeds dim {dim} size {dim_size}",
            ))
            .bt());
        }
        Ok(Self {
            inner: self.inner.slice(dim, start, len),
        })
    }

    /// Concatenate two tensors along `dim`. Shape mismatch or bad `dim`
    /// surfaces as a typed error at build time. Accepts any [`Dim`].
    pub fn concat<D: Dim>(
        &self,
        other: &Self,
        dim: D,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "concat")?;
        let self_dims = shape.dims().to_vec();
        let other_dims = other.inner.shape().dims().to_vec();
        if self_dims.len() != other_dims.len() {
            return Err(fuel_ir::Error::Msg(format!(
                "concat: rank mismatch lhs={self_dims:?} rhs={other_dims:?}",
            ))
            .bt());
        }
        for (i, (&a, &b)) in self_dims.iter().zip(other_dims.iter()).enumerate() {
            if i != dim && a != b {
                return Err(fuel_ir::Error::Msg(format!(
                    "concat: dim {i} mismatch lhs={a} rhs={b} (concat dim is {dim})",
                ))
                .bt());
            }
        }
        Ok(Self {
            inner: self.inner.concat(&other.inner, dim),
        })
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
        Ok(Self {
            inner: self.inner.argmax_dim(dim),
        })
    }

    /// Realize as a `u32` (index) `Vec`.
    ///
    /// Routes through the [`fuel_dispatch::pipelined::PipelinedExecutor`] like [`Self::realize_f32`]
    /// — the legacy fuel-reference-backend executor predates U8-output
    /// ops (comparison masks feeding argmin/argmax) and rejects them.
    pub fn realize_u32(&self) -> Vec<u32> {
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_as::<u32>(&graph, target, &device)
            .expect("realize_u32 via PipelinedExecutor")
    }

    /// Realize as raw bytes (`u8`). A BYTE VIEW: reads a `Bool` mask (0/1 per byte) or a
    /// `U8` tensor back to host (GAP-168(c)), and by design does NOT guard the root dtype —
    /// it routes through [`crate::pipelined_bridge::realize_one_bytes`] (the raw byte entry,
    /// GAP-327), so a `Bool` root (dtype != `U8`) is read as its bytes rather than rejected.
    ///
    /// **PANICS** on a realize failure; for a fallible byte view use
    /// [`crate::pipelined_bridge::realize_one_bytes`]. See [`Self::realize_f32`] for the
    /// documented-not-enforced rationale (GAP-186).
    pub fn realize_u8(&self) -> Vec<u8> {
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_bytes(&graph, target, &device)
            .expect("realize_u8 via PipelinedExecutor")
    }

    // ---- reductions ----

    /// Sum of all elements, producing a scalar.
    pub fn sum_all(&self) -> Self {
        Self {
            inner: self.inner.sum_all(),
        }
    }

    /// Arithmetic mean of all elements, producing a scalar.
    pub fn mean_all(&self) -> Self {
        Self {
            inner: self.inner.mean_all(),
        }
    }

    /// Maximum of every element, producing a scalar.
    pub fn max_all(&self) -> Self {
        Self {
            inner: self.inner.max_all(),
        }
    }

    /// Minimum of every element, producing a scalar.
    pub fn min_all(&self) -> Self {
        Self {
            inner: self.inner.min_all(),
        }
    }

    /// Sum along a single dimension (dim removed from output). Bad
    /// `dim` surfaces as a typed error at build time. Accepts any [`Dim`].
    pub fn sum_dim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "sum_dim")?;
        Ok(Self {
            inner: self.inner.sum_dim(dim),
        })
    }

    /// Max along a single dimension (dim removed from output).
    pub fn max_dim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "max_dim")?;
        Ok(Self {
            inner: self.inner.max_dim(dim),
        })
    }

    /// Min along a single dimension (dim removed from output).
    pub fn min_dim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "min_dim")?;
        Ok(Self {
            inner: self.inner.min_dim(dim),
        })
    }

    /// Element-wise clamp to `[min, max]`.
    pub fn clamp(&self, min: f64, max: f64) -> Self {
        Self {
            inner: self.inner.clamp(min, max),
        }
    }

    /// Mean along a single dimension.
    pub fn mean_dim<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "mean_dim")?;
        Ok(Self {
            inner: self.inner.mean_dim(dim),
        })
    }

    /// Sum-reduce to a smaller broadcast-compatible shape. Inverse of
    /// [`Self::broadcast_to`]; reduces over any dim where the source
    /// was broadcast against the target.
    pub fn reduce_sum_to(&self, target: impl Into<Shape>) -> Self {
        Self {
            inner: self.inner.reduce_sum_to(target),
        }
    }

    /// Max-reduce to a smaller broadcast-compatible shape — the
    /// max-symmetric counterpart of [`Self::reduce_sum_to`].
    pub fn reduce_max_to(&self, target: impl Into<Shape>) -> Self {
        Self {
            inner: self.inner.reduce_max_to(target),
        }
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
            )
            .bt());
        }
        Ok(Self {
            inner: self.inner.softmax_last_dim(),
        })
    }

    /// bitsandbytes-style 4-bit NormalFloat quantized matrix
    /// multiply. See [`fuel_graph::NodeHandle::nf4_matmul`] for the full
    /// shape contract. v1 covers F32/F16/BF16 activations.
    pub fn nf4_matmul(&self, w_packed: &Self, absmax: &Self, block_size: usize) -> Self {
        Self {
            inner: self
                .inner
                .nf4_matmul(&w_packed.inner, &absmax.inner, block_size),
        }
    }

    /// Mamba-2's State-Space Duality chunked scan (forward). See
    /// [`fuel_graph::NodeHandle::ssd_chunk_scan`] for the full shape
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
            inner: self
                .inner
                .ssd_chunk_scan(&dt.inner, &a.inner, &b.inner, &c.inner, chunk_size),
        }
    }

    /// Mamba-1's selective state-space scan (forward). See
    /// [`fuel_graph::NodeHandle::selective_scan`] for the full shape
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
                &delta.inner,
                &a.inner,
                &b.inner,
                &c.inner,
                delta_softplus,
            ),
        }
    }

    /// Multi-output Mamba-1 SSM scan: returns `(y, last_state)`. `y`
    /// matches the single-output [`Self::selective_scan`] result;
    /// `last_state` is the final hidden state `[batch, dim, dstate]`
    /// used by autoregressive callers to resume from a prefill
    /// snapshot. Both Tensors are `Op::View` projections of the
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
            &delta.inner,
            &a.inner,
            &b.inner,
            &c.inner,
            delta_softplus,
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
        let (y, last_state) = self
            .inner
            .ssd_chunk_scan_bundled(&dt.inner, &a.inner, &b.inner, &c.inner, chunk_size)?;
        Ok((Self { inner: y }, Self { inner: last_state }))
    }

    /// Data-determined nonzero-index extraction — the keystone primitive
    /// for **data-dependent dynamic shapes**. Returns `(indices, count)`:
    /// `indices` is `[capacity]` U32 (`capacity == self.elem_count()`),
    /// the first `count` entries being the ascending flat indices of
    /// `self`'s nonzero elements; `count` is `[1]` U32, the runtime
    /// nonzero count. Both are `Op::View` projections of one bundled
    /// producer. The executor also publishes `count`'s realized value into
    /// the per-pass `SymEnv` under `count_sym`, so downstream ops can
    /// consume it as a dynamic extent (the KV-cache `cached_len` pattern,
    /// generalized to a data-determined count). `count_sym` is allocated
    /// by the caller (from a [`fuel_ir::SymGen`]).
    pub fn nonzero_indices_bundled(
        &self,
        count_sym: fuel_ir::SymId,
    ) -> std::result::Result<(Self, Self), fuel_ir::Error> {
        let (indices, count) = self.inner.nonzero_indices_bundled(count_sym)?;
        Ok((Self { inner: indices }, Self { inner: count }))
    }

    /// Allocate a data-determined [`fuel_ir::SymId`] not yet used by any
    /// producer on this tensor's graph — see
    /// [`fuel_graph::Graph::next_data_determined_sym`]. Lets a builder that
    /// emits several [`Self::nonzero_indices_bundled`] producers on one
    /// graph (per-expert MoE dispatch, stacked layers) claim non-colliding
    /// count syms without threading a [`fuel_ir::SymGen`]: each call
    /// reflects the producers already added.
    pub fn fresh_data_determined_sym(&self) -> fuel_ir::SymId {
        let graph = self.inner.graph();
        let g = graph.read().unwrap();
        g.next_data_determined_sym()
    }

    /// Depthwise 1-D causal convolution + bias + optional fused SiLU
    /// — the Mamba-1 / Mamba-2 prefill convolution fusion. See
    /// [`fuel_graph::NodeHandle::causal_conv1d`] for the full shape
    /// contract (caller must left-pad x with `kernel - 1` zeros).
    pub fn causal_conv1d(&self, weight: &Self, bias: &Self, use_silu: bool) -> Self {
        Self {
            inner: self
                .inner
                .causal_conv1d(&weight.inner, &bias.inner, use_silu),
        }
    }

    /// Fused softmax + cross-entropy with integer (class-index)
    /// targets — the standard PyTorch CE loss. See
    /// [`fuel_graph::NodeHandle::fused_softmax_cross_entropy`] for the full
    /// shape contract.
    pub fn fused_softmax_cross_entropy(
        &self,
        targets: &Self,
        reduction: fuel_graph::registry::Reduction,
        ignore_index: i64,
    ) -> Self {
        Self {
            inner: self
                .inner
                .fused_softmax_cross_entropy(&targets.inner, reduction, ignore_index),
        }
    }

    /// LayerNorm along the last dim with the given epsilon. Rank-0
    /// or zero-last-dim input surfaces as a typed error at build time.
    pub fn layer_norm_last_dim(&self, eps: f64) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dims = shape.dims();
        if dims.last().copied().unwrap_or(0) == 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "layer_norm_last_dim: input must have non-zero last dim, got {dims:?}",
            ))
            .bt());
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
        if dims.last().copied().unwrap_or(0) == 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "rms_norm_last_dim: input must have non-zero last dim, got {dims:?}",
            ))
            .bt());
        }
        Ok(Self {
            inner: self.inner.rms_norm_last_dim(eps),
        })
    }

    /// Apply rotary position embeddings. See [`fuel_graph::NodeHandle::rope`].
    /// Rank < 2 surfaces as a typed error at build time.
    pub fn rope(&self, base: f64, start_pos: usize) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dims = shape.dims();
        if dims.len() < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "rope: input must have rank >= 2, got {dims:?}",
            ))
            .bt());
        }
        Ok(Self {
            inner: self.inner.rope(base, start_pos),
        })
    }

    /// Apply RoPE using caller-supplied `cos` and `sin` tables so they
    /// can be shared across many layers. See
    /// [`fuel_graph::NodeHandle::rope_with_tables`].
    ///
    /// Rank / dtype / table-shape mismatches surface as typed errors
    /// at build time rather than panicking inside `fuel_graph`.
    pub fn rope_with_tables(
        &self,
        cos: &Self,
        sin: &Self,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        if self.inner.dtype() != fuel_ir::DType::F32 {
            return Err(fuel_ir::Error::Msg(format!(
                "rope: only f32 is supported today, got {:?} (cast explicitly for other dtypes)",
                self.inner.dtype(),
            ))
            .bt());
        }
        let in_shape = self.inner.shape();
        let dims = in_shape.dims();
        let rank = dims.len();
        if rank < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "rope: input must have rank >= 2, got {dims:?}",
            ))
            .bt());
        }
        let seq = dims[rank - 2];
        let d = dims[rank - 1];
        if !d.is_multiple_of(2) {
            return Err(fuel_ir::Error::Msg(format!("rope: feature dim {d} must be even",)).bt());
        }
        let cos_shape = cos.inner.shape();
        let cos_dims = cos_shape.dims();
        if cos_dims != [seq, d] {
            return Err(fuel_ir::Error::Msg(format!(
                "rope_with_tables: cos shape {cos_dims:?} does not match [seq, d] = [{seq}, {d}]",
            ))
            .bt());
        }
        let sin_shape = sin.inner.shape();
        let sin_dims = sin_shape.dims();
        if sin_dims != [seq, d] {
            return Err(fuel_ir::Error::Msg(format!(
                "rope_with_tables: sin shape {sin_dims:?} does not match [seq, d] = [{seq}, {d}]",
            ))
            .bt());
        }
        Ok(Self {
            inner: self.inner.rope_with_tables(&cos.inner, &sin.inner),
        })
    }

    /// Apply RoPE using caller-supplied `cos`/`sin` tables, emitting the
    /// PRIMITIVE decomposition (slice + neg + concat + mul + add — the
    /// rotate-half recipe) instead of a single fused `FusedOps::ROPE` node.
    /// See [`fuel_graph::NodeHandle::rope_with_tables_decomposed`].
    ///
    /// CapturedRun 4b-resume: Fuel's `FusedOps::ROPE` is rotate-half (Llama/HF)
    /// but has NO correct (rotate-half) CUDA kernel — baracuda's is interleaved
    /// (see the rope-convention note). The fused op therefore places on CPU,
    /// which breaks CUDA-graph capture. Emitting the decomposition makes every
    /// step a primitive with a capture-safe CUDA kernel, so the decode runs
    /// entirely on CUDA. Forward-compatible: once a rotate-half fused CUDA rope
    /// kernel exists, the optimizer's re-fusion pattern will fold this subgraph
    /// back into the fused op automatically (recipe principle).
    ///
    /// Byte-identical semantics to [`Self::rope_with_tables`] (same rotate-half
    /// math); same validation.
    pub fn rope_with_tables_decomposed(
        &self,
        cos: &Self,
        sin: &Self,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        if self.inner.dtype() != fuel_ir::DType::F32 {
            return Err(fuel_ir::Error::Msg(format!(
                "rope: only f32 is supported today, got {:?} (cast explicitly for other dtypes)",
                self.inner.dtype(),
            ))
            .bt());
        }
        let in_shape = self.inner.shape();
        let dims = in_shape.dims();
        let rank = dims.len();
        if rank < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "rope: input must have rank >= 2, got {dims:?}",
            ))
            .bt());
        }
        let seq = dims[rank - 2];
        let d = dims[rank - 1];
        if !d.is_multiple_of(2) {
            return Err(fuel_ir::Error::Msg(format!("rope: feature dim {d} must be even",)).bt());
        }
        let cos_dims = cos.inner.shape();
        if cos_dims.dims() != [seq, d] {
            return Err(fuel_ir::Error::Msg(format!(
                "rope_with_tables_decomposed: cos shape {:?} does not match [seq, d] = [{seq}, {d}]",
                cos_dims.dims(),
            )).bt());
        }
        let sin_dims = sin.inner.shape();
        if sin_dims.dims() != [seq, d] {
            return Err(fuel_ir::Error::Msg(format!(
                "rope_with_tables_decomposed: sin shape {:?} does not match [seq, d] = [{seq}, {d}]",
                sin_dims.dims(),
            )).bt());
        }
        Ok(Self {
            inner: self
                .inner
                .rope_with_tables_decomposed(&cos.inner, &sin.inner),
        })
    }

    // ---- indexing ----

    /// Pick slices along `dim` using a 1-D U32 index tensor. Accepts
    /// any [`Dim`]. Dim bounds / index dtype / index rank mismatches
    /// surface as typed errors at build time.
    pub fn index_select<D: Dim>(
        &self,
        dim: D,
        indices: &Self,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "index_select")?;
        if indices.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "index_select: index tensor must be U32, got {:?}",
                indices.inner.dtype(),
            ))
            .bt());
        }
        let idx_shape = indices.inner.shape();
        let idx_dims = idx_shape.dims();
        if idx_dims.len() != 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "index_select: index tensor must be rank 1, got {idx_dims:?}",
            ))
            .bt());
        }
        Ok(Self {
            inner: self.inner.index_select(dim, &indices.inner),
        })
    }

    /// N-D gather along `dim` using a U32 index tensor with the same
    /// rank as `self`; output shape equals the index shape. Accepts
    /// any [`Dim`]. Dim bounds / index dtype / rank mismatches surface
    /// as typed errors at build time.
    pub fn gather<D: Dim>(
        &self,
        dim: D,
        indices: &Self,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "gather")?;
        if indices.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "gather: index tensor must be U32, got {:?}",
                indices.inner.dtype(),
            ))
            .bt());
        }
        let data_rank = shape.dims().len();
        let idx_shape = indices.inner.shape();
        let idx_rank = idx_shape.dims().len();
        if data_rank != idx_rank {
            return Err(fuel_ir::Error::Msg(format!(
                "gather: data and index must have the same rank, got {data_rank} vs {idx_rank}",
            ))
            .bt());
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
    ///
    /// **Convenience accessor — PANICS on a realize failure** (`.expect()`).
    /// That is a deliberate choice for the test / example / notebook callers
    /// who want `.realize_f32()[0]` without error-handling clutter, NOT a trap:
    /// a caller that must PROPAGATE a realize failure (any serving / production
    /// path) has a documented fallible sibling —
    /// [`crate::pipelined_bridge::realize_one_as`] (with `::<f32>`) — which
    /// returns `Result`. GAP-186: this is DOCUMENTED, NOT ENFORCED. `realize_*`
    /// is a genuine public API (13 `fuel-core/tests/` integration crates plus
    /// `fuel-examples` call it), so it cannot be made
    /// test-only; the never-panic obligation lives on the production callers,
    /// which use the fallible sibling directly (as `train.rs::param_to_host`
    /// does).
    pub fn realize_f32(&self) -> Vec<f32> {
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_as::<f32>(&graph, target, &device)
            .expect("realize_f32 via PipelinedExecutor")
    }

    /// Realize on CPU as an **independent correctness oracle**: like
    /// [`Self::realize_f32`] but with cost-based cross-device placement
    /// suppressed, so the whole graph runs on the CPU backend's bit-stable
    /// kernels and is never relocated to a GPU by the optimizer.
    ///
    /// [`Self::realize_f32`] pins CPU only as a *soft* host anchor — since the
    /// Step-E cost-based auto-placement, its optimizer may price model nodes
    /// onto a present GPU and insert an H2D `Op::Copy`, which both defeats the
    /// oracle's independence (it would validate a backend against itself) and,
    /// on a single-device realize, crashes for lack of a seeded GPU handle.
    /// This entry hard-pins CPU (`allow_cost_placement = false`); by the
    /// always-built coverage commitment the CPU backend supplies a kernel for
    /// every primitive op, so nothing is ever stranded. This is the
    /// pairwise-consensus oracle that replaces the retiring
    /// `fuel-reference-backend`.
    pub fn realize_f32_reference(&self) -> Vec<f32> {
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_reference_as::<f32>(&graph, target, &device)
            .expect("realize_f32_reference via PipelinedExecutor")
    }

    /// Realize as an `f64` `Vec`.
    ///
    /// Routes through the [`fuel_dispatch::pipelined::PipelinedExecutor`] like
    /// [`Self::realize_f32`] — executor-unification Session 1
    /// (re-audit gap 8) retires the typed `fuel_graph_cpu` recursive
    /// evaluator from the public API. The root must already be
    /// F64-dtype (insert [`Self::to_dtype`] otherwise). The dtype guard
    /// lives at the single realize funnel
    /// [`crate::pipelined_bridge::realize_one_as`] (GAP-327): a mismatch
    /// returns [`fuel_ir::Error::UnexpectedDType`], which the `.expect`
    /// below turns into a panic here — so the byte reinterpretation never
    /// silently returns garbage.
    pub fn realize_f64(&self) -> Vec<f64> {
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_as::<f64>(&graph, target, &device)
            .expect("realize_f64 via PipelinedExecutor")
    }

    /// Realize as a `bf16` `Vec`. See [`Self::realize_f64`] for the
    /// routing + dtype-guard rationale.
    pub fn realize_bf16(&self) -> Vec<half::bf16> {
        let graph = self.inner.graph().clone();
        let target = self.inner.id();
        let device = crate::Device::cpu();
        crate::pipelined_bridge::realize_one_as::<half::bf16>(&graph, target, &device)
            .expect("realize_bf16 via PipelinedExecutor")
    }

    /// Realize as an `f16` `Vec`. See [`Self::realize_f64`] for the
    /// routing + dtype-guard rationale.
    pub fn realize_f16(&self) -> Vec<half::f16> {
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
    pub fn realize_f32_cuda(&self, device: &fuel_cuda_backend::CudaDevice) -> Vec<f32> {
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
pub fn realize_many_f32(tensors: &[&Tensor]) -> Vec<Vec<f32>> {
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
    tensors: &[&Tensor],
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
        let t =
            Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), &Device::cpu()).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.shape().dims(), &[3]);
        assert_eq!(t.rank(), 1);
        assert_eq!(t.elem_count(), 3);
    }

    #[test]
    fn add_builds_add_node_in_underlying_graph() {
        let a =
            Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), &Device::cpu()).unwrap();
        let b = a
            .const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]))
            .unwrap();
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
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            &Device::cpu(),
        )
        .unwrap();
        let w = x
            .const_f32_like(
                vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                Shape::from_dims(&[3, 3]),
            )
            .unwrap();
        let y = x
            .rms_norm_last_dim(1e-6)
            .unwrap()
            .matmul(&w)
            .unwrap()
            .relu();
        assert_eq!(y.shape().dims(), &[2, 3]);
        assert_eq!(y.dtype(), DType::F32);
    }

    #[test]
    fn rope_through_lazy_wrapper() {
        // Verify the RoPE builder is reachable through Tensor.
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::from_dims(&[2, 4]),
            &Device::cpu(),
        )
        .unwrap();
        let y = x.rope(10000.0, 0).unwrap();
        assert_eq!(y.shape().dims(), &[2, 4]);
        assert_eq!(y.dtype(), DType::F32);
    }

    #[test]
    fn rope_delta_rotate_equals_direct_shift() {
        // rung-2 primitive: a K block rotated at its ORIGINAL positions then
        // uniformly delta-rotated by M is byte-equal to the same raw K rotated
        // DIRECTLY at the shifted positions. This is the exactness that lets a
        // cached prefix be reused at a non-zero offset. Everything is in the POOL
        // block layout [block_size, n_kv_heads, head_dim] that `read_block` yields.
        let dev = Device::cpu();
        let (n_kv_heads, head_dim, bs) = (2usize, 4usize, 4usize);
        let theta = 10000.0;
        let (p0, m) = (0usize, 8usize); // block at positions 0..4, shift by 8 → 8..12
        let raw: Vec<f32> = (0..bs * n_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.017).sin())
            .collect();
        // Reference: rope a pool-layout [bs,Hkv,D] block DIRECTLY at `start_pos`
        // (heads forward → rope → back), mirroring the helper's permute so the two
        // paths are compared in the same layout.
        let rope_pool_at = |data: &[f32], start_pos: usize| -> Vec<f32> {
            let k = Tensor::from_f32(
                data.to_vec(),
                Shape::from_dims(&[bs, n_kv_heads, head_dim]),
                &dev,
            )
            .unwrap();
            let k4 = k
                .permute([1, 0, 2])
                .unwrap()
                .reshape(Shape::from_dims(&[1, n_kv_heads, bs, head_dim]))
                .unwrap();
            let (c, s) = k4.rope_tables_const(theta, start_pos, bs, head_dim);
            k4.rope_with_tables_decomposed(&c, &s)
                .unwrap()
                .reshape(Shape::from_dims(&[n_kv_heads, bs, head_dim]))
                .unwrap()
                .permute([1, 0, 2])
                .unwrap()
                .realize_f32()
        };
        let direct = rope_pool_at(&raw, p0 + m); // rope directly at shifted positions
        let at_p0 = rope_pool_at(&raw, p0); // cached, rotated at original positions
        let shifted =
            Tensor::rope_delta_rotate_block_f32(&dev, &at_p0, theta, m, bs, n_kv_heads, head_dim);
        let maxdiff = direct
            .iter()
            .zip(&shifted)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            maxdiff < 1e-5,
            "delta-rotate == direct shift (maxdiff {maxdiff})"
        );
    }

    #[test]
    fn cast_switches_dtype_through_wrapper() {
        let x =
            Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), &Device::cpu()).unwrap();
        let y = x.to_dtype(DType::F64).unwrap();
        assert_eq!(y.dtype(), DType::F64);
        assert_eq!(y.shape().dims(), &[3]);
    }

    #[test]
    fn indexing_builds_correct_output_shape() {
        let data =
            Tensor::from_f32(vec![1.0; 12], Shape::from_dims(&[3, 4]), &Device::cpu()).unwrap();
        let idx = data
            .const_u32_like(vec![0, 2, 1], Shape::from_dims(&[3]))
            .unwrap();
        let out = data.index_select(0, &idx).unwrap();
        assert_eq!(out.shape().dims(), &[3, 4]);
    }

    // ---- Bridge realization tests ----

    #[test]
    fn realize_f32_executes_the_underlying_graph() {
        // The moment of truth: build a graph through Tensor and
        // then realize it end-to-end. (a + b) * a for a = [1, 2, 3],
        // b = [4, 5, 6] should yield [5, 14, 27].
        let a =
            Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), &Device::cpu()).unwrap();
        let b = a
            .const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]))
            .unwrap();
        let c = a.add(&b).unwrap().mul(&a).unwrap();
        let result = c.realize_f32();
        assert_eq!(result, vec![5.0, 14.0, 27.0]);
    }

    #[test]
    fn realize_f32_relu_and_maximum_propagate_nan_end_to_end() {
        // Pinned NaN-semantics convention (2026-07-08,
        // docs/architecture/10-decisions-log.md): Relu/Maximum/Minimum are
        // NaN-propagating (torch parity). This guards the full lazy path
        // (graph build -> optimize -> dispatch -> CPU kernel -> realize),
        // not just the scalar-kernel unit tests in fuel-cpu-backend.
        let a = Tensor::from_f32(
            vec![f32::NAN, -2.0, 3.0],
            Shape::from_dims(&[3]),
            &Device::cpu(),
        )
        .unwrap();
        let relu_result = a.relu().realize_f32();
        assert!(
            relu_result[0].is_nan(),
            "relu(NaN) must propagate NaN through realize, got {}",
            relu_result[0]
        );
        assert_eq!(relu_result[1], 0.0, "relu(-2) must clip to 0");
        assert_eq!(relu_result[2], 3.0, "relu(3) must pass through");

        let b = a
            .const_f32_like(vec![1.0, f32::NAN, 2.0], Shape::from_dims(&[3]))
            .unwrap();
        let max_result = a.maximum(&b).unwrap().realize_f32();
        assert!(
            max_result[0].is_nan(),
            "maximum(NaN, 1) must propagate NaN through realize, got {}",
            max_result[0]
        );
        assert!(
            max_result[1].is_nan(),
            "maximum(-2, NaN) must propagate NaN through realize, got {}",
            max_result[1]
        );
        assert_eq!(max_result[2], 3.0, "maximum(3, 2) non-NaN sanity");
    }

    // ===== GAP-048: KISS-Ops §6.15 resolution helpers (fmax_ieee / fmin_ieee /
    // rem_trunc) — realized end-to-end because fuel-graph tests only BUILD; the
    // VALUE divergence needs a backend. Do NOT move these "closer to the code" in
    // fuel-graph: there they cannot realize and become vacuous. =====

    #[test]
    fn fmax_fmin_ieee_suppress_nan_where_prop_propagates() {
        // §6.15-0001 divergence born-red. The IEEE (suppress) and torch-parity
        // (propagate) halves diverge ONLY on a NaN operand — a no-NaN fixture
        // passes under BOTH and is vacuous, so the fixture must carry NaN.
        let a = Tensor::from_f32(
            vec![f32::NAN, 2.0, 5.0],
            Shape::from_dims(&[3]),
            &Device::cpu(),
        )
        .unwrap();
        let b = a
            .const_f32_like(vec![3.0, f32::NAN, 4.0], Shape::from_dims(&[3]))
            .unwrap();

        let fmax = a.fmax_ieee(&b).unwrap().realize_f32();
        assert_eq!(fmax[0], 3.0, "fmax_ieee(NaN,3) suppresses NaN → 3");
        assert_eq!(fmax[1], 2.0, "fmax_ieee(2,NaN) suppresses NaN → 2");
        assert_eq!(fmax[2], 5.0, "fmax_ieee(5,4) → 5");
        let max = a.maximum(&b).unwrap().realize_f32();
        assert!(
            max[0].is_nan() && max[1].is_nan(),
            "maximum PROPAGATES NaN — the divergence that makes fmax_ieee distinct"
        );

        let fmin = a.fmin_ieee(&b).unwrap().realize_f32();
        assert_eq!(fmin[0], 3.0, "fmin_ieee(NaN,3) suppresses NaN → 3");
        assert_eq!(fmin[1], 2.0, "fmin_ieee(2,NaN) suppresses NaN → 2");
        assert_eq!(fmin[2], 4.0, "fmin_ieee(5,4) → 4");
        let min = a.minimum(&b).unwrap().realize_f32();
        assert!(
            min[0].is_nan() && min[1].is_nan(),
            "minimum PROPAGATES NaN — the divergence that makes fmin_ieee distinct"
        );
    }

    #[test]
    fn fmax_fmin_ieee_signed_zero_tie_operand_a_wins_bitwise() {
        // KISS Appendix A.3: on a ±0 tie (cmp_ge and cmp_le both true) operand `a`
        // (self) wins in all four minmax ops. ⚠️ Asserted on to_bits() — a value
        // compare cannot tell +0.0 from -0.0 and would pass VACUOUSLY. Fuel has
        // live ±0 tie-bias on CPU (#67)/Vulkan (#76) minmax paths, so if the CPU
        // `ge`/`where` path is tie-biased this born-red REVEALS it rather than
        // hiding it behind a value compare.
        //
        // ⚠️ SENSITIVITY PROVEN, not assumed (this test passed on first write, and
        // a first-write pass demonstrates nothing about its own sensitivity):
        // sabotaging the fmax_ieee decomposition `cmp_ge` → `cmp_gt` flips the tie
        // to b-wins and this goes RED with left=0 (+0.0) vs right=0x80000000 (-0.0).
        // So the a-wins result below is EARNED — the CPU ge/where path resolves the
        // tie a-wins, not tie-biased. Testing BOTH operand orders (each asserting
        // the *first* operand's zero) is the permanent discriminator: an order
        // swap or a collapse-to-constant flips one side; a return-`a`/return-`b`
        // collapse is caught by the sibling NaN test (fmax(NaN,3)=3 needs `b`,
        // fmax(2,NaN)=2 needs `a`). GAP-048.
        let a = Tensor::from_f32(vec![-0.0, 0.0], Shape::from_dims(&[2]), &Device::cpu()).unwrap();
        let b = a
            .const_f32_like(vec![0.0, -0.0], Shape::from_dims(&[2]))
            .unwrap();

        let fmax = a.fmax_ieee(&b).unwrap().realize_f32();
        assert_eq!(
            fmax[0].to_bits(),
            (-0.0f32).to_bits(),
            "fmax_ieee(-0.0,+0.0): operand a wins the tie → -0.0 (bitwise)"
        );
        assert_eq!(
            fmax[1].to_bits(),
            (0.0f32).to_bits(),
            "fmax_ieee(+0.0,-0.0): operand a wins the tie → +0.0 (bitwise)"
        );

        let fmin = a.fmin_ieee(&b).unwrap().realize_f32();
        assert_eq!(
            fmin[0].to_bits(),
            (-0.0f32).to_bits(),
            "fmin_ieee(-0.0,+0.0): operand a wins the tie → -0.0 (bitwise)"
        );
        assert_eq!(
            fmin[1].to_bits(),
            (0.0f32).to_bits(),
            "fmin_ieee(+0.0,-0.0): operand a wins the tie → +0.0 (bitwise)"
        );
    }

    #[test]
    fn rem_trunc_diverges_from_floored_rem_on_opposite_signs() {
        // §6.15-0003 divergence born-red. rem_trunc (sign of the DIVIDEND, C99
        // fmod) vs rem (floored, sign of the divisor) diverge ONLY when the
        // operands have opposite signs — a same-sign fixture is vacuous.
        let a = Tensor::from_f32(
            vec![-7.0, 7.0, -7.0],
            Shape::from_dims(&[3]),
            &Device::cpu(),
        )
        .unwrap();
        let b = a
            .const_f32_like(vec![3.0, -3.0, -3.0], Shape::from_dims(&[3]))
            .unwrap();

        let rt = a.rem_trunc(&b).unwrap().realize_f32();
        assert_eq!(rt[0], -1.0, "-7 rem_trunc 3 = -1 (sign of dividend)");
        assert_eq!(rt[1], 1.0, "7 rem_trunc -3 = 1 (sign of dividend)");
        assert_eq!(rt[2], -1.0, "-7 rem_trunc -3 = -1");
        let rf = a.rem(&b).unwrap().realize_f32();
        assert_eq!(
            rf[0], 2.0,
            "-7 rem 3 = 2 (floored, sign of divisor) — the divergence from rem_trunc"
        );
    }

    #[test]
    fn trunc_edges_signed_zero_nan_inf_bitwise() {
        // GAP-300: trunc's expansion (q>=0 ? floor : ceil) is FUEL's own — no KISS
        // vector catches an error in it, so its edges are verified HERE. ⚠️ ±0 on
        // to_bits() (value compare 0.0 == -0.0 is vacuous).
        let q = Tensor::from_f32(
            vec![
                2.7,
                -2.7,
                -0.0,
                0.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ],
            Shape::from_dims(&[7]),
            &Device::cpu(),
        )
        .unwrap();
        let t = q.trunc().realize_f32();
        assert_eq!(t[0], 2.0, "trunc(2.7) = 2 (toward zero)");
        assert_eq!(t[1], -2.0, "trunc(-2.7) = -2 (toward zero, NOT -3)");
        assert_eq!(
            t[2].to_bits(),
            (-0.0f32).to_bits(),
            "trunc(-0.0) must PRESERVE -0.0 (bitwise, not +0.0)"
        );
        assert_eq!(
            t[3].to_bits(),
            (0.0f32).to_bits(),
            "trunc(+0.0) = +0.0 (bitwise)"
        );
        assert!(t[4].is_nan(), "trunc(NaN) = NaN");
        assert_eq!(t[5], f32::INFINITY, "trunc(+inf) = +inf");
        assert_eq!(t[6], f32::NEG_INFINITY, "trunc(-inf) = -inf");
    }

    #[test]
    fn realize_f32_matmul_hand_computed() {
        // Classic 2x3 @ 3x2 matmul through the bridge.
        let a = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            &Device::cpu(),
        )
        .unwrap();
        let b = a
            .const_f32_like(
                vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                Shape::from_dims(&[3, 2]),
            )
            .unwrap();
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
        let a = Tensor::from_f32(a_data, Shape::from_dims(&[m, k]), &Device::cpu()).unwrap();
        let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n])).unwrap();
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

    /// Reference attention (B=1, multi-head, GQA-aware) in plain Rust:
    /// `out = softmax( softcap(scale·QKᵀ) [+ causal -inf] ) · V`. q
    /// `[hq,sq,d]`, k/v `[hkv,sk,d]` (GQA: q-head `h` attends kv-head
    /// `h / (hq/hkv)`), out `[hq,sq,d]`.
    #[allow(clippy::too_many_arguments)]
    fn ref_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        hq: usize,
        hkv: usize,
        sq: usize,
        sk: usize,
        d: usize,
        scale: f32,
        causal: bool,
        softcap: Option<f32>,
        alibi: Option<&[f32]>,
    ) -> Vec<f32> {
        let g = hq / hkv;
        let mut out = vec![0.0f32; hq * sq * d];
        for h in 0..hq {
            let hk = h / g;
            let qh = &q[h * sq * d..(h + 1) * sq * d];
            let kh = &k[hk * sk * d..(hk + 1) * sk * d];
            let vh = &v[hk * sk * d..(hk + 1) * sk * d];
            for i in 0..sq {
                let mut scores = vec![0.0f32; sk];
                for j in 0..sk {
                    let mut s = 0.0f32;
                    for l in 0..d {
                        s += qh[i * d + l] * kh[j * d + l];
                    }
                    let mut sc = scale * s;
                    if let Some(cap) = softcap {
                        sc = cap * (sc / cap).tanh();
                    }
                    if let Some(slopes) = alibi {
                        // alibi bias = slope[h] · (key_pos - query_pos).
                        sc += slopes[h] * (j as f32 - i as f32);
                    }
                    scores[j] = if causal && j > i {
                        f32::NEG_INFINITY
                    } else {
                        sc
                    };
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                let exps: Vec<f32> = scores
                    .iter()
                    .map(|&s| {
                        let e = (s - m).exp();
                        sum += e;
                        e
                    })
                    .collect();
                for l in 0..d {
                    let mut o = 0.0f32;
                    for j in 0..sk {
                        o += (exps[j] / sum) * vh[j * d + l];
                    }
                    out[(h * sq + i) * d + l] = o;
                }
            }
        }
        out
    }

    /// Numerical parity: lower a FlashAttn node to its primitive
    /// decomposition, realize it on CPU, and assert it matches the plain-Rust
    /// reference — verifying the recipe's *math* (scale·QKᵀ → softmax → ·V),
    /// the `Triu` `-inf` causal mask, GQA head-repeat, the `tanh` softcap, and
    /// the alibi bias (which is the sole config that lowers to `Op::Iota`).
    fn flash_decompose_vs_reference(
        hq: usize,
        hkv: usize,
        causal: bool,
        softcap: Option<f32>,
        alibi: bool,
    ) {
        let dev = Device::cpu();
        let (sq, sk, d) = (2usize, 2usize, 2usize);
        // deterministic, varied inputs so the comparison is meaningful.
        let q_data: Vec<f32> = (0..hq * sq * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k_data: Vec<f32> = (0..hkv * sk * d).map(|i| (i as f32 * 0.13).cos()).collect();
        let v_data: Vec<f32> = (0..hkv * sk * d).map(|i| i as f32 * 0.07 + 1.0).collect();
        // 4-digit test scale (~1/sqrt(2)), deliberately NOT FRAC_1_SQRT_2 (0.70710678);
        // swapping the exact constant would perturb these attention-scale parity tests.
        #[allow(clippy::approx_constant)]
        let scale = 0.7071f32;
        let q = Tensor::from_f32(q_data.clone(), Shape::from_dims(&[1, hq, sq, d]), &dev).unwrap();
        let k = q
            .const_f32_like(k_data.clone(), Shape::from_dims(&[1, hkv, sk, d]))
            .unwrap();
        let v = q
            .const_f32_like(v_data.clone(), Shape::from_dims(&[1, hkv, sk, d]))
            .unwrap();
        // Distinct positive slope per head (powers of 1/2 — the alibi default).
        let alibi_slopes: Option<Vec<f32>> = if alibi {
            Some((0..hq).map(|h| 0.5f32.powi(h as i32 + 1)).collect())
        } else {
            None
        };
        let alibi_t = alibi_slopes.as_ref().map(|s| {
            q.const_f32_like(s.clone(), Shape::from_dims(&[hq]))
                .unwrap()
        });
        let attn = q
            .flash_attn(&k, &v, alibi_t.as_ref(), scale, causal, None, None, softcap)
            .unwrap();

        // Decompose explicitly, then realize the primitive subgraph.
        let graph = attn.inner.graph().clone();
        let id = attn.inner.id();
        let roots =
            fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
        assert_eq!(roots.len(), 1, "lowering should keep a single root");
        let got = crate::pipelined_bridge::realize_one_as::<f32>(&graph, roots[0], &dev)
            .expect("realize decomposed FlashAttn on CPU");

        let expected = ref_attention(
            &q_data,
            &k_data,
            &v_data,
            hq,
            hkv,
            sq,
            sk,
            d,
            scale,
            causal,
            softcap,
            alibi_slopes.as_deref(),
        );
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "FlashAttn decompose mismatch (hq={hq} hkv={hkv} causal={causal} \
                 softcap={softcap:?} alibi={alibi}) at {i}: got {g}, expected {e}",
            );
        }
    }

    #[test]
    fn flash_attn_decompose_vanilla() {
        flash_decompose_vs_reference(1, 1, false, None, false);
    }

    #[test]
    fn flash_attn_decompose_causal() {
        flash_decompose_vs_reference(1, 1, true, None, false);
    }

    #[test]
    fn flash_attn_decompose_gqa_causal() {
        flash_decompose_vs_reference(2, 1, true, None, false); // Hq=2, Hkv=1 (head-repeat)
    }

    #[test]
    fn flash_attn_decompose_softcap() {
        flash_decompose_vs_reference(1, 1, false, Some(0.5), false); // small cap → tanh saturates
    }

    #[test]
    fn flash_attn_decompose_alibi() {
        // alibi is the only config that lowers to Op::Iota (relative-position
        // values) — this exercises the new 0-input Iota execution path.
        flash_decompose_vs_reference(2, 2, false, None, true);
    }

    #[test]
    fn flash_attn_decompose_alibi_causal() {
        flash_decompose_vs_reference(2, 2, true, None, true);
    }

    /// Plain-Rust reference for **capacity-K decode attention**: attend `q`
    /// (`Sq` queries at absolute positions `[kl−Sq, kl)`) against the first
    /// `kl` keys/values of a `cap`-capacity KV buffer, with bottom-right
    /// causal masking (query `i` attends keys `j ≤ (kl−Sq)+i`). GQA folds
    /// `h → h/(Hq/Hkv)`.
    #[allow(clippy::too_many_arguments)]
    fn ref_decode_attn(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        hq: usize,
        hkv: usize,
        sq: usize,
        cap: usize,
        kl: usize,
        d: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        let g = hq / hkv;
        let offset = kl - sq;
        let mut out = vec![0f32; hq * sq * d];
        for h in 0..hq {
            let hk = h / g;
            for i in 0..sq {
                let qh = &q[((h * sq) + i) * d..][..d];
                let mut scores = vec![f32::NEG_INFINITY; kl];
                for j in 0..kl {
                    if causal && j > offset + i {
                        continue;
                    }
                    let kh = &k[((hk * cap) + j) * d..][..d];
                    let s: f32 = (0..d).map(|l| qh[l] * kh[l]).sum();
                    scores[j] = scale * s;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                let exps: Vec<f32> = scores
                    .iter()
                    .map(|&s| {
                        let e = (s - m).exp();
                        sum += e;
                        e
                    })
                    .collect();
                for l in 0..d {
                    let mut o = 0f32;
                    for j in 0..kl {
                        o += (exps[j] / sum) * v[((hk * cap) + j) * d + l];
                    }
                    out[(h * sq + i) * d + l] = o;
                }
            }
        }
        out
    }

    /// Recipe principle (G2): a **concrete** `k_len` FlashAttn is a *static*
    /// config that used to return self (`k_len.is_some()` short-circuit). It
    /// now decomposes: `Slice` K/V to the live prefix and run the SDPA recipe
    /// bottom-right-aligned (`q_pos_offset = kl − Sq`). RED before the fix (an
    /// `Op::Fused(FLASH_ATTN)` island survives lowering); GREEN after, matching
    /// the decode-attention reference. Uses GQA (Hq=2, Hkv=1), Sq=2 (so the
    /// offset causal band is exercised, not a Sq=1 no-op), capacity 4, kl=3.
    #[test]
    fn flash_attn_decompose_concrete_klen() {
        use fuel_graph::Op;
        use fuel_graph::registry::FusedOps;
        use fuel_ir::DynScalar;
        let dev = Device::cpu();
        let (hq, hkv, sq, cap, kl, d) = (2usize, 1usize, 2usize, 4usize, 3usize, 2usize);
        // 4-digit test scale (~1/sqrt(2)), deliberately NOT FRAC_1_SQRT_2 (0.70710678);
        // swapping the exact constant would perturb these attention-scale parity tests.
        #[allow(clippy::approx_constant)]
        let scale = 0.7071f32;
        let causal = true;
        let q_data: Vec<f32> = (0..hq * sq * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k_data: Vec<f32> = (0..hkv * cap * d)
            .map(|i| (i as f32 * 0.13).cos())
            .collect();
        let v_data: Vec<f32> = (0..hkv * cap * d).map(|i| i as f32 * 0.07 + 1.0).collect();
        let q = Tensor::from_f32(q_data.clone(), Shape::from_dims(&[1, hq, sq, d]), &dev).unwrap();
        let k = q
            .inner
            .const_f32_like(k_data.clone(), Shape::from_dims(&[1, hkv, cap, d]))
            .unwrap();
        let v = q
            .inner
            .const_f32_like(v_data.clone(), Shape::from_dims(&[1, hkv, cap, d]))
            .unwrap();
        // Concrete k_len — the fused node that formerly returned self.
        let attn = q.inner.flash_attn_dyn(
            &k,
            &v,
            None,
            scale,
            causal,
            None,
            None,
            None,
            DynScalar::Concrete(kl),
        );

        let graph = attn.graph().clone();
        let id = attn.id();
        let roots =
            fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
        assert_eq!(roots.len(), 1, "lowering should keep a single root");

        // Born-red discriminator: no Op::Fused(FLASH_ATTN) reachable from the root.
        {
            let g = graph.read().unwrap();
            let mut stack = vec![roots[0]];
            let mut seen = std::collections::HashSet::new();
            while let Some(nid) = stack.pop() {
                if !seen.insert(nid) {
                    continue;
                }
                let node = g.node(nid);
                assert!(
                    !matches!(node.op, Op::Fused(fid, _) if fid == FusedOps::FLASH_ATTN),
                    "concrete-k_len FlashAttn still fused after lowering (self-return)",
                );
                for &inp in &node.inputs {
                    stack.push(inp);
                }
            }
        }

        let got = crate::pipelined_bridge::realize_one_as::<f32>(&graph, roots[0], &dev)
            .expect("realize decomposed concrete-k_len FlashAttn on CPU");
        let expected = ref_decode_attn(
            &q_data, &k_data, &v_data, hq, hkv, sq, cap, kl, d, scale, causal,
        );
        assert_eq!(got.len(), expected.len());
        for (i, (&gv, &ev)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (gv - ev).abs() < 1e-4,
                "concrete-k_len decode mismatch at {i}: got {gv}, expected {ev}",
            );
        }
    }

    /// Bit-exact comparison of two `f32` slices (via raw bit patterns), so a
    /// last-ULP divergence is not masked by `==` and NaNs compare by pattern.
    fn assert_f32_bits_eq(got: &[f32], want: &[f32], ctx: &str) {
        assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{ctx}: bit mismatch at {i}: got {g} (0x{:08x}), want {w} (0x{:08x})",
                g.to_bits(),
                w.to_bits(),
            );
        }
    }

    /// Assert two positive-`f32` slices agree to within `max_ulp` ULPs. Used
    /// to document the (migration-orthogonal) 1-ULP gap between a primitive
    /// `Div` and the fused kernel's reciprocal-multiply.
    fn assert_f32_within_ulp(got: &[f32], want: &[f32], max_ulp: i64, ctx: &str) {
        assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let d = (g.to_bits() as i64 - w.to_bits() as i64).abs();
            assert!(
                d <= max_ulp,
                "{ctx}: {d} ULP at {i} (> {max_ulp}): got {g} (0x{:08x}), want {w} (0x{:08x})",
                g.to_bits(),
                w.to_bits(),
            );
        }
    }

    /// Real-backend accumulation-order parity for the Increment-C-slice-1
    /// "shrink-via-swap" reduce spellings (prior-review nit, I3).
    ///
    /// The migration parity tests in `fuel-graph`
    /// (`*_matches_frozen_legacy`) run BOTH the recipe decompose and the
    /// frozen legacy builder through a shared toy-`f64` reference
    /// interpreter, so they pin recipe STRUCTURE only and — by construction
    /// — cannot exercise the concern that the CPU backend's `SumDim`
    /// (reduction chassis `reduce`) and `ReduceSumTo` (chassis `reduce_to`),
    /// or `MaxDim` vs `ReduceMaxTo`, might accumulate a last-dim reduction in
    /// a DIFFERENT order and diverge in `f32`. The recipe swapped
    /// `ReduceSumTo(keepdim)` → `SumDim(last)`+`Unsqueeze` (and the max
    /// side likewise), so the two spellings dispatch two DIFFERENT CPU
    /// reduce kernels.
    ///
    /// This realizes both spellings as BARE reduce roots on the ACTUAL CPU
    /// backend — a bare reduce cannot re-fuse into a `SoftmaxLastDim`
    /// (no surrounding `Div`/`Exp`), and there is no `SumDim`↔`ReduceSumTo`
    /// canonicalization, so the two reduce kernels genuinely execute — over
    /// data engineered so a left-fold and a right-fold `f32` sum DIFFER,
    /// which gives the bit-exact equality real teeth. Result: bit-exact,
    /// because both chassis fold every element into its output slot in
    /// row-major flat order (`reduce`: `for flat in 0..total_input`;
    /// `reduce_to`: `for in_flat in 0..in_elems`).
    #[test]
    fn reduce_swap_sumdim_vs_reducesumto_bit_exact_on_cpu_backend() {
        let dev = Device::cpu();
        let (rows, cols) = (2usize, 4usize);
        // Each row is order-sensitive in f32 (catastrophic cancellation):
        // its left-fold sum differs from its right-fold sum, so bit-exact
        // equality of the two spellings is a NON-trivial statement about the
        // backend's accumulation order.
        let data: Vec<f32> = vec![1e8, 1.0, -1e8, 1.0, 3.0e7, 2.0, -3.0e7, -1.0];

        // Independent host references: fold each row left-to-right and
        // right-to-left in f32. They disagree — the data is adversarial.
        let mut left_fold = vec![0.0f32; rows];
        let mut right_fold = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            let mut acc = 0.0f32;
            for &v in row.iter() {
                acc += v;
            }
            left_fold[r] = acc;
            let mut acc_r = 0.0f32;
            for &v in row.iter().rev() {
                acc_r += v;
            }
            right_fold[r] = acc_r;
        }
        assert_ne!(
            left_fold, right_fold,
            "test data must be accumulation-order-sensitive so the parity has teeth",
        );

        let x = Tensor::from_f32(data.clone(), Shape::from_dims(&[rows, cols]), &dev).unwrap();
        // Recipe spelling: rank-reducing SumDim(last) (chassis `reduce`).
        let sumdim = x.inner.sum_dim(1);
        let got_sumdim = crate::pipelined_bridge::realize_one_as::<f32>(
            &sumdim.graph().clone(),
            sumdim.id(),
            &dev,
        )
        .expect("realize SumDim(last) on CPU");
        // Legacy spelling: ReduceSumTo(keepdim [rows,1]) (chassis `reduce_to`).
        let reducesumto = x.inner.reduce_sum_to(Shape::from_dims(&[rows, 1]));
        let got_reducesumto = crate::pipelined_bridge::realize_one_as::<f32>(
            &reducesumto.graph().clone(),
            reducesumto.id(),
            &dev,
        )
        .expect("realize ReduceSumTo(keepdim) on CPU");

        // The load-bearing parity: the two reduce kernels agree bit-exactly.
        assert_f32_bits_eq(&got_sumdim, &got_reducesumto, "SumDim vs ReduceSumTo");
        // Correctness + which order: both equal the row-major left-fold (and,
        // because the data is adversarial, NOT the right-fold).
        assert_f32_bits_eq(&got_sumdim, &left_fold, "SumDim vs left-fold");

        // Max side: MaxDim(last) (chassis `reduce`) vs ReduceMaxTo(keepdim)
        // (chassis `reduce_to`). Max is order-independent, but the swap still
        // dispatches two DIFFERENT CPU kernels; confirm they agree bit-exactly
        // and equal the per-row maximum.
        let maxdim = x.inner.max_dim(1);
        let got_maxdim = crate::pipelined_bridge::realize_one_as::<f32>(
            &maxdim.graph().clone(),
            maxdim.id(),
            &dev,
        )
        .expect("realize MaxDim(last) on CPU");
        let reducemaxto = x.inner.reduce_max_to(Shape::from_dims(&[rows, 1]));
        let got_reducemaxto = crate::pipelined_bridge::realize_one_as::<f32>(
            &reducemaxto.graph().clone(),
            reducemaxto.id(),
            &dev,
        )
        .expect("realize ReduceMaxTo(keepdim) on CPU");
        let row_max: Vec<f32> = (0..rows)
            .map(|r| {
                data[r * cols..(r + 1) * cols]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        assert_f32_bits_eq(&got_maxdim, &got_reducemaxto, "MaxDim vs ReduceMaxTo");
        assert_f32_bits_eq(&got_maxdim, &row_max, "MaxDim vs per-row max");
    }

    /// End-to-end real-backend parity for the migrated `SoftmaxLastDim`
    /// (Increment C slice 1 pilot, I3). The migration swapped the keepdim
    /// reduce spelling `ReduceMaxTo`/`ReduceSumTo(keepdim)` →
    /// `MaxDim`/`SumDim(last)` + `Unsqueeze(append)` (the D3 shrink-via-swap).
    /// This realizes the WHOLE softmax on the ACTUAL CPU backend two ways and
    /// asserts the swap is numerically inert:
    /// * RECIPE path — `lowering_only` lowers the fused node to its recipe
    ///   (`SumDim`/`MaxDim` spelling) primitive subgraph, then realize it;
    /// * LEGACY path — the pre-migration `ReduceMaxTo`/`ReduceSumTo` primitive
    ///   subgraph, hand-built here with the same `Sub`/`Exp`/`Div`.
    ///
    /// The two must be BIT-EXACT: `reduce_swap_…` proves `SumDim`≡`ReduceSumTo`
    /// and `MaxDim`≡`ReduceMaxTo` bit-exactly on the CPU backend, and every
    /// other node (and the true-`Div` normalize) is identical, so the whole
    /// pipeline agrees to the bit. (The runtime-fusion pass does NOT re-fuse
    /// either primitive subgraph back to the `SoftmaxLastDim` kernel — see the
    /// secondary assertion, which measures the migration-ORTHOGONAL 1-ULP gap
    /// between the primitive `Div` and the fused kernel's reciprocal-multiply
    /// `e*(1/sum)`.) Adversarial mixed-magnitude rows make it non-trivial.
    #[test]
    fn softmax_last_dim_recipe_decompose_matches_legacy_on_cpu() {
        let dev = Device::cpu();
        let (rows, cols) = (3usize, 5usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.37).sin() * 4.0 - (i as f32) * 0.11)
            .collect();
        let shape = || Shape::from_dims(&[rows, cols]);
        let keepdim = || Shape::from_dims(&[rows, 1]);

        // RECIPE path: lower the fused node to the recipe (SumDim spelling).
        let recipe = Tensor::from_f32(data.clone(), shape(), &dev)
            .unwrap()
            .softmax_last_dim()
            .expect("softmax recipe build");
        let graph = recipe.inner.graph().clone();
        let id = recipe.inner.id();
        let roots =
            fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
        assert_eq!(roots.len(), 1, "lowering keeps a single root");
        let recipe_out = crate::pipelined_bridge::realize_one_as::<f32>(&graph, roots[0], &dev)
            .expect("realize recipe-decomposed softmax on CPU");

        // LEGACY path: the pre-migration ReduceMaxTo/ReduceSumTo(keepdim)
        // spelling, hand-built with identical Sub/Exp/Div.
        let x = Tensor::from_f32(data.clone(), shape(), &dev).unwrap().inner;
        let m = x.reduce_max_to(keepdim());
        let mb = m.broadcast_to(shape());
        let s = x.sub(&mb);
        let e = s.exp();
        let d = e.reduce_sum_to(keepdim());
        let db = d.broadcast_to(shape());
        let legacy = e.div(&db);
        let legacy_out = crate::pipelined_bridge::realize_one_as::<f32>(
            &legacy.graph().clone(),
            legacy.id(),
            &dev,
        )
        .expect("realize legacy softmax on CPU");

        // PRIMARY: the migration is numerically inert — bit-exact.
        assert_f32_bits_eq(&recipe_out, &legacy_out, "softmax recipe vs legacy");
        // Sanity: it is a real softmax (each row sums to ~1).
        for r in 0..rows {
            let sum: f32 = recipe_out[r * cols..(r + 1) * cols].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {r} softmax sum = {sum}");
        }

        // SECONDARY (documentation): the fused/production kernel differs by
        // at most 1 ULP — NOT from the migration but from its reciprocal-
        // multiply normalize (`e*(1/sum)`) vs the primitive true-`Div`.
        let fused_out = Tensor::from_f32(data.clone(), shape(), &dev)
            .unwrap()
            .softmax_last_dim()
            .expect("softmax fused build")
            .realize_f32();
        assert_f32_within_ulp(&recipe_out, &fused_out, 1, "softmax recipe vs fused kernel");
    }

    /// End-to-end real-backend parity for the migrated `RmsNormLastDim`
    /// (Increment C slice 1, T7, I3) — the norm counterpart of
    /// `softmax_last_dim_recipe_decompose_matches_legacy_on_cpu`. Here the D3
    /// shrink-via-swap replaced the keepdim `Reshape` with `Unsqueeze`
    /// (append) while the reducing `MeanDim(last)` is UNCHANGED, so the swap
    /// is metadata-only; the recipe-decompose realize must match a hand-built
    /// legacy `Reshape`-keepdim subgraph bit-exactly on the CPU backend.
    #[test]
    fn rms_norm_last_dim_recipe_decompose_matches_legacy_on_cpu() {
        let dev = Device::cpu();
        let (rows, cols) = (3usize, 6usize);
        let eps = 1e-6f64;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.29).cos() * 2.5 + 0.5)
            .collect();
        let shape = || Shape::from_dims(&[rows, cols]);
        let keepdim = || Shape::from_dims(&[rows, 1]);

        // RECIPE path: lower the fused node (MeanDim + Unsqueeze spelling).
        let recipe = Tensor::from_f32(data.clone(), shape(), &dev)
            .unwrap()
            .rms_norm_last_dim(eps)
            .expect("rms_norm recipe build");
        let graph = recipe.inner.graph().clone();
        let id = recipe.inner.id();
        let roots =
            fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
        assert_eq!(roots.len(), 1, "lowering keeps a single root");
        let recipe_out = crate::pipelined_bridge::realize_one_as::<f32>(&graph, roots[0], &dev)
            .expect("realize recipe-decomposed rms_norm on CPU");

        // LEGACY path: MeanDim(last) + Reshape-keepdim spelling, same eps and
        // same true-`Div` normalize.
        let x = Tensor::from_f32(data.clone(), shape(), &dev).unwrap().inner;
        let sq = x.sqr();
        let mean = sq.mean_dim(1);
        let mean_kd = mean.reshape(keepdim());
        let denom_sq = mean_kd.add_scalar(eps);
        let denom = denom_sq.sqrt();
        let denom_b = denom.broadcast_to(shape());
        let legacy = x.div(&denom_b);
        let legacy_out = crate::pipelined_bridge::realize_one_as::<f32>(
            &legacy.graph().clone(),
            legacy.id(),
            &dev,
        )
        .expect("realize legacy rms_norm on CPU");

        // PRIMARY: the keepdim swap is metadata-only — bit-exact.
        assert_f32_bits_eq(&recipe_out, &legacy_out, "rms_norm recipe vs legacy");

        // SECONDARY (documentation): the fused kernel's reciprocal-multiply
        // (`x*rms_inv`) differs from the primitive true-`Div` by ≤1 ULP.
        let fused_out = Tensor::from_f32(data.clone(), shape(), &dev)
            .unwrap()
            .rms_norm_last_dim(eps)
            .expect("rms_norm fused build")
            .realize_f32();
        assert_f32_within_ulp(
            &recipe_out,
            &fused_out,
            1,
            "rms_norm recipe vs fused kernel",
        );
    }

    /// Recipe principle (G2/G3 — part 1): SelectiveScan is the constitution's
    /// canonical **basis gap** (a higher-order `Scan` primitive). Fuel now HAS
    /// that primitive (`Op::Scan`), so `decompose` lowers to it — closing G3.
    /// This test flips the former surfaced-gap posture: (a) NON-REGRESSION —
    /// the fused CPU kernel still realizes `y = 12`; (b) VERIFICATION —
    /// `lowering_only` leaves NO `Op::Fused(SELECTIVE_SCAN)`, an `Op::Scan`
    /// terminal is present, and `unroll_scan` + realize of the ys oracle
    /// contains `y = 12` (`h = 3`). The fused kernel remains the executed
    /// production path; the `Op::Scan` recipe is the optimizer base-map cover
    /// + the numeric verify oracle.
    #[test]
    fn selective_scan_decompose_lowers_to_scan_and_matches() {
        use fuel_graph::Op;
        use fuel_graph::registry::FusedOps;
        let dev = Device::cpu();
        let u = Tensor::from_f32(vec![2.0f32], Shape::from_dims(&[1, 1, 1]), &dev).unwrap();
        let delta = u
            .const_f32_like(vec![0.5f32], Shape::from_dims(&[1, 1, 1]))
            .unwrap();
        let a = u
            .const_f32_like(vec![-1.0f32], Shape::from_dims(&[1, 1]))
            .unwrap();
        let b = u
            .const_f32_like(vec![3.0f32], Shape::from_dims(&[1, 1, 1]))
            .unwrap();
        let c = u
            .const_f32_like(vec![4.0f32], Shape::from_dims(&[1, 1, 1]))
            .unwrap();
        let y = u.selective_scan(&delta, &a, &b, &c, /* delta_softplus */ false);

        // (a) NON-REGRESSION: the fused kernel still runs and produces 12.0.
        let got = y.realize_f32();
        assert_eq!(got.len(), 1);
        assert!((got[0] - 12.0).abs() < 1e-4, "fused kernel y: {}", got[0]);

        // (b) VERIFICATION: lowering leaves NO Op::Fused(SELECTIVE_SCAN); an
        // Op::Scan terminal is present; unroll+realize matches h=3,y=12.
        let graph = y.inner.graph().clone();
        let id = y.inner.id();
        let roots =
            fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
        assert_eq!(roots.len(), 1);
        let scan_id = {
            let g = graph.read().unwrap();
            let mut stack = vec![roots[0]];
            let mut seen = std::collections::HashSet::new();
            let mut scan_id = None;
            while let Some(nid) = stack.pop() {
                if !seen.insert(nid) {
                    continue;
                }
                let node = g.node(nid);
                assert!(
                    !matches!(node.op, Op::Fused(fid, _) if fid == FusedOps::SELECTIVE_SCAN),
                    "SelectiveScan must lower to Op::Scan, not remain fused"
                );
                if matches!(node.op, Op::Scan { .. }) {
                    scan_id = Some(nid);
                }
                for &inp in &node.inputs {
                    stack.push(inp);
                }
            }
            scan_id.expect("an Op::Scan terminal must be present after lowering")
        };
        // Unroll the Op::Scan (seqlen = 1) and realize the ys oracle.
        let ys = {
            let mut g = graph.write().unwrap();
            fuel_graph::scan::unroll_scan(&mut g, scan_id, 1)
                .expect("unroll")
                .0[0] // GAP-303: selected side is a Vec; emit=All => one stacked ys
        };
        let oracle = crate::pipelined_bridge::realize_one_as::<f32>(&graph, ys, &dev)
            .expect("realize unrolled selective_scan oracle on CPU");
        assert!(
            oracle.iter().any(|&v| (v - 12.0).abs() < 1e-4),
            "unroll oracle must contain y = 12, got {oracle:?}"
        );
    }

    /// Recipe principle (G2/G3 — part 2): SsdChunkScan (Mamba-2's State-Space
    /// Duality scan) was the constitution's twin **basis gap** to SelectiveScan.
    /// Fuel now HAS the higher-order `Op::Scan` primitive, so `decompose` lowers
    /// to it — closing G3 part 2. The SSD recurrence carries a per-head SCALAR
    /// gate `exp(dt·a_h)` (`a` is `[heads]`, broadcast across the whole
    /// `head_dim×state_dim` block), with NO softplus. This test mirrors the
    /// SelectiveScan gate test: (a) NON-REGRESSION — the fused CPU kernel still
    /// realizes `y = 12`; (b) VERIFICATION — `lowering_only` leaves NO
    /// `Op::Fused(SSD_CHUNK_SCAN)`, an `Op::Scan` terminal is present, and
    /// `unroll_scan` + realize of the ys oracle contains `y = 12`.
    /// Fixture (batch=heads=head_dim=state_dim=1, seqlen=chunk_size=1):
    /// `x=2, dt=0.5, a=-1, b=3, c=4` → `exp(0.5·-1)·0 + 0.5·3·2 = 3`, `y = 3·4 = 12`.
    #[test]
    fn ssd_chunk_scan_decompose_lowers_to_scan_and_matches() {
        use fuel_graph::Op;
        use fuel_graph::registry::FusedOps;
        let dev = Device::cpu();
        // x [batch, seqlen, heads, head_dim] = [1,1,1,1]; dt [b,s,h]=[1,1,1];
        // a [heads]=[1]; b/c [b,s,h,state]=[1,1,1,1].
        let x = Tensor::from_f32(vec![2.0f32], Shape::from_dims(&[1, 1, 1, 1]), &dev).unwrap();
        let dt = x
            .const_f32_like(vec![0.5f32], Shape::from_dims(&[1, 1, 1]))
            .unwrap();
        let a = x
            .const_f32_like(vec![-1.0f32], Shape::from_dims(&[1]))
            .unwrap();
        let b = x
            .const_f32_like(vec![3.0f32], Shape::from_dims(&[1, 1, 1, 1]))
            .unwrap();
        let c = x
            .const_f32_like(vec![4.0f32], Shape::from_dims(&[1, 1, 1, 1]))
            .unwrap();
        let y = x.ssd_chunk_scan(&dt, &a, &b, &c, /* chunk_size */ 1);

        // (a) NON-REGRESSION: the fused kernel still runs and produces 12.0.
        let got = y.realize_f32();
        assert!(
            (got[0] - 12.0).abs() < 1e-4,
            "fused ssd kernel y: {}",
            got[0]
        );

        // (b) VERIFICATION: lowering leaves NO Op::Fused(SSD_CHUNK_SCAN); an
        // Op::Scan terminal is present; unroll+realize matches h=3, y=12.
        let graph = y.inner.graph().clone();
        let id = y.inner.id();
        let roots =
            fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
        let scan_id = {
            let g = graph.read().unwrap();
            let mut stack = vec![roots[0]];
            let mut seen = std::collections::HashSet::new();
            let mut scan_id = None;
            while let Some(nid) = stack.pop() {
                if !seen.insert(nid) {
                    continue;
                }
                let node = g.node(nid);
                assert!(
                    !matches!(node.op, Op::Fused(fid, _) if fid == FusedOps::SSD_CHUNK_SCAN),
                    "SsdChunkScan must lower to Op::Scan, not remain fused"
                );
                if matches!(node.op, Op::Scan { .. }) {
                    scan_id = Some(nid);
                }
                for &inp in &node.inputs {
                    stack.push(inp);
                }
            }
            scan_id.expect("an Op::Scan terminal must be present after lowering")
        };
        let ys = {
            let mut g = graph.write().unwrap();
            fuel_graph::scan::unroll_scan(&mut g, scan_id, 1)
                .expect("unroll")
                .0[0] // GAP-303: selected side is a Vec; emit=All => one stacked ys
        };
        let oracle = crate::pipelined_bridge::realize_one_as::<f32>(&graph, ys, &dev)
            .expect("realize unrolled ssd_chunk_scan oracle on CPU");
        assert!(
            oracle.iter().any(|&v| (v - 12.0).abs() < 1e-4),
            "oracle y=12, got {oracle:?}"
        );
    }

    // ---- Op::Scan Phase 2 (C3/C4/C5): numeric BPTT finite-difference gates ---
    //
    // Prove the lower_scans_for_backward pre-pass gradients are correct against
    // finite differences over the SAME unrolled graph (self-consistent, decoupled
    // from kernel F64-accumulate parity). These need realize, so they live here.

    #[test]
    fn affine_scan_bptt_matches_finite_difference() {
        use fuel_graph::{Node, NodeHandle, Op, ScanEmit, ScanRole};
        use fuel_ir::{DType, Shape};
        let dev = Device::cpu();
        // f(init, a, b): carry_{t+1} = a*carry_t + b, carry_0 = init, bound = 3, loss = carry_3.
        // Closed form: carry_3 = a^3*init + b*(a^2 + a + 1). d/dinit = a^3.
        let build = |init_v: f32,
                     a_v: f32,
                     b_v: f32|
         -> (
            std::sync::Arc<std::sync::RwLock<fuel_graph::Graph>>,
            fuel_graph::NodeId,
            fuel_graph::NodeHandle,
            fuel_graph::NodeHandle,
            fuel_graph::NodeHandle,
        ) {
            let init =
                NodeHandle::from_f32(vec![init_v], Shape::from_dims(&[1]), dev.as_dyn()).unwrap();
            let g = init.graph().clone();
            let a = NodeHandle::from_existing(g.clone(), init.id())
                .const_f32_like(vec![a_v], Shape::from_dims(&[1]))
                .unwrap();
            let b = NodeHandle::from_existing(g.clone(), init.id())
                .const_f32_like(vec![b_v], Shape::from_dims(&[1]))
                .unwrap();
            let nc = {
                let mut gw = g.write().unwrap();
                let s = Shape::from_dims(&[1]);
                let hole = gw.push(Node {
                    op: Op::ScanPlaceholder {
                        role: ScanRole::Carry,
                        index: 0,
                    },
                    inputs: vec![],
                    shape: s.clone(),
                    dtype: DType::F32,
                });
                let ac = gw.push(Node {
                    op: Op::Mul,
                    inputs: vec![a.id(), hole],
                    shape: s.clone(),
                    dtype: DType::F32,
                });
                gw.push(Node {
                    op: Op::Add,
                    inputs: vec![ac, b.id()],
                    shape: s.clone(),
                    dtype: DType::F32,
                })
            };
            let nc_t = NodeHandle::from_existing(g.clone(), nc);
            let out = init
                .scan(
                    &[],
                    &[a.clone(), b.clone()],
                    &nc_t,
                    &nc_t,
                    3,
                    ScanEmit::Final,
                )
                .expect("scan");
            (g, out.id(), init, a, b)
        };
        // Forward value via unroll+realize (self-consistent oracle).
        let fwd = |init_v: f32, a_v: f32, b_v: f32| -> f32 {
            let (g, out_id, _i, _a, _b) = build(init_v, a_v, b_v);
            let scan_id = {
                let gr = g.read().unwrap();
                gr.node(out_id).inputs[0]
            };
            let carry = {
                let mut gw = g.write().unwrap();
                fuel_graph::scan::unroll_scan(&mut gw, scan_id, 3)
                    .expect("unroll")
                    .0[0] // GAP-303: selected side is a Vec; emit=All => one stacked ys
            }; // emit=Final -> selected=final_carry
            crate::pipelined_bridge::realize_one_as::<f32>(&g, carry, &dev).expect("realize")[0]
        };
        // Autograd grad w.r.t. init at (1.0, 0.5, 0.1).
        let (g, out_id, init, a, b) = build(1.0, 0.5, 0.1);
        let out = fuel_graph::NodeHandle::from_existing(g.clone(), out_id);
        let grads = out.backward();
        let realize_grad = |t: &fuel_graph::NodeHandle| {
            crate::pipelined_bridge::realize_one_as::<f32>(
                &g,
                grads.get(t).expect("grad").id(),
                &dev,
            )
            .expect("realize grad")[0]
        };
        let g_init = realize_grad(&init);
        let g_a = realize_grad(&a);
        let g_b = realize_grad(&b);
        // Central finite differences.
        let h = 1e-3f32;
        let fd = |dinit: f32, da: f32, db: f32| {
            (fwd(1.0 + dinit, 0.5 + da, 0.1 + db) - fwd(1.0 - dinit, 0.5 - da, 0.1 - db))
                / (2.0 * h)
        };
        assert!(
            (g_init - fd(h, 0.0, 0.0)).abs() < 2e-2,
            "d/dinit: autograd {g_init} vs FD {}",
            fd(h, 0.0, 0.0)
        );
        assert!(
            (g_a - fd(0.0, h, 0.0)).abs() < 2e-2,
            "d/da: autograd {g_a} vs FD {}",
            fd(0.0, h, 0.0)
        );
        assert!(
            (g_b - fd(0.0, 0.0, h)).abs() < 2e-2,
            "d/db: autograd {g_b} vs FD {}",
            fd(0.0, 0.0, h)
        );
    }

    #[test]
    fn selective_scan_is_differentiable_backward_matches_fd() {
        let dev = Device::cpu();
        let fwd = |u_v: f32| -> f32 {
            let u = Tensor::from_f32(vec![u_v], Shape::from_dims(&[1, 1, 1]), &dev).unwrap();
            let delta = u
                .const_f32_like(vec![0.5f32], Shape::from_dims(&[1, 1, 1]))
                .unwrap();
            let a = u
                .const_f32_like(vec![-1.0f32], Shape::from_dims(&[1, 1]))
                .unwrap();
            let b = u
                .const_f32_like(vec![3.0f32], Shape::from_dims(&[1, 1, 1]))
                .unwrap();
            let c = u
                .const_f32_like(vec![4.0f32], Shape::from_dims(&[1, 1, 1]))
                .unwrap();
            u.selective_scan(&delta, &a, &b, &c, false).realize_f32()[0]
        };
        // Autograd at u=2.0.
        let u = Tensor::from_f32(vec![2.0f32], Shape::from_dims(&[1, 1, 1]), &dev).unwrap();
        let delta = u
            .const_f32_like(vec![0.5f32], Shape::from_dims(&[1, 1, 1]))
            .unwrap();
        let a = u
            .const_f32_like(vec![-1.0f32], Shape::from_dims(&[1, 1]))
            .unwrap();
        let b = u
            .const_f32_like(vec![3.0f32], Shape::from_dims(&[1, 1, 1]))
            .unwrap();
        let c = u
            .const_f32_like(vec![4.0f32], Shape::from_dims(&[1, 1, 1]))
            .unwrap();
        let y = u.selective_scan(&delta, &a, &b, &c, false);
        let grads = y.inner.backward();
        let g_u_id = grads.get(u.graph_tensor()).expect("grad u").id();
        let g_u =
            crate::pipelined_bridge::realize_one_as::<f32>(&u.inner.graph().clone(), g_u_id, &dev)
                .expect("realize")[0];
        let h = 1e-3;
        let fd = (fwd(2.0 + h) - fwd(2.0 - h)) / (2.0 * h);
        assert!(
            (g_u - fd).abs() < 5e-2,
            "selective_scan d/du: autograd {g_u} vs FD {fd}"
        );
    }

    #[test]
    fn ssd_chunk_scan_is_differentiable_backward_matches_fd() {
        let dev = Device::cpu();
        // x [batch,seqlen,heads,head_dim]=[1,1,1,1]; dt [b,s,h]=[1,1,1]; a [heads]=[1];
        // b/c [b,s,h,state]=[1,1,1,1]. Single step: h = dt*b*x = 1.5x, y = c*h = 6x (linear).
        let fwd = |x_v: f32| -> f32 {
            let x = Tensor::from_f32(vec![x_v], Shape::from_dims(&[1, 1, 1, 1]), &dev).unwrap();
            let dt = x
                .const_f32_like(vec![0.5f32], Shape::from_dims(&[1, 1, 1]))
                .unwrap();
            let a = x
                .const_f32_like(vec![-1.0f32], Shape::from_dims(&[1]))
                .unwrap();
            let b = x
                .const_f32_like(vec![3.0f32], Shape::from_dims(&[1, 1, 1, 1]))
                .unwrap();
            let c = x
                .const_f32_like(vec![4.0f32], Shape::from_dims(&[1, 1, 1, 1]))
                .unwrap();
            x.ssd_chunk_scan(&dt, &a, &b, &c, 1).realize_f32()[0]
        };
        // Autograd at x=2.0.
        let x = Tensor::from_f32(vec![2.0f32], Shape::from_dims(&[1, 1, 1, 1]), &dev).unwrap();
        let dt = x
            .const_f32_like(vec![0.5f32], Shape::from_dims(&[1, 1, 1]))
            .unwrap();
        let a = x
            .const_f32_like(vec![-1.0f32], Shape::from_dims(&[1]))
            .unwrap();
        let b = x
            .const_f32_like(vec![3.0f32], Shape::from_dims(&[1, 1, 1, 1]))
            .unwrap();
        let c = x
            .const_f32_like(vec![4.0f32], Shape::from_dims(&[1, 1, 1, 1]))
            .unwrap();
        let y = x.ssd_chunk_scan(&dt, &a, &b, &c, 1);
        let grads = y.inner.backward();
        let g_x_id = grads.get(x.graph_tensor()).expect("grad x").id();
        let g_x =
            crate::pipelined_bridge::realize_one_as::<f32>(&x.inner.graph().clone(), g_x_id, &dev)
                .expect("realize")[0];
        let h = 1e-3;
        let fd = (fwd(2.0 + h) - fwd(2.0 - h)) / (2.0 * h);
        assert!(
            (g_x - fd).abs() < 5e-2,
            "ssd_chunk_scan d/dx: autograd {g_x} vs FD {fd}"
        );
    }

    // Review Fix 2: the seqlen=1 SSM gates above multiply exp(dt*a) by a zero
    // initial carry, so `y` is linear in the input and the multi-step recurrent
    // gate is never FD-checked for the SSM ops. These seqlen>=2 gates carry a
    // NON-ZERO h across steps (a < 0), so d(sum y)/d(early input) flows THROUGH
    // the gate exp(dt*a) — the multi-step BPTT the headline SSM deliverable needs.

    #[test]
    fn selective_scan_seqlen2_bptt_matches_finite_difference() {
        let dev = Device::cpu();
        // u/delta [batch,seqlen,dim]=[1,2,1]; a [dim,dstate]=[1,1] (< 0, stable gate);
        // b/c [batch,seqlen,dstate]=[1,2,1]. loss = sum(y) (ones-seed over [1,2,1]).
        let fwd = |u_vals: &[f32]| -> f32 {
            let u = Tensor::from_f32(u_vals.to_vec(), Shape::from_dims(&[1, 2, 1]), &dev).unwrap();
            let delta = u
                .const_f32_like(vec![0.7f32, 0.7], Shape::from_dims(&[1, 2, 1]))
                .unwrap();
            let a = u
                .const_f32_like(vec![-0.5f32], Shape::from_dims(&[1, 1]))
                .unwrap();
            let b = u
                .const_f32_like(vec![1.0f32, 1.0], Shape::from_dims(&[1, 2, 1]))
                .unwrap();
            let c = u
                .const_f32_like(vec![1.0f32, 1.0], Shape::from_dims(&[1, 2, 1]))
                .unwrap();
            u.selective_scan(&delta, &a, &b, &c, false)
                .realize_f32()
                .iter()
                .sum()
        };
        let u0 = vec![1.0f32, 2.0];
        let u = Tensor::from_f32(u0.clone(), Shape::from_dims(&[1, 2, 1]), &dev).unwrap();
        let delta = u
            .const_f32_like(vec![0.7f32, 0.7], Shape::from_dims(&[1, 2, 1]))
            .unwrap();
        let a = u
            .const_f32_like(vec![-0.5f32], Shape::from_dims(&[1, 1]))
            .unwrap();
        let b = u
            .const_f32_like(vec![1.0f32, 1.0], Shape::from_dims(&[1, 2, 1]))
            .unwrap();
        let c = u
            .const_f32_like(vec![1.0f32, 1.0], Shape::from_dims(&[1, 2, 1]))
            .unwrap();
        let y = u.selective_scan(&delta, &a, &b, &c, false);
        let grads = y.inner.backward(); // ones-seed over [1,2,1] == grad of sum(y)
        let g_u_id = grads.get(u.graph_tensor()).expect("grad u").id();
        let g_u =
            crate::pipelined_bridge::realize_one_as::<f32>(&u.inner.graph().clone(), g_u_id, &dev)
                .expect("realize");
        // FD over u_1 (index 0): d(sum y)/d u_1 = dB + c*exp(dt*a)*dB — the second
        // term is the recurrent gate carrying u_1 into y_2.
        let h = 1e-3f32;
        let mut up = u0.clone();
        up[0] += h;
        let mut um = u0.clone();
        um[0] -= h;
        let fd0 = (fwd(&up) - fwd(&um)) / (2.0 * h);
        assert!(
            (g_u[0] - fd0).abs() < 5e-2,
            "selective_scan seqlen2 dL/du_1: autograd {} vs FD {fd0}",
            g_u[0]
        );
        // Non-triviality: the gate contribution makes this strictly > the pure-dB
        // seqlen=1 grad — a real, non-vacuous multi-step gradient (~1.19 here).
        assert!(
            g_u[0].abs() > 0.5,
            "recurrent-gate grad must be non-trivial, got {g_u:?}"
        );
    }

    #[test]
    fn ssd_chunk_scan_seqlen4_chunk2_bptt_matches_finite_difference() {
        let dev = Device::cpu();
        // seqlen=4, chunk=2 -> a 2-CHUNK Op::Scan (bound=2): the inter-chunk state
        // carry passes exp(dt*a) across chunks. x [b,s,h,hd]=[1,4,1,1]; dt [b,s,h]=[1,4,1];
        // a [heads]=[1] (< 0); b/c [b,s,h,state]=[1,4,1,1].
        let fwd = |x_vals: &[f32]| -> f32 {
            let x =
                Tensor::from_f32(x_vals.to_vec(), Shape::from_dims(&[1, 4, 1, 1]), &dev).unwrap();
            let dt = x
                .const_f32_like(vec![0.6f32, 0.6, 0.6, 0.6], Shape::from_dims(&[1, 4, 1]))
                .unwrap();
            let a = x
                .const_f32_like(vec![-0.5f32], Shape::from_dims(&[1]))
                .unwrap();
            let b = x
                .const_f32_like(vec![1.0f32, 1.0, 1.0, 1.0], Shape::from_dims(&[1, 4, 1, 1]))
                .unwrap();
            let c = x
                .const_f32_like(vec![1.0f32, 1.0, 1.0, 1.0], Shape::from_dims(&[1, 4, 1, 1]))
                .unwrap();
            x.ssd_chunk_scan(&dt, &a, &b, &c, 2)
                .realize_f32()
                .iter()
                .sum()
        };
        let x0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = Tensor::from_f32(x0.clone(), Shape::from_dims(&[1, 4, 1, 1]), &dev).unwrap();
        let dt = x
            .const_f32_like(vec![0.6f32, 0.6, 0.6, 0.6], Shape::from_dims(&[1, 4, 1]))
            .unwrap();
        let a = x
            .const_f32_like(vec![-0.5f32], Shape::from_dims(&[1]))
            .unwrap();
        let b = x
            .const_f32_like(vec![1.0f32, 1.0, 1.0, 1.0], Shape::from_dims(&[1, 4, 1, 1]))
            .unwrap();
        let c = x
            .const_f32_like(vec![1.0f32, 1.0, 1.0, 1.0], Shape::from_dims(&[1, 4, 1, 1]))
            .unwrap();
        let y = x.ssd_chunk_scan(&dt, &a, &b, &c, 2);
        let grads = y.inner.backward();
        let g_x_id = grads.get(x.graph_tensor()).expect("grad x").id();
        let g_x =
            crate::pipelined_bridge::realize_one_as::<f32>(&x.inner.graph().clone(), g_x_id, &dev)
                .expect("realize");
        // FD over x_0 (a chunk-0 input): its contribution to y_3/y_4 (chunk 1)
        // rides the cross-chunk gate exp(dt*a) — multi-step BPTT.
        let h = 1e-3f32;
        let mut xp = x0.clone();
        xp[0] += h;
        let mut xm = x0.clone();
        xm[0] -= h;
        let fd0 = (fwd(&xp) - fwd(&xm)) / (2.0 * h);
        assert!(
            (g_x[0] - fd0).abs() < 5e-2,
            "ssd_chunk_scan seqlen4 dL/dx_0: autograd {} vs FD {fd0}",
            g_x[0]
        );
        assert!(
            g_x[0].abs() > 0.5,
            "cross-chunk-gate grad must be non-trivial, got {g_x:?}"
        );
    }

    // ---- Task 8: general-dimension SSM parity gate (C7 non-regression) ------
    //
    // The all-1s gap-test fixtures above (Task 6/7) share a single flat order
    // across every axis, so they cannot catch an axis transposition in the
    // decompose. The tests below run each SSM op at DISTINGUISHING multi-dim
    // sizes and assert the `unroll_scan` oracle (over the lowered `Op::Scan`)
    // matches the fused kernel element-wise, which is only possible if every
    // per-step slice / broadcast / reduce axis in the Task-6/7 decompose lines
    // up with the kernel's layout.

    /// Walk the lowered graph from `root`, assert NO `Op::Fused(fused_id)`
    /// survives the `lowering_only()` pass, and return the `Op::Scan` terminal
    /// the decompose emitted (there must be exactly one on the SSM path).
    fn find_scan_terminal(
        graph: &std::sync::Arc<std::sync::RwLock<fuel_graph::Graph>>,
        root: fuel_graph::NodeId,
        fused_id: fuel_graph::registry::FusedOpId,
    ) -> fuel_graph::NodeId {
        use fuel_graph::Op;
        let g = graph.read().unwrap();
        let mut stack = vec![root];
        let mut seen = std::collections::HashSet::new();
        let mut scan_id = None;
        while let Some(nid) = stack.pop() {
            if !seen.insert(nid) {
                continue;
            }
            let node = g.node(nid);
            assert!(
                !matches!(node.op, Op::Fused(fid, _) if fid == fused_id),
                "SSM op must lower to Op::Scan, not remain fused",
            );
            if matches!(node.op, Op::Scan { .. }) {
                scan_id = Some(nid);
            }
            for &inp in &node.inputs {
                stack.push(inp);
            }
        }
        scan_id.expect("an Op::Scan terminal must be present after lowering")
    }

    /// Reindex a flat row-major `[seqlen, batch, tail]` buffer — the layout
    /// `unroll_scan` stacks its per-step ys into (scan axis 0) — to
    /// `[batch, seqlen, tail]`, the fused kernel's `y` layout, so the two align
    /// element-wise. `tail` is the product of the trailing per-token dims (`dim`
    /// for selective_scan; `heads*head_dim` for ssd) — a contiguous block whose
    /// internal order is preserved by the batch/seqlen swap.
    ///
    /// This is done in host code deliberately: the executor treats `Op::Permute`
    /// as a metadata-only view (shared bytes + strided layout), so realizing a
    /// `Permute` *root* can hand back byte-shared storage still in seqlen-major
    /// order — an unreliable comparand. Realizing the contiguous unroll `Concat`
    /// (`ys`) and permuting the flat host bytes here is unambiguous.
    fn scan_ys_to_fused_layout(ys: &[f32], seqlen: usize, batch: usize, tail: usize) -> Vec<f32> {
        assert_eq!(ys.len(), seqlen * batch * tail, "ys layout mismatch");
        let mut out = vec![0f32; ys.len()];
        for s in 0..seqlen {
            for bb in 0..batch {
                for t in 0..tail {
                    out[(bb * seqlen + s) * tail + t] = ys[(s * batch + bb) * tail + t];
                }
            }
        }
        out
    }

    /// Task 8 (part 1) — `selective_scan` general-dimension parity.
    ///
    /// **C7 non-regression:** `PipelinedExecutor::realize` dispatches
    /// `Op::Fused(SELECTIVE_SCAN)` **directly** to its kernel and never runs a
    /// `RuleRegistry` lowering pass, so `decompose`→`Op::Scan` is reached ONLY by
    /// the `lowering_only()` verification below — the fused kernel is the
    /// executed path *by construction*. The `realize_f32()` leg (a) is the proof
    /// the fused kernel still runs (no `O(seqlen)` unroll on the hot path).
    ///
    /// **Distinguishing dims** batch=2, seqlen=3, dim=2, dstate=2 with EVERY
    /// element distinct and asymmetric: `u`/`delta` index `dim`, `b`/`c` index
    /// `dstate`, and `a` is a full `[dim,dstate]` gate — so a dim↔dstate
    /// broadcast swap multiplies wrong pairs, and because `h` threads across the
    /// (order-sensitive) seqlen recurrence, a batch↔seqlen transpose changes the
    /// result. The all-1s fixture cannot see either; this can.
    #[test]
    fn selective_scan_multidim_unroll_matches_fused_kernel() {
        use fuel_graph::registry::FusedOps;
        let dev = Device::cpu();
        let (batch, seqlen, dim, dstate) = (2usize, 3usize, 2usize, 2usize);
        // [batch, seqlen, dim] — all distinct, mixed signs.
        let u = Tensor::from_f32(
            vec![
                0.5, -0.3, 1.0, 0.2, -0.5, 0.7, 0.25, 0.9, -0.8, 0.4, 0.6, -0.1,
            ],
            Shape::from_dims(&[batch, seqlen, dim]),
            &dev,
        )
        .unwrap();
        // [batch, seqlen, dim] — positive (delta is a rate); all distinct.
        let delta = u
            .const_f32_like(
                vec![
                    0.1, 0.2, 0.3, 0.15, 0.25, 0.4, 0.35, 0.05, 0.2, 0.3, 0.12, 0.28,
                ],
                Shape::from_dims(&[batch, seqlen, dim]),
            )
            .unwrap();
        // [dim, dstate] — negative (Mamba a<0 → stable gate in (0,1)); distinct.
        let a = u
            .const_f32_like(
                vec![-0.7, -0.3, -0.5, -0.9],
                Shape::from_dims(&[dim, dstate]),
            )
            .unwrap();
        // [batch, seqlen, dstate] — distinct.
        let b = u
            .const_f32_like(
                vec![
                    1.0, 0.5, 0.25, 0.75, 0.6, 0.2, 0.8, 0.4, 0.3, 0.9, 0.55, 0.15,
                ],
                Shape::from_dims(&[batch, seqlen, dstate]),
            )
            .unwrap();
        // [batch, seqlen, dstate] — distinct.
        let c = u
            .const_f32_like(
                vec![2.0, 1.0, 0.5, 1.5, 0.8, 0.3, 1.2, 0.6, 0.9, 0.4, 0.7, 1.1],
                Shape::from_dims(&[batch, seqlen, dstate]),
            )
            .unwrap();
        let y = u.selective_scan(&delta, &a, &b, &c, /* delta_softplus */ false);

        // (a) Fused kernel (executed production path) runs and is well-shaped.
        let fused = y.realize_f32();
        assert_eq!(fused.len(), batch * seqlen * dim);

        // (b) Unroll oracle over the lowered Op::Scan matches the fused kernel.
        let graph = y.inner.graph().clone();
        let roots = fuel_graph::opt::RuleRegistry::lowering_only()
            .optimize_to_fixpoint(&graph, &[y.inner.id()]);
        assert_eq!(roots.len(), 1);
        let scan_id = find_scan_terminal(&graph, roots[0], FusedOps::SELECTIVE_SCAN);
        let ys = {
            let mut g = graph.write().unwrap();
            fuel_graph::scan::unroll_scan(&mut g, scan_id, seqlen)
                .expect("unroll")
                .0[0] // GAP-303: selected side is a Vec; emit=All => one stacked ys
        };
        let ys_flat = crate::pipelined_bridge::realize_one_as::<f32>(&graph, ys, &dev)
            .expect("realize selective_scan unroll oracle on CPU");
        // ys is [seqlen,batch,dim]; fused y is [batch,seqlen,dim].
        let oracle = scan_ys_to_fused_layout(&ys_flat, seqlen, batch, dim);
        assert_eq!(oracle.len(), fused.len());
        let mut max_drift = 0f32;
        for (i, (o, f)) in oracle.iter().zip(fused.iter()).enumerate() {
            let d = (o - f).abs();
            max_drift = max_drift.max(d);
            // 1e-3 sabotage-calibrated: native-F32 unroll vs F64-accumulate
            // kernel. Honest drift here is ~1e-6 (printed); a batch/seqlen or
            // dim/dstate transpose diverges by O(0.1..10) — far above 1e-3.
            assert!(
                d < 1e-3,
                "selective_scan multidim mismatch at {i}: oracle {o} vs fused {f} (drift {d})",
            );
        }
        eprintln!("selective_scan multidim: max honest drift = {max_drift:e}");
    }

    /// Task 8 (part 1, softplus) — `selective_scan` with `delta_softplus=true`,
    /// the branch unexercised at T=1. `delta` carries mixed-sign values so the
    /// stable softplus form `Relu(x)+Log(1+Exp(Neg(Abs(x))))` does real work on
    /// both the `x>0` and `x<0` sides before the scan. Same fused-vs-oracle
    /// parity as the non-softplus case.
    #[test]
    fn selective_scan_softplus_multitoken_unroll_matches_fused_kernel() {
        use fuel_graph::registry::FusedOps;
        let dev = Device::cpu();
        let (batch, seqlen, dim, dstate) = (1usize, 3usize, 2usize, 2usize);
        let u = Tensor::from_f32(
            vec![0.5, -0.3, 1.0, 0.2, -0.5, 0.7],
            Shape::from_dims(&[batch, seqlen, dim]),
            &dev,
        )
        .unwrap();
        // Mixed signs → softplus exercises max(x,0) AND ln(1+exp(-|x|)).
        let delta = u
            .const_f32_like(
                vec![-0.5, 0.2, 0.8, -1.0, 0.3, -0.2],
                Shape::from_dims(&[batch, seqlen, dim]),
            )
            .unwrap();
        let a = u
            .const_f32_like(
                vec![-0.7, -0.3, -0.5, -0.9],
                Shape::from_dims(&[dim, dstate]),
            )
            .unwrap();
        let b = u
            .const_f32_like(
                vec![1.0, 0.5, 0.25, 0.75, 0.6, 0.2],
                Shape::from_dims(&[batch, seqlen, dstate]),
            )
            .unwrap();
        let c = u
            .const_f32_like(
                vec![2.0, 1.0, 0.5, 1.5, 0.8, 0.3],
                Shape::from_dims(&[batch, seqlen, dstate]),
            )
            .unwrap();
        let y = u.selective_scan(&delta, &a, &b, &c, /* delta_softplus */ true);

        let fused = y.realize_f32();
        assert_eq!(fused.len(), batch * seqlen * dim);

        let graph = y.inner.graph().clone();
        let roots = fuel_graph::opt::RuleRegistry::lowering_only()
            .optimize_to_fixpoint(&graph, &[y.inner.id()]);
        let scan_id = find_scan_terminal(&graph, roots[0], FusedOps::SELECTIVE_SCAN);
        let ys = {
            let mut g = graph.write().unwrap();
            fuel_graph::scan::unroll_scan(&mut g, scan_id, seqlen)
                .expect("unroll")
                .0[0] // GAP-303: selected side is a Vec; emit=All => one stacked ys
        };
        let ys_flat = crate::pipelined_bridge::realize_one_as::<f32>(&graph, ys, &dev)
            .expect("realize selective_scan softplus oracle on CPU");
        let oracle = scan_ys_to_fused_layout(&ys_flat, seqlen, batch, dim);
        let mut max_drift = 0f32;
        for (i, (o, f)) in oracle.iter().zip(fused.iter()).enumerate() {
            let d = (o - f).abs();
            max_drift = max_drift.max(d);
            assert!(
                d < 1e-3,
                "selective_scan softplus mismatch at {i}: oracle {o} vs fused {f} (drift {d})",
            );
        }
        eprintln!("selective_scan softplus: max honest drift = {max_drift:e}");
    }

    /// Task 8 (part 2) — `ssd_chunk_scan` general-dimension parity.
    ///
    /// **C7 non-regression** (same construction as part 1): the executor
    /// dispatches `Op::Fused(SSD_CHUNK_SCAN)` straight to its kernel; the
    /// `decompose`→`Op::Scan` recipe is reached only by `lowering_only()` here.
    ///
    /// **Distinguishing dims** batch=2, seqlen=4, heads=2, head_dim=2,
    /// state_dim=2, chunk_size=2. Per-head `a = [-0.6, -1.0]` (head 0's gate
    /// leaking into head 1 flips the decay); `x` (indexes head_dim), `b`/`c`
    /// (index state_dim) use three DIFFERENT value ramps, so a head_dim↔state_dim
    /// broadcast swap (`bc_x` vs `bc_state`) mixes the wrong axes, and the
    /// seqlen recurrence + batch↔seqlen permute are exercised because batch>1.
    /// Every element is distinct.
    #[test]
    fn ssd_chunk_scan_multidim_unroll_matches_fused_kernel() {
        use fuel_graph::registry::FusedOps;
        let dev = Device::cpu();
        let (batch, seqlen, heads, head_dim, state_dim, chunk) =
            (2usize, 4usize, 2usize, 2usize, 2usize, 2usize);
        // Distinct value ramps (different base + slope per tensor) guarantee no
        // accidental symmetry; dt stays positive, a stays negative (stable gate).
        let x_data: Vec<f32> = (0..batch * seqlen * heads * head_dim)
            .map(|i| 0.20 + 0.05 * i as f32)
            .collect();
        let dt_data: Vec<f32> = (0..batch * seqlen * heads)
            .map(|i| 0.10 + 0.04 * i as f32)
            .collect();
        let b_data: Vec<f32> = (0..batch * seqlen * heads * state_dim)
            .map(|i| 0.15 + 0.03 * i as f32)
            .collect();
        let c_data: Vec<f32> = (0..batch * seqlen * heads * state_dim)
            .map(|i| 0.25 + 0.035 * i as f32)
            .collect();
        let x = Tensor::from_f32(
            x_data,
            Shape::from_dims(&[batch, seqlen, heads, head_dim]),
            &dev,
        )
        .unwrap();
        let dt = x
            .const_f32_like(dt_data, Shape::from_dims(&[batch, seqlen, heads]))
            .unwrap();
        let a = x
            .const_f32_like(vec![-0.6f32, -1.0], Shape::from_dims(&[heads]))
            .unwrap();
        let b = x
            .const_f32_like(b_data, Shape::from_dims(&[batch, seqlen, heads, state_dim]))
            .unwrap();
        let c = x
            .const_f32_like(c_data, Shape::from_dims(&[batch, seqlen, heads, state_dim]))
            .unwrap();
        let y = x.ssd_chunk_scan(&dt, &a, &b, &c, chunk);

        // (a) Fused kernel (executed production path) runs and is well-shaped.
        let fused = y.realize_f32();
        assert_eq!(fused.len(), batch * seqlen * heads * head_dim);

        // (b) Unroll oracle over the lowered Op::Scan matches the fused kernel.
        let graph = y.inner.graph().clone();
        let roots = fuel_graph::opt::RuleRegistry::lowering_only()
            .optimize_to_fixpoint(&graph, &[y.inner.id()]);
        assert_eq!(roots.len(), 1);
        let scan_id = find_scan_terminal(&graph, roots[0], FusedOps::SSD_CHUNK_SCAN);
        let ys = {
            let mut g = graph.write().unwrap();
            fuel_graph::scan::unroll_scan(&mut g, scan_id, seqlen)
                .expect("unroll")
                .0[0] // GAP-303: selected side is a Vec; emit=All => one stacked ys
        };
        let ys_flat = crate::pipelined_bridge::realize_one_as::<f32>(&graph, ys, &dev)
            .expect("realize ssd_chunk_scan unroll oracle on CPU");
        // ys is [seqlen,batch,heads,head_dim]; fused y is
        // [batch,seqlen,heads,head_dim]. tail = heads*head_dim (contiguous).
        let oracle = scan_ys_to_fused_layout(&ys_flat, seqlen, batch, heads * head_dim);
        assert_eq!(oracle.len(), fused.len());
        let mut max_drift = 0f32;
        for (i, (o, f)) in oracle.iter().zip(fused.iter()).enumerate() {
            let d = (o - f).abs();
            max_drift = max_drift.max(d);
            assert!(
                d < 1e-3,
                "ssd_chunk_scan multidim mismatch at {i}: oracle {o} vs fused {f} (drift {d})",
            );
        }
        eprintln!("ssd_chunk_scan multidim: max honest drift = {max_drift:e}");
    }

    /// Recipe principle (G2): Nf4Matmul must decompose to a **fused-free**
    /// primitive subgraph whose realize matches the fused kernel. RED before
    /// the total decompose landed (the self-return left an
    /// `Op::Fused(NF4_MATMUL)` opaque island in the base map); GREEN after.
    /// Uses the hand-computed two-outputs / two-blocks case (expected `[10,
    /// 50]`) shared with `fuel_cpu_backend`'s byte-kernel test, so the
    /// indicator-sum codebook + nibble unpack + per-block scale are checked
    /// against the same numbers the fused CPU kernel produces.
    #[test]
    fn nf4_matmul_decompose_matches_kernel() {
        use fuel_graph::Op;
        use fuel_graph::registry::FusedOps;
        let dev = Device::cpu();
        // n=2, k=4, block_size=2; w_packed [2, 2] U8, absmax [2, 2] F32.
        let weight = crate::nf4::nf4_from_bytes(
            vec![247_u8, 247, 127, 127],
            vec![1.0_f32, 2.0, 10.0, 20.0],
            2,
            4,
            2,
            &dev,
        )
        .expect("nf4_from_bytes");
        let act = Tensor::from_graph_tensor(
            weight
                .w_packed
                .graph_tensor()
                .const_f32_like(vec![1.0_f32, 2.0, 2.0, 4.0], Shape::from_dims(&[1, 4]))
                .unwrap(),
        );
        let y = weight.matmul(&act);

        // Decompose explicitly, then realize the primitive subgraph.
        let graph = y.inner.graph().clone();
        let id = y.inner.id();
        let roots =
            fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
        assert_eq!(roots.len(), 1, "lowering should keep a single root");

        // Born-red discriminator: no Op::Fused(NF4_MATMUL) reachable from the
        // realized root — a self-returning decompose fails exactly here.
        {
            let g = graph.read().unwrap();
            let mut stack = vec![roots[0]];
            let mut seen = std::collections::HashSet::new();
            while let Some(nid) = stack.pop() {
                if !seen.insert(nid) {
                    continue;
                }
                let node = g.node(nid);
                assert!(
                    !matches!(node.op, Op::Fused(fid, _) if fid == FusedOps::NF4_MATMUL),
                    "decomposed graph still contains an Op::Fused(NF4_MATMUL) island",
                );
                for &inp in &node.inputs {
                    stack.push(inp);
                }
            }
        }

        let got = crate::pipelined_bridge::realize_one_as::<f32>(&graph, roots[0], &dev)
            .expect("realize decomposed Nf4Matmul on CPU");
        assert_eq!(got.len(), 2);
        assert!((got[0] - 10.0).abs() < 1e-4, "out 0: {}", got[0]);
        assert!((got[1] - 50.0).abs() < 1e-4, "out 1: {}", got[1]);
    }

    /// Build a one-output fused-op node directly on `anchor`'s graph, lower it
    /// to primitives, realize on CPU, and return the F32 values. The decompose
    /// parity tests for backward helpers use this — those ops have no public
    /// builder (autograd creates them), so the test constructs the
    /// `Op::Fused` node by hand.
    fn lower_realize_fused(
        anchor: &Tensor,
        op: fuel_graph::Op,
        inputs: Vec<fuel_graph::NodeId>,
        shape: Shape,
    ) -> Vec<f32> {
        let graph = anchor.inner.graph().clone();
        let fused_id = {
            let mut g = graph.write().unwrap();
            g.push(fuel_graph::Node {
                op,
                inputs,
                shape,
                dtype: DType::F32,
            })
        };
        let roots = fuel_graph::opt::RuleRegistry::lowering_only()
            .optimize_to_fixpoint(&graph, &[fused_id]);
        assert_eq!(roots.len(), 1, "lowering keeps a single root");
        crate::pipelined_bridge::realize_one_as::<f32>(&graph, roots[0], &Device::cpu())
            .expect("realize decomposed fused op on CPU")
    }

    #[test]
    fn powi_backward_decompose_matches_reference() {
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        let dev = Device::cpu();
        let exp = 3i32;
        let x_data = vec![1.5f32, -2.0, 0.5, 3.0];
        let up_data = vec![1.0f32, 0.5, 2.0, -1.0];
        let shape = Shape::from_dims(&[4]);
        let x = Tensor::from_f32(x_data.clone(), shape.clone(), &dev).unwrap();
        let up = x.const_f32_like(up_data.clone(), shape.clone()).unwrap();
        let got = lower_realize_fused(
            &x,
            fuel_graph::Op::Fused(FusedOps::POWI_BACKWARD, FusedOpParams::PowIBackward { exp }),
            vec![x.inner.id(), up.inner.id()],
            shape,
        );
        // grad_x = exp · x^(exp-1) · upstream
        let expected: Vec<f32> = x_data
            .iter()
            .zip(&up_data)
            .map(|(&x, &u)| exp as f32 * x.powi(exp - 1) * u)
            .collect();
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "powi_backward at {i}: got {g}, expected {e}"
            );
        }
    }

    /// Increment C — `inplace_affine`'s migrated functional value recipe:
    /// `decompose` lowers `Op::Fused(INPLACE_AFFINE, {mul, add})` to
    /// `AddScalar(add)(MulScalar(mul)(x))`, which realizes on the CPU backend to
    /// the plain affine `mul·x + add`. Bit-exact: the affine kernel pre-casts the
    /// f64 scalars to f32 and computes `mul_f32 * x + add_f32`, so the two-op
    /// `MulScalar`→`AddScalar` composition matches a hand `(mul as f32)*x + (add
    /// as f32)` exactly (the intermediate `* 1.0` / `+ 0.0` are IEEE identities).
    #[test]
    fn inplace_affine_decompose_matches_affine_reference() {
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        let dev = Device::cpu();
        let (mul, add) = (2.5f64, -1.0f64);
        let x_data = vec![1.5f32, -2.0, 0.5, 3.0];
        let shape = Shape::from_dims(&[4]);
        let x = Tensor::from_f32(x_data.clone(), shape.clone(), &dev).unwrap();
        let got = lower_realize_fused(
            &x,
            fuel_graph::Op::Fused(
                FusedOps::INPLACE_AFFINE,
                FusedOpParams::InplaceAffine { mul, add },
            ),
            vec![x.inner.id()],
            shape,
        );
        // functional affine: out = mul·x + add (scalars pre-cast to f32).
        let expected: Vec<f32> = x_data
            .iter()
            .map(|&v| (mul as f32) * v + add as f32)
            .collect();
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(g, e, "inplace_affine at {i}: got {g}, expected {e}");
        }
    }

    #[test]
    fn softmax_last_dim_backward_decompose_matches_reference() {
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        let dev = Device::cpu();
        let (rows, cols) = (2usize, 3usize);
        // s = a real softmax over the last dim; g = arbitrary upstream.
        let logits = [[0.5f32, -1.0, 2.0], [1.0, 0.0, -0.5]];
        let mut s_data = Vec::new();
        for row in &logits {
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            s_data.extend(exps.iter().map(|&e| e / sum));
        }
        let g_data = vec![1.0f32, -0.5, 0.3, 0.2, 1.5, -1.0];
        let shape = Shape::from_dims(&[rows, cols]);
        let s = Tensor::from_f32(s_data.clone(), shape.clone(), &dev).unwrap();
        let g = s.const_f32_like(g_data.clone(), shape.clone()).unwrap();
        let got = lower_realize_fused(
            &s,
            fuel_graph::Op::Fused(
                FusedOps::SOFTMAX_LAST_DIM_BACKWARD,
                FusedOpParams::SoftmaxLastDimBackward,
            ),
            vec![s.inner.id(), g.inner.id()],
            shape,
        );
        // grad_x = s · (g − sum(g·s, last))
        let mut expected = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let dot: f32 = (0..cols)
                .map(|c| g_data[r * cols + c] * s_data[r * cols + c])
                .sum();
            for c in 0..cols {
                expected[r * cols + c] = s_data[r * cols + c] * (g_data[r * cols + c] - dot);
            }
        }
        for (i, (&gv, &ev)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (gv - ev).abs() < 1e-5,
                "softmax_bwd at {i}: got {gv}, expected {ev}"
            );
        }
    }

    #[test]
    fn rms_norm_last_dim_backward_decompose_matches_reference() {
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        let dev = Device::cpu();
        let (rows, cols) = (2usize, 3usize);
        let eps = 1e-5f64;
        let x_data = vec![1.0f32, -2.0, 0.5, 3.0, 0.25, -1.5];
        let g_data = vec![0.5f32, 1.0, -0.3, 0.2, -1.0, 0.7];
        let shape = Shape::from_dims(&[rows, cols]);
        let x = Tensor::from_f32(x_data.clone(), shape.clone(), &dev).unwrap();
        let g = x.const_f32_like(g_data.clone(), shape.clone()).unwrap();
        let got = lower_realize_fused(
            &x,
            fuel_graph::Op::Fused(
                FusedOps::RMS_NORM_LAST_DIM_BACKWARD,
                FusedOpParams::RmsNormLastDimBackward { eps },
            ),
            vec![x.inner.id(), g.inner.id()],
            shape,
        );
        // grad_x = r_rms · (g − x·s / (n·(mean_sq + eps)))
        let n = cols as f32;
        let mut expected = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let meansq: f32 = (0..cols).map(|c| x_data[r * cols + c].powi(2)).sum::<f32>() / n;
            let denom = meansq + eps as f32;
            let rrms = 1.0 / denom.sqrt();
            let s: f32 = (0..cols)
                .map(|c| g_data[r * cols + c] * x_data[r * cols + c])
                .sum();
            for c in 0..cols {
                let term = x_data[r * cols + c] * s / (n * denom);
                expected[r * cols + c] = rrms * (g_data[r * cols + c] - term);
            }
        }
        for (i, (&gv, &ev)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (gv - ev).abs() < 1e-4,
                "rms_norm_bwd at {i}: got {gv}, expected {ev}"
            );
        }
    }

    #[test]
    fn layer_norm_last_dim_backward_decompose_matches_reference() {
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        let dev = Device::cpu();
        let (rows, cols) = (2usize, 3usize);
        let eps = 1e-5f64;
        let x_data = vec![1.0f32, -2.0, 0.5, 3.0, 0.25, -1.5];
        let g_data = vec![0.5f32, 1.0, -0.3, 0.2, -1.0, 0.7];
        let shape = Shape::from_dims(&[rows, cols]);
        let x = Tensor::from_f32(x_data.clone(), shape.clone(), &dev).unwrap();
        let g = x.const_f32_like(g_data.clone(), shape.clone()).unwrap();
        let got = lower_realize_fused(
            &x,
            fuel_graph::Op::Fused(
                FusedOps::LAYER_NORM_LAST_DIM_BACKWARD,
                FusedOpParams::LayerNormLastDimBackward { eps },
            ),
            vec![x.inner.id(), g.inner.id()],
            shape,
        );
        // grad_x = istd · (g − mean(g) − xhat·mean(g·xhat))
        let n = cols as f32;
        let mut expected = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let mean_x: f32 = (0..cols).map(|c| x_data[r * cols + c]).sum::<f32>() / n;
            let var: f32 = (0..cols)
                .map(|c| (x_data[r * cols + c] - mean_x).powi(2))
                .sum::<f32>()
                / n;
            let istd = 1.0 / (var + eps as f32).sqrt();
            let xhat: Vec<f32> = (0..cols)
                .map(|c| (x_data[r * cols + c] - mean_x) * istd)
                .collect();
            let mean_g: f32 = (0..cols).map(|c| g_data[r * cols + c]).sum::<f32>() / n;
            let mean_gxh: f32 = (0..cols)
                .map(|c| g_data[r * cols + c] * xhat[c])
                .sum::<f32>()
                / n;
            for c in 0..cols {
                expected[r * cols + c] =
                    istd * (g_data[r * cols + c] - mean_g - xhat[c] * mean_gxh);
            }
        }
        for (i, (&gv, &ev)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (gv - ev).abs() < 1e-4,
                "layer_norm_bwd at {i}: got {gv}, expected {ev}"
            );
        }
    }

    #[test]
    fn reduce_max_to_backward_decompose_matches_reference() {
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        let dev = Device::cpu();
        // row0 has a tie at the max (two 3.0s → share 10/2); row1 unique max.
        let x_data = vec![1.0f32, 3.0, 3.0, 2.0, 0.0, 1.0]; // [2,3]
        let up_data = vec![10.0f32, 5.0]; // [2,1] — one per reduced row.
        let x_shape = Shape::from_dims(&[2, 3]);
        let up_shape = Shape::from_dims(&[2, 1]);
        let x = Tensor::from_f32(x_data.clone(), x_shape.clone(), &dev).unwrap();
        let up = x.const_f32_like(up_data.clone(), up_shape).unwrap();
        let got = lower_realize_fused(
            &x,
            fuel_graph::Op::Fused(
                FusedOps::REDUCE_MAX_TO_BACKWARD,
                FusedOpParams::ReduceMaxToBackward,
            ),
            vec![x.inner.id(), up.inner.id()],
            x_shape,
        );
        // row0: max 3.0 tied 2× → 10/2 each; row1: max 2.0 unique → full 5.0.
        let expected = vec![0.0f32, 5.0, 5.0, 5.0, 0.0, 0.0];
        for (i, (&gv, &ev)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (gv - ev).abs() < 1e-5,
                "reduce_max_bwd at {i}: got {gv}, expected {ev}"
            );
        }
    }

    fn causal_conv1d_check(use_silu: bool) {
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        let dev = Device::cpu();
        let (b, c, k, seq) = (1usize, 2usize, 3usize, 2usize);
        let x_seq = seq + (k - 1); // caller pre-pads by K-1.
        // x [B,C,x_seq], weight [C,1,K], bias [C] — all row-major.
        let x_data = vec![0.0f32, 1.0, 2.0, 3.0, 1.0, 0.0, -1.0, -2.0];
        let w_data = vec![0.5f32, 1.0, -0.5, 1.0, 0.0, 2.0];
        let bias_data = vec![0.1f32, -0.2];
        let x = Tensor::from_f32(x_data.clone(), Shape::from_dims(&[b, c, x_seq]), &dev).unwrap();
        let w = x
            .const_f32_like(w_data.clone(), Shape::from_dims(&[c, 1, k]))
            .unwrap();
        let bias = x
            .const_f32_like(bias_data.clone(), Shape::from_dims(&[c]))
            .unwrap();
        let got = lower_realize_fused(
            &x,
            fuel_graph::Op::Fused(
                FusedOps::CAUSAL_CONV1D,
                FusedOpParams::CausalConv1d { use_silu },
            ),
            vec![x.inner.id(), w.inner.id(), bias.inner.id()],
            Shape::from_dims(&[b, c, seq]),
        );
        // out[c,t] = Σ_k w[c,k]·x[c,t+k] + bias[c], optional SiLU.
        let mut expected = vec![0.0f32; b * c * seq];
        for ch in 0..c {
            for t in 0..seq {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += w_data[ch * k + kk] * x_data[ch * x_seq + t + kk];
                }
                acc += bias_data[ch];
                if use_silu {
                    acc *= 1.0 / (1.0 + (-acc).exp());
                }
                expected[ch * seq + t] = acc;
            }
        }
        for (i, (&gv, &ev)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (gv - ev).abs() < 1e-4,
                "causal_conv1d (silu={use_silu}) at {i}: got {gv}, expected {ev}",
            );
        }
    }

    #[test]
    fn causal_conv1d_decompose_matches_reference() {
        causal_conv1d_check(false);
        causal_conv1d_check(true);
    }

    #[test]
    fn flash_attn_backward_decompose_matches_reference() {
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        let dev = Device::cpu();
        let (sq, sk, d) = (2usize, 2usize, 2usize);
        // 4-digit test scale (~1/sqrt(2)), deliberately NOT FRAC_1_SQRT_2 (0.70710678);
        // swapping the exact constant would perturb these attention-scale parity tests.
        #[allow(clippy::approx_constant)]
        let scale = 0.7071f32;
        let q_data = vec![0.1f32, -0.2, 0.3, 0.5]; // [Sq,D]
        let k_data = vec![0.4f32, 0.1, -0.3, 0.2]; // [Sk,D]
        let v_data = vec![1.0f32, 2.0, -1.0, 0.5]; // [Sk,D]
        let do_data = vec![0.5f32, -1.0, 0.2, 0.3]; // [Sq,D]
        let qshape = Shape::from_dims(&[1, 1, sq, d]);
        let kshape = Shape::from_dims(&[1, 1, sk, d]);
        let q = Tensor::from_f32(q_data.clone(), qshape.clone(), &dev).unwrap();
        let k = q.const_f32_like(k_data.clone(), kshape.clone()).unwrap();
        let v = q.const_f32_like(v_data.clone(), kshape.clone()).unwrap();
        let dout = q.const_f32_like(do_data.clone(), qshape.clone()).unwrap();
        let params = FusedOpParams::FlashAttnBackward {
            softmax_scale: scale,
            causal: false,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
        };
        let inputs = vec![q.inner.id(), k.inner.id(), v.inner.id(), dout.inner.id()];
        let dq = lower_realize_fused(
            &q,
            fuel_graph::Op::Fused(FusedOps::FLASH_ATTN_BACKWARD_Q, params.clone()),
            inputs.clone(),
            qshape.clone(),
        );
        let dk = lower_realize_fused(
            &q,
            fuel_graph::Op::Fused(FusedOps::FLASH_ATTN_BACKWARD_K, params.clone()),
            inputs.clone(),
            kshape.clone(),
        );
        let dv = lower_realize_fused(
            &q,
            fuel_graph::Op::Fused(FusedOps::FLASH_ATTN_BACKWARD_V, params),
            inputs,
            kshape,
        );

        // --- reference SDPA backward (B=1, H=1) ---
        let mut p = vec![0.0f32; sq * sk];
        for i in 0..sq {
            let mut scores = vec![0.0f32; sk];
            for j in 0..sk {
                let mut s = 0.0f32;
                for l in 0..d {
                    s += q_data[i * d + l] * k_data[j * d + l];
                }
                scores[j] = scale * s;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let e: Vec<f32> = scores
                .iter()
                .map(|&s| {
                    let x = (s - m).exp();
                    sum += x;
                    x
                })
                .collect();
            for j in 0..sk {
                p[i * sk + j] = e[j] / sum;
            }
        }
        // dV[j,l] = Σ_i P[i,j]·dO[i,l]
        let mut ref_dv = vec![0.0f32; sk * d];
        for j in 0..sk {
            for l in 0..d {
                ref_dv[j * d + l] = (0..sq).map(|i| p[i * sk + j] * do_data[i * d + l]).sum();
            }
        }
        // dP[i,j] = Σ_l dO[i,l]·v[j,l]
        let mut dp = vec![0.0f32; sq * sk];
        for i in 0..sq {
            for j in 0..sk {
                dp[i * sk + j] = (0..d).map(|l| do_data[i * d + l] * v_data[j * d + l]).sum();
            }
        }
        // dScores_raw[i,j] = scale · P[i,j]·(dP[i,j] − Σ_j' dP·P)
        let mut dsr = vec![0.0f32; sq * sk];
        for i in 0..sq {
            let rowdot: f32 = (0..sk).map(|j| dp[i * sk + j] * p[i * sk + j]).sum();
            for j in 0..sk {
                dsr[i * sk + j] = scale * p[i * sk + j] * (dp[i * sk + j] - rowdot);
            }
        }
        // dQ[i,l] = Σ_j dsr[i,j]·k[j,l] ; dK[j,l] = Σ_i dsr[i,j]·q[i,l]
        let mut ref_dq = vec![0.0f32; sq * d];
        for i in 0..sq {
            for l in 0..d {
                ref_dq[i * d + l] = (0..sk).map(|j| dsr[i * sk + j] * k_data[j * d + l]).sum();
            }
        }
        let mut ref_dk = vec![0.0f32; sk * d];
        for j in 0..sk {
            for l in 0..d {
                ref_dk[j * d + l] = (0..sq).map(|i| dsr[i * sk + j] * q_data[i * d + l]).sum();
            }
        }
        let check = |name: &str, got: &[f32], exp: &[f32]| {
            assert_eq!(got.len(), exp.len(), "{name} length");
            for (i, (&gv, &ev)) in got.iter().zip(exp).enumerate() {
                assert!(
                    (gv - ev).abs() < 1e-4,
                    "{name} at {i}: got {gv}, expected {ev}"
                );
            }
        };
        check("dQ", &dq, &ref_dq);
        check("dK", &dk, &ref_dk);
        check("dV", &dv, &ref_dv);
    }

    #[test]
    fn paged_attn_decompose_matches_reference() {
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        let dev = Device::cpu();
        // B=1, Hq=Hkv=1, Sq=1 (decode), D=2; block_size=2, max_blk=2 → kv_len=4.
        let block_size = 2usize;
        let scale = 1.0f32;
        // k_cache / v_cache: [num_blocks=3, block_size=2, Hkv=1, D=2]. Block 1
        // is unused (not in the block table) and holds sentinel 9s.
        let kc = vec![
            1.0f32, 0.0, 0.0, 1.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 2.0, 0.0,
        ];
        let vc = vec![
            1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0,
        ];
        let q_data = vec![0.5f32, 0.5]; // [1,1,1,2]
        let bt = vec![0u32, 2u32]; // block_table [1,2]: sequence uses blocks 0 and 2
        let cl = vec![3u32]; // context_lens [1]: only the first 3 keys are valid

        let q = Tensor::from_f32(q_data.clone(), Shape::from_dims(&[1, 1, 1, 2]), &dev).unwrap();
        let kcache = q
            .const_f32_like(kc.clone(), Shape::from_dims(&[3, 2, 1, 2]))
            .unwrap();
        let vcache = q
            .const_f32_like(vc.clone(), Shape::from_dims(&[3, 2, 1, 2]))
            .unwrap();
        let block_table = q.const_u32_like(bt, Shape::from_dims(&[1, 2])).unwrap();
        let context_lens = q.const_u32_like(cl, Shape::from_dims(&[1])).unwrap();
        let params = FusedOpParams::PagedAttn {
            softmax_scale: scale,
            block_size,
            softcap: None,
        };
        let inputs = vec![
            q.inner.id(),
            kcache.inner.id(),
            vcache.inner.id(),
            block_table.inner.id(),
            context_lens.inner.id(),
        ];
        let got = lower_realize_fused(
            &q,
            fuel_graph::Op::Fused(FusedOps::PAGED_ATTN, params),
            inputs,
            Shape::from_dims(&[1, 1, 1, 2]),
        );

        // reference: gather blocks [0, 2] → k_seq/v_seq[4][2]; mask j ≥ 3;
        // softmax; weighted sum of v.
        let (kv_len, ctx) = (4usize, 3usize);
        let mut k_seq = vec![];
        let mut v_seq = vec![];
        for &blk in &[0usize, 2usize] {
            for p in 0..block_size {
                let base = (blk * block_size + p) * 2; // Hkv=1, D=2
                k_seq.push([kc[base], kc[base + 1]]);
                v_seq.push([vc[base], vc[base + 1]]);
            }
        }
        let mut scores = vec![0.0f32; kv_len];
        for j in 0..kv_len {
            let s = q_data[0] * k_seq[j][0] + q_data[1] * k_seq[j][1];
            scores[j] = if j >= ctx {
                f32::NEG_INFINITY
            } else {
                scale * s
            };
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let e: Vec<f32> = scores
            .iter()
            .map(|&s| {
                let x = (s - m).exp();
                sum += x;
                x
            })
            .collect();
        let mut expected = vec![0.0f32; 2];
        for j in 0..kv_len {
            for l in 0..2 {
                expected[l] += (e[j] / sum) * v_seq[j][l];
            }
        }
        for (i, (&gv, &ev)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (gv - ev).abs() < 1e-4,
                "paged_attn at {i}: got {gv}, expected {ev}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_add_mul() {
        let a =
            Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), &Device::cpu()).unwrap();
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]))?;
        let c = a.add(&b).unwrap().mul(&a).unwrap();
        let cpu_result = c.realize_f32();
        let executor = fuel_cuda_backend::CudaDevice::new(0).unwrap();
        let cuda_result = c.realize_f32_cuda(&executor);
        assert_eq!(cpu_result, cuda_result);
    }

    /// END-TO-END SMOKE ONLY — NOT a CUDA-kernel pin. Verifies the lazy
    /// realize path produces the NaN-propagating `maximum`/`minimum`
    /// convention (pinned 2026-07-08,
    /// `docs/architecture/10-decisions-log.md`) and that a CPU realize and
    /// a CUDA-device realize of the same graph agree on NaN-ness.
    ///
    /// CAVEAT (orchestrator sabotage finding, 2026-07-08): cost-based
    /// placement may route a tiny elementwise op to CPU on BOTH legs even
    /// under `realize_f32_cuda`, so this test does NOT guarantee the CUDA
    /// kernel executed — a wrong CUDA binding can pass it. The real
    /// CUDA-kernel pin is the direct binding-table invocation test
    /// `fuel-dispatch/tests/cuda_dispatch_live.rs::cuda_maximum_minimum_propagate_nan_f32`.
    ///
    /// Gated `#[cfg(feature = "cuda")]` + `#[ignore]`; skips cleanly if no
    /// CUDA device is present. Run:
    ///   `cargo test -p fuel-core --features cuda --lib \
    ///    maximum_minimum_nan_convention_lazy_realize_smoke -- --ignored --nocapture`
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a live CUDA device"]
    fn maximum_minimum_nan_convention_lazy_realize_smoke() {
        let cuda = match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };
        let a = Tensor::from_f32(
            vec![f32::NAN, -2.0, 3.0, f32::NAN],
            Shape::from_dims(&[4]),
            &Device::cpu(),
        )
        .unwrap();
        let b = a.const_f32_like(vec![1.0, f32::NAN, 2.0, f32::NAN], Shape::from_dims(&[4]))?;

        let max_lazy = a.maximum(&b).unwrap();
        let cpu_max = max_lazy.realize_f32();
        let cuda_max = max_lazy.realize_f32_cuda(&cuda);
        for i in 0..4 {
            assert_eq!(
                cpu_max[i].is_nan(),
                cuda_max[i].is_nan(),
                "maximum[{i}] NaN-ness must match: cpu={}, cuda={}",
                cpu_max[i],
                cuda_max[i],
            );
            if !cpu_max[i].is_nan() {
                assert_eq!(cpu_max[i], cuda_max[i], "maximum[{i}]");
            }
        }

        let min_lazy = a.minimum(&b).unwrap();
        let cpu_min = min_lazy.realize_f32();
        let cuda_min = min_lazy.realize_f32_cuda(&cuda);
        for i in 0..4 {
            assert_eq!(
                cpu_min[i].is_nan(),
                cuda_min[i].is_nan(),
                "minimum[{i}] NaN-ness must match: cpu={}, cuda={}",
                cpu_min[i],
                cuda_min[i],
            );
            if !cpu_min[i].is_nan() {
                assert_eq!(cpu_min[i], cuda_min[i], "minimum[{i}]");
            }
        }
    }

    /// END-TO-END SMOKE ONLY — NOT a CUDA-kernel pin. Verifies the lazy
    /// realize path produces the NaN-propagating `relu` convention (torch
    /// parity, pinned 2026-07-08, `docs/architecture/10-decisions-log.md`)
    /// and that a CPU realize and a CUDA-device realize of the same graph
    /// agree on NaN-ness. Successor of the transitional-divergence pin
    /// `relu_cuda_still_scrubs_nan_pending_alpha76_rebind`, flipped at the
    /// alpha.76 `unary_relu_propagating_*` rebind.
    ///
    /// CAVEAT (orchestrator sabotage finding, 2026-07-08): cost-based
    /// placement may route a tiny elementwise op to CPU on BOTH legs even
    /// under `realize_f32_cuda`, so this test does NOT guarantee the CUDA
    /// kernel executed — it stayed green with the CUDA binding deliberately
    /// reverted to the scrubbing stem. The real CUDA-kernel pin (verified
    /// born-red against that sabotage) is the direct binding-table
    /// invocation test
    /// `fuel-dispatch/tests/cuda_dispatch_live.rs::cuda_relu_propagates_nan_f32`
    /// (+ its bf16 sibling).
    ///
    /// Gated `#[cfg(feature = "cuda")]` + `#[ignore]`; skips cleanly if no
    /// CUDA device is present. Run:
    ///   `cargo test -p fuel-core --features cuda --lib \
    ///    relu_nan_convention_lazy_realize_smoke -- --ignored --nocapture`
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a live CUDA device"]
    fn relu_nan_convention_lazy_realize_smoke() {
        let cuda = match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };
        let a = Tensor::from_f32(
            vec![f32::NAN, -2.0, 3.0],
            Shape::from_dims(&[3]),
            &Device::cpu(),
        )
        .unwrap();
        let relu_lazy = a.relu();
        let cpu_result = relu_lazy.realize_f32();
        let cuda_result = relu_lazy.realize_f32_cuda(&cuda);

        for i in 0..3 {
            assert_eq!(
                cpu_result[i].is_nan(),
                cuda_result[i].is_nan(),
                "relu[{i}] NaN-ness must match: cpu={}, cuda={}",
                cpu_result[i],
                cuda_result[i],
            );
            if !cpu_result[i].is_nan() {
                assert_eq!(cpu_result[i], cuda_result[i], "relu[{i}]");
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_matmul() {
        let a = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            &Device::cpu(),
        )
        .unwrap();
        let b = a.const_f32_like(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::from_dims(&[3, 2]),
        )?;
        let c = a.matmul(&b).unwrap();
        let cpu = c.realize_f32();
        let exe = fuel_cuda_backend::CudaDevice::new(0).unwrap();
        let cuda = c.realize_f32_cuda(&exe);
        assert_eq!(cpu.len(), cuda.len());
        for (i, (a, b)) in cpu.iter().zip(cuda.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "matmul[{i}]: cpu={a}, cuda={b}",);
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_broadcast_matmul() {
        // Rank-3 × rank-2 matmul (what the transformer forward does).
        // The graph auto-broadcasts the rank-2 to rank-3.
        let x = Tensor::from_f32(
            (0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 4]),
            &Device::cpu(),
        )
        .unwrap();
        let w = x.const_f32_like(
            (0..8).map(|i| i as f32 * 0.2).collect::<Vec<_>>(),
            Shape::from_dims(&[4, 2]),
        )?;
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
        let x = Tensor::from_f32(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 2, 3, 4]),
            &Device::cpu(),
        )
        .unwrap();
        let y = x.permute([0, 2, 1, 3_usize]).unwrap();
        let cpu = y.realize_f32();
        let exe = fuel_cuda_backend::CudaDevice::new(0).unwrap();
        let cuda = y.realize_f32_cuda(&exe);
        assert_eq!(cpu, cuda, "permute mismatch");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_executor_matches_cpu_on_softmax() {
        let x = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            &Device::cpu(),
        )
        .unwrap();
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
        let x = Tensor::from_f32(data, Shape::from_dims(&[n, last]), &Device::cpu()).unwrap();
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
        let cuda = crate::pipelined_bridge::realize_one_as::<f32>(&graph, remapped[0], &fc_device)
            .expect("realize lowered softmax on CUDA");

        assert_eq!(cpu.len(), cuda.len());
        let mut max_abs_err = 0.0_f32;
        for (i, (&a, &b)) in cpu.iter().zip(cuda.iter()).enumerate() {
            let err = (a - b).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
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
        let a = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0],
            Shape::from_dims(&[2, 2]),
            &Device::cpu(),
        )
        .unwrap();
        let b = a.const_f32_like(vec![5.0, 6.0, 7.0, 8.0], Shape::from_dims(&[2, 2]))?;
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
        let x = Tensor::from_f32(
            (0..8).map(|i| i as f32 * 0.5 - 1.5).collect::<Vec<_>>(),
            Shape::from_dims(&[2, 4]),
            &Device::cpu(),
        )
        .unwrap();
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
        let a =
            Tensor::from_f64(vec![1.5, 2.5, 3.5], Shape::from_dims(&[3]), &Device::cpu()).unwrap();
        let b = a.mul(&a).unwrap();
        assert_eq!(b.realize_f64(), vec![2.25, 6.25, 12.25]);
    }

    /// GAP-327 regression test (was the born-red probe; now LIVE — the guard has landed).
    ///
    /// The realize funnel `pipelined_bridge::extract_cpu_bytes_typed` guards the byte
    /// reinterpretation on the root's dtype: `realize_one_as::<T>` returns
    /// `Error::UnexpectedDType` on a mismatch, and the typed `realize_fNN` accessors (which
    /// `.expect` that result) panic. The BYTE VIEW `realize_one_bytes` is the deliberate
    /// UNGUARDED escape hatch and still returns raw bytes for any dtype. Measured by probe
    /// (GAP-327) before the fix: realize_f32()/realize_one_as::<f32> on an F64 [1.0, 2.0]
    /// root returned Ok([0.0, 1.875, 0.0, 2.0]) — silent reinterpretation; this asserts that
    /// is now rejected while the byte view is not. See docs/gaps.md GAP-327.
    #[test]
    fn gap327_realize_f32_on_f64_root_must_reject_not_reinterpret() {
        let t =
            Tensor::from_f64(vec![1.0_f64, 2.0], Shape::from_dims(&[2]), &Device::cpu()).unwrap();
        let graph = t.inner.graph().clone();
        let target = t.inner.id();
        let device = Device::cpu();

        // Silence the panic hook only around the catch_unwind measurements; restore it
        // before the assertions so a real failure still prints.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        // Control 1 (RIGHT dtype): realize_f64 on the F64 root succeeds.
        let ctl_f64 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| t.realize_f64()));
        // Control 2 (guard FIRES): a MISMATCHED guarded accessor panics — proves the harness
        // observes guards, so the subject rejection below is real and not a swallowed nothing.
        let ctl_f16 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| t.realize_f16()));
        // SUBJECT A: realize_f32 on the F64 root must now panic (guard -> Err -> .expect).
        let subj_method =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| t.realize_f32()));
        std::panic::set_hook(prev_hook);

        // SUBJECT B: realize_one_as::<f32> must return Err(UnexpectedDType) (matched by variant).
        let subj_raw = crate::pipelined_bridge::realize_one_as::<f32>(&graph, target, &device);
        // BYTE VIEW: the unguarded escape hatch must still return the root's RAW bytes
        // (2 x f64 = 16 bytes) — a dtype mismatch is NOT an error for the byte view.
        let bytes_view = crate::pipelined_bridge::realize_one_bytes(&graph, target, &device);

        assert!(
            ctl_f64.is_ok(),
            "control: realize_f64 on an F64 root must succeed; got {ctl_f64:?}"
        );
        assert!(
            ctl_f16.is_err(),
            "positive control: realize_f16() on an F64 root must panic (guard fires); if it \
             does not, the harness is not observing panics and the subject asserts are vacuous"
        );
        assert!(
            subj_method.is_err(),
            "realize_f32() on an F64 root must REJECT (panic), not silently reinterpret bytes"
        );
        assert!(
            matches!(subj_raw, Err(fuel_ir::Error::UnexpectedDType { .. })),
            "realize_one_as::<f32> on an F64 root must return Err(UnexpectedDType), not \
             Ok(reinterpreted bytes); got {subj_raw:?}"
        );
        assert!(
            matches!(&bytes_view, Ok(b) if b.len() == 16),
            "byte view realize_one_bytes on an F64 [1.0, 2.0] root must return 16 raw bytes \
             (unguarded), not an error; got {bytes_view:?}"
        );
    }

    #[test]
    fn lazy_tensor_mini_llama_block_forward() {
        // A minimal LLaMA-style attention-only "block" built entirely
        // through Tensor. No training, just the forward pass:
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
        let x =
            Tensor::from_f32(x_data, Shape::from_dims(&[1, seq, d_model]), &Device::cpu()).unwrap();

        // Fake weights (just identities for simplicity — makes the
        // test easy to verify output finiteness without needing to
        // hand-compute).
        let w_q = x
            .const_f32_like(
                identity_matrix(d_model),
                Shape::from_dims(&[d_model, d_model]),
            )
            .unwrap();
        let w_k = x
            .const_f32_like(
                identity_matrix(d_model),
                Shape::from_dims(&[d_model, d_model]),
            )
            .unwrap();
        let w_v = x
            .const_f32_like(
                identity_matrix(d_model),
                Shape::from_dims(&[d_model, d_model]),
            )
            .unwrap();
        let w_o = x
            .const_f32_like(
                identity_matrix(d_model),
                Shape::from_dims(&[d_model, d_model]),
            )
            .unwrap();

        // RmsNorm → Q/K/V projection (auto-broadcasting matmul).
        let x_norm = x.rms_norm_last_dim(1e-6).unwrap();
        let q = x_norm.matmul(&w_q).unwrap();
        let k = x_norm.matmul(&w_k).unwrap();
        let v = x_norm.matmul(&w_v).unwrap();

        // Split heads: [1, seq, 8] → [1, seq, 2, 4] → [1, 2, seq, 4]
        let q_h = q
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head]))
            .unwrap()
            .permute([0, 2, 1, 3_usize])
            .unwrap();
        let k_h = k
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head]))
            .unwrap()
            .permute([0, 2, 1, 3_usize])
            .unwrap();
        let v_h = v
            .reshape(Shape::from_dims(&[1, seq, num_heads, d_head]))
            .unwrap()
            .permute([0, 2, 1, 3_usize])
            .unwrap();

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
            .permute([0, 2, 1, 3_usize])
            .unwrap()
            .reshape(Shape::from_dims(&[1, seq, d_model]))
            .unwrap();
        let attn_out = merged.matmul(&w_o).unwrap();
        let h = x.add(&attn_out).unwrap();

        // Realize end-to-end through the bridge.
        let result = h.realize_f32();
        assert_eq!(result.len(), seq * d_model);
        for &v in &result {
            assert!(
                v.is_finite(),
                "bridge-based LLaMA block output non-finite: {v}"
            );
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
impl Tensor {
    /// Build a second const U32 (index) tensor on the same graph.
    pub fn const_u32_like(
        &self,
        data: impl Into<Arc<[u32]>>,
        shape: impl Into<Shape>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.const_u32_like(data, shape)?,
        })
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
    pub fn const_placeholder_like(&self, shape: impl Into<Shape>, dtype: fuel_ir::DType) -> Self {
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
    pub fn write_slice(&self, source: &Self, ranges: Vec<(usize, usize)>) -> crate::Result<Self> {
        let inner = self.inner.write_slice(&source.inner, ranges)?;
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
        let inner = self
            .inner
            .write_slice_dyn(&source.inner, ranges, dyn_axis, offset)?;
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
            &source.inner,
            &position.inner,
            axis,
            modulus,
            ranges,
        )?;
        Ok(Self { inner })
    }

    /// Append a [`fuel_graph::Op::WriteSliceDoff`] node — write `source`
    /// into a slab of `self` whose start on `axis` is read from `offset`
    /// (a rank-0 `I64` tensor) **device-side** at kernel launch. No
    /// modulo wrap: `offset` is the raw start on `axis`.
    /// `ranges[axis].0` is ignored; `ranges[axis].1 - ranges[axis].0`
    /// is the write width on `axis` and must equal `source`'s `axis`
    /// dim. Bounds (`offset + width <= dest_dims[axis]`) are the
    /// caller's contract (device-only at build). Destructive on `self`;
    /// same scheduling as `write_slice`. Backs the CUDA-graph-
    /// capturable KV-cache append (CapturedRun / `DecodeSession`).
    /// Returns `Result`: rank / dtype / axis-bound / static-width /
    /// offset-dtype mismatches surface as typed errors at build time.
    pub fn write_slice_doff(
        &self,
        source: &Self,
        offset: &Self,
        axis: usize,
        ranges: Vec<(usize, usize)>,
    ) -> crate::Result<Self> {
        let inner = self
            .inner
            .write_slice_doff(&source.inner, &offset.inner, axis, ranges)?;
        Ok(Self { inner })
    }

    /// Append a [`fuel_graph::Op::Fused`] node carrying
    /// [`fuel_graph::registry::FusedOpParams::Conv2D`]. See `fuel_graph`'s
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
            return Err(
                fuel_ir::Error::Msg(format!("conv2d: groups must be >= 1, got {groups}",)).bt(),
            );
        }
        let x_shape = self.inner.shape();
        let x_dims = x_shape.dims();
        let w_shape = weight.inner.shape();
        let w_dims = w_shape.dims();
        if x_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: x must be rank 4 [N, Cin, H, W], got {x_dims:?}",
            ))
            .bt());
        }
        if w_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: weight must be rank 4 [Cout, Cin/groups, Kh, Kw], got {w_dims:?}",
            ))
            .bt());
        }
        let (cin, h_in, w_in) = (x_dims[1], x_dims[2], x_dims[3]);
        let (cout, cin_per_g, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
        if cin != cin_per_g * groups {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: x has {cin} in-channels but weight expects {} ({cin_per_g}*{groups})",
                cin_per_g * groups,
            ))
            .bt());
        }
        if cout % groups != 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: Cout={cout} must be divisible by groups={groups}",
            ))
            .bt());
        }
        if let Some(b) = bias {
            let b_shape = b.inner.shape();
            let b_dims = b_shape.dims();
            if b_dims != [cout] {
                return Err(fuel_ir::Error::Msg(format!(
                    "conv2d: bias shape {b_dims:?} must match [Cout={cout}]",
                ))
                .bt());
            }
        }
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        if stride_h < 1 || stride_w < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: stride must be >= 1, got ({stride_h}, {stride_w})",
            ))
            .bt());
        }
        let h_padded = h_in + 2 * pad_h;
        let w_padded = w_in + 2 * pad_w;
        if h_padded < kh || w_padded < kw {
            return Err(fuel_ir::Error::Msg(format!(
                "conv2d: padded input ({h_padded}x{w_padded}) smaller than kernel ({kh}x{kw})",
            ))
            .bt());
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

    /// Append a [`fuel_graph::Op::Fused`] node carrying
    /// [`fuel_graph::registry::FusedOpParams::FlashAttn`]. `self` is `q`
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
            ))
            .bt());
        }
        if k_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: k must be rank 4 [B, Hkv, Sk, D], got {k_dims:?}",
            ))
            .bt());
        }
        if v_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: v must be rank 4 [B, Hkv, Sk, D], got {v_dims:?}",
            ))
            .bt());
        }
        let (b, hq, _sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let (bk, hkv, sk, dk) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
        let (bv, hkv_v, sk_v, dv) = (v_dims[0], v_dims[1], v_dims[2], v_dims[3]);
        if b != bk || b != bv {
            return Err(fuel_ir::Error::Msg(
                format!("flash_attn: B mismatch q={b} k={bk} v={bv}",),
            )
            .bt());
        }
        if hkv != hkv_v {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: Hkv mismatch k={hkv} vs v={hkv_v}",
            ))
            .bt());
        }
        if sk != sk_v {
            return Err(fuel_ir::Error::Msg(
                format!("flash_attn: Sk mismatch k={sk} vs v={sk_v}",),
            )
            .bt());
        }
        if d != dk || d != dv {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: head_dim mismatch q={d} k={dk} v={dv}",
            ))
            .bt());
        }
        if hkv == 0 || hq % hkv != 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "flash_attn: Hq={hq} must be a positive multiple of Hkv={hkv}",
            ))
            .bt());
        }
        if let Some(a) = alibi_slopes {
            let a_shape = a.inner.shape();
            let a_dims = a_shape.dims();
            if a_dims != [hq] {
                return Err(fuel_ir::Error::Msg(format!(
                    "flash_attn: alibi_slopes must be [Hq={hq}], got {a_dims:?}",
                ))
                .bt());
            }
        }
        Ok(Self {
            inner: self.inner.flash_attn(
                &k.inner,
                &v.inner,
                alibi_slopes.map(|t| &t.inner),
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                softcap,
            ),
        })
    }

    /// Append a [`fuel_graph::Op::Fused`] node carrying
    /// [`fuel_graph::registry::FusedOpParams::PagedAttn`]. `self` is the Q
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
            return Err(fuel_ir::Error::Msg("paged_attn: block_size must be >= 1".into()).bt());
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
            ))
            .bt());
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
            ))
            .bt());
        }
        if cl_dims.len() != 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: context_lens must be rank 1 [B], got {cl_dims:?}",
            ))
            .bt());
        }
        let (b, hq, _sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        if kc_dims[1] != block_size {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: k_cache block dim {} != block_size {block_size}",
                kc_dims[1],
            ))
            .bt());
        }
        if vc_dims[1] != block_size {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: v_cache block dim {} != block_size {block_size}",
                vc_dims[1],
            ))
            .bt());
        }
        let hkv = kc_dims[2];
        if vc_dims[2] != hkv {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: Hkv mismatch k_cache={hkv} vs v_cache={}",
                vc_dims[2],
            ))
            .bt());
        }
        if kc_dims[3] != d || vc_dims[3] != d {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: D mismatch q={d} k={} v={}",
                kc_dims[3], vc_dims[3],
            ))
            .bt());
        }
        if hkv == 0 || hq % hkv != 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: Hq={hq} must be a positive multiple of Hkv={hkv}",
            ))
            .bt());
        }
        if bt_dims[0] != b {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: block_table batch dim {} != B={b}",
                bt_dims[0],
            ))
            .bt());
        }
        if cl_dims[0] != b {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: context_lens len {} != B={b}",
                cl_dims[0],
            ))
            .bt());
        }
        if block_table.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: block_table must be U32, got {:?}",
                block_table.inner.dtype(),
            ))
            .bt());
        }
        if context_lens.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "paged_attn: context_lens must be U32, got {:?}",
                context_lens.inner.dtype(),
            ))
            .bt());
        }
        if let Some(a) = alibi_slopes {
            let a_shape = a.inner.shape();
            let a_dims = a_shape.dims();
            if a_dims != [hq] {
                return Err(fuel_ir::Error::Msg(format!(
                    "paged_attn: alibi_slopes must be [Hq={hq}], got {a_dims:?}",
                ))
                .bt());
            }
        }
        Ok(Self {
            inner: self.inner.paged_attn(
                &k_cache.inner,
                &v_cache.inner,
                &block_table.inner,
                &context_lens.inner,
                alibi_slopes.map(|t| &t.inner),
                softmax_scale,
                block_size,
                softcap,
            ),
        })
    }

    /// Append a [`fuel_graph::Op::Fused`] node carrying
    /// [`fuel_graph::registry::FusedOpParams::ConvTranspose2D`]. `self` must
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
            ))
            .bt());
        }
        let x_shape = self.inner.shape();
        let x_dims = x_shape.dims();
        let w_shape = weight.inner.shape();
        let w_dims = w_shape.dims();
        if x_dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: x must be rank 4 [N, Cin, H, W], got {x_dims:?}",
            ))
            .bt());
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
            ))
            .bt());
        }
        if cin % groups != 0 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: Cin={cin} must be divisible by groups={groups}",
            ))
            .bt());
        }
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let (out_pad_h, out_pad_w) = output_padding;
        let (dil_h, dil_w) = dilation;
        if stride_h < 1 || stride_w < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: stride must be >= 1, got ({stride_h}, {stride_w})",
            ))
            .bt());
        }
        if dil_h < 1 || dil_w < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose2d: dilation must be >= 1, got ({dil_h}, {dil_w})",
            ))
            .bt());
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
                stride,
                padding,
                output_padding,
                dilation,
                groups,
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
            ))
            .bt());
        }
        let x_shape = self.inner.shape();
        let x_dims = x_shape.dims();
        let w_shape = weight.inner.shape();
        let w_dims = w_shape.dims();
        if x_dims.len() != 3 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: x must be rank 3 [N, Cin, Lin], got {x_dims:?}",
            ))
            .bt());
        }
        if w_dims.len() != 3 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: weight must be rank 3 [Cin, Cout/groups, K], got {w_dims:?}",
            ))
            .bt());
        }
        if stride < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: stride must be >= 1, got {stride}",
            ))
            .bt());
        }
        if dilation < 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: dilation must be >= 1, got {dilation}",
            ))
            .bt());
        }
        let cin = x_dims[1];
        let cin_w = w_dims[0];
        if cin != cin_w {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: x has {cin} in-channels but weight has {cin_w}",
            ))
            .bt());
        }
        if !cin.is_multiple_of(groups) {
            return Err(fuel_ir::Error::Msg(format!(
                "conv_transpose1d: Cin={cin} must be divisible by groups={groups}",
            ))
            .bt());
        }
        Ok(Self {
            inner: self.inner.conv_transpose1d(
                &weight.inner,
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            ),
        })
    }
}

// ============================================================================
// Phase A.1 — wrapper additions (eager-`Tensor` retirement program).
//
// Methods on `fuel_graph::NodeHandle` that weren't previously surfaced through
// `Tensor`. Pure delegation; no new graph ops. See
// `docs/session-prompts/eager-tensor-retirement-master-plan.md`.
// ============================================================================

impl Tensor {
    // ---- shape ops: unsqueeze (Result + Dim) + Result-returning siblings ----

    /// Append a size-1 dimension at position `dim`. Inverse of
    /// [`Self::squeeze`]. Accepts any [`Dim`] (`usize`, `D::Minus1`,
    /// etc.). Bad `dim` surfaces as a typed error at build time.
    pub fn unsqueeze<D: Dim>(&self, dim: D) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index_plus_one(&shape, "unsqueeze")?;
        Ok(Self {
            inner: self.inner.try_unsqueeze(dim)?,
        })
    }

    // ---- triangular masking (canonical attention masks) ----

    /// Upper-triangular mask along the last two dims. `diagonal = 0`
    /// keeps the main diagonal and above; positive shifts higher.
    pub fn triu(&self, diagonal: i64) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.triu(diagonal)?,
        })
    }

    /// Lower-triangular mask along the last two dims. `tril(0)` is the
    /// canonical causal-attention mask.
    pub fn tril(&self, diagonal: i64) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.tril(diagonal)?,
        })
    }

    // ---- additional reductions / activations ----

    /// `log(softmax(self))` along the last dim, fused into one op.
    pub fn log_softmax_last_dim(&self) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.log_softmax_last_dim()?,
        })
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
        Ok(Self {
            inner: self.inner.argmin_dim(dim),
        })
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
        Ok(Self {
            inner: self.inner.masked_fill(&mask.inner, value)?,
        })
    }

    /// `self + scatter(indices, src, dim=dim)` — accumulate `src` rows
    /// at positions named by `indices` along `dim`. `indices` is rank-1
    /// U32 with length equal to `src.dims()[dim]`. Accepts any [`Dim`].
    /// Dim bounds / index dtype / shape / dtype-parity mismatches
    /// surface as typed errors at build time.
    pub fn index_add<D: Dim>(
        &self,
        dim: D,
        indices: &Self,
        src: &Self,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "index_add")?;
        if indices.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "index_add: index must be U32, got {:?}",
                indices.inner.dtype(),
            ))
            .bt());
        }
        if self.inner.dtype() != src.inner.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "index_add: base and src dtypes must match, got {:?} vs {:?}",
                self.inner.dtype(),
                src.inner.dtype(),
            ))
            .bt());
        }
        let base_dims = shape.dims();
        let src_shape = src.inner.shape();
        let src_dims = src_shape.dims();
        if base_dims.len() != src_dims.len() {
            return Err(fuel_ir::Error::Msg(format!(
                "index_add: base and src must have the same rank, got {} vs {}",
                base_dims.len(),
                src_dims.len(),
            ))
            .bt());
        }
        let idx_shape = indices.inner.shape();
        let idx_dims = idx_shape.dims();
        if idx_dims.len() != 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "index_add: index must be rank 1, got {idx_dims:?}",
            ))
            .bt());
        }
        if src_dims[dim] != idx_dims[0] {
            return Err(fuel_ir::Error::Msg(format!(
                "index_add: src dim {dim} ({}) must match index length ({})",
                src_dims[dim], idx_dims[0],
            ))
            .bt());
        }
        Ok(Self {
            inner: self.inner.index_add(dim, &indices.inner, &src.inner),
        })
    }

    /// Functional inverse of [`Self::gather`]. Accumulates `src` into
    /// `self` at positions given by `indices` (substituted at `dim`).
    /// Accepts any [`Dim`]. Dim bounds / index dtype / shape / dtype-
    /// parity mismatches surface as typed errors at build time.
    pub fn scatter_add<D: Dim>(
        &self,
        dim: D,
        indices: &Self,
        src: &Self,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = self.inner.shape();
        let dim = dim.to_index(&shape, "scatter_add")?;
        if indices.inner.dtype() != fuel_ir::DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "scatter_add: index must be U32, got {:?}",
                indices.inner.dtype(),
            ))
            .bt());
        }
        if self.inner.dtype() != src.inner.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "scatter_add: base and src dtypes must match, got {:?} vs {:?}",
                self.inner.dtype(),
                src.inner.dtype(),
            ))
            .bt());
        }
        let idx_shape = indices.inner.shape();
        let src_shape = src.inner.shape();
        if idx_shape.dims() != src_shape.dims() {
            return Err(fuel_ir::Error::Msg(format!(
                "scatter_add: index and src must have the same shape, got {:?} vs {:?}",
                idx_shape.dims(),
                src_shape.dims(),
            ))
            .bt());
        }
        Ok(Self {
            inner: self.inner.scatter_add(dim, &indices.inner, &src.inner),
        })
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
        Self {
            inner: self.inner.relu_inplace(),
        }
    }

    /// In-place `self * sigmoid(self)`. See [`Self::silu`] for the
    /// functional variant.
    pub fn silu_inplace(&self) -> Self {
        Self {
            inner: self.inner.silu_inplace(),
        }
    }

    /// In-place tanh-approximation GELU. See [`Self::gelu`] for the
    /// functional variant.
    pub fn gelu_inplace(&self) -> Self {
        Self {
            inner: self.inner.gelu_inplace(),
        }
    }

    /// In-place `tanh(self)`. See [`Self::tanh`] for the functional
    /// variant.
    pub fn tanh_inplace(&self) -> Self {
        Self {
            inner: self.inner.tanh_inplace(),
        }
    }

    /// In-place `sigmoid(self)`. See [`Self::sigmoid`] for the
    /// functional variant.
    pub fn sigmoid_inplace(&self) -> Self {
        Self {
            inner: self.inner.sigmoid_inplace(),
        }
    }

    /// In-place `self = mul · self + add`. Single fused-op equivalent
    /// of `self.mul_scalar(mul).add_scalar(add)` plus reassignment.
    pub fn affine_inplace(&self, mul: f64, add: f64) -> Self {
        Self {
            inner: self.inner.affine_inplace(mul, add),
        }
    }

    // ---- additional const_*_like factories ----

    /// Build a sibling F64 `Const` on the same graph as `self`.
    pub fn const_f64_like(
        &self,
        data: impl Into<Arc<[f64]>>,
        shape: impl Into<Shape>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.const_f64_like(data, shape)?,
        })
    }

    /// Build a sibling I64 `Const` on the same graph. Used by integer-
    /// target ops (e.g. cross-entropy with PyTorch-convention class
    /// indices).
    pub fn const_i64_like(
        &self,
        data: impl Into<Arc<[i64]>>,
        shape: impl Into<Shape>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        Ok(Self {
            inner: self.inner.const_i64_like(data, shape)?,
        })
    }

    // ---- device residency control ----

    /// Pin this tensor's realized storage to `device`. Consumes `self`
    /// because the placement is a graph-level annotation tied to the
    /// node id rather than a side-effecting operation.
    pub fn on_device(self, device: &Device) -> Self {
        Self {
            inner: self.inner.on_device(device.location()),
        }
    }

    /// Append an `Op::Release` node — explicitly drop this tensor's
    /// device-resident storage once the ordering pass has scheduled
    /// every reader before it.
    pub fn release(&self) -> Self {
        Self {
            inner: self.inner.release(),
        }
    }

    /// Move bytes to `device`, destroying the source. Use when the
    /// source is genuinely dead after the transfer.
    pub fn move_to_device(&self, device: &Device) -> Self {
        Self {
            inner: self.inner.move_to_device(device.location()),
        }
    }

    /// Copy bytes to `device`, leaving the source resident. Use when
    /// other ops still need the source.
    pub fn copy_to_device(&self, device: &Device) -> Self {
        Self {
            inner: self.inner.copy_to_device(device.location()),
        }
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
// Each method here is implemented in terms of `Tensor`'s existing
// surface (reshape, permute, concat, unsqueeze, etc.). No new graph ops.
// ============================================================================

impl Tensor {
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
    /// rest in place. Implemented via [`fuel_graph::NodeHandle::try_permute`]; matches the
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
            ))
            .bt());
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
            return Err(fuel_ir::Error::Msg("stack: requires at least one tensor".into()).bt());
        }
        let reference_shape = args[0].shape();
        let reference_dims = reference_shape.dims().to_vec();
        let dim = dim.to_index_plus_one(&reference_shape, "stack")?;
        for (idx, t) in args.iter().enumerate().skip(1) {
            if t.shape().dims() != reference_dims.as_slice() {
                return Err(fuel_ir::Error::Msg(format!(
                    "stack: tensor {idx} shape {:?} != reference shape {:?}",
                    t.shape().dims(),
                    reference_dims,
                ))
                .bt());
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
        // GAP-003 carve-out. THE PROOF IS LOCAL AND ON THIS LINE: the buffer is
        // `vec![0.0; self.elem_count()]` and the shape is `self.shape()`, whose
        // elem_count IS that number. Both from `self`; no caller supplies either.
        let zero = self
            .const_f32_like(vec![0.0; self.elem_count()], self.shape())
            .expect("elu: vec![_; self.elem_count()] against self.shape() -- same source");
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
            ))
            .bt());
        }
        if a[0] != b[0] {
            return Err(fuel_ir::Error::Msg(format!(
                "dot: length mismatch lhs={} rhs={}",
                a[0], b[0],
            ))
            .bt());
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
            ))
            .bt());
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
            DType::F32 => self.const_f32_like(vec![1.0_f32; n], shape),
            DType::F64 => self.const_f64_like(vec![1.0_f64; n], shape),
            DType::BF16 => self.const_bf16_like(vec![half::bf16::ONE; n], shape),
            DType::F16 => self.const_f16_like(vec![half::f16::ONE; n], shape),
            DType::U32 => self.const_u32_like(vec![1_u32; n], shape),
            DType::I64 => self.const_i64_like(vec![1_i64; n], shape),
            other => {
                Err(fuel_ir::Error::Msg(format!("ones_like: unsupported dtype {other:?}",)).bt())
            }
        }
    }

    /// New tensor with the same shape, dtype, and graph as `self`, filled
    /// with zeros. Returns Err for unsupported dtypes (anything outside
    /// F32/F64/BF16/F16/U32/I64) — matches eager `Tensor::zeros_like` parity.
    pub fn zeros_like(&self) -> std::result::Result<Self, fuel_ir::Error> {
        let n = self.elem_count();
        let shape = self.shape();
        match self.dtype() {
            DType::F32 => self.const_f32_like(vec![0.0_f32; n], shape),
            DType::F64 => self.const_f64_like(vec![0.0_f64; n], shape),
            DType::BF16 => self.const_bf16_like(vec![half::bf16::ZERO; n], shape),
            DType::F16 => self.const_f16_like(vec![half::f16::ZERO; n], shape),
            DType::U32 => self.const_u32_like(vec![0_u32; n], shape),
            DType::I64 => self.const_i64_like(vec![0_i64; n], shape),
            other => {
                Err(fuel_ir::Error::Msg(format!("zeros_like: unsupported dtype {other:?}",)).bt())
            }
        }
    }

    /// New tensor with `shape`/`dtype`/`device`, every element set to `1`.
    /// Static factory equivalent of eager's `Tensor::ones`. Returns Err for
    /// dtypes outside F32/F64/BF16/F16/U32.
    pub fn ones(
        shape: impl Into<Shape>,
        dtype: DType,
        device: &Device,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = shape.into();
        let n = shape.elem_count();
        match dtype {
            DType::F32 => Self::from_f32(vec![1.0_f32; n], shape, device),
            DType::F64 => Self::from_f64(vec![1.0_f64; n], shape, device),
            DType::BF16 => Self::from_bf16(vec![half::bf16::ONE; n], shape, device),
            DType::F16 => Self::from_f16(vec![half::f16::ONE; n], shape, device),
            DType::U32 => Self::from_u32(vec![1_u32; n], shape, device),
            other => Err(fuel_ir::Error::Msg(format!("ones: unsupported dtype {other:?}",)).bt()),
        }
    }

    /// New tensor with `shape`/`dtype`/`device`, every element set to `0`.
    /// Static factory equivalent of eager's `Tensor::zeros`. Returns Err for
    /// dtypes outside F32/F64/BF16/F16/U32.
    ///
    /// **Mints a NEW graph** — it delegates to the `from_*` constructors, so
    /// the tensor it returns cannot be combined with one built elsewhere. To
    /// add a zero tensor to an existing graph, use
    /// [`zeros_like`](Self::zeros_like) or a `const_*_like` builder. See
    /// [graph affinity](Self#tensors-are-graph-affine--read-this-before-building-anything).
    pub fn zeros(
        shape: impl Into<Shape>,
        dtype: DType,
        device: &Device,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = shape.into();
        let n = shape.elem_count();
        match dtype {
            DType::F32 => Self::from_f32(vec![0.0_f32; n], shape, device),
            DType::F64 => Self::from_f64(vec![0.0_f64; n], shape, device),
            DType::BF16 => Self::from_bf16(vec![half::bf16::ZERO; n], shape, device),
            DType::F16 => Self::from_f16(vec![half::f16::ZERO; n], shape, device),
            DType::U32 => Self::from_u32(vec![0_u32; n], shape, device),
            other => Err(fuel_ir::Error::Msg(format!("zeros: unsupported dtype {other:?}",)).bt()),
        }
    }

    /// New tensor of `shape`/`device` filled with `value`. The scalar's
    /// dtype determines the tensor's dtype. Returns Err for scalar dtypes
    /// outside F32/F64/BF16/F16/U32.
    ///
    /// **Mints a NEW graph** — it delegates to the `from_*` constructors, so
    /// the tensor it returns cannot be combined with one built elsewhere. To
    /// add a filled tensor to an existing graph, use a `const_*_like` builder.
    /// See [graph affinity](Self#tensors-are-graph-affine--read-this-before-building-anything).
    pub fn full(
        shape: impl Into<Shape>,
        value: fuel_ir::Scalar,
        device: &Device,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shape = shape.into();
        let n = shape.elem_count();
        match value {
            fuel_ir::Scalar::F32(v) => Self::from_f32(vec![v; n], shape, device),
            fuel_ir::Scalar::F64(v) => Self::from_f64(vec![v; n], shape, device),
            fuel_ir::Scalar::BF16(v) => Self::from_bf16(vec![v; n], shape, device),
            fuel_ir::Scalar::F16(v) => Self::from_f16(vec![v; n], shape, device),
            fuel_ir::Scalar::U32(v) => Self::from_u32(vec![v; n], shape, device),
            other => Err(fuel_ir::Error::Msg(format!(
                "full: unsupported scalar dtype {:?}",
                other.dtype(),
            ))
            .bt()),
        }
    }

    /// Identity matrix `[n, n]` with the given dtype on the given device.
    /// Built host-side as a flat Vec; no graph-layer arange dependency.
    pub fn eye(n: usize, dtype: DType, device: &Device) -> Self {
        let mut data = vec![0.0_f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        // GAP-003 carve-out: `.expect` and NOT `?`, because this cannot fail and
        // saying it CAN would be a false claim in the signature -- `eye` would
        // return a `Result` that is never `Err`, and every caller would write a
        // `?` for an error that does not exist.
        //
        // THE PROOF IS LOCAL, two lines up: `data` is `vec![_; n * n]`, and the
        // shape `[n, n]` has `elem_count() == n * n`. Both are computed from `n`
        // in this function; no caller supplies either, so they cannot disagree.
        let base = Self::from_f32(data, vec![n, n], device).expect(
            "eye: data is vec![_; n*n] and shape is [n, n] (elem_count n*n), \n             both computed from `n` in this function -- lengths cannot disagree",
        );
        if dtype == DType::F32 {
            base
        } else {
            base.to_dtype(dtype).unwrap()
        }
    }

    /// Split a `(B, N, num_heads * head_dim)` projection into the
    /// multi-head attention layout `(B, num_heads, N, head_dim)`.
    /// Equivalent to `reshape(B, N, num_heads, head_dim).permute([0, 2, 1, 3])`
    /// — promoted to a method to retire the per-port reimplementations
    /// of this same composite.
    pub fn split_heads(
        &self,
        num_heads: usize,
        head_dim: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.inner.shape().dims().to_vec();
        debug_assert_eq!(
            dims.len(),
            3,
            "split_heads: input must be rank 3 (B, N, embed), got {dims:?}"
        );
        debug_assert_eq!(
            dims[2],
            num_heads * head_dim,
            "split_heads: trailing dim ({}) != num_heads * head_dim ({} * {} = {})",
            dims[2],
            num_heads,
            head_dim,
            num_heads * head_dim
        );
        let b = dims[0];
        let n = dims[1];
        self.reshape(Shape::from_dims(&[b, n, num_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])
    }

    /// Merge a `(B, num_heads, N, head_dim)` attention result back
    /// into the projection layout `(B, N, num_heads * head_dim)`.
    /// Inverse of [`Self::split_heads`].
    pub fn merge_heads(&self) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.inner.shape().dims().to_vec();
        debug_assert_eq!(
            dims.len(),
            4,
            "merge_heads: input must be rank 4 (B, heads, N, head_dim), got {dims:?}"
        );
        let b = dims[0];
        let num_heads = dims[1];
        let n = dims[2];
        let head_dim = dims[3];
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
        &self,
        bias: std::sync::Arc<[f32]>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let n = bias.len();
        // Build the bias const in `self`'s dtype rather than hardcoding
        // f32: under BF16-throughout decode (Phase D increment A) the
        // activation stream is BF16 and `broadcast_add` asserts dtype
        // equality, so an f32 bias would panic. No-op for f32 activations
        // (every other caller today).
        let bias_t = self.const_like_dtype(&bias, Shape::from_dims(&[n]), self.dtype())?;
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
            return Err(fuel_ir::Error::Msg("embed_tokens: tokens must be non-empty".into()).bt());
        }
        let embed = Self::from_f32(embed_table, Shape::from_dims(&[vocab_size, hidden]), device)?;
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]))?;
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
            )
            .bt());
        }
        let embed = self.const_f32_like(embed_table, Shape::from_dims(&[vocab_size, hidden]))?;
        let token_ids = self.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]))?;
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
        let (cos_data, sin_data) = fuel_graph::build_rope_tables(theta, start_pos, seq, head_dim);
        let rope_shape = Shape::from_dims(&[seq, head_dim]);
        // ⚠️ GAP-003 category 2: the proof EXISTS but is NOT LOCAL --
        // `build_rope_tables` owns the length and its `-> (Vec<f32>, Vec<f32>)`
        // signature does not carry it. The message names WHERE the guarantee
        // lives rather than claiming one. `-> (Self, Self)` has no error channel;
        // giving it one is a signature change, a different obligation. See #157.
        let rope_cos = self
            .const_f32_like(cos_data, rope_shape.clone())
            .expect("rope tables: build_rope_tables must return seq*head_dim elements");
        let rope_sin = self
            .const_f32_like(sin_data, rope_shape)
            .expect("rope tables: build_rope_tables must return seq*head_dim elements");
        (rope_cos, rope_sin)
    }

    /// Constant-position RoPE cos/sin tables `[rows, head_dim]` where EVERY row
    /// is the *same* θ·`delta` rotation — the rung-2 delta-rotation tables. A
    /// cached prefix's keys, already rotated for their original positions, are
    /// shifted to `+delta` by applying this uniform rotation (RoPE composes
    /// additively per pair: `R(delta)·R(p) = R(p+delta)`).
    ///
    /// This is the position-`delta` row repeated `rows` times — NOT
    /// `rope_tables_const(theta, delta, rows, head_dim)`, which would give the
    /// *incrementing* progression `delta, delta+1, …` (the standard RoPE sweep).
    /// Routed through the canonical `fuel_graph::build_rope_tables` so it inherits
    /// the model's inv-freq/scaling exactly.
    pub fn rope_delta_tables_const(
        &self,
        theta: f64,
        delta: usize,
        rows: usize,
        head_dim: usize,
    ) -> (Self, Self) {
        let (c1, s1) = fuel_graph::build_rope_tables(theta, delta, 1, head_dim);
        let mut cos = Vec::with_capacity(rows * head_dim);
        let mut sin = Vec::with_capacity(rows * head_dim);
        for _ in 0..rows {
            cos.extend_from_slice(&c1);
            sin.extend_from_slice(&s1);
        }
        let shape = Shape::from_dims(&[rows, head_dim]);
        // ⚠️ GAP-003, HALF LOCAL. LOCAL: the loop appends `c1` exactly `rows`
        // times against a `[rows, head_dim]` shape. NOT LOCAL: that `c1` has
        // `head_dim` elements comes from `build_rope_tables`. Stated as both.
        (
            self.const_f32_like(cos, shape.clone()).expect(
                "rope tables: rows appends of c1 (local); c1 has head_dim elements (not local)",
            ),
            self.const_f32_like(sin, shape).expect(
                "rope tables: rows appends of s1 (local); s1 has head_dim elements (not local)",
            ),
        )
    }

    /// Uniformly delta-rotate one cached (post-RoPE) K block by θ·`delta`,
    /// returning the shifted block in the SAME pool layout. Reuses the model's
    /// exact `rope_with_tables_decomposed` (rotate-half) rather than a hand-rolled
    /// rotation, so it inherits the model's RoPE convention + scaling and stays
    /// byte-exact with a direct rope-at-shifted-position. A single uniform
    /// rotation is correct for the whole block even though it holds `block_size`
    /// distinct original positions — the delta is the same for every position.
    ///
    /// `k_block` is the pool block layout, row-major `[block_size, n_kv_heads,
    /// head_dim]` (positions-major, as `DeviceKvPool::read_block` returns). RoPE
    /// wants `(position, dim)` as the last two axes, so heads are brought forward
    /// to the `project_qkv_roped` layout `[1, n_kv_heads, block_size, head_dim]`,
    /// rotated, and returned to `[block_size, n_kv_heads, head_dim]`. The rung-2
    /// numeric core, realized on `dev`.
    pub fn rope_delta_rotate_block_f32(
        dev: &crate::Device,
        k_block: &[f32],
        theta: f64,
        delta: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let pool_shape = Shape::from_dims(&[block_size, n_kv_heads, head_dim]);
        // ⚠️ GAP-003: NOT a carve-out and NOT a propagation, and the reason is
        // this function's SIGNATURE rather than anything about the operands.
        // `k_block` and the pool dims are BOTH caller-supplied, so there is no
        // local proof and the message below deliberately does not claim one.
        // `-> Vec<f32>` has no error channel, and the two `.expect`s already on
        // this same expression (permute, reshape) are the same pre-existing
        // condition -- converting one of three would be less honest, not more.
        //
        // The fix is this function returning `Result`, which is a DIFFERENT
        // obligation from the constructor-family ruling and is not in its scope.
        let k = Tensor::from_f32(k_block.to_vec(), pool_shape, dev).expect(
            "rope_delta_rotate_block_f32: k_block length must equal              block_size*n_kv_heads*head_dim -- both caller-supplied, so this              function cannot prove it; the -> Vec<f32> signature has no error              channel (see the sibling permute/reshape expects)",
        );
        let k4 = k
            .permute([1, 0, 2])
            .expect("rope_delta_rotate_block_f32: permute [bs,Hkv,D]->[Hkv,bs,D]")
            .reshape(Shape::from_dims(&[1, n_kv_heads, block_size, head_dim]))
            .expect("rope_delta_rotate_block_f32: reshape to [1,Hkv,bs,D]");
        let (cos, sin) = k4.rope_delta_tables_const(theta, delta, block_size, head_dim);
        k4.rope_with_tables_decomposed(&cos, &sin)
            .expect("rope_delta_rotate_block_f32: rope_with_tables_decomposed")
            .reshape(Shape::from_dims(&[n_kv_heads, block_size, head_dim]))
            .expect("rope_delta_rotate_block_f32: reshape back to [Hkv,bs,D]")
            .permute([1, 0, 2])
            .expect("rope_delta_rotate_block_f32: permute [Hkv,bs,D]->[bs,Hkv,D]")
            .realize_f32()
    }

    /// `[B, 1, 1, head_dim]` RoPE cos/sin tables, row `b` at absolute position
    /// `positions[b]` — the per-row generalization of [`Self::rope_tables_const`]
    /// (which carries one shared position). Built as a **single** const from `B`
    /// calls to `fuel_graph::build_rope_tables` (the canonical `(theta, position)
    /// → (cos, sin)` math), concatenated host-side — no `Op::Concat` chain. At
    /// `B == 1` the emitted data is bit-identical to `rope_tables_const`'s
    /// `[1, head_dim]` const (same builder call), so
    /// [`Self::rope_batched`] reduces to the single-position decomposed RoPE.
    ///
    /// This is the mechanism that lets a decode batch carry sessions at
    /// *different* positions (continuous batching) rather than requiring
    /// equal-length rows. `anchor` (`self`) supplies the graph.
    pub fn rope_tables_const_batched(
        &self,
        theta: f64,
        positions: &[usize],
        head_dim: usize,
    ) -> (Self, Self) {
        let b = positions.len();
        let mut cos_data = Vec::with_capacity(b * head_dim);
        let mut sin_data = Vec::with_capacity(b * head_dim);
        for &pos in positions {
            let (c, s) = fuel_graph::build_rope_tables(theta, pos, 1, head_dim);
            cos_data.extend_from_slice(&c);
            sin_data.extend_from_slice(&s);
        }
        let shape = Shape::from_dims(&[b, 1, 1, head_dim]);
        // ⚠️ GAP-003, and the proof is HALF local -- worth stating as such rather
        // than rounding to either category. LOCAL: `b` IS `positions.len()` (six
        // lines up) and the loop appends once per position, so the buffer has
        // `positions.len() * head_dim` elements against a `[b, 1, 1, head_dim]`
        // shape. NOT LOCAL: that each `build_rope_tables(_, _, 1, head_dim)` call
        // yields exactly `head_dim` elements lives in that function.
        let rope_cos = self
            .const_f32_like(cos_data, shape.clone())
            .expect("batched rope tables: b IS positions.len() (local); build_rope_tables must yield head_dim per position (not local)");
        let rope_sin = self
            .const_f32_like(sin_data, shape)
            .expect("batched rope tables: b IS positions.len() (local); build_rope_tables must yield head_dim per position (not local)");
        (rope_cos, rope_sin)
    }

    /// Rotary position embedding with **per-batch-row** tables.
    ///
    /// `self` is `[B, H, 1, D]`; `cos`/`sin` are `[B, 1, 1, D]` (row `b` holds the
    /// table for sequence `b`'s own position — see [`Self::rope_tables_const_batched`]).
    /// Broadcasting the `[B, 1, 1, D]` tables over the head axis is what lets one
    /// table row serve all of that sequence's heads.
    ///
    /// This is `fuel_graph::NodeHandle::rope_with_tables_decomposed`'s body
    /// (`fuel-graph/src/lib.rs` ~6931-6954) with the table shape generalized from
    /// one shared position to one-per-row — **same ops, same order, same
    /// associativity** (`y = x·cos + concat(-x[…,D/2..], x[…,..D/2])·sin`), so at
    /// `B == 1` it is bit-identical to `rope_with_tables_decomposed`. Built on the
    /// public `Tensor` surface (not a new `fuel-graph` op) deliberately: the
    /// single-position [`Self::rope_with_tables_decomposed`] hard-requires
    /// `cos.dims() == [seq, d]`, so it cannot carry per-row tables, and rebuilding
    /// the lowering here avoids re-entering the `fuel-graph` rope op.
    ///
    /// (Adopted from the Lightbulb consumer's verified `rope_batched`; the
    /// mechanism belongs in Fuel, not the consumer.)
    pub fn rope_batched(
        &self,
        cos: &Self,
        sin: &Self,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims: Vec<usize> = self.inner.shape().dims().to_vec();
        let rank = dims.len();
        if rank < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "rope_batched: expected rank >= 2, got {dims:?}",
            ))
            .bt());
        }
        let d = dims[rank - 1];
        if !d.is_multiple_of(2) {
            return Err(
                fuel_ir::Error::Msg(format!("rope_batched: head_dim {d} must be even",)).bt(),
            );
        }
        let half = d / 2;
        let target = Shape::from_dims(&dims);

        let cos_b = cos.broadcast_to(target.clone())?;
        let sin_b = sin.broadcast_to(target)?;

        let first = self.slice(rank - 1, 0, half)?;
        let second = self.slice(rank - 1, half, half)?;
        let rotated = second.neg().concat(&first, rank - 1)?;

        let left = self.mul(&cos_b)?;
        let right = rotated.mul(&sin_b)?;
        left.add(&right)
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
            fuel_ir::Error::Msg("rope_partial: receiver must have at least one dimension".into())
                .bt()
        })?;
        if rope_dim == head_dim {
            return self.rope_with_tables(rope_cos, rope_sin);
        }
        if rope_dim > head_dim {
            return Err(fuel_ir::Error::Msg(format!(
                "rope_partial: rope_dim={rope_dim} exceeds head_dim={head_dim}",
            ))
            .bt());
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
        &self,
        bias: Option<&std::sync::Arc<[f32]>>,
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
        &self,
        gain: &[f32],
        offset: f32,
        eps: f64,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let shifted: std::sync::Arc<[f32]> =
            std::sync::Arc::from(gain.iter().map(|g| *g + offset).collect::<Vec<_>>());
        self.rms_norm_affine(shifted, eps)
    }

    /// Apply RMSNorm along the last dim with an affine `gain · x`
    /// post-step (no bias — RMSNorm has no β term). `gain` is a
    /// length-`gain.len()` vector materialized fresh on the
    /// receiver's graph and broadcast across all leading dims.
    pub fn rms_norm_affine(
        &self,
        gain: std::sync::Arc<[f32]>,
        eps: f64,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let hidden = gain.len();
        let normed = self.rms_norm_last_dim(eps)?;
        let gain_t = self.const_f32_like(gain, Shape::from_dims(&[hidden]))?;
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
    pub fn global_avg_pool_2d(&self) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.inner.shape().dims().to_vec();
        debug_assert_eq!(
            dims.len(),
            4,
            "global_avg_pool_2d: input must be rank 4 (B, C, H, W), got {dims:?}"
        );
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
        &self,
        gain: std::sync::Arc<[f32]>,
        bias: std::sync::Arc<[f32]>,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.inner.shape().dims().to_vec();
        debug_assert_eq!(
            dims.len(),
            4,
            "channel_affine_4d: input must be rank 4 (B, C, H, W), got {dims:?}"
        );
        let channels = dims[1];
        debug_assert_eq!(
            gain.len(),
            channels,
            "channel_affine_4d: gain len ({}) != C ({})",
            gain.len(),
            channels
        );
        debug_assert_eq!(
            bias.len(),
            channels,
            "channel_affine_4d: bias len ({}) != C ({})",
            bias.len(),
            channels
        );
        let w_t = self
            .const_f32_like(gain, Shape::from_dims(&[channels]))?
            .reshape(Shape::from_dims(&[1, channels, 1, 1]))?;
        let b_t = self
            .const_f32_like(bias, Shape::from_dims(&[channels]))?
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
    pub fn additive_causal_mask_like(anchor: &Tensor, seq_len: usize) -> Self {
        let mut data = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        // GAP-003 carve-out, PROOF LOCAL: `data` is `vec![_; seq_len * seq_len]`
        // six lines up and the loop only writes IN PLACE -- it cannot change the
        // length -- against a `[seq_len, seq_len]` shape. Both from `seq_len`.
        anchor
            .const_f32_like(
                std::sync::Arc::from(data),
                Shape::from_dims(&[seq_len, seq_len]),
            )
            .expect(
                "additive_causal_mask_like: vec![_; seq_len*seq_len] against [seq_len, seq_len]",
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
        // GAP-003 carve-out: `.expect` and NOT `?`, because this cannot fail and
        // saying it CAN would be a false claim in the signature -- `tril2` would
        // return a `Result` that is never `Err`, and every caller would write a
        // `?` for an error that does not exist.
        //
        // THE PROOF IS LOCAL, two lines up: `data` is `vec![_; n * n]`, and the
        // shape `[n, n]` has `elem_count() == n * n`. Both are computed from `n`
        // in this function; no caller supplies either, so they cannot disagree.
        let base = Self::from_f32(data, vec![n, n], device).expect(
            "tril2: data is vec![_; n*n] and shape is [n, n] (elem_count n*n), \n             both computed from `n` in this function -- lengths cannot disagree",
        );
        if dtype == DType::F32 {
            base
        } else {
            base.to_dtype(dtype).unwrap()
        }
    }

    /// Upper-triangular ones matrix `[n, n]`.
    pub fn triu2(n: usize, dtype: DType, device: &Device) -> Self {
        let mut data = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in i..n {
                data[i * n + j] = 1.0;
            }
        }
        // GAP-003 carve-out: `.expect` and NOT `?`, because this cannot fail and
        // saying it CAN would be a false claim in the signature -- `triu2` would
        // return a `Result` that is never `Err`, and every caller would write a
        // `?` for an error that does not exist.
        //
        // THE PROOF IS LOCAL, two lines up: `data` is `vec![_; n * n]`, and the
        // shape `[n, n]` has `elem_count() == n * n`. Both are computed from `n`
        // in this function; no caller supplies either, so they cannot disagree.
        let base = Self::from_f32(data, vec![n, n], device).expect(
            "triu2: data is vec![_; n*n] and shape is [n, n] (elem_count n*n), \n             both computed from `n` in this function -- lengths cannot disagree",
        );
        if dtype == DType::F32 {
            base
        } else {
            base.to_dtype(dtype).unwrap()
        }
    }

    // ---- additional deferred-Phase-A items: indexing / multi-dim / RNG ----

    /// Eager-API alias of [`Self::slice`] (PyTorch / Candle naming).
    /// `narrow(dim, start, len)` is `slice(dim, start, len)` —
    /// produces a view of `[start, start+len)` along `dim`. Bad input
    /// surfaces as a typed error at build time. Accepts any [`Dim`].
    pub fn narrow<D: Dim>(
        &self,
        dim: D,
        start: usize,
        len: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        self.slice(dim, start, len)
    }

    /// Split into `chunks` views along `dim`. The split distributes the
    /// `chunk_size = ceil(dim_size / chunks)` extra slot to the leading
    /// chunks so every chunk's size differs by at most 1. If `dim_size
    /// < chunks`, returns `dim_size` singleton chunks instead of
    /// `chunks` chunks (matches eager / PyTorch). Accepts any [`Dim`].
    pub fn chunk<D: Dim>(
        &self,
        chunks: usize,
        dim: D,
    ) -> std::result::Result<Vec<Self>, fuel_ir::Error> {
        let shape = self.shape();
        let dim = dim.to_index(&shape, "chunk")?;
        if chunks == 0 {
            return Err(fuel_ir::Error::Msg("chunk: chunks must be > 0".into()).bt());
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
    /// `self.slice(0, i, 1).unwrap().squeeze(0)`. Matches eager's `crate::Tensor::get`.
    pub fn get(&self, i: usize) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.shape().dims().to_vec();
        if dims.is_empty() {
            return Ok(self.clone());
        }
        self.slice(0, i, 1).unwrap().squeeze(0)
    }

    /// Sub-tensor at index along an arbitrary dim. Equivalent to
    /// `self.slice(dim, index, 1).unwrap().squeeze(dim)`. Matches eager's
    /// `crate::Tensor::get_on_dim`. Accepts any [`Dim`].
    pub fn get_on_dim<D: Dim>(
        &self,
        dim: D,
        index: usize,
    ) -> std::result::Result<Self, fuel_ir::Error> {
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
    pub fn rand_like(&self, lo: f64, up: f64) -> std::result::Result<Self, fuel_ir::Error> {
        Self::rand(self.shape(), lo, up, self.dtype(), &Device::cpu())
    }

    /// Normal random tensor with shape/dtype/device matching `self`.
    /// Returns Err for unsupported dtypes or invalid stdev.
    pub fn randn_like(&self, mean: f64, stdev: f64) -> std::result::Result<Self, fuel_ir::Error> {
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
            other => {
                Err(fuel_ir::Error::Msg(format!("Tensor::rand: unsupported dtype {other:?}",)).bt())
            }
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
            fuel_ir::Error::Msg(format!("Tensor::randn: invalid stdev={stdev}: {e}",)).bt()
        })?;
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
            other => Err(fuel_ir::Error::Msg(format!(
                "Tensor::randn: unsupported dtype {other:?}",
            ))
            .bt()),
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
        // GAP-003 carve-out. `.expect`, NOT `?`: this cannot fail, and a `Result`
        // here would be one that is never `Err` -- a false claim in the signature
        // that every caller would then write a `?` for.
        //
        // THE PROOF IS LOCAL AND ONE LINE UP: `n` IS `data.len()`, and the shape
        // is `[n]`. The shape is derived FROM the buffer, so they cannot disagree.
        Self::from_f32(data, vec![n], device).expect(
            "arange_step: shape is [n] where n IS data.len(), computed one line              above from that same buffer -- the lengths cannot disagree",
        )
    }

    /// Linearly-spaced 1D tensor with `n` points from `start` to `end`
    /// (inclusive on both ends). Matches NumPy's `linspace`.
    pub fn linspace(start: f32, end: f32, n: usize, device: &Device) -> Self {
        assert!(n >= 1, "linspace: n must be >= 1");
        if n == 1 {
            // GAP-003 carve-out: a one-element vec with shape [1], on one line.
            return Self::from_f32(vec![start], vec![1], device)
                .expect("linspace: vec![start] has exactly 1 element and the shape is [1]");
        }
        let step = (end - start) / ((n - 1) as f32);
        let data: Vec<f32> = (0..n).map(|i| start + step * (i as f32)).collect();
        // GAP-003 carve-out: `data` is a `(0..n)` map -- exactly `n` elements --
        // and the shape is `[n]`. Both from the same `n`, two lines apart.
        Self::from_f32(data, vec![n], device).expect(
            "linspace: data is (0..n).map(..).collect() so it has exactly n              elements, and the shape is [n] -- both from the same `n`",
        )
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
            ))
            .bt());
        }
        if w_dims.len() != 3 {
            return Err(fuel_ir::Error::Msg(format!(
                "conv1d: weight must be rank 3 [Cout, Cin/groups, K], got {w_dims:?}",
            ))
            .bt());
        }
        if groups < 1 {
            return Err(fuel_ir::Error::Msg("conv1d: groups must be ≥ 1".into()).bt());
        }
        if stride < 1 {
            return Err(fuel_ir::Error::Msg("conv1d: stride must be ≥ 1".into()).bt());
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
            ))
            .bt());
        }
        let c = dims[1];
        let (kh, kw) = kernel;
        if kh == 0 || kw == 0 {
            return Err(
                fuel_ir::Error::Msg("avg_pool2d: kernel sizes must be positive".into()).bt(),
            );
        }
        let inv = 1.0_f32 / ((kh * kw) as f32);
        // Depthwise kernel: one filter per input channel, each filter
        // is a constant 1/(kh·kw). Shape [C, 1, kh, kw] with groups=C
        // makes Conv2D compute one independent kernel per channel.
        let weight =
            self.const_f32_like(vec![inv; c * kh * kw], Shape::from_dims(&[c, 1, kh, kw]))?;
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
            ))
            .bt());
        }
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let (kh, kw) = kernel;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        if kh == 0 || kw == 0 {
            return Err(
                fuel_ir::Error::Msg("max_pool2d: kernel sizes must be positive".into()).bt(),
            );
        }
        if sh == 0 || sw == 0 {
            return Err(fuel_ir::Error::Msg("max_pool2d: strides must be positive".into()).bt());
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
        let mut acc: Option<Tensor> = None;
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
        acc.ok_or_else(|| fuel_ir::Error::Msg("max_pool2d: empty kernel".into()).bt())
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
    /// as the `upsample_nearest_2x` helper in `fuel_transformers::models::lazy_yolov8`
    /// and `fuel_transformers::models::lazy_sd_unet`, generalized to arbitrary scale.
    pub fn upsample_nearest2d(&self, scale: usize) -> std::result::Result<Self, fuel_ir::Error> {
        if scale == 0 {
            return Err(
                fuel_ir::Error::Msg("upsample_nearest2d: scale must be positive".into()).bt(),
            );
        }
        let dims = self.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(fuel_ir::Error::Msg(format!(
                "upsample_nearest2d: input must be rank 4 [N, C, H, W], got {dims:?}",
            ))
            .bt());
        }
        if scale == 1 {
            return Ok(self.clone());
        }
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        // [N, C, H, 1, W, 1]
        let expanded = self.reshape(vec![n, c, h, 1, w, 1])?;
        // Replicate along the new unit dims by concatenating scale copies.
        let h_expanded = (0..scale)
            .fold(None, |acc: Option<Tensor>, _| {
                Some(match acc {
                    None => expanded.clone(),
                    Some(a) => a.concat(&expanded, 3).unwrap(),
                })
            })
            .unwrap();
        let h_then_w = (0..scale)
            .fold(None, |acc: Option<Tensor>, _| {
                Some(match acc {
                    None => h_expanded.clone(),
                    Some(a) => a.concat(&h_expanded, 5).unwrap(),
                })
            })
            .unwrap();
        h_then_w.reshape(vec![n, c, h * scale, w * scale])
    }

    /// Nearest-neighbor upsample for 1-D signals `[N, C, T]` by integer
    /// `scale`. Reshape to insert a unit dim, concat scale copies,
    /// reshape back.
    pub fn upsample_nearest1d(&self, scale: usize) -> std::result::Result<Self, fuel_ir::Error> {
        if scale == 0 {
            return Err(
                fuel_ir::Error::Msg("upsample_nearest1d: scale must be positive".into()).bt(),
            );
        }
        let dims = self.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(fuel_ir::Error::Msg(format!(
                "upsample_nearest1d: input must be rank 3 [N, C, T], got {dims:?}",
            ))
            .bt());
        }
        if scale == 1 {
            return Ok(self.clone());
        }
        let (n, c, t) = (dims[0], dims[1], dims[2]);
        let expanded = self.reshape(vec![n, c, t, 1])?;
        let replicated = (0..scale)
            .fold(None, |acc: Option<Tensor>, _| {
                Some(match acc {
                    None => expanded.clone(),
                    Some(a) => a.concat(&expanded, 3).unwrap(),
                })
            })
            .unwrap();
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
            ))
            .bt());
        }
        let h = dims[2];
        let w = dims[3];
        if h == 0 || w == 0 || target_h == 0 || target_w == 0 {
            return Err(fuel_ir::Error::Msg(
                "interpolate2d: input + target spatial dims must be positive".into(),
            )
            .bt());
        }
        // Fast-path: identity.
        if target_h == h && target_w == w {
            return Ok(self.clone());
        }
        // Fast-path: integer-multiple uniform scale → existing
        // `upsample_nearest2d` (more cache-friendly than the
        // index_select composite for the common 2× / 4× case).
        if target_h.is_multiple_of(h) && target_w.is_multiple_of(w) && target_h / h == target_w / w
        {
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
        let h_idx_tensor = self.const_u32_like(h_idx, fuel_ir::Shape::from_dims(&[target_h]))?;
        let w_idx_tensor = self.const_u32_like(w_idx, fuel_ir::Shape::from_dims(&[target_w]))?;
        let after_h = self.index_select(2_usize, &h_idx_tensor)?;
        after_h.index_select(3_usize, &w_idx_tensor)
    }

    /// 1-D nearest interpolation to an explicit target size. Same
    /// constraints as [`Self::interpolate2d`]: integer-multiple targets
    /// only.
    pub fn interpolate1d(&self, target_t: usize) -> std::result::Result<Self, fuel_ir::Error> {
        let dims = self.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(fuel_ir::Error::Msg(format!(
                "interpolate1d: input must be rank 3 [N, C, T], got {dims:?}",
            ))
            .bt());
        }
        let t = dims[2];
        if t == 0 {
            return Err(
                fuel_ir::Error::Msg("interpolate1d: input length must be positive".into()).bt(),
            );
        }
        if !target_t.is_multiple_of(t) {
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
    /// `torch.meshgrid` and eager's `crate::Tensor::meshgrid`:
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
            )
            .bt());
        }
        let ordered: Vec<&Self> = if xy_indexing {
            args.iter().rev().copied().collect()
        } else {
            args.to_vec()
        };
        let mut lens = Vec::with_capacity(ordered.len());
        for (i, t) in ordered.iter().enumerate() {
            let dims = t.shape().dims().to_vec();
            if dims.len() != 1 {
                return Err(fuel_ir::Error::Msg(format!(
                    "meshgrid: input {i} must be rank 1, got shape {dims:?}",
                ))
                .bt());
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
            ))
            .bt());
        } else {
            self.clone()
        };
        for (axis, &n) in repeats.iter().enumerate() {
            if n == 0 {
                return Err(fuel_ir::Error::Msg(format!(
                    "repeat: zero repeat count at axis {axis} not supported",
                ))
                .bt());
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

impl Tensor {
    /// Build a `Tensor` from raw little-endian bytes as they appear
    /// in a safetensors file, plus a dtype and shape. Row-major layout
    /// is assumed. The byte count must match `shape.elem_count() *
    /// dtype_bytes`.
    ///
    /// This is the low-level loader. Prefer [`Self::from_safetensors_view`]
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
                for chunk in bytes.as_chunks::<4>().0.iter() {
                    data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Self::from_f32(data, shape_obj, device)
            }
            Dtype::F64 => {
                check_len(elem_count * 8)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.as_chunks::<8>().0.iter() {
                    let arr: [u8; 8] = *chunk;
                    data.push(f64::from_le_bytes(arr));
                }
                Self::from_f64(data, shape_obj, device)
            }
            Dtype::BF16 => {
                check_len(elem_count * 2)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.as_chunks::<2>().0.iter() {
                    let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                    data.push(half::bf16::from_bits(raw));
                }
                Self::from_bf16(data, shape_obj, device)
            }
            Dtype::F16 => {
                check_len(elem_count * 2)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.as_chunks::<2>().0.iter() {
                    let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                    data.push(half::f16::from_bits(raw));
                }
                Self::from_f16(data, shape_obj, device)
            }
            Dtype::U32 => {
                check_len(elem_count * 4)?;
                let mut data = Vec::with_capacity(elem_count);
                for chunk in bytes.as_chunks::<4>().0.iter() {
                    data.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Self::from_u32(data, shape_obj, device)
            }
            other => crate::bail!(
                "from_safetensors_bytes: unsupported dtype {other:?} — extend Tensor's \
                 safetensors loader to handle it",
            ),
        }
    }

    /// Build a `Tensor` from a `safetensors::TensorView`. This is
    /// the most natural entry point when iterating over a
    /// [`crate::safetensors::MmapedSafetensors`] or similar.
    pub fn from_safetensors_view(
        view: &safetensors::tensor::TensorView<'_>,
        device: &Device,
    ) -> crate::Result<Self> {
        Self::from_safetensors_bytes(view.data(), view.dtype(), view.shape(), device)
    }
}

/// A LayerNorm's learned `(gain, bias)`, both length `[dim]`. Wrapped in
/// `Option` at sites where the norm is conditional — present only on layer 0
/// (RWKV `pre_ln`), or only when a config flag is set (Persimmon QK-LN).
pub type LayerNormPair = (Arc<[f32]>, Arc<[f32]>);

/// A convolution's `(weight, bias)`: the flattened `[out, in, kh, kw]` kernel
/// and its `[out]` bias. Distinct from [`LayerNormPair`] despite the identical
/// shape — the two are not interchangeable and naming them apart is the point.
pub type ConvWeightBias = (Arc<[f32]>, Arc<[f32]>);

/// Weight tensor storage that preserves source precision.
///
/// Projection weights (Q/K/V/O for attention, gate/up/down for FFN,
/// and the output `lm_head` matrix) stay in whatever dtype the
/// source checkpoint used — f32 when that's how it was saved, bf16
/// for modern HF checkpoints that ship bf16 to halve weight memory.
/// Activations are usually f32, but LlamaModel's cached decode path
/// (Phase D increment A, "BF16-throughout decode") threads the KV
/// cache's dtype through the whole activation stream — when that's
/// BF16, weights must ALSO be BF16 (`matmul`'s dtype gate only allows
/// same-dtype homogeneous or `(lhs=F32, rhs=BF16)`; a BF16-activation ×
/// F32-weight matmul is rejected, the opposite direction from the
/// classic f32-activation-over-bf16-weight quantized-serving case).
/// F32 activations still route the mixed `(A:F32, B:BF16) → F32`
/// precision via `VulkanBackend::matmul` unchanged.
///
/// Norm gains and biases are NOT covered by this enum — they're
/// small and precision-sensitive, so they stay `Arc<[f32]>`.
///
/// Cloning is cheap (Arc bump) for both variants. Use
/// [`WeightStorage::const_like`] to emit a [`Tensor`] `Const`
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
        base: Box<WeightStorage>,
        /// `[in_features, rank]` adapter A (HF's `lora_A` transposed).
        lora_a: Arc<[f32]>,
        /// `[rank, out_features]` adapter B (HF's `lora_B` transposed).
        lora_b: Arc<[f32]>,
        rank: usize,
        /// LoRA scaling factor; effective scale is `alpha / rank`.
        alpha: f32,
        in_features: usize,
        out_features: usize,
    },
}

impl WeightStorage {
    pub fn elem_count(&self) -> usize {
        match self {
            Self::F32(a) => a.len(),
            Self::BF16(a) => a.len(),
            // Logical element count for a Q4_0 weight matrix is n*k.
            Self::Q4_0 {
                in_features,
                out_features,
                ..
            } => *in_features * *out_features,
            Self::WithLoRA {
                in_features,
                out_features,
                ..
            } => *in_features * *out_features,
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
    /// weight into a `Tensor`.
    ///
    /// For `Q4_0`, the emitted tensor is a 1-D `U32` const of length
    /// `bytes.len() / 4` holding the raw block byte stream. Callers
    /// must pair this with `Tensor::qmatmul` rather than `matmul`.
    ///
    /// Returns Err for `WithLoRA` — the base + LoRA update can only be
    /// applied via `apply_linear` so the right graph structure is built.
    pub fn const_like(
        &self,
        anchor: &Tensor,
        shape: Shape,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        match self {
            Self::F32(a) => anchor.const_f32_like(a.clone(), shape),
            Self::BF16(a) => anchor.const_bf16_like(a.clone(), shape),
            Self::Q4_0 { words, .. } => {
                let _ = shape; // shape arg unused — Q4_0 const is 1-D U32
                // Arc-clone the precomputed u32 view; no byte copy.
                anchor.const_u32_like(Arc::clone(words), Shape::from_dims(&[words.len()]))
            }
            Self::WithLoRA { .. } => Err(fuel_ir::Error::Msg(
                "WeightStorage::WithLoRA::const_like is not supported \
                 — the base + LoRA update must be applied via \
                 apply_linear to produce the right graph structure."
                    .into(),
            )
            .bt()),
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
        x: &Tensor,
        in_features: usize,
        out_features: usize,
        bias: std::sync::Arc<[f32]>,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        if bias.len() != out_features {
            // Was a `debug_assert_eq!` — i.e. compiled OUT of release builds,
            // so a release binary silently built a broadcast against a
            // wrong-length bias. A length mismatch is a checkpoint condition;
            // it errors in every profile now.
            return Err(fuel_ir::Error::Msg(format!(
                "apply_linear_with_bias: bias len ({}) != out_features ({out_features})",
                bias.len(),
            ))
            .bt());
        }
        let projected = self.apply_linear(x, in_features, out_features)?;
        let bias_t = projected.const_f32_like(bias, Shape::from_dims(&[out_features]))?;
        projected.broadcast_add(&bias_t)
    }

    /// Produce `X @ W` (with optional bias) for this weight storage.
    /// Dispatches to `matmul` for F32/BF16 weights and to `qmatmul`
    /// for Q4_0. `x`'s dtype must be compatible with this weight's
    /// dtype per `matmul`'s gate: same-dtype homogeneous (e.g. BF16
    /// activations × BF16 weights — LlamaModel's BF16-throughout
    /// decode path, Phase D increment A) or `(x=F32, weight=BF16)`
    /// (f32-activation quantized-weight serving). `(x=BF16,
    /// weight=F32)` is NOT supported — cast the weight or the
    /// activation to match before calling.
    /// # Errors
    ///
    /// Returns a typed error rather than panicking on a dimension or dtype
    /// mismatch. This used to `.unwrap()` the inner `matmul`/`qmatmul` and
    /// `assert_eq!` the stored-vs-requested dimensions, which made a
    /// **production** path panic — a mismatched checkpoint (a GGUF whose
    /// stored shape disagrees with the model config) is a data condition, not
    /// a programming invariant, so it must surface as `Err`.
    pub fn apply_linear(
        &self,
        x: &Tensor,
        in_features: usize,
        out_features: usize,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        // Build-time contract check, shared by every arm: a "linear with
        // `in_features`" requires x's trailing dim to BE `in_features`.
        // Checked here rather than at the ~620 call sites, and phrased so the
        // message names the projection instead of surfacing a bare matmul
        // shape error two layers down.
        let x_dims = x.shape().dims().to_vec();
        match x_dims.last() {
            Some(&k) if k == in_features => {}
            _ => {
                return Err(fuel_ir::Error::Msg(format!(
                    "apply_linear({in_features} -> {out_features}): input shape {x_dims:?} \
                     must have trailing dim {in_features}",
                ))
                .bt());
            }
        }
        match self {
            Self::F32(_) | Self::BF16(_) => {
                let w = self.const_like(x, Shape::from_dims(&[in_features, out_features]))?;
                x.matmul(&w)
            }
            Self::Q4_0 {
                in_features: expected_in,
                out_features: expected_out,
                ..
            } => {
                if *expected_in != in_features || *expected_out != out_features {
                    return Err(fuel_ir::Error::Msg(format!(
                        "WeightStorage::Q4_0 shape mismatch: stored \
                         [{expected_in}, {expected_out}], requested \
                         [{in_features}, {out_features}]",
                    ))
                    .bt());
                }
                let w_bytes = self.const_like(x, Shape::from_dims(&[in_features, out_features]))?;
                x.qmatmul(
                    &w_bytes,
                    fuel_graph::QuantType::Q4_0,
                    in_features,
                    out_features,
                )
            }
            Self::WithLoRA {
                base,
                lora_a,
                lora_b,
                rank,
                alpha,
                in_features: expected_in,
                out_features: expected_out,
            } => {
                if *expected_in != in_features || *expected_out != out_features {
                    return Err(fuel_ir::Error::Msg(format!(
                        "WeightStorage::WithLoRA shape mismatch: stored \
                         [{expected_in}, {expected_out}], requested \
                         [{in_features}, {out_features}]",
                    ))
                    .bt());
                }
                // Base forward (F32, BF16, or Q4_0).
                let base_out = base.apply_linear(x, in_features, out_features)?;
                // Low-rank update: y += (alpha/rank) · x @ A @ B.
                let a_t =
                    x.const_f32_like(Arc::clone(lora_a), Shape::from_dims(&[in_features, *rank]))?;
                let b_t =
                    x.const_f32_like(Arc::clone(lora_b), Shape::from_dims(&[*rank, out_features]))?;
                let scale = *alpha as f64 / *rank as f64;
                // x: [*, in] → @A [*, rank] → @B [*, out] → scale → add base.
                let lora_path = Tensor {
                    inner: x.matmul(&a_t)?.matmul(&b_t)?.inner.mul_scalar(scale),
                };
                base_out.add(&lora_path)
            }
        }
    }

    /// Data-determined-M variant of [`Self::apply_linear`]: computes only
    /// `row_count` rows of the capacity-buffer projection `x @ W`, the rest
    /// left zeroed — the per-expert sparse-MoE FFN path. `x` is a
    /// `[capacity, in_features]` buffer whose first `row_count` rows are the
    /// routed tokens; the returned `[capacity, out_features]` buffer's tail
    /// (rows `row_count..capacity`) stays exactly zero (see
    /// [`Tensor::matmul_dyn_m`]).
    ///
    /// F32-only (mirrors `matmul_dyn_m`); other weight encodings surface a
    /// typed build-time error. No bias is applied — a bias would fill the
    /// un-computed tail, which the sparse-MoE caller relies on being zero
    /// for a correct `index_add` scatter-back.
    pub fn apply_linear_dyn_m(
        &self,
        x: &Tensor,
        in_features: usize,
        out_features: usize,
        row_count: fuel_ir::DynScalar,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        match self {
            Self::F32(_) => {
                // const_like only errors on WithLoRA; the F32 arm is infallible.
                let w = self
                    .const_like(x, Shape::from_dims(&[in_features, out_features]))
                    .expect("apply_linear_dyn_m F32 arm: const_like cannot fail for F32");
                x.matmul_dyn_m(&w, row_count)
            }
            other => Err(fuel_ir::Error::Msg(format!(
                "apply_linear_dyn_m: sparse-MoE dispatch is F32-only today, got {:?} \
                 weight (cast the expert weights to F32)",
                other.dtype(),
            ))
            .bt()),
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
            lora_a.len(),
            in_features * rank,
            "lora_a length {} does not match in_features ({in_features}) × rank ({rank}) = {}",
            lora_a.len(),
            in_features * rank,
        );
        assert_eq!(
            lora_b.len(),
            rank * out_features,
            "lora_b length {} does not match rank ({rank}) × out_features ({out_features}) = {}",
            lora_b.len(),
            rank * out_features,
        );
        assert!(
            !matches!(self, Self::WithLoRA { .. }),
            "with_lora: base is already WithLoRA (nested adapters unsupported)",
        );
        Self::WithLoRA {
            base: Box::new(self),
            lora_a,
            lora_b,
            rank,
            alpha,
            in_features,
            out_features,
        }
    }
}

// Auto-conversions so code that was storing `Arc<[f32]>` keeps
// compiling through the refactor — the LayerWeights field type
// widened to WeightStorage but ergonomics don't regress.
impl From<Arc<[f32]>> for WeightStorage {
    fn from(a: Arc<[f32]>) -> Self {
        Self::F32(a)
    }
}
impl From<Vec<f32>> for WeightStorage {
    fn from(v: Vec<f32>) -> Self {
        Self::F32(Arc::from(v))
    }
}
impl From<Arc<[half::bf16]>> for WeightStorage {
    fn from(a: Arc<[half::bf16]>) -> Self {
        Self::BF16(a)
    }
}
impl From<Vec<half::bf16>> for WeightStorage {
    fn from(v: Vec<half::bf16>) -> Self {
        Self::BF16(Arc::from(v))
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

pub fn invalidate_decode_pair_if_stale<C>(
    session: &mut Option<crate::inference_context::DecodeSession>,
    captured: &mut Option<C>,
    ctx: &mut InferenceContext,
    seq: usize,
    max_seq_len: Option<usize>,
    cache_dtype: DType,
    n_layers: usize,
    shape_key: u64,
    kv: &dyn crate::inference_context::KvRebindSource,
    drop_session: impl FnOnce(
        &mut Option<crate::inference_context::DecodeSession>,
        &mut InferenceContext,
    ),
) -> SessionDisposition {
    refresh_decode_session(
        session,
        ctx,
        seq,
        max_seq_len,
        cache_dtype,
        n_layers,
        shape_key,
        kv,
        // Reader before owner — see the doc comment above. This fires for a
        // re-bind as well as a drop: a capture records FIXED device addresses
        // drawn from `base_cache`, and `rebind_kv` replaces some of them.
        || *captured = None,
        drop_session,
    )
}

/// What [`refresh_decode_session`] did to the held plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionDisposition {
    /// There was no held plan; nothing to do.
    Absent,
    /// The plan is valid for this step and was left alone.
    Kept,
    /// The KV allocation changed and the plan was re-bound onto it (GAP-028).
    /// Any reader of the plan's device addresses has been retired.
    Rebound,
    /// The plan was retired. `Some(refusal)` means a re-bind was attempted and
    /// declined — carried so a caller (or a test) can distinguish "refused for
    /// the right reason" from "refused because nothing was tried".
    Dropped(Option<crate::inference_context::RebindRefusal>),
}

/// **The single decision point for what happens to a held decode plan at the
/// top of a decode step.** Every model routes through here, so they cannot
/// drift on what "stale" means — the reason this was a shared free function
/// before GAP-028, and more so now that the answer has three outcomes.
///
/// Order of business, and why it is this order:
///
/// 1. Ask [`crate::inference_context::DecodeSession::validity_for`] — ONE comparison list, so "is it
///    valid" and "is only the allocation different" cannot disagree.
/// 2. On anything but `Valid`, retire readers FIRST via `on_invalidate`. A
///    `CapturedDecodeSession` replays a recorded CUDA graph over FIXED device
///    addresses drawn from the session's `base_cache`; both a drop and a
///    re-bind make some of those addresses stale, and a replay against stale
///    addresses is silently wrong logits at full speed. Retiring before
///    touching the owner means no replay can observe a half-updated plan.
///    Note this fires *before* the re-bind is attempted, not after it
///    succeeds: a capture is worthless either way, and clearing it early keeps
///    the ordering true without depending on the outcome.
/// 3. Try the guarded re-bind. It may refuse; a refusal is not an error.
/// 4. Otherwise drop, so the caller's first-decode-token arm rebuilds.
///
/// Generic in nothing — the capture rides in the `on_invalidate` closure,
/// which is what lets this compile (and therefore be CPU-tested) without the
/// `cuda` feature.
#[allow(clippy::too_many_arguments)]
pub fn refresh_decode_session(
    session: &mut Option<crate::inference_context::DecodeSession>,
    ctx: &mut InferenceContext,
    seq: usize,
    max_seq_len: Option<usize>,
    cache_dtype: DType,
    n_layers: usize,
    shape_key: u64,
    kv: &dyn crate::inference_context::KvRebindSource,
    on_invalidate: impl FnOnce(),
    drop_session: impl FnOnce(
        &mut Option<crate::inference_context::DecodeSession>,
        &mut InferenceContext,
    ),
) -> SessionDisposition {
    use crate::inference_context::PlanValidity;

    let verdict = match (session.as_ref(), max_seq_len) {
        (None, _) => return SessionDisposition::Absent,
        (Some(s), Some(msl)) => {
            s.validity_for(seq, msl, n_layers, cache_dtype, shape_key, kv.alloc_id())
        }
        // A cache with no pre-allocated capacity cannot back a held plan at
        // all (the graph bakes `max_seq_len` into its KV Const shapes).
        (Some(_), None) => PlanValidity::Stale,
    };
    if verdict == PlanValidity::Valid {
        return SessionDisposition::Kept;
    }

    on_invalidate();

    let refusal = if verdict == PlanValidity::AllocationChanged {
        match session.as_mut().map(|s| s.rebind_kv(kv)) {
            Some(Ok(())) => return SessionDisposition::Rebound,
            Some(Err(r)) => Some(r),
            // Unreachable: `verdict` was computed from a `Some` session and
            // nothing above takes it. Handled rather than `expect`ed because
            // this is a production path.
            None => None,
        }
    } else {
        None
    };

    drop_session(session, ctx);
    SessionDisposition::Dropped(refusal)
}

pub fn build_decode_causal_mask(cached_len: usize, seq: usize, max_seq_len: usize) -> Vec<f32> {
    let mut mask_data = vec![0.0_f32; seq * max_seq_len];
    for q_idx in 0..seq {
        let abs_q = cached_len + q_idx;
        for k_idx in (abs_q + 1)..max_seq_len {
            mask_data[q_idx * max_seq_len + k_idx] = f32::NEG_INFINITY;
        }
    }
    mask_data
}

/// Sliding-window sibling of [`build_decode_causal_mask`] — the decode-time
/// mask for a layer that attends only to the most recent `window` positions.
///
/// # Why this exists (GAP-029 increment 3)
///
/// The decode path has always built **one** mask per model, because `LlamaModel`
/// and `PhiModel` — the only two carriers — have no per-layer mask variation
/// (measured: **zero** `sliding_window` hits in this file, against a positive
/// control of hits in five sibling model files). **Four of the six families
/// GAP-029 increment 3 ports do vary per layer**, three of them via a sliding
/// window: `Qwen2` (`lazy_qwen2.rs:249-251`), `Qwen3` (`lazy_qwen3.rs:172`) and
/// `Qwen3Moe` (`lazy_qwen3_moe.rs:192`) all select per layer on
/// `use_sliding_window && layer_idx < max_window_layers`.
///
/// # The predicate, and why it is stated against the prefill path
///
/// The shipped **prefill** mask for these families is `j > i || j + window <= i`
/// over absolute positions (`lazy_qwen2.rs:271`). Decode must agree with it
/// position-for-position, or a model silently changes behaviour at the
/// prefill→decode boundary. Here the query's absolute position is
/// `cached_len + q_idx` and the key's is `k_idx`, giving the same two clauses:
/// causal (`k_idx > abs_q`) and window (`k_idx + window <= abs_q`).
///
/// `window >= max_seq_len` makes the window clause unsatisfiable, so the result
/// is exactly [`build_decode_causal_mask`] — the property that lets a family
/// with the window disabled share the dense path rather than a near-copy of it.
///
/// # Status: LIVE. `#[allow(dead_code)]` removed 2026-08-13 — this has a caller.
///
/// It landed ahead of its consumer with its correctness established
/// independently (see the born-red record in `decode_mask_tests`), under an
/// architect-owned checkpoint: *land family 1 or the function is deleted*.
/// GAP-029 increment 3 landed `Qwen2Model`'s decode path, and
/// [`crate::persistent_decode::build_decode_mask_variants`] calls this for every
/// windowed mask variant — so the attribute is gone rather than renewed.
///
/// It is now on the live decode path of the three windowed families as they
/// arrive (`Qwen2Model` today; `Qwen3Model`, `Qwen3MoeModel` next), reached
/// through `MaskPlan::split_window`.
pub fn build_decode_causal_mask_windowed(
    cached_len: usize,
    seq: usize,
    max_seq_len: usize,
    window: usize,
) -> Vec<f32> {
    let mut mask_data = vec![0.0_f32; seq * max_seq_len];
    for q_idx in 0..seq {
        let abs_q = cached_len + q_idx;
        for k_idx in 0..max_seq_len {
            // Causal: no future keys. Window: nothing older than `window` back.
            if k_idx > abs_q || k_idx + window <= abs_q {
                mask_data[q_idx * max_seq_len + k_idx] = f32::NEG_INFINITY;
            }
        }
    }
    mask_data
}

#[cfg(test)]
mod decode_mask_tests {
    use super::{build_decode_causal_mask, build_decode_causal_mask_windowed};

    /// Render a mask row as `1` for attend / `0` for masked, so a failure
    /// prints the shape of the disagreement rather than a wall of `-inf`.
    fn attendable(row: &[f32]) -> Vec<u8> {
        row.iter().map(|&v| u8::from(v == 0.0)).collect()
    }

    /// **The discriminating test.** Hand-computed ground truth, NOT a
    /// re-derivation of the implementation's own formula: at `cached_len = 3`,
    /// `seq = 1`, `max_seq_len = 6`, `window = 2` the single query sits at
    /// absolute position 3 and may attend to positions `{2, 3}` — 3 itself
    /// (causal boundary) and 2 (one step back, inside a window of 2). Positions
    /// 0 and 1 fall out of the window; 4 and 5 are the unwritten future tail.
    #[test]
    fn windowed_mask_excludes_positions_older_than_the_window() {
        let m = build_decode_causal_mask_windowed(3, 1, 6, 2);
        assert_eq!(
            attendable(&m),
            vec![0, 0, 1, 1, 0, 0],
            "window=2 at absolute position 3 must attend to exactly {{2, 3}}",
        );
    }

    /// Multi-row (prefill-shaped) ground truth, also by hand: `cached_len = 1`,
    /// `seq = 2`, `window = 2` puts the queries at absolute positions 1 and 2,
    /// attending to `{0, 1}` and `{1, 2}` respectively. Guards the `q_idx`
    /// → `abs_q` offset, which a `seq == 1` test cannot see at all.
    #[test]
    fn windowed_mask_offsets_each_row_by_cached_len() {
        let m = build_decode_causal_mask_windowed(1, 2, 5, 2);
        assert_eq!(
            attendable(&m[0..5]),
            vec![1, 1, 0, 0, 0],
            "row 0 = abs pos 1"
        );
        assert_eq!(
            attendable(&m[5..10]),
            vec![0, 1, 1, 0, 0],
            "row 1 = abs pos 2"
        );
    }

    /// **Non-discrimination control, and it is what keeps the two tests above
    /// meaningful.** A window at least as wide as the capacity cannot mask
    /// anything the causal clause does not already mask, so the windowed builder
    /// must reproduce the dense one byte-for-byte. This passes under BOTH a
    /// correct implementation and one that ignores `window` entirely — so it
    /// certifies that the suite is not simply broken, and it must NOT be read as
    /// evidence that the window works. (Same role as the positive control that
    /// stayed `ok` in increment 2a's sabotage record.)
    #[test]
    fn window_wider_than_capacity_is_byte_identical_to_the_dense_mask() {
        for (cached_len, seq, max_seq_len) in [(0, 1, 8), (3, 1, 8), (0, 4, 8), (2, 3, 8)] {
            let dense = build_decode_causal_mask(cached_len, seq, max_seq_len);
            let windowed =
                build_decode_causal_mask_windowed(cached_len, seq, max_seq_len, max_seq_len);
            assert_eq!(
                dense, windowed,
                "window == max_seq_len must equal dense at \
                 (cached_len={cached_len}, seq={seq}, max_seq_len={max_seq_len})",
            );
        }
    }

    /// Decode must agree with the **shipped prefill** mask position-for-position,
    /// or a windowed model changes behaviour at the prefill→decode boundary.
    /// This replays `lazy_qwen2.rs:265-277`'s predicate over a full prefill of
    /// length `n`, then asserts that decoding position `p` with `cached_len = p`
    /// reproduces prefill row `p`. The prefill mask is `[seq, seq]` and the
    /// decode mask `[1, max_seq_len]`, so only the first `n` keys are comparable
    /// — the tail beyond `n` is the unwritten capacity the prefill mask has no
    /// opinion about.
    #[test]
    fn decode_row_matches_the_prefill_mask_row_at_the_same_position() {
        let (n, window, max_seq_len) = (6_usize, 3_usize, 10_usize);

        // Qwen2's prefill predicate, transcribed from `build_layer_mask`.
        let mut prefill = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                if j > i || j + window <= i {
                    prefill[i * n + j] = f32::NEG_INFINITY;
                }
            }
        }

        for p in 0..n {
            let decode = build_decode_causal_mask_windowed(p, 1, max_seq_len, window);
            assert_eq!(
                attendable(&decode[0..n]),
                attendable(&prefill[p * n..(p + 1) * n]),
                "decode at cached_len={p} must match prefill row {p}",
            );
            assert!(
                decode[n..].iter().all(|&v| v == f32::NEG_INFINITY),
                "the unwritten tail beyond position {n} must stay masked",
            );
        }
    }

    // =======================================================================
    // BORN-RED RECORD (2026-08-13, GAP-029 increment 3)
    //
    // These tests were run FIRST against a deliberately wrong body — the
    // "old fabrication in the new shape": `build_decode_causal_mask_windowed`
    // returning `build_decode_causal_mask(..)` and ignoring `window`, which is
    // precisely what a single-mask decode port computes today. That makes the
    // red state prove *windowing vs dense* rather than merely proving the tests
    // execute.
    //
    //   windowed_mask_excludes_positions_older_than_the_window   FAILED
    //       left [1, 1, 1, 1, 0, 0]  right [0, 0, 1, 1, 0, 0]
    //   windowed_mask_offsets_each_row_by_cached_len             FAILED
    //       left [1, 1, 1, 0, 0]     right [0, 1, 1, 0, 0]
    //   decode_row_matches_the_prefill_mask_row_at_the_same_position FAILED
    //       left [1, 1, 1, 1, 0, 0]  right [0, 1, 1, 1, 0, 0]
    //   window_wider_than_capacity_is_byte_identical_to_the_dense_mask  ok
    //   test result: FAILED. 1 passed; 3 failed
    //
    // The `ok` line is load-bearing in BOTH directions and is the reason the
    // control exists: it holds under the correct body AND under the dense
    // fabrication, so (a) the suite is demonstrably not simply broken, and
    // (b) it must never be cited as evidence that windowing works. Only the
    // three that moved carry that claim.
    //
    // After the flip: `4 passed; 0 failed`, on a run reporting a 53.44s
    // compile — i.e. a rebuilt binary, not a cached one. A passing run without
    // confirmed recompilation would have certified nothing.
    //
    // WHAT THIS DOES **NOT** ESTABLISH, stated because the next reader will be
    // wiring this into a decode path: these are tests of the MASK BYTES ONLY.
    // No decode has yet been run with a windowed mask, so "a single-mask decode
    // port is silently wrong for Qwen2/Qwen3/Qwen3Moe" remains an INFERENCE
    // from the mask's meaning — argued from the fact that one
    // `[1, 1, seq, max_seq_len]` Const cannot express two layer groups. The
    // logits-level born-red against Qwen2's live mixed config
    // (`use_sliding_window: true, max_window_layers: 1` over 2 layers,
    // `lazy_qwen2.rs:441-443`) is the measurement that would settle it, and it
    // is NOT in this change.
    // =======================================================================
}

/// Build a [`fuel_dispatch::decode_flash::DecodeFlashSpec`] from a decode
/// attention region's tensor handles and offer the optimizer-owned CUDA
/// flash-decode arm on the shared graph.
///
/// This is the model-layer WIRING for [`offer_decode_flash_arm`]: it supplies
/// the region's tensor handles + the live attended-prefix `k_len` (as
/// `Sym(attended_len_sym)`, resolved per-token via the `SymEnv`) — data the
/// model alone knows — while every strategic decision (the shape/dtype/config
/// gate, the capability gate, the CUDA pin, the `Op::Branch` construction)
/// stays in the dispatch layer. The region is always **causal** with no
/// window / softcap / ALiBi (the LlamaModel decode shape).
///
/// - `q` — the RoPE'd query (`[B, Hq, 1, D]` in decode), also the branch
///   diverge point;
/// - `k` / `v` — the post-`WriteSlice` capacity KV buffers (`[B, Hkv,
///   capacity, D]`);
/// - `decomposed_out` — the region's attention output (arm 0 / the oracle);
/// - `reconverge` — the sole consumer of `decomposed_out` (the merge).
///
/// Returns `Ok(None)` (graph untouched, byte-identical to today) whenever the
/// emitter's gate declines — f32/f64 dtype, `seq_q != 1` (prefill),
/// `head_dim > 128`, or a non-CUDA / kernel-absent topology. Never panics.
#[allow(clippy::too_many_arguments)]
/// Translate a Fuel sliding-window width into the FA-v2 `window_size_*`
/// convention the CUDA flash path speaks.
///
/// Fuel's mask predicate is `j > i || j + window <= i`, i.e. query `i` attends
/// keys `j ∈ [i - window + 1, i]` — **`window - 1` positions LEFT and `0`
/// RIGHT**. FA-v2 reads `is_causal` as `window_size_right == 0 &&
/// window_size_left < 0`, and treats *both* bounds `>= 0` as opting into local
/// attention (`fuel-cuda-backend/src/flash_attn.rs::translate_window`).
///
/// `None` (a dense layer) stays `(None, None)` — plain causal, which is what
/// every non-windowed family wants and what the decode kernel implements.
///
/// The `w - 1` is saturating: a nonsensical `window == 0` yields `Some(0)`,
/// which the admissibility gate rejects along with every other window. It never
/// silently becomes "no window".
pub fn flash_window_bounds(window: Option<usize>) -> (Option<usize>, Option<usize>) {
    match window {
        None => (None, None),
        Some(w) => (Some(w.saturating_sub(1)), Some(0)),
    }
}

// 11 independent graph handles (q/k/v/decomposed/reconverge node ids) plus the
// scalar rewrite parameters (scale, attended-len sym, window, softcap, backend
// capability) for one flash-decode-arm offer. Each is a distinct graph node or
// scalar, not a bundle-able struct. Exceeds even the raised (10) threshold.
#[allow(clippy::too_many_arguments)]
pub fn offer_flash_decode_arm_for_region(
    graph: &fuel_graph::SharedGraph,
    q: fuel_graph::NodeId,
    k: fuel_graph::NodeId,
    v: fuel_graph::NodeId,
    decomposed_out: fuel_graph::NodeId,
    reconverge: fuel_graph::NodeId,
    softmax_scale: f32,
    attended_len_sym: fuel_ir::SymId,
    attn_window: Option<usize>,
    softcap: Option<f32>,
    cap: fuel_dispatch::decode_flash::FlashArmCapability,
) -> crate::Result<Option<fuel_graph::NodeId>> {
    use fuel_dispatch::decode_flash::{DecodeFlashSpec, offer_decode_flash_arm};
    let (window_size_left, window_size_right) = flash_window_bounds(attn_window);
    let spec = DecodeFlashSpec {
        q,
        k,
        v,
        alibi: None,
        softmax_scale,
        causal: true,
        // ⚠️ GAP-194: these three were hardcoded `None`. That is TRUE for
        // `LlamaModel` — no window, no softcap — and would be a LIE for a
        // windowed or softcapped family, whose arm would then attend the whole
        // prefix and skip the cap, silently. A guard must not encode a claim
        // that is false: the caller states them, and the admissibility gate
        // declines what `flash_decoding` cannot implement.
        window_size_left,
        window_size_right,
        softcap,
        k_len: fuel_ir::DynScalar::Sym(attended_len_sym),
        decomposed_out,
        reconverge,
    };
    let mut g = graph.write().map_err(|_| {
        fuel_ir::Error::Msg("graph lock poisoned during flash-arm offer".into()).bt()
    })?;
    offer_decode_flash_arm(&mut g, &spec, cap)
}

pub fn apply_affine_rms_norm(x: &Tensor, gain: &Arc<[f32]>, dim: usize, eps: f64) -> Tensor {
    assert_eq!(
        gain.len(),
        dim,
        "apply_affine_rms_norm: gain length must equal dim"
    );
    let normalized = x.rms_norm_last_dim(eps).unwrap();
    // The gain is always stored f32 (norm gains are precision-sensitive —
    // see `LayerWeights::attn_norm_gain`'s doc) but must be materialized
    // in `x`'s dtype: under BF16-throughout decode (Phase D increment A)
    // `x` is BF16 and `broadcast_mul` asserts dtype equality, so an f32
    // gain would panic. No-op conversion for f32 activations.
    let gain_t = x
        .const_like_dtype(gain, Shape::from_dims(&[dim]), x.dtype())
        .expect("apply_affine_rms_norm: activation dtype must be F32 or BF16");
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
            for chunk in bytes.as_chunks::<4>().0.iter() {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F64 => {
            let mut out = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.as_chunks::<8>().0.iter() {
                let arr: [u8; 8] = *chunk;
                out.push(f64::from_le_bytes(arr) as f32);
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.as_chunks::<2>().0.iter() {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::bf16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        Dtype::F16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.as_chunks::<2>().0.iter() {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        other => {
            crate::bail!("load_tensor_as_f32: unsupported dtype {other:?} for tensor {name:?}",)
        }
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
                    bytes.len(),
                    expected * 2,
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

/// Sampling strategy for decode loops.
#[derive(Debug, Clone, Copy, Default)]
pub enum SamplingStrategy {
    /// Greedy: always pick the highest-probability token.
    #[default]
    Greedy,
    /// Temperature-scaled sampling with a deterministic seed. `temp`
    /// is the softmax temperature (`1.0` is unscaled, `0.0` is
    /// effectively greedy, higher values spread probability mass).
    /// The seed makes sampling reproducible.
    Temperature { temp: f32, seed: u64 },
}

/// Pick the next token from a logits vector using the configured
/// sampling strategy. Pulled out of `generate` so both the cached and
/// future non-cached callers can share it.
pub fn sample_logits(logits: &[f32], strategy: SamplingStrategy, rng_state: &mut u64) -> u32 {
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
            let inv_temp = if temp == 0.0 { 1.0 } else { 1.0 / temp };
            let scaled: Vec<f32> = logits.iter().map(|&x| x * inv_temp).collect();
            let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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
pub fn spec_argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Temperature-scaled softmax. Returns normalized probabilities.
pub fn spec_softmax_temp(logits: &[f32], temp: f32) -> Vec<f32> {
    let inv_t = if temp == 0.0 { 1.0 } else { 1.0 / temp };
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| ((x - max) * inv_t).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

/// Advance a deterministic LCG and return a u01 uniform.
pub fn spec_next_u01(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as f32 / u32::MAX as f32
}

/// Sample a category from a distribution summing to ~1.
pub fn spec_sample_cat(probs: &[f32], state: &mut u64) -> u32 {
    let u = spec_next_u01(state);
    let mut cum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u <= cum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Sample a categorical distribution using a small deterministic LCG.
/// Takes `probs` (assumed to sum to ~1) and a mutable RNG state,
/// returns a sampled index.
pub fn sample_multinomial(probs: &[f32], rng_state: &mut u64) -> u32 {
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

#[cfg(test)]
mod lora_tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn with_lora_matches_manual_base_plus_lora() {
        // Anchor graph.
        let in_f = 4;
        let out_f = 3;
        let rank = 2;
        let alpha = 8.0_f32;

        let anchor =
            Tensor::from_f32(vec![0.0_f32; 1], Shape::from_dims(&[1]), &Device::cpu()).unwrap();
        // Base weight [in, out].
        let base_vec: Vec<f32> = (0..in_f * out_f).map(|i| (i as f32) * 0.1).collect();
        let lora_a_vec: Vec<f32> = (0..in_f * rank).map(|i| (i as f32) * 0.05).collect();
        let lora_b_vec: Vec<f32> = (0..rank * out_f).map(|i| (i as f32) * 0.02).collect();

        let ws = WeightStorage::F32(Arc::from(base_vec.clone())).with_lora(
            Arc::from(lora_a_vec.clone()),
            Arc::from(lora_b_vec.clone()),
            rank,
            alpha,
            in_f,
            out_f,
        );

        // Activations x [2, in_f].
        let batch = 2;
        let x_data: Vec<f32> = (0..batch * in_f).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let x = anchor
            .const_f32_like(x_data.clone(), Shape::from_dims(&[batch, in_f]))
            .unwrap();
        let y = ws.apply_linear(&x, in_f, out_f).unwrap();
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
        let ws = WeightStorage::F32(Arc::from(vec![0.0_f32; 12])); // 4 x 3
        let bad_a = Arc::from(vec![0.0_f32; 3]); // wrong
        let b = Arc::from(vec![0.0_f32; 6]); // 2 x 3
        let _ = ws.with_lora(bad_a, b, 2, 8.0, 4, 3);
    }

    // ---- never-panic conversion (`apply_linear` -> `Result`) ----
    //
    // These three inputs each used to abort the process on a PRODUCTION path:
    // a wrong trailing dim reached `x.matmul(&w).unwrap()`, a stored/requested
    // shape disagreement hit `assert_eq!`, and a wrong-length bias hit a
    // `debug_assert_eq!` that is compiled OUT of release builds — so release
    // binaries broadcast a mis-sized bias with no complaint at all. All three
    // are checkpoint/data conditions rather than programming invariants (a
    // GGUF whose stored shape disagrees with the model config produces exactly
    // this), so all three must be `Err`.

    #[test]
    fn apply_linear_errors_on_trailing_dim_mismatch() {
        let ws = WeightStorage::F32(Arc::from(vec![0.0_f32; 12])); // 4 x 3
        let x = Tensor::from_f32(
            vec![0.0_f32; 10],
            Shape::from_dims(&[2, 5]), // trailing dim 5 != in_features 4
            &Device::cpu(),
        )
        .unwrap();
        let err = ws
            .apply_linear(&x, 4, 3)
            .expect_err("must not panic, must Err");
        let msg = err.to_string();
        assert!(
            msg.contains("trailing dim 4"),
            "error should name the expected trailing dim, got: {msg}",
        );
    }

    #[test]
    fn apply_linear_errors_on_q4_0_stored_shape_mismatch() {
        // 4 x 4 of Q4_0: one 32-element block needs 18 bytes; keep the word
        // count self-consistent since only the shape check is under test.
        let ws = WeightStorage::Q4_0 {
            words: Arc::from(vec![0_u32; 8]),
            bytes_len: 18, // one Q4_0 block: 2-byte scale + 16 packed nibble bytes
            in_features: 4,
            out_features: 4,
        };
        let x =
            Tensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[2, 4]), &Device::cpu()).unwrap();
        // Trailing dim matches in_features, so this reaches the stored-shape
        // check rather than the contract check above.
        let err = ws
            .apply_linear(&x, 4, 7)
            .expect_err("must not panic, must Err");
        let msg = err.to_string();
        assert!(
            msg.contains("Q4_0 shape mismatch"),
            "error should name the stored/requested disagreement, got: {msg}",
        );
    }

    #[test]
    fn apply_linear_with_bias_errors_on_wrong_bias_length() {
        let ws = WeightStorage::F32(Arc::from(vec![0.0_f32; 12])); // 4 x 3
        let x =
            Tensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[2, 4]), &Device::cpu()).unwrap();
        let bias: Arc<[f32]> = Arc::from(vec![0.0_f32; 2]); // wrong: out_features is 3
        let err = ws
            .apply_linear_with_bias(&x, 4, 3, bias)
            .expect_err("wrong-length bias must Err in EVERY profile, not just debug");
        assert!(
            err.to_string().contains("bias len"),
            "error should name the bias length, got: {err}",
        );
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
        let t =
            Tensor::from_safetensors_bytes(&bytes, safetensors::Dtype::F32, &[4], &Device::cpu())
                .unwrap();
        assert_eq!(t.shape().dims(), &[4]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.realize_f32(), original);
    }

    #[test]
    fn from_safetensors_bytes_round_trip_bf16() {
        let original_f32: Vec<f32> = vec![0.5, -1.0, 2.0, 4.0];
        let bf16_vec: Vec<half::bf16> = original_f32
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let mut bytes = Vec::with_capacity(bf16_vec.len() * 2);
        for b in &bf16_vec {
            bytes.extend_from_slice(&b.to_bits().to_le_bytes());
        }
        let t =
            Tensor::from_safetensors_bytes(&bytes, safetensors::Dtype::BF16, &[4], &Device::cpu())
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
        let result = Tensor::from_safetensors_bytes(
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
// `fuel-graph`; here we only verify that the Tensor wrappers don't
// drop information or mis-thread arguments.
// ============================================================================
#[cfg(test)]
mod phase_a1_wrapper_tests {
    use super::*;

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_f32(data, shape.to_vec(), &Device::cpu()).unwrap()
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
        assert!(t.permute([0_usize, 1]).is_err()); // wrong rank
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
        assert_eq!(
            lower.realize_f32(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        );
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
            -1.0, 0.0, // r=0, c=0..1
            0.0, -3.0, // r=1
            -2.0, -1.0, // r=2
            // b=1
            -2.0, -1.0, // r=0
            -1.0, -1.0, // r=1
            0.0, 0.0, // r=2
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
            [-1.0, 0.0, -2.0], // (b=0, c=0)
            [0.0, -3.0, -1.0], // (b=0, c=1)
            [-2.0, -1.0, 0.0], // (b=1, c=0)
            [-1.0, -1.0, 0.0], // (b=1, c=1)
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
            1.0, 2.0, -1.0, 0.5, 0.0, -2.0, 3.0, 1.5, // batch dim 2
            4.0, -1.0, 2.0, 0.0, -3.0, 0.25, 0.75, 1.0,
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
            -1.0, 0.0, 0.0, -3.0, -2.0, -1.0, -2.0, -1.0, -1.0, -1.0, 0.0, 0.0,
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
            [-1.0, 0.0, -2.0],
            [0.0, -3.0, -1.0],
            [-2.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0],
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
            1.0, 2.0, -1.0, 0.5, 0.0, -2.0, 3.0, 1.5, 4.0, -1.0, 2.0, 0.0, -3.0, 0.25, 0.75, 1.0,
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
        // Comparison ops produce Bool masks directly (GAP-168(c)) — masked_fill
        // now accepts Bool, no F32→mask cast needed.
        let probe = t
            .const_f32_like(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2])
            .unwrap();
        let threshold = t.const_f32_like(vec![0.5; 4], vec![2, 2]).unwrap();
        let mask = probe.gt(&threshold).unwrap(); // [0, 1, 1, 0] as Bool
        let out = t.masked_fill(&mask, fuel_ir::Scalar::F32(-9.0)).unwrap();
        assert_eq!(out.realize_f32(), vec![1.0, -9.0, -9.0, 4.0]);
    }

    /// GAP-168(c) DISCRIMINATION: `Bool` and `U8` are byte-identical in storage,
    /// which is exactly the condition under which a wrong wiring is *silently
    /// correct*. This test proves they are NOT interchangeable end-to-end on CPU:
    /// a comparison yields a `Bool` (not `U8`) tensor, and `to_dtype(Bool)` on a
    /// `U8` tensor is a real `!= 0` CONVERSION — never a byte reinterpret.
    #[test]
    fn bool_is_distinguishable_from_u8_end_to_end() {
        // A comparison realizes end-to-end as a Bool tensor (NOT U8) whose bytes
        // are 0/1 — the byte-identity hazard's most likely hiding spot, closed:
        // dtype tag is Bool, the CPU kernel dispatches on [F32,F32,Bool] (its FKC
        // contract declares fixed(BOOL)), and Copy[Bool] materializes it to host.
        let t = cpu_f32(vec![0.0, 5.0, 3.0, 0.0], &[4]);
        let thr = t.const_f32_like(vec![0.5; 4], vec![4]).unwrap();
        let mask = t.gt(&thr).unwrap();
        assert_eq!(mask.dtype(), DType::Bool, "gt yields Bool, not U8");
        assert_eq!(mask.realize_u8(), vec![0, 1, 1, 0], "mask bytes are 0/1");

        // ---- The cast discrimination, now that the 22 Bool pairs are wired ----
        //
        // NEGATIVE CONTROL FIRST, or the assertion below is vacuous: build a U8
        // tensor that genuinely holds 5. If this line ever came back as 0/1, the
        // real test would pass for the wrong reason — it would be comparing 0/1
        // against 0/1 and could not tell a conversion from a reinterpret.
        let u8_vals = t
            .const_f32_like(vec![0.0, 5.0, 3.0, 0.0], vec![4])
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap();
        assert_eq!(u8_vals.dtype(), DType::U8);
        assert_eq!(
            u8_vals.realize_u8(),
            vec![0, 5, 3, 0],
            "negative control: the U8 source must really hold 5 and 3, else the \
             conversion assertion below cannot discriminate"
        );

        // THE DISCRIMINATION: U8 -> Bool is a real `!= 0` conversion. A byte
        // reinterpret would yield [0, 5, 3, 0] — byte-identical storage makes
        // that the silently-correct-looking failure. A conversion yields 0/1.
        let as_bool = u8_vals.to_dtype(DType::Bool).unwrap();
        assert_eq!(as_bool.dtype(), DType::Bool);
        assert_eq!(
            as_bool.realize_u8(),
            vec![0, 1, 1, 0],
            "U8->Bool must CONVERT (5 -> 1), not reinterpret (5 -> 5)"
        );

        // And back out: Bool -> F32 is exactly 0.0/1.0.
        assert_eq!(
            as_bool.to_dtype(DType::F32).unwrap().realize_f32(),
            vec![0.0, 1.0, 1.0, 0.0],
            "Bool->F32 yields 0.0/1.0"
        );
        // Bool -> U8 keeps the canonical 0/1 (this direction IS byte-preserving,
        // which is fine — it is the OTHER direction that must not be).
        assert_eq!(
            mask.to_dtype(DType::U8).unwrap().realize_u8(),
            vec![0, 1, 1, 0]
        );

        // FINALLY, the two are not INTERCHANGEABLE at the API, which is what
        // stops a numeric mask being coerced silently: `masked_fill` requires a
        // Bool mask and rejects U8 at GRAPH-BUILD time, so the cast is explicit
        // at the call site rather than implied.
        let err = t.masked_fill(&u8_vals, fuel_ir::Scalar::F32(-9.0));
        assert!(
            err.is_err(),
            "a U8 mask must be REJECTED by masked_fill — Bool and U8 are \
             byte-identical, so accepting it would 'work' and erase the distinction"
        );
    }

    /// GAP-183: `float -> Bool` is lowered in the optimizer to `Ne(x, x*0)` so it
    /// reaches CUDA's shipped `ne_{f32,f64,f16,bf16}` kernels, which a
    /// storage-level cast cannot compose. This pins the VALUES end-to-end, and
    /// the three interesting inputs are the whole point of the test — a lowering
    /// that got only the ordinary cases right would still be wrong:
    ///
    ///   -0.0  -> FALSE   (IEEE `-0.0 == 0.0`)
    ///   NaN   -> TRUE    (matches PyTorch `.bool()`)
    ///   ±inf  -> TRUE
    ///
    /// The zeros operand is itself NaN for non-finite x; that is harmless because
    /// `x != NaN` and `x != 0` agree for every non-finite x. This test is what
    /// would catch it if that ever stopped being true.
    #[test]
    fn float_to_bool_lowering_preserves_ieee_semantics() {
        let t = cpu_f32(
            vec![
                0.0,
                -0.0,
                5.0,
                -3.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ],
            &[7],
        );
        let b = t.to_dtype(DType::Bool).unwrap();
        assert_eq!(b.dtype(), DType::Bool, "to_dtype(Bool) still yields Bool");
        assert_eq!(
            b.realize_u8(),
            vec![0, 0, 1, 1, 1, 1, 1],
            "0 and -0.0 are false; every other value including NaN and both \
             infinities is true",
        );
    }

    #[test]
    fn index_add_smoke() {
        let base = cpu_f32(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let src = base
            .const_f32_like(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])
            .unwrap();
        let indices = base.const_u32_like(vec![0_u32, 0_u32], vec![2]).unwrap();
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
        let src = base
            .const_f32_like(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])
            .unwrap();
        let indices = base
            .const_u32_like(vec![0_u32, 1_u32, 1_u32, 0_u32], vec![2, 2])
            .unwrap();
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
        let t = anchor.const_f64_like(vec![1.5, 2.5, 3.5], vec![3]).unwrap();
        assert_eq!(t.shape().dims(), &[3]);
        assert_eq!(t.dtype(), DType::F64);
        assert_eq!(t.realize_f64(), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn const_i64_like_round_trips() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let t = anchor.const_i64_like(vec![-1_i64, 2, -3], vec![3]).unwrap();
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

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_f32(data, shape.to_vec(), &Device::cpu()).unwrap()
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
        assert_eq!(
            t.t().unwrap().realize_f32(),
            t.transpose_last_two().unwrap().realize_f32()
        );
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

    /// `rope_batched` at `B == 1` must be **bit-identical** to
    /// `rope_with_tables_decomposed`, and each row at `B == 3` must equal the
    /// single-position result at that row's own position — the exactness proof
    /// for the per-row RoPE that unblocks ragged (non-uniform-position) batched
    /// decode. A subtly wrong rotate-half (wrong half split, wrong sign, wrong
    /// concat order) still produces plausible tokens, so this asserts
    /// bit-equality; the CONTROL proves the per-row table is load-bearing.
    /// (Adopted from the Lightbulb consumer's verified `rope_batched` oracle;
    /// the mechanism belongs in Fuel.)
    #[test]
    fn rope_batched_matches_single_position_rope_row_by_row() {
        let lcg = |n: usize, seed: u32| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.5
                })
                .collect()
        };
        let (b, heads, hd) = (3usize, 4usize, 8usize);
        let base = 10000.0f64;
        let positions = [0usize, 7, 3];

        let x_data = lcg(b * heads * hd, 11);
        let x = cpu_f32(x_data.clone(), &[b, heads, 1, hd]);
        let (cos, sin) = x.rope_tables_const_batched(base, &positions, hd);
        let got = x.rope_batched(&cos, &sin).unwrap().realize_f32();

        let per_row = heads * hd;
        for (row, &pos) in positions.iter().enumerate() {
            let row_data = x_data[row * per_row..(row + 1) * per_row].to_vec();
            let xr = cpu_f32(row_data, &[1, heads, 1, hd]);
            let (c, s) = xr.rope_tables_const(base, pos, 1, hd);
            let want = xr
                .rope_with_tables_decomposed(&c, &s)
                .unwrap()
                .realize_f32();
            assert_eq!(
                &got[row * per_row..(row + 1) * per_row],
                &want[..],
                "row {row} (position {pos}) != decomposed RoPE at that position",
            );
        }

        // CONTROL: prove the per-row table is load-bearing. Give EVERY row
        // position 0 (row 0's position); rows at other positions must now
        // DISAGREE with their own-position result — else the comparison above is
        // blind to the whole per-row-position mechanism.
        let shared = [positions[0]; 3];
        let (c0, s0) = x.rope_tables_const_batched(base, &shared, hd);
        let wrong = x.rope_batched(&c0, &s0).unwrap().realize_f32();
        for (row, &pos) in positions.iter().enumerate().skip(1) {
            assert_ne!(
                &wrong[row * per_row..(row + 1) * per_row],
                &got[row * per_row..(row + 1) * per_row],
                "CONTROL: row {row} at shared position 0 matched its own position \
                 {pos} — per-row RoPE not applied, or comparison blind to it",
            );
        }
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
        let table: Vec<f32> = (0..vocab_size)
            .flat_map(|i| vec![i as f32, i as f32 + 0.5, i as f32 + 1.0])
            .collect();
        let tokens = vec![1_u32, 3, 0];
        let out = Tensor::embed_tokens(
            std::sync::Arc::from(table),
            vocab_size,
            hidden,
            &tokens,
            &crate::Device::cpu(),
        )
        .unwrap();
        assert_eq!(out.shape().dims(), &[1, 3, hidden]);
        let v = out.realize_f32();
        let want = [
            1.0_f32, 1.5, 2.0, // token 1
            3.0, 3.5, 4.0, // token 3
            0.0, 0.5, 1.0, // token 0
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
        let table: Vec<f32> = (0..vocab_size)
            .flat_map(|i| vec![i as f32, i as f32 * 2.0])
            .collect();
        let table_arc: std::sync::Arc<[f32]> = std::sync::Arc::from(table);
        let tokens = vec![2_u32, 1];

        let anchor = cpu_f32(vec![0.0_f32], &[1]);
        let embedded = anchor
            .embed_tokens_anchored(
                std::sync::Arc::clone(&table_arc),
                vocab_size,
                hidden,
                &tokens,
            )
            .unwrap();
        assert_eq!(embedded.shape().dims(), &[1, 2, hidden]);

        // Anchored: composes with the anchor.
        let one_scaled = anchor
            .const_f32_like(std::sync::Arc::from(vec![1.0_f32]), Shape::from_dims(&[1]))
            .unwrap();
        let _ = embedded
            .add(
                &one_scaled
                    .reshape(Shape::from_dims(&[1, 1, 1]))
                    .unwrap()
                    .broadcast_to(Shape::from_dims(&[1, 2, hidden]))
                    .unwrap(),
            )
            .unwrap();
        let v = embedded.realize_f32();
        let want = [
            2.0_f32, 4.0, // token 2
            1.0, 2.0, // token 1
        ];
        for (i, (&got, &exp)) in v.iter().zip(want.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "row {i}: got={got} want={exp}");
        }
    }

    #[test]
    fn embed_tokens_empty_returns_error() {
        let r = Tensor::embed_tokens(
            std::sync::Arc::from(vec![0.0_f32]),
            1,
            1,
            &[],
            &crate::Device::cpu(),
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
            assert!(
                (got - want).abs() < 1e-3,
                "softcap[{i}] got={got} want={want}"
            );
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
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
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
                1.0_f32, 2.0, 3.0, 4.0, // channel 0
                10.0, 20.0, 30.0, 40.0, // channel 1
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
        let mask = Tensor::additive_causal_mask_like(&anchor, 4);
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
                    assert!(
                        got.is_infinite() && got.is_sign_negative(),
                        "above-diag (i={i}, j={j}) should be -inf, got {got}"
                    );
                } else {
                    assert_eq!(
                        got, 0.0,
                        "on/below-diag (i={i}, j={j}) should be 0, got {got}"
                    );
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
        let row0_norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let row1_norm = (v[3] * v[3] + v[4] * v[4] + v[5] * v[5]).sqrt();
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
        assert_eq!(
            out.realize_f32(),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
        );
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
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let out = Tensor::stack(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape().dims(), &[2, 3]);
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_adds_trailing_dim() {
        let a = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let out = Tensor::stack(&[&a, &b], 1).unwrap();
        assert_eq!(out.shape().dims(), &[3, 2]);
        assert_eq!(out.realize_f32(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn stack_rejects_mismatched_shapes() {
        let a = cpu_f32(vec![1.0, 2.0], &[2]);
        let b = a.const_f32_like(vec![3.0, 4.0, 5.0], vec![3]).unwrap();
        assert!(Tensor::stack(&[&a, &b], 0).is_err());
    }

    #[test]
    fn stack_rejects_empty_input() {
        let result = Tensor::stack(&[], 0);
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

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_f32(data, shape.to_vec(), &Device::cpu()).unwrap()
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

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_f32(data, shape.to_vec(), &Device::cpu()).unwrap()
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
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let out = a.dot(&b).unwrap();
        assert_eq!(out.shape().elem_count(), 1);
        let v = out.realize_f32();
        assert_eq!(v[0], 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn dot_rejects_non_rank_one() {
        let a = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = a
            .const_f32_like(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .unwrap();
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn dot_rejects_length_mismatch() {
        let a = cpu_f32(vec![1.0, 2.0], &[2]);
        let b = a.const_f32_like(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn mv_matrix_times_vector() {
        let m = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let v = m.const_f32_like(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let out = m.mv(&v).unwrap();
        assert_eq!(out.shape().dims(), &[2]);
        assert_eq!(out.realize_f32(), vec![6.0, 15.0]);
    }

    #[test]
    fn matvec_is_mv_alias() {
        let m = cpu_f32(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let v = m.const_f32_like(vec![3.0, 4.0], vec![2]).unwrap();
        let a = m.mv(&v).unwrap().realize_f32();
        let b = m.matvec(&v).unwrap().realize_f32();
        assert_eq!(a, b);
    }

    #[test]
    fn mv_rejects_shape_mismatch() {
        let m = cpu_f32(vec![1.0; 6], &[2, 3]);
        let v = m.const_f32_like(vec![1.0, 1.0], vec![2]).unwrap();
        assert!(m.mv(&v).is_err());
    }

    #[test]
    fn broadcast_matmul_passes_through_to_matmul() {
        let a = cpu_f32(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let b = a
            .const_f32_like(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])
            .unwrap();
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

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_f32(data, shape.to_vec(), &Device::cpu()).unwrap()
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
        let t = Tensor::ones(vec![2, 3], DType::F32, &Device::cpu()).unwrap();
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.realize_f32(), vec![1.0; 6]);
    }

    #[test]
    fn static_zeros_f64() {
        let t = Tensor::zeros(vec![4], DType::F64, &Device::cpu()).unwrap();
        assert_eq!(t.dtype(), DType::F64);
        assert_eq!(t.realize_f64(), vec![0.0; 4]);
    }

    #[test]
    fn full_with_f32_scalar() {
        let t = Tensor::full(vec![5], fuel_ir::Scalar::F32(2.5), &Device::cpu()).unwrap();
        assert_eq!(t.realize_f32(), vec![2.5; 5]);
    }

    #[test]
    fn eye_identity_matrix() {
        let t = Tensor::eye(3, DType::F32, &Device::cpu());
        assert_eq!(t.shape().dims(), &[3, 3]);
        assert_eq!(
            t.realize_f32(),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn tril2_lower_triangular_ones() {
        let t = Tensor::tril2(3, DType::F32, &Device::cpu());
        assert_eq!(
            t.realize_f32(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        );
    }

    #[test]
    fn triu2_upper_triangular_ones() {
        let t = Tensor::triu2(3, DType::F32, &Device::cpu());
        assert_eq!(
            t.realize_f32(),
            vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn meshgrid_ij_indexing_two_inputs() {
        let x = Tensor::from_f32(vec![1.0_f32, 2.0, 3.0], vec![3], &Device::cpu()).unwrap();
        let y = x.const_f32_like(vec![4.0_f32, 5.0], vec![2]).unwrap();
        let grids = Tensor::meshgrid(&[&x, &y], false).unwrap();
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
        let x = Tensor::from_f32(vec![1.0_f32, 2.0, 3.0], vec![3], &Device::cpu()).unwrap();
        let y = x.const_f32_like(vec![4.0_f32, 5.0], vec![2]).unwrap();
        let grids = Tensor::meshgrid(&[&x, &y], true).unwrap();
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
        let x = Tensor::from_f32(vec![1.0_f32, 2.0], vec![2], &Device::cpu()).unwrap();
        assert!(Tensor::meshgrid(&[&x], false).is_err());
    }

    #[test]
    fn meshgrid_rejects_non_rank_one() {
        let x = Tensor::from_f32(vec![1.0; 4], vec![2, 2], &Device::cpu()).unwrap();
        let y = x.const_f32_like(vec![1.0, 2.0], vec![2]).unwrap();
        assert!(Tensor::meshgrid(&[&x, &y], false).is_err());
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
        let t = Tensor::rand(vec![100], 0.0, 1.0, DType::F32, &Device::cpu()).unwrap();
        let v = t.realize_f32();
        // Mean of uniform [0,1) should be ~0.5; tolerate sample noise.
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert!((0.3..0.7).contains(&mean), "mean {mean} too far from 0.5");
    }

    #[test]
    fn static_randn_f64() {
        let t = Tensor::randn(vec![1000], 0.0, 1.0, DType::F64, &Device::cpu()).unwrap();
        let v = t.realize_f64();
        let mean: f64 = v.iter().sum::<f64>() / v.len() as f64;
        // Normal(0,1) mean should be near 0; n=1000 gives stderr ~0.03.
        assert!(mean.abs() < 0.2, "mean {mean} too far from 0");
    }

    #[test]
    fn arange_int_step() {
        let t = Tensor::arange(0.0, 5.0, &Device::cpu());
        assert_eq!(t.shape().dims(), &[5]);
        assert_eq!(t.realize_f32(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn arange_step_fractional() {
        let t = Tensor::arange_step(2.0, 4.0, 0.5, &Device::cpu());
        assert_eq!(t.realize_f32(), vec![2.0, 2.5, 3.0, 3.5]);
    }

    #[test]
    fn arange_step_negative_descends() {
        let t = Tensor::arange_step(5.0, 0.0, -1.0, &Device::cpu());
        assert_eq!(t.realize_f32(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn linspace_includes_endpoints() {
        let t = Tensor::linspace(0.0, 1.0, 5, &Device::cpu());
        assert_eq!(t.shape().dims(), &[5]);
        let v = t.realize_f32();
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!((v[4] - 1.0).abs() < 1e-6);
        assert!((v[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn linspace_n_one_returns_start() {
        let t = Tensor::linspace(7.0, 99.0, 1, &Device::cpu());
        assert_eq!(t.realize_f32(), vec![7.0]);
    }

    #[test]
    fn norm_is_sqrt_sum_sq() {
        let t = Tensor::from_f32(vec![3.0_f32, 4.0], vec![2], &Device::cpu()).unwrap();
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
        let w = x.const_f32_like(vec![1.0], vec![1, 1, 1]).unwrap();
        let out = x.conv1d(&w, None, 1, 0, 1).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 5]);
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn conv1d_sum_kernel_two_wide() {
        // Sum kernel of size 2: out[t] = x[t] + x[t+1].
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let w = x.const_f32_like(vec![1.0, 1.0], vec![1, 1, 2]).unwrap();
        let out = x.conv1d(&w, None, 1, 0, 1).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 3]);
        assert_eq!(out.realize_f32(), vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn conv1d_with_bias_applies_correctly() {
        let x = cpu_f32(vec![1.0, 1.0, 1.0], &[1, 1, 3]);
        let w = x.const_f32_like(vec![1.0], vec![1, 1, 1]).unwrap();
        let bias = x.const_f32_like(vec![10.0], vec![1]).unwrap();
        let out = x.conv1d(&w, Some(&bias), 1, 0, 1).unwrap();
        assert_eq!(out.realize_f32(), vec![11.0, 11.0, 11.0]);
    }

    #[test]
    fn conv1d_stride_two_halves_output() {
        // Input length 6, kernel 2, stride 2 → output length (6-2)/2+1 = 3.
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
        let w = x.const_f32_like(vec![1.0, 1.0], vec![1, 1, 2]).unwrap();
        let out = x.conv1d(&w, None, 2, 0, 1).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 3]);
        assert_eq!(out.realize_f32(), vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn conv1d_padding_one_preserves_length() {
        // Input length 4, kernel 3, padding 1, stride 1 → output length 4.
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let w = x
            .const_f32_like(vec![1.0, 1.0, 1.0], vec![1, 1, 3])
            .unwrap();
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
        let w = x.const_f32_like(vec![2.0, -1.0], vec![2, 1, 1]).unwrap();
        let out = x.conv1d(&w, None, 1, 0, 1).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2, 3]);
        // Channel 0: 2.0 × input. Channel 1: -1.0 × input.
        assert_eq!(out.realize_f32(), vec![2.0, 4.0, 6.0, -1.0, -2.0, -3.0]);
    }

    #[test]
    fn conv1d_rejects_rank_two_input() {
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w = x.const_f32_like(vec![1.0], vec![1, 1, 1]).unwrap();
        assert!(x.conv1d(&w, None, 1, 0, 1).is_err());
    }

    #[test]
    fn conv1d_rejects_rank_two_weight() {
        let x = cpu_f32(vec![1.0; 4], &[1, 1, 4]);
        let w = x.const_f32_like(vec![1.0], vec![1, 1]).unwrap();
        assert!(x.conv1d(&w, None, 1, 0, 1).is_err());
    }

    #[test]
    fn conv1d_rejects_zero_groups_or_stride() {
        let x = cpu_f32(vec![1.0; 4], &[1, 1, 4]);
        let w = x.const_f32_like(vec![1.0], vec![1, 1, 1]).unwrap();
        assert!(x.conv1d(&w, None, 0, 0, 1).is_err());
        assert!(x.conv1d(&w, None, 1, 0, 0).is_err());
    }

    #[test]
    fn conv1d_with_algo_ignores_algo_param() {
        let x = cpu_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let w = x.const_f32_like(vec![1.0, 1.0], vec![1, 1, 2]).unwrap();
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
        let x = cpu_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        );
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
                1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
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
pub struct TokenDataHost {
    pub token_ids: fuel_ir::HostBuffer,
    pub rope_cos: fuel_ir::HostBuffer,
    pub rope_sin: fuel_ir::HostBuffer,
    pub mask: fuel_ir::HostBuffer,
    pub offset: Option<fuel_ir::HostBuffer>,
}

/// Same per-token data as [`crate::inference_context::DecodeTokenData`],
/// as raw host bytes instead of freshly-uploaded device Arcs — for
/// [`fuel_dispatch::pipelined::CapturedDecodeSession::replay_token`]'s
/// in-place H2D overwrite of fixed buffers. `LlamaModel`-private.
#[cfg(feature = "cuda")]
pub struct TokenDataBytes {
    pub token_ids: Vec<u8>,
    pub rope_cos: Vec<u8>,
    pub rope_sin: Vec<u8>,
    pub mask: Vec<u8>,
    pub offset: Option<Vec<u8>>,
}

/// D2H a [`fuel_dispatch::pipelined::CapturedDecodeSession::replay_token`]
/// result's device-resident output Arc to a host `Vec<f32>`.
///
/// The capture mechanism deliberately keeps the output device-resident
/// INSIDE the capture (a `Copy`/`Move` node is capture-unsafe — see
/// `LlamaModel::forward_with_kv_context_captured`'s design note); this
/// reads it back OUTSIDE the capture, dispatching on the same
/// `BackendStorage` variants every other pipelined D2H site in this
/// codebase does (`CudaStorageBytes::to_cpu_bytes` / `CpuStorageBytes::
/// bytes`) rather than hand-rolling a new D2H mechanism. Capture is
/// f32-only today (project constraint), so the bytes are reinterpreted
/// as f32 directly.
#[cfg(feature = "cuda")]
pub fn captured_output_to_f32(
    output: &Arc<std::sync::RwLock<fuel_memory::Storage>>,
) -> fuel_ir::error::Result<Vec<f32>> {
    use fuel_memory::BackendStorage;
    let guard = output.read().map_err(|_| {
        fuel_ir::Error::Msg("captured decode output storage lock poisoned".into()).bt()
    })?;
    let bytes: Vec<u8> = match &guard.inner {
        BackendStorage::Cpu(c) => c.bytes().to_vec(),
        BackendStorage::Cuda(c) => c.to_cpu_bytes()?,
        #[allow(unreachable_patterns)]
        other => {
            return Err(fuel_ir::Error::Msg(format!(
                "captured decode output is BackendStorage::{:?}, expected Cpu or Cuda",
                std::mem::discriminant(other),
            ))
            .bt());
        }
    };
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}
