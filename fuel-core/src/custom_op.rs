use crate::dyn_backend::DynBackendStorage;
use crate::op::{BackpropOp, Op};
use crate::storage::StorageApplyOps;
use crate::tensor::{from_storage, Tensor};
use crate::{Layout, Result, Shape};
use std::sync::Arc;

/// A custom unary operation (one input tensor, one output tensor).
///
/// Implement this trait to define your own operations that can be applied to a single tensor
/// and participate in the autograd computation graph. The [`fwd`] method receives the storage
/// as a `&dyn DynBackendStorage` trait object — use [`as_any()`] and `downcast_ref` to access
/// concrete backend types when needed. Override [`bwd`] to enable gradient computation through
/// this operation.
///
/// Apply the operation to a tensor with [`Tensor::apply_op1`] (with backward support) or
/// [`Tensor::apply_op1_no_bwd`] (forward only, no graph tracking).
///
/// [`fwd`]: CustomOp1::fwd
/// [`bwd`]: CustomOp1::bwd
/// [`as_any()`]: DynBackendStorage::as_any
///
/// # Example
///
/// ```no_run
/// use fuel_core::{Layout, Shape, CustomOp1, Result};
/// use fuel_core::dyn_backend::DynBackendStorage;
/// struct Negate;
/// impl CustomOp1 for Negate {
///     fn name(&self) -> &'static str { "negate" }
///     fn fwd(&self, s: &dyn DynBackendStorage, l: &Layout)
///         -> Result<(Box<dyn DynBackendStorage>, Shape)> { todo!() }
/// }
/// ```
pub trait CustomOp1 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass. The storage is a trait object; use `storage.as_any().downcast_ref()`
    /// to access backend-specific types (e.g. `CpuBackendStorage`, `CudaBackendStorage`).
    /// Note that the storage can use arbitrary strides, offsets etc so the associated
    /// layout should be used to access it.
    fn fwd(
        &self,
        storage: &dyn DynBackendStorage,
        layout: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)>;

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    fn bwd(&self, _arg: &Tensor, _res: &Tensor, _grad_res: &Tensor) -> Result<Option<Tensor>> {
        Err(crate::Error::BackwardNotSupported { op: self.name() })
    }
}

/// A custom binary operation (two input tensors, one output tensor).
///
/// This is the two-input variant of [`CustomOp1`]. Implement this trait to define operations
/// that combine two tensors into a new one (e.g., a custom distance metric or attention score).
/// The [`fwd`] method receives both storages as trait objects. Override [`bwd`] to return
/// gradients for both inputs.
///
/// Apply with [`Tensor::apply_op2`] or [`Tensor::apply_op2_no_bwd`].
///
/// [`fwd`]: CustomOp2::fwd
/// [`bwd`]: CustomOp2::bwd
///
/// # Example
///
/// ```no_run
/// use fuel_core::{Layout, Shape, CustomOp2, Result};
/// use fuel_core::dyn_backend::DynBackendStorage;
/// struct Add;
/// impl CustomOp2 for Add {
///     fn name(&self) -> &'static str { "add" }
///     fn fwd(&self, s1: &dyn DynBackendStorage, l1: &Layout,
///            s2: &dyn DynBackendStorage, l2: &Layout)
///         -> Result<(Box<dyn DynBackendStorage>, Shape)> { todo!() }
/// }
/// ```
pub trait CustomOp2 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass. Both storages are trait objects; use `as_any().downcast_ref()` to
    /// access backend-specific types.
    fn fwd(
        &self,
        s1: &dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)>;

    /// Computes gradients for both inputs during backpropagation.
    ///
    /// Given the two original arguments (`arg1`, `arg2`), the forward result (`res`), and
    /// the upstream gradient (`grad_res`), returns the gradient with respect to each argument.
    /// Return `None` for an argument that does not need a gradient.
    fn bwd(
        &self,
        _arg1: &Tensor,
        _arg2: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Err(crate::Error::BackwardNotSupported { op: self.name() })
    }
}

/// A custom ternary operation (three input tensors, one output tensor).
///
/// This is the three-input variant of [`CustomOp1`]. Useful for operations like
/// `where_cond(condition, on_true, on_false)` or fused attention primitives that need
/// three tensors. Override [`bwd`] to return gradients for all three inputs.
///
/// Apply with [`Tensor::apply_op3`] or [`Tensor::apply_op3_no_bwd`].
///
/// [`bwd`]: CustomOp3::bwd
///
/// # Example
///
/// ```no_run
/// use fuel_core::{Layout, Shape, CustomOp3, Result};
/// use fuel_core::dyn_backend::DynBackendStorage;
/// struct Select;
/// impl CustomOp3 for Select {
///     fn name(&self) -> &'static str { "select" }
///     fn fwd(&self, s1: &dyn DynBackendStorage, l1: &Layout,
///            s2: &dyn DynBackendStorage, l2: &Layout,
///            s3: &dyn DynBackendStorage, l3: &Layout)
///         -> Result<(Box<dyn DynBackendStorage>, Shape)> { todo!() }
/// }
/// ```
pub trait CustomOp3 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass. All three storages are trait objects; use `as_any().downcast_ref()`
    /// to access backend-specific types.
    fn fwd(
        &self,
        s1: &dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
        s3: &dyn DynBackendStorage,
        l3: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)>;

    /// Computes gradients for all three inputs during backpropagation.
    fn bwd(
        &self,
        _arg1: &Tensor,
        _arg2: &Tensor,
        _arg3: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        Err(crate::Error::BackwardNotSupported { op: self.name() })
    }
}

impl Tensor {
    /// Applies a unary custom op without backward support
    pub fn apply_op1_no_bwd<C: CustomOp1>(&self, c: &C) -> Result<Self> {
        let self_arc = self.storage()?;
        let (storage, shape) = self_arc.read().unwrap().apply_op1(self.layout(), c)?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Applies a binary custom op without backward support
    pub fn apply_op2_no_bwd<C: CustomOp2>(&self, rhs: &Self, c: &C) -> Result<Self> {
        let self_arc = self.storage()?;
        let rhs_arc = rhs.storage()?;
        let (storage, shape) = self_arc.read().unwrap().apply_op2(
            self.layout(),
            &rhs_arc.read().unwrap(),
            rhs.layout(),
            c,
        )?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Applies a ternary custom op without backward support
    pub fn apply_op3_no_bwd<C: CustomOp3>(&self, t2: &Self, t3: &Self, c: &C) -> Result<Self> {
        let self_arc = self.storage()?;
        let t2_arc = t2.storage()?;
        let t3_arc = t3.storage()?;
        let (storage, shape) = self_arc.read().unwrap().apply_op3(
            self.layout(),
            &t2_arc.read().unwrap(),
            t2.layout(),
            &t3_arc.read().unwrap(),
            t3.layout(),
            c,
        )?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Applies a unary custom op, recording it in the computation graph for backpropagation.
    ///
    /// This is the `Arc`-based variant; prefer [`apply_op1`](Tensor::apply_op1) unless you
    /// need to share the operation object.
    pub fn apply_op1_arc(&self, c: Arc<dyn CustomOp1 + Send + Sync>) -> Result<Self> {
        let self_arc = self.storage()?;
        let (storage, shape) = self_arc.read().unwrap().apply_op1(self.layout(), c.as_ref())?;
        let op = BackpropOp::new1(self, |s| Op::CustomOp1(s, c.clone()));
        Ok(from_storage(storage, shape, op, false))
    }

    /// Applies a unary custom op, recording it in the computation graph for backpropagation.
    ///
    /// The operation `c` must implement [`CustomOp1`]. If you do not need gradient tracking,
    /// use [`apply_op1_no_bwd`](Tensor::apply_op1_no_bwd) instead.
    pub fn apply_op1<C: 'static + CustomOp1 + Send + Sync>(&self, c: C) -> Result<Self> {
        self.apply_op1_arc(Arc::new(c))
    }

    /// Applies a binary custom op, recording it in the computation graph for backpropagation.
    ///
    /// This is the `Arc`-based variant; prefer [`apply_op2`](Tensor::apply_op2) unless you
    /// need to share the operation object.
    pub fn apply_op2_arc(
        &self,
        rhs: &Self,
        c: Arc<dyn CustomOp2 + Send + Sync>,
    ) -> Result<Self> {
        let self_arc = self.storage()?;
        let rhs_arc = rhs.storage()?;
        let (storage, shape) = self_arc.read().unwrap().apply_op2(
            self.layout(),
            &rhs_arc.read().unwrap(),
            rhs.layout(),
            c.as_ref(),
        )?;
        let op = BackpropOp::new2(self, rhs, |t1, t2| Op::CustomOp2(t1, t2, c.clone()));
        Ok(from_storage(storage, shape, op, false))
    }

    /// Applies a binary custom op, recording it in the computation graph for backpropagation.
    pub fn apply_op2<C: 'static + CustomOp2 + Send + Sync>(&self, r: &Self, c: C) -> Result<Self> {
        self.apply_op2_arc(r, Arc::new(c))
    }

    /// Applies a ternary custom op, recording it in the computation graph for backpropagation.
    ///
    /// This is the `Arc`-based variant; prefer [`apply_op3`](Tensor::apply_op3) unless you
    /// need to share the operation object.
    pub fn apply_op3_arc(
        &self,
        t2: &Self,
        t3: &Self,
        c: Arc<dyn CustomOp3 + Send + Sync>,
    ) -> Result<Self> {
        let self_arc = self.storage()?;
        let t2_arc = t2.storage()?;
        let t3_arc = t3.storage()?;
        let (storage, shape) = self_arc.read().unwrap().apply_op3(
            self.layout(),
            &t2_arc.read().unwrap(),
            t2.layout(),
            &t3_arc.read().unwrap(),
            t3.layout(),
            c.as_ref(),
        )?;
        let op = BackpropOp::new3(self, t2, t3, |t1, t2, t3| {
            Op::CustomOp3(t1, t2, t3, c.clone())
        });
        Ok(from_storage(storage, shape, op, false))
    }

    /// Applies a ternary custom op, recording it in the computation graph for backpropagation.
    pub fn apply_op3<C: 'static + CustomOp3 + Send + Sync>(
        &self,
        t2: &Self,
        t3: &Self,
        c: C,
    ) -> Result<Self> {
        self.apply_op3_arc(t2, t3, Arc::new(c))
    }
}

// In-place op traits live in fuel-core-types so backend crates can implement
// them without depending on fuel-core (which would create a cycle).
pub use fuel_ir::inplace_op::{InplaceOp1, InplaceOp2, InplaceOp3};

impl Tensor {
    /// Applies a unary custom op in place.
    pub fn inplace_op1<C: InplaceOp1>(&self, c: &C) -> Result<()> {
        let self_arc = self.storage_mut()?;
        self_arc.write().unwrap().inplace_op1(self.layout(), c)
    }

    /// Applies a unary custom op in place (for the first tensor).
    pub fn inplace_op2<C: InplaceOp2>(&self, rhs: &Self, c: &C) -> Result<()> {
        let self_arc = self.storage_mut()?;
        let rhs_arc = rhs.storage()?;
        self_arc.write().unwrap().inplace_op2(
            self.layout(),
            &rhs_arc.read().unwrap(),
            rhs.layout(),
            c,
        )
    }

    /// Applies a ternary custom op in place (for the first tensor).
    pub fn inplace_op3<C: InplaceOp3>(&self, t2: &Self, t3: &Self, c: &C) -> Result<()> {
        let self_arc = self.storage_mut()?;
        let t2_arc = t2.storage()?;
        let t3_arc = t3.storage()?;
        self_arc.write().unwrap().inplace_op3(
            self.layout(),
            &t2_arc.read().unwrap(),
            t2.layout(),
            &t3_arc.read().unwrap(),
            t3.layout(),
            c,
        )
    }
}

// `UgIOp1` was split into per-backend types and moved out of fuel-core in
// step B2 of the backend extraction:
//   - CUDA: `fuel_cuda_backend::ug::CudaUgIOp1`
//   - Metal: `fuel_metal_backend::ug::MetalUgIOp1`
// Both impl `InplaceOp1` (re-exported above from `fuel-core-types`).
