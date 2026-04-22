use crate::dyn_backend::DynBackendStorage;
use crate::op::{BackpropOp, Op};
use crate::tensor::from_storage;
use crate::{Layout, Result, Shape, Tensor};
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
        let (storage, shape) = self.storage().apply_op1(self.layout(), c)?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Applies a binary custom op without backward support
    pub fn apply_op2_no_bwd<C: CustomOp2>(&self, rhs: &Self, c: &C) -> Result<Self> {
        let (storage, shape) =
            self.storage()
                .apply_op2(self.layout(), &rhs.storage(), rhs.layout(), c)?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Applies a ternary custom op without backward support
    pub fn apply_op3_no_bwd<C: CustomOp3>(&self, t2: &Self, t3: &Self, c: &C) -> Result<Self> {
        let (storage, shape) = self.storage().apply_op3(
            self.layout(),
            &t2.storage(),
            t2.layout(),
            &t3.storage(),
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
        let (storage, shape) = self.storage().apply_op1(self.layout(), c.as_ref())?;
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
        let (storage, shape) = self.storage().apply_op2(
            self.layout(),
            &rhs.storage(),
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
        let (storage, shape) = self.storage().apply_op3(
            self.layout(),
            &t2.storage(),
            t2.layout(),
            &t3.storage(),
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

// In place ops.

/// A custom in-place unary operation that modifies tensor storage directly.
///
/// Unlike [`CustomOp1`], in-place operations mutate the input tensor's storage rather than
/// producing a new tensor. Because they modify data in place, they cannot participate in
/// backpropagation and are not recorded in the computation graph.
///
/// The [`fwd`] method receives the storage as a `&mut dyn DynBackendStorage` trait object.
/// Apply with [`Tensor::inplace_op1`].
///
/// [`fwd`]: InplaceOp1::fwd
///
/// # Example
///
/// ```no_run
/// use fuel_core::{Layout, InplaceOp1, Result};
/// use fuel_core::dyn_backend::DynBackendStorage;
/// struct ZeroOut;
/// impl InplaceOp1 for ZeroOut {
///     fn name(&self) -> &'static str { "zero_out" }
///     fn fwd(&self, s: &mut dyn DynBackendStorage, l: &Layout) -> Result<()> { todo!() }
/// }
/// ```
pub trait InplaceOp1 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass. The storage is a mutable trait object; use
    /// `storage.as_any_mut().downcast_mut()` to access backend-specific types.
    fn fwd(&self, storage: &mut dyn DynBackendStorage, layout: &Layout) -> Result<()>;
}

/// A custom in-place binary operation that modifies the first tensor using data from a second.
///
/// The first tensor's storage is mutated; the second tensor is read-only. Because this is an
/// in-place operation, it does not participate in backpropagation.
///
/// Apply with [`Tensor::inplace_op2`].
///
/// # Example
///
/// ```no_run
/// use fuel_core::{Layout, InplaceOp2, Result};
/// use fuel_core::dyn_backend::DynBackendStorage;
/// struct CopyFrom;
/// impl InplaceOp2 for CopyFrom {
///     fn name(&self) -> &'static str { "copy_from" }
///     fn fwd(&self, dst: &mut dyn DynBackendStorage, dl: &Layout,
///            src: &dyn DynBackendStorage, sl: &Layout) -> Result<()> { todo!() }
/// }
/// ```
pub trait InplaceOp2 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass. The first storage is mutable; the second is read-only.
    fn fwd(
        &self,
        s1: &mut dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
    ) -> Result<()>;
}

/// A custom in-place ternary operation that modifies the first tensor using data from two others.
///
/// The first tensor's storage is mutated; the second and third tensors are read-only. Because
/// this is an in-place operation, it does not participate in backpropagation.
///
/// Apply with [`Tensor::inplace_op3`].
///
/// # Example
///
/// ```no_run
/// use fuel_core::{Layout, InplaceOp3, Result};
/// use fuel_core::dyn_backend::DynBackendStorage;
/// struct MaskedFill;
/// impl InplaceOp3 for MaskedFill {
///     fn name(&self) -> &'static str { "masked_fill" }
///     fn fwd(&self, s1: &mut dyn DynBackendStorage, l1: &Layout,
///            s2: &dyn DynBackendStorage, l2: &Layout,
///            s3: &dyn DynBackendStorage, l3: &Layout) -> Result<()> { todo!() }
/// }
/// ```
pub trait InplaceOp3 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass. The first storage is mutable; the second and third are read-only.
    fn fwd(
        &self,
        s1: &mut dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
        s3: &dyn DynBackendStorage,
        l3: &Layout,
    ) -> Result<()>;
}

impl Tensor {
    /// Applies a unary custom op in place.
    pub fn inplace_op1<C: InplaceOp1>(&self, c: &C) -> Result<()> {
        self.storage_mut().inplace_op1(self.layout(), c)
    }

    /// Applies a unary custom op in place (for the first tensor).
    pub fn inplace_op2<C: InplaceOp2>(&self, rhs: &Self, c: &C) -> Result<()> {
        self.storage_mut()
            .inplace_op2(self.layout(), &rhs.storage(), rhs.layout(), c)
    }

    /// Applies a ternary custom op in place (for the first tensor).
    pub fn inplace_op3<C: InplaceOp3>(&self, t2: &Self, t3: &Self, c: &C) -> Result<()> {
        self.storage_mut().inplace_op3(
            self.layout(),
            &t2.storage(),
            t2.layout(),
            &t3.storage(),
            t3.layout(),
            c,
        )
    }
}

#[cfg(feature = "ug")]
pub struct UgIOp1 {
    name: &'static str,
    #[cfg(feature = "cuda")]
    func: cudarc::driver::CudaFunction,
    #[cfg(feature = "metal")]
    func: fuel_metal_kernels::metal::ComputePipeline,
}

#[cfg(feature = "ug")]
impl UgIOp1 {
    #[allow(unused)]
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn new(
        name: &'static str,
        kernel: fuel_ug::lang::ssa::Kernel,
        device: &crate::Device,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let device = device.as_cuda_device()?;
            let func = device.compile(name, kernel)?;
            Ok(Self {
                name,
                func: func.into_cuda_function(),
            })
        }
        #[cfg(feature = "metal")]
        {
            let device = device.as_metal_device()?;
            let func = device.compile(name, kernel)?;
            Ok(Self { name, func })
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            Ok(Self { name })
        }
    }
}

#[cfg(feature = "ug")]
impl InplaceOp1 for UgIOp1 {
    fn name(&self) -> &'static str {
        self.name
    }

    fn fwd(&self, storage: &mut dyn DynBackendStorage, layout: &Layout) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Some(sto) = storage
            .as_any_mut()
            .downcast_mut::<fuel_metal::MetalBackendStorage>()
        {
            let sto = &mut sto.storage;
            use crate::backend::BackendStorage;
            use objc2_metal;

            let elem_count = layout.shape().elem_count();
            if sto.dtype() != crate::DType::F32 {
                // TODO: support more dtypes.
                crate::bail!("input is not a f32 tensor")
            }
            let device = sto.device();
            let encoder = device.command_encoder()?;
            encoder.set_compute_pipeline_state(&self.func);
            let (g, b) = if elem_count.is_multiple_of(32) {
                (elem_count / 32, 32)
            } else {
                (elem_count, 1)
            };
            let grid_dims = objc2_metal::MTLSize {
                width: g,
                height: 1,
                depth: 1,
            };
            let group_dims = fuel_metal_kernels::utils::get_block_dims(b, 1, 1);
            fuel_metal_kernels::utils::set_param(&encoder, 0, (sto.buffer(), 0usize));

            encoder.use_resource(sto.buffer(), objc2_metal::MTLResourceUsage::Write);
            encoder.dispatch_threads(grid_dims, group_dims);

            return Ok(());
        }

        #[cfg(feature = "cuda")]
        if let Some(sto) = storage
            .as_any_mut()
            .downcast_mut::<fuel_graph_cuda::CudaBackendStorage>()
        {
            let sto = &mut sto.storage;
            use crate::cuda_backend::WrapErr;
            use cudarc::driver::PushKernelArg;

            let elem_count = layout.shape().elem_count();
            let stream = sto.device.cuda_stream();
            // TODO: support more dtypes.
            let sto = sto.as_cuda_slice::<f32>()?;
            let sto = match layout.contiguous_offsets() {
                None => crate::bail!("input has to be contiguous"),
                Some((o1, o2)) => sto.slice(o1..o2),
            };
            let (g, b) = if elem_count % 32 == 0 {
                (elem_count / 32, 32)
            } else {
                (elem_count, 1)
            };
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (g as u32, 1, 1),
                block_dim: (b as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&self.func);
            builder.arg(&sto);
            unsafe { builder.launch(cfg) }.w()?;
            return Ok(());
        }

        crate::bail!("ug ops are only supported on metal/cuda at the moment")
    }
}
