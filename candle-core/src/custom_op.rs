use crate::op::{BackpropOp, Op};
use crate::tensor::from_storage;
use crate::{CpuStorage, CudaStorage, Layout, MetalStorage, Result, Shape, Tensor};
use std::sync::Arc;

/// A custom unary operation (one input tensor, one output tensor).
///
/// Implement this trait to define your own operations that can be applied to a single tensor
/// and participate in the autograd computation graph. At minimum you must provide [`cpu_fwd`];
/// the CUDA and Metal methods have default implementations that return an error, so you can
/// add GPU support incrementally. Override [`bwd`] to enable gradient computation through
/// this operation.
///
/// Apply the operation to a tensor with [`Tensor::apply_op1`] (with backward support) or
/// [`Tensor::apply_op1_no_bwd`] (forward only, no graph tracking).
///
/// [`cpu_fwd`]: CustomOp1::cpu_fwd
/// [`bwd`]: CustomOp1::bwd
///
/// # Example
///
/// ```no_run
/// use candle_core::{CpuStorage, Layout, Shape, CustomOp1, Result};
/// struct Negate;
/// impl CustomOp1 for Negate {
///     fn name(&self) -> &'static str { "negate" }
///     fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
///         todo!()
///     }
/// }
/// ```
pub trait CustomOp1 {
    /// Returns a human-readable name for this operation, used in error messages.
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(&self, _storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _storage: &MetalStorage,
        _layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }

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
/// At minimum you must provide [`cpu_fwd`]; CUDA and Metal have default error-returning
/// implementations. Override [`bwd`] to return gradients for both inputs.
///
/// Apply with [`Tensor::apply_op2`] or [`Tensor::apply_op2_no_bwd`].
///
/// [`cpu_fwd`]: CustomOp2::cpu_fwd
/// [`bwd`]: CustomOp2::bwd
///
/// # Example
///
/// ```no_run
/// use candle_core::{CpuStorage, Layout, Shape, CustomOp2, Result};
/// struct Add;
/// impl CustomOp2 for Add {
///     fn name(&self) -> &'static str { "add" }
///     fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
///         todo!()
///     }
/// }
/// ```
pub trait CustomOp2 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }

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
/// three tensors. At minimum you must provide [`cpu_fwd`]; CUDA and Metal have default
/// error-returning implementations. Override [`bwd`] to return gradients for all three inputs.
///
/// Apply with [`Tensor::apply_op3`] or [`Tensor::apply_op3_no_bwd`].
///
/// [`cpu_fwd`]: CustomOp3::cpu_fwd
/// [`bwd`]: CustomOp3::bwd
///
/// # Example
///
/// ```no_run
/// use candle_core::{CpuStorage, Layout, Shape, CustomOp3, Result};
/// struct Select;
/// impl CustomOp3 for Select {
///     fn name(&self) -> &'static str { "select" }
///     fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout, s3: &CpuStorage, l3: &Layout) -> Result<(CpuStorage, Shape)> {
///         todo!()
///     }
/// }
/// ```
pub trait CustomOp3 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }

    /// Computes gradients for all three inputs during backpropagation.
    ///
    /// Given the three original arguments, the forward result (`res`), and the upstream
    /// gradient (`grad_res`), returns the gradient with respect to each argument. Return
    /// `None` for any argument that does not need a gradient.
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
        let (storage, shape) = self
            .storage()
            .apply_op1(self.layout(), c.as_ref())?;
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
    ///
    /// The operation `c` must implement [`CustomOp2`]. If you do not need gradient tracking,
    /// use [`apply_op2_no_bwd`](Tensor::apply_op2_no_bwd) instead.
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
    ///
    /// The operation `c` must implement [`CustomOp3`]. If you do not need gradient tracking,
    /// use [`apply_op3_no_bwd`](Tensor::apply_op3_no_bwd) instead.
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
/// At minimum you must implement [`cpu_fwd`]; CUDA and Metal have default error-returning
/// implementations. Apply with [`Tensor::inplace_op1`].
///
/// [`cpu_fwd`]: InplaceOp1::cpu_fwd
///
/// # Example
///
/// ```no_run
/// use candle_core::{CpuStorage, Layout, InplaceOp1, Result};
/// struct ZeroOut;
/// impl InplaceOp1 for ZeroOut {
///     fn name(&self) -> &'static str { "zero_out" }
///     fn cpu_fwd(&self, s: &mut CpuStorage, l: &Layout) -> Result<()> { todo!() }
/// }
/// ```
pub trait InplaceOp1 {
    /// Returns a human-readable name for this operation, used in error messages.
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> Result<()>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(&self, _storage: &mut CudaStorage, _layout: &Layout) -> Result<()> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(&self, _storage: &mut MetalStorage, _layout: &Layout) -> Result<()> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }
}

/// A custom in-place binary operation that modifies the first tensor using data from a second.
///
/// The first tensor's storage is mutated; the second tensor is read-only. Because this is an
/// in-place operation, it does not participate in backpropagation.
///
/// At minimum you must implement [`cpu_fwd`]; CUDA and Metal have default error-returning
/// implementations. Apply with [`Tensor::inplace_op2`].
///
/// [`cpu_fwd`]: InplaceOp2::cpu_fwd
///
/// # Example
///
/// ```no_run
/// use candle_core::{CpuStorage, Layout, InplaceOp2, Result};
/// struct CopyFrom;
/// impl InplaceOp2 for CopyFrom {
///     fn name(&self) -> &'static str { "copy_from" }
///     fn cpu_fwd(&self, dst: &mut CpuStorage, dl: &Layout, src: &CpuStorage, sl: &Layout) -> Result<()> { todo!() }
/// }
/// ```
pub trait InplaceOp2 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout)
        -> Result<()>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(&self, _: &mut CudaStorage, _: &Layout, _: &CudaStorage, _: &Layout) -> Result<()> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _: &mut MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
    ) -> Result<()> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }
}

/// A custom in-place ternary operation that modifies the first tensor using data from two others.
///
/// The first tensor's storage is mutated; the second and third tensors are read-only. Because
/// this is an in-place operation, it does not participate in backpropagation.
///
/// At minimum you must implement [`cpu_fwd`]; CUDA and Metal have default error-returning
/// implementations. Apply with [`Tensor::inplace_op3`].
///
/// [`cpu_fwd`]: InplaceOp3::cpu_fwd
///
/// # Example
///
/// ```no_run
/// use candle_core::{CpuStorage, Layout, InplaceOp3, Result};
/// struct MaskedFill;
/// impl InplaceOp3 for MaskedFill {
///     fn name(&self) -> &'static str { "masked_fill" }
///     fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout, s3: &CpuStorage, l3: &Layout) -> Result<()> { todo!() }
/// }
/// ```
pub trait InplaceOp3 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<()>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(
        &self,
        _: &mut CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<()> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _: &mut MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
    ) -> Result<()> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }
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
    func: candle_metal_kernels::metal::ComputePipeline,
}

#[cfg(feature = "ug")]
impl UgIOp1 {
    #[allow(unused)]
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn new(
        name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
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

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        crate::bail!("ug ops are only supported on metal/cuda at the moment")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, sto: &mut MetalStorage, layout: &Layout) -> Result<()> {
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
        let group_dims = candle_metal_kernels::utils::get_block_dims(b, 1, 1);
        candle_metal_kernels::utils::set_param(&encoder, 0, (sto.buffer(), 0usize));

        encoder.use_resource(sto.buffer(), objc2_metal::MTLResourceUsage::Write);
        encoder.dispatch_threads(grid_dims, group_dims);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, sto: &mut CudaStorage, layout: &Layout) -> Result<()> {
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
        Ok(())
    }
}
