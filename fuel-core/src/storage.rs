use crate::dyn_backend::DynBackendStorage;
use crate::op::{self, CmpOp, ReduceOp};
use crate::scalar::Scalar;
use crate::{CpuStorage, DType, Device, Error, Layout, Result, Shape};
use crate::{CustomOp1, CustomOp2, CustomOp3, InplaceOp1, InplaceOp2, InplaceOp3};
use fuel_core_types::DeviceLocation;
use fuel_cpu_backend::dyn_impl::CpuBackendStorage;

// We do not want to implement Clone on Storage as cloning may fail because of
// out of memory. Instead try_clone should be used.
#[derive(Debug)]
pub struct Storage(pub(crate) Box<dyn DynBackendStorage>);

impl Storage {
    /// Construct storage wrapping a CPU buffer.
    pub fn from_cpu(s: CpuStorage) -> Self {
        Storage(Box::new(CpuBackendStorage(s)))
    }

    /// Construct storage wrapping a CUDA buffer.
    #[cfg(feature = "cuda")]
    pub fn from_cuda(s: crate::CudaStorage) -> Self {
        Storage(Box::new(fuel_cuda::CudaBackendStorage::new(s)))
    }

    /// Construct storage wrapping a Metal buffer.
    #[cfg(feature = "metal")]
    pub fn from_metal(s: crate::MetalStorage) -> Self {
        Storage(Box::new(fuel_metal::MetalBackendStorage::new(s)))
    }

    pub fn try_clone(&self, layout: &Layout) -> Result<Self> {
        Ok(Storage(self.0.try_clone_dyn(layout)?))
    }

    pub fn device(&self) -> Device {
        Device {
            inner: self.0.device_arc_dyn(),
        }
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype_dyn()
    }

    pub(crate) fn same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs_device = self.device();
        let rhs_device = rhs.device();
        let lhs = lhs_device.location();
        let rhs = rhs_device.location();
        let same_device = if lhs_device.is_metal() {
            // On metal, we require the device to be exactly the same rather than
            // having the same location.
            lhs_device.same_device(&rhs_device)
        } else {
            lhs == rhs
        };
        if !same_device {
            Err(Error::DeviceMismatchBinaryOp { lhs, rhs, op }.bt())
        } else {
            Ok(())
        }
    }

    pub(crate) fn same_dtype(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.dtype();
        let rhs = rhs.dtype();
        if lhs != rhs {
            Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op }.bt())
        } else {
            Ok(())
        }
    }

    pub(crate) fn const_set(&mut self, v: Scalar, l: &Layout) -> Result<()> {
        self.0.const_set_dyn(v, l)
    }

    pub(crate) fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        Ok(Storage(self.0.affine_dyn(layout, mul, add)?))
    }

    pub(crate) fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        Ok(Storage(self.0.powf_dyn(layout, e)?))
    }

    pub(crate) fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        Ok(Storage(self.0.elu_dyn(layout, alpha)?))
    }

    pub(crate) fn cmp(
        &self,
        op: CmpOp,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        Ok(Storage(
            self.0.cmp_dyn(op, &*rhs.0, lhs_layout, rhs_layout)?,
        ))
    }

    pub(crate) fn reduce_op(
        &self,
        op: ReduceOp,
        layout: &Layout,
        reduce_dims: &[usize],
    ) -> Result<Self> {
        Ok(Storage(self.0.reduce_op_dyn(op, layout, reduce_dims)?))
    }

    pub(crate) fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        Ok(Storage(self.0.to_dtype_dyn(layout, dtype)?))
    }

    pub(crate) fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.0.to_cpu_storage_dyn()
    }

    // -----------------------------------------------------------------------
    // CustomOp bridge (temporary — will be removed when CustomOp traits are
    // redesigned in Step 9 to use &dyn DynBackendStorage directly)
    // -----------------------------------------------------------------------

    /// Downcast helper: get a `&CpuStorage` from the inner trait object.
    pub fn as_cpu_storage(&self) -> Result<&CpuStorage> {
        self.0
            .as_any()
            .downcast_ref::<CpuBackendStorage>()
            .map(|s| &s.0)
            .ok_or_else(|| Error::Msg("expected cpu storage".into()).bt())
    }

    /// Downcast helper: get a `&mut CpuStorage` from the inner trait object.
    fn as_cpu_storage_mut(&mut self) -> Result<&mut CpuStorage> {
        self.0
            .as_any_mut()
            .downcast_mut::<CpuBackendStorage>()
            .map(|s| &mut s.0)
            .ok_or_else(|| Error::Msg("expected cpu storage".into()).bt())
    }

    /// Downcast helper: get a `&CudaStorage` from the inner trait object.
    #[cfg(feature = "cuda")]
    pub fn as_cuda_storage(&self) -> Option<&crate::CudaStorage> {
        self.0
            .as_any()
            .downcast_ref::<fuel_cuda::CudaBackendStorage>()
            .map(|s| s.inner())
    }

    /// Downcast helper: get a `&MetalStorage` from the inner trait object.
    #[cfg(feature = "metal")]
    pub fn as_metal_storage(&self) -> Option<&crate::MetalStorage> {
        self.0
            .as_any()
            .downcast_ref::<fuel_metal::MetalBackendStorage>()
            .map(|s| s.inner())
    }

    pub(crate) fn apply_op1(&self, l: &Layout, c: &dyn CustomOp1) -> Result<(Self, Shape)> {
        let location = self.0.device_dyn().location_dyn();
        match location {
            DeviceLocation::Cpu => {
                let s = self.as_cpu_storage()?;
                let (storage, shape) = c.cpu_fwd(s, l)?;
                Ok((Self::from_cpu(storage), shape))
            }
            DeviceLocation::Cuda { .. } => {
                #[cfg(feature = "cuda")]
                {
                    let s = self
                        .0
                        .as_any()
                        .downcast_ref::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    let (storage, shape) = c.cuda_fwd(s.inner(), l)?;
                    Ok((Self::from_cuda(storage), shape))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(Error::NotCompiledWithCudaSupport.bt())
                }
            }
            DeviceLocation::Metal { .. } => {
                #[cfg(feature = "metal")]
                {
                    let s = self
                        .0
                        .as_any()
                        .downcast_ref::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    let (storage, shape) = c.metal_fwd(s.inner(), l)?;
                    Ok((Self::from_metal(storage), shape))
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(Error::NotCompiledWithMetalSupport.bt())
                }
            }
            _ => Err(Error::Msg(
                "custom-op is not supported on this backend".to_string(),
            )
            .bt()),
        }
    }

    pub(crate) fn apply_op2(
        &self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        c: &dyn CustomOp2,
    ) -> Result<(Self, Shape)> {
        self.same_device(t2, c.name())?;
        let location = self.0.device_dyn().location_dyn();
        match location {
            DeviceLocation::Cpu => {
                let s1 = self.as_cpu_storage()?;
                let s2 = t2.as_cpu_storage()?;
                let (s, shape) = c.cpu_fwd(s1, l1, s2, l2)?;
                Ok((Self::from_cpu(s), shape))
            }
            DeviceLocation::Cuda { .. } => {
                #[cfg(feature = "cuda")]
                {
                    let s1 = self
                        .0
                        .as_any()
                        .downcast_ref::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    let s2 = t2
                        .0
                        .as_any()
                        .downcast_ref::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    let (s, shape) = c.cuda_fwd(s1.inner(), l1, s2.inner(), l2)?;
                    Ok((Self::from_cuda(s), shape))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(Error::NotCompiledWithCudaSupport.bt())
                }
            }
            DeviceLocation::Metal { .. } => {
                #[cfg(feature = "metal")]
                {
                    let s1 = self
                        .0
                        .as_any()
                        .downcast_ref::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    let s2 = t2
                        .0
                        .as_any()
                        .downcast_ref::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    let (s, shape) = c.metal_fwd(s1.inner(), l1, s2.inner(), l2)?;
                    Ok((Self::from_metal(s), shape))
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(Error::NotCompiledWithMetalSupport.bt())
                }
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn apply_op3(
        &self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        t3: &Self,
        l3: &Layout,
        c: &dyn CustomOp3,
    ) -> Result<(Self, Shape)> {
        self.same_device(t2, c.name())?;
        self.same_device(t3, c.name())?;
        let location = self.0.device_dyn().location_dyn();
        match location {
            DeviceLocation::Cpu => {
                let s1 = self.as_cpu_storage()?;
                let s2 = t2.as_cpu_storage()?;
                let s3 = t3.as_cpu_storage()?;
                let (s, shape) = c.cpu_fwd(s1, l1, s2, l2, s3, l3)?;
                Ok((Self::from_cpu(s), shape))
            }
            DeviceLocation::Cuda { .. } => {
                #[cfg(feature = "cuda")]
                {
                    let s1 = self
                        .0
                        .as_any()
                        .downcast_ref::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    let s2 = t2
                        .0
                        .as_any()
                        .downcast_ref::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    let s3 = t3
                        .0
                        .as_any()
                        .downcast_ref::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    let (s, shape) =
                        c.cuda_fwd(s1.inner(), l1, s2.inner(), l2, s3.inner(), l3)?;
                    Ok((Self::from_cuda(s), shape))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(Error::NotCompiledWithCudaSupport.bt())
                }
            }
            DeviceLocation::Metal { .. } => {
                #[cfg(feature = "metal")]
                {
                    let s1 = self
                        .0
                        .as_any()
                        .downcast_ref::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    let s2 = t2
                        .0
                        .as_any()
                        .downcast_ref::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    let s3 = t3
                        .0
                        .as_any()
                        .downcast_ref::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    let (s, shape) =
                        c.metal_fwd(s1.inner(), l1, s2.inner(), l2, s3.inner(), l3)?;
                    Ok((Self::from_metal(s), shape))
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(Error::NotCompiledWithMetalSupport.bt())
                }
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn inplace_op1(&mut self, l: &Layout, c: &dyn InplaceOp1) -> Result<()> {
        let location = self.0.device_dyn().location_dyn();
        match location {
            DeviceLocation::Cpu => {
                let s = self.as_cpu_storage_mut()?;
                c.cpu_fwd(s, l)
            }
            DeviceLocation::Cuda { .. } => {
                #[cfg(feature = "cuda")]
                {
                    let s = self
                        .0
                        .as_any_mut()
                        .downcast_mut::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    c.cuda_fwd(s.inner_mut(), l)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(Error::NotCompiledWithCudaSupport.bt())
                }
            }
            DeviceLocation::Metal { .. } => {
                #[cfg(feature = "metal")]
                {
                    let s = self
                        .0
                        .as_any_mut()
                        .downcast_mut::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    c.metal_fwd(s.inner_mut(), l)
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(Error::NotCompiledWithMetalSupport.bt())
                }
            }
            _ => Err(Error::Msg(
                "inplace-op is not supported on this backend".to_string(),
            )
            .bt()),
        }
    }

    pub(crate) fn inplace_op2(
        &mut self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        c: &dyn InplaceOp2,
    ) -> Result<()> {
        self.same_device(t2, c.name())?;
        let location = self.0.device_dyn().location_dyn();
        match location {
            DeviceLocation::Cpu => {
                let s2 = t2.as_cpu_storage()?;
                let s1 = self.as_cpu_storage_mut()?;
                c.cpu_fwd(s1, l1, s2, l2)
            }
            DeviceLocation::Cuda { .. } => {
                #[cfg(feature = "cuda")]
                {
                    let s2 = t2
                        .0
                        .as_any()
                        .downcast_ref::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    let s2_inner: &crate::CudaStorage = s2.inner();
                    // SAFETY: s2_inner borrows from t2 (not self), so it's fine
                    // to borrow self mutably next.
                    let s2_ptr = s2_inner as *const crate::CudaStorage;
                    let s1 = self
                        .0
                        .as_any_mut()
                        .downcast_mut::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    c.cuda_fwd(s1.inner_mut(), l1, unsafe { &*s2_ptr }, l2)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(Error::NotCompiledWithCudaSupport.bt())
                }
            }
            DeviceLocation::Metal { .. } => {
                #[cfg(feature = "metal")]
                {
                    let s2 = t2
                        .0
                        .as_any()
                        .downcast_ref::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    let s2_inner: &crate::MetalStorage = s2.inner();
                    let s2_ptr = s2_inner as *const crate::MetalStorage;
                    let s1 = self
                        .0
                        .as_any_mut()
                        .downcast_mut::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    c.metal_fwd(s1.inner_mut(), l1, unsafe { &*s2_ptr }, l2)
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(Error::NotCompiledWithMetalSupport.bt())
                }
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn inplace_op3(
        &mut self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        t3: &Self,
        l3: &Layout,
        c: &dyn InplaceOp3,
    ) -> Result<()> {
        self.same_device(t2, c.name())?;
        self.same_device(t3, c.name())?;
        let location = self.0.device_dyn().location_dyn();
        match location {
            DeviceLocation::Cpu => {
                let s2 = t2.as_cpu_storage()?;
                let s3 = t3.as_cpu_storage()?;
                let s1 = self.as_cpu_storage_mut()?;
                c.cpu_fwd(s1, l1, s2, l2, s3, l3)
            }
            DeviceLocation::Cuda { .. } => {
                #[cfg(feature = "cuda")]
                {
                    let s2 = t2
                        .0
                        .as_any()
                        .downcast_ref::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    let s3 = t3
                        .0
                        .as_any()
                        .downcast_ref::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    let s2_ptr = s2.inner() as *const crate::CudaStorage;
                    let s3_ptr = s3.inner() as *const crate::CudaStorage;
                    let s1 = self
                        .0
                        .as_any_mut()
                        .downcast_mut::<fuel_cuda::CudaBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected cuda storage".into()).bt())?;
                    c.cuda_fwd(
                        s1.inner_mut(),
                        l1,
                        unsafe { &*s2_ptr },
                        l2,
                        unsafe { &*s3_ptr },
                        l3,
                    )
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(Error::NotCompiledWithCudaSupport.bt())
                }
            }
            DeviceLocation::Metal { .. } => {
                #[cfg(feature = "metal")]
                {
                    let s2 = t2
                        .0
                        .as_any()
                        .downcast_ref::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    let s3 = t3
                        .0
                        .as_any()
                        .downcast_ref::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    let s2_ptr = s2.inner() as *const crate::MetalStorage;
                    let s3_ptr = s3.inner() as *const crate::MetalStorage;
                    let s1 = self
                        .0
                        .as_any_mut()
                        .downcast_mut::<fuel_metal::MetalBackendStorage>()
                        .ok_or_else(|| Error::Msg("expected metal storage".into()).bt())?;
                    c.metal_fwd(
                        s1.inner_mut(),
                        l1,
                        unsafe { &*s2_ptr },
                        l2,
                        unsafe { &*s3_ptr },
                        l3,
                    )
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(Error::NotCompiledWithMetalSupport.bt())
                }
            }
            _ => unreachable!(),
        }
    }

    // -----------------------------------------------------------------------
    // Unary / Binary dispatch
    // -----------------------------------------------------------------------

    pub(crate) fn unary_impl<B: op::UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let op = op::UnaryOp::from_name(B::NAME).ok_or_else(|| {
            Error::Msg(format!("unknown unary op '{}'", B::NAME))
        })?;
        Ok(Storage(self.0.unary_op_dyn(layout, op)?))
    }

    pub(crate) fn binary_impl<B: op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, B::NAME)?;
        self.same_dtype(rhs, B::NAME)?;
        let op = op::BinaryOp::from_name(B::NAME).ok_or_else(|| {
            Error::Msg(format!("unknown binary op '{}'", B::NAME))
        })?;
        Ok(Storage(
            self.0
                .binary_op_dyn(&*rhs.0, lhs_layout, rhs_layout, op)?,
        ))
    }

    // -----------------------------------------------------------------------
    // Convolutions, pooling, upsampling
    // -----------------------------------------------------------------------

    pub(crate) fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv1d")?;
        self.same_dtype(kernel, "conv1d")?;
        Ok(Storage(
            self.0.conv1d_dyn(l, &*kernel.0, kernel_l, params)?,
        ))
    }

    pub(crate) fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv-transpose1d")?;
        self.same_dtype(kernel, "conv-transpose1d")?;
        Ok(Storage(
            self.0
                .conv_transpose1d_dyn(l, &*kernel.0, kernel_l, params)?,
        ))
    }

    pub(crate) fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv2d")?;
        self.same_dtype(kernel, "conv2d")?;
        Ok(Storage(
            self.0.conv2d_dyn(l, &*kernel.0, kernel_l, params)?,
        ))
    }

    pub(crate) fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv_transpose2d")?;
        self.same_dtype(kernel, "conv_transpose2d")?;
        Ok(Storage(
            self.0
                .conv_transpose2d_dyn(l, &*kernel.0, kernel_l, params)?,
        ))
    }

    pub(crate) fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        Ok(Storage(
            self.0.avg_pool2d_dyn(layout, kernel_size, stride)?,
        ))
    }

    pub(crate) fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        Ok(Storage(
            self.0.max_pool2d_dyn(layout, kernel_size, stride)?,
        ))
    }

    pub(crate) fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        Ok(Storage(self.0.upsample_nearest1d_dyn(layout, sz)?))
    }

    pub(crate) fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        Ok(Storage(self.0.upsample_nearest2d_dyn(layout, h, w)?))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        Ok(Storage(
            self.0
                .upsample_bilinear2d_dyn(layout, h, w, align_corners, scale_h, scale_w)?,
        ))
    }

    // -----------------------------------------------------------------------
    // Gather / Scatter / Index
    // -----------------------------------------------------------------------

    pub(crate) fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        layout_t: &Layout,
        f: &Self,
        layout_f: &Layout,
    ) -> Result<Self> {
        self.same_device(t, "where")?;
        self.same_device(f, "where")?;
        t.same_dtype(f, "where")?;
        Ok(Storage(
            self.0
                .where_cond_dyn(layout, &*t.0, layout_t, &*f.0, layout_f)?,
        ))
    }

    pub(crate) fn gather(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(indexes, "index-add")?;
        Ok(Storage(
            self.0.gather_dyn(l, &*indexes.0, indexes_l, d)?,
        ))
    }

    pub(crate) fn scatter_set(
        &mut self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<()> {
        self.same_device(indexes, "scatter-set")?;
        self.same_device(source, "scatter-set")?;
        self.0
            .scatter_set_dyn(l, &*source.0, source_l, &*indexes.0, indexes_l, d)
    }

    pub(crate) fn scatter_add(
        &mut self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<()> {
        self.same_device(indexes, "scatter-add")?;
        self.same_device(source, "scatter-add")?;
        self.0
            .scatter_add_set_dyn(l, &*source.0, source_l, &*indexes.0, indexes_l, d)
    }

    pub(crate) fn index_add(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(indexes, "index-add")?;
        self.same_device(source, "index-add")?;
        Ok(Storage(
            self.0
                .index_add_dyn(l, &*indexes.0, indexes_l, &*source.0, source_l, d)?,
        ))
    }

    pub(crate) fn index_select(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(rhs, "index-select")?;
        Ok(Storage(
            self.0.index_select_dyn(&*rhs.0, lhs_l, rhs_l, d)?,
        ))
    }

    // -----------------------------------------------------------------------
    // Matmul and copy
    // -----------------------------------------------------------------------

    pub(crate) fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, "matmul")?;
        self.same_dtype(rhs, "matmul")?;
        Ok(Storage(
            self.0
                .matmul_dyn(&*rhs.0, bmnk, lhs_layout, rhs_layout)?,
        ))
    }

    // self, the source can be strided whereas dst is contiguous.
    pub(crate) fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()> {
        self.0.copy_strided_src_dyn(&mut *dst.0, dst_offset, src_l)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        self.0
            .copy2d_dyn(&mut *dst.0, d1, d2, src_s, dst_s, src_o, dst_o)
    }
}
