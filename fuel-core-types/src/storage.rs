//! Type-erased tensor storage wrapper (Phase 7.5 work item G fix-up).
//!
//! `Storage` was originally `fuel-core::Storage` and lived alongside the
//! eager-dispatch op methods that mutate it. Phase 7.5 work item G moves
//! the struct and the methods that depend only on fuel-core-types here so
//! that `fuel_graph::Graph` can own a NodeId-keyed map of
//! `Arc<RwLock<Storage>>` slots without inverting the dependency graph.
//!
//! The eager-dispatch methods that need fuel-core types (`CustomOp1/2/3`)
//! stay in fuel-core via the `StorageApplyOps` trait extension. They are
//! scheduled for removal in Phase 7.5 work item B6 (drop eager dispatch).
//!
//! `Storage::device()` returns `Arc<dyn DynBackendDevice>` rather than
//! the `Device` wrapper (which still lives in fuel-core); callers wrap as
//! needed. This is the one API change vs. the pre-G surface.

use crate::dyn_backend::{DynBackendDevice, DynBackendStorage};
use crate::op::{self, BinaryOp, CmpOp, ReduceOp};
use crate::scalar::Scalar;
use crate::{
    conv, DType, Error, HostBuffer, InplaceOp1, InplaceOp2, InplaceOp3, Layout, Result,
};
use std::sync::Arc;

/// Owns a typed contiguous buffer on one device. The boxed
/// `DynBackendStorage` is the actual byte holder; `Storage` is a thin
/// wrapper that gives the eager-dispatch op methods (matmul, conv,
/// unary, binary, …) somewhere to live.
///
/// We do not implement `Clone` because cloning storage may fail on
/// out-of-memory; use [`Self::try_clone`] for the fallible version.
#[derive(Debug)]
pub struct Storage(pub Box<dyn DynBackendStorage>);

impl Storage {
    /// Construct storage from any concrete `DynBackendStorage` implementor.
    ///
    /// This is the backend-agnostic entry point — backends provide a type
    /// implementing `DynBackendStorage`, and `Storage::new` boxes it.
    pub fn new<B: DynBackendStorage + 'static>(b: B) -> Self {
        Storage(Box::new(b))
    }

    /// Wrap an already-boxed `dyn DynBackendStorage`. Used by callers
    /// (notably the quantized fast-paths) that produce a `Box<dyn ..>`
    /// directly from trait dispatch.
    pub fn from_dyn(b: Box<dyn DynBackendStorage>) -> Self {
        Storage(b)
    }

    /// Borrow the inner storage as a `DynBackendStorage` trait object.
    ///
    /// Backends that need to peel back to their concrete storage type can
    /// downcast via `storage.as_dyn().as_any().downcast_ref::<MyStorage>()`.
    pub fn as_dyn(&self) -> &dyn DynBackendStorage {
        &*self.0
    }

    /// Mutable variant of [`as_dyn`].
    pub fn as_dyn_mut(&mut self) -> &mut dyn DynBackendStorage {
        &mut *self.0
    }

    /// Downcast the inner storage to a concrete backend type.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.0.as_any().downcast_ref::<T>()
    }

    /// Mutable variant of [`downcast_ref`](Self::downcast_ref).
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.0.as_any_mut().downcast_mut::<T>()
    }

    pub fn try_clone(&self, layout: &Layout) -> Result<Self> {
        Ok(Storage(self.0.try_clone_dyn(layout)?))
    }

    /// Return an `Arc` to the owning device as a trait object.
    /// fuel-core wraps this in its `Device` newtype; other consumers
    /// (fuel-graph, tests) can use the trait object directly.
    pub fn device(&self) -> Arc<dyn DynBackendDevice> {
        self.0.device_arc_dyn()
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype_dyn()
    }

    /// Pre-G this method consulted `Device::same_device` for the Metal
    /// pointer-identity check; post-G it goes through the
    /// `DynBackendDevice::same_device_dyn` trait method which has the
    /// same semantics.
    pub fn same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs_dev = self.0.device_dyn();
        let rhs_dev = rhs.0.device_dyn();
        let lhs_loc = lhs_dev.location_dyn();
        let rhs_loc = rhs_dev.location_dyn();
        let same = if matches!(lhs_loc, crate::DeviceLocation::Metal { .. }) {
            // On metal, require physical identity (matches pre-G behaviour).
            lhs_dev.same_device_dyn(rhs_dev)
        } else {
            lhs_loc == rhs_loc
        };
        if !same {
            Err(Error::DeviceMismatchBinaryOp { lhs: lhs_loc, rhs: rhs_loc, op }.bt())
        } else {
            Ok(())
        }
    }

    pub fn same_dtype(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.dtype();
        let rhs = rhs.dtype();
        if lhs != rhs {
            Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op }.bt())
        } else {
            Ok(())
        }
    }

    pub fn const_set(&mut self, v: Scalar, l: &Layout) -> Result<()> {
        self.0.const_set_dyn(v, l)
    }

    pub fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        Ok(Storage(self.0.affine_dyn(layout, mul, add)?))
    }

    pub fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        Ok(Storage(self.0.powf_dyn(layout, e)?))
    }

    pub fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        Ok(Storage(self.0.elu_dyn(layout, alpha)?))
    }

    pub fn cmp(
        &self,
        op: CmpOp,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        Ok(Storage(self.0.cmp_dyn(op, &*rhs.0, lhs_layout, rhs_layout)?))
    }

    pub fn reduce_op(
        &self,
        op: ReduceOp,
        layout: &Layout,
        reduce_dims: &[usize],
    ) -> Result<Self> {
        Ok(Storage(self.0.reduce_op_dyn(op, layout, reduce_dims)?))
    }

    pub fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        Ok(Storage(self.0.to_dtype_dyn(layout, dtype)?))
    }

    pub fn to_cpu_storage(&self) -> Result<HostBuffer> {
        self.0.to_host_buffer_dyn()
    }

    pub fn inplace_op1(&mut self, l: &Layout, c: &dyn InplaceOp1) -> Result<()> {
        c.fwd(&mut *self.0, l)
    }

    pub fn inplace_op2(
        &mut self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        c: &dyn InplaceOp2,
    ) -> Result<()> {
        self.same_device(t2, c.name())?;
        c.fwd(&mut *self.0, l1, &*t2.0, l2)
    }

    pub fn inplace_op3(
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
        c.fwd(&mut *self.0, l1, &*t2.0, l2, &*t3.0, l3)
    }

    // -----------------------------------------------------------------------
    // Unary / Binary dispatch
    // -----------------------------------------------------------------------

    pub fn unary_impl<B: op::UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let op = op::UnaryOp::from_name(B::NAME).ok_or_else(|| {
            Error::Msg(format!("unknown unary op '{}'", B::NAME))
        })?;
        Ok(Storage(self.0.unary_op_dyn(layout, op)?))
    }

    pub fn binary_impl<B: op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, B::NAME)?;
        self.same_dtype(rhs, B::NAME)?;
        let op = BinaryOp::from_name(B::NAME).ok_or_else(|| {
            Error::Msg(format!("unknown binary op '{}'", B::NAME))
        })?;
        Ok(Storage(self.0.binary_op_dyn(&*rhs.0, lhs_layout, rhs_layout, op)?))
    }

    // -----------------------------------------------------------------------
    // Convolutions, pooling, upsampling
    // -----------------------------------------------------------------------

    pub fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &conv::ParamsConv1D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv1d")?;
        self.same_dtype(kernel, "conv1d")?;
        Ok(Storage(self.0.conv1d_dyn(l, &*kernel.0, kernel_l, params)?))
    }

    pub fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv-transpose1d")?;
        self.same_dtype(kernel, "conv-transpose1d")?;
        Ok(Storage(self.0.conv_transpose1d_dyn(l, &*kernel.0, kernel_l, params)?))
    }

    pub fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &conv::ParamsConv2D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv2d")?;
        self.same_dtype(kernel, "conv2d")?;
        Ok(Storage(self.0.conv2d_dyn(l, &*kernel.0, kernel_l, params)?))
    }

    pub fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv_transpose2d")?;
        self.same_dtype(kernel, "conv_transpose2d")?;
        Ok(Storage(self.0.conv_transpose2d_dyn(l, &*kernel.0, kernel_l, params)?))
    }

    pub fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        Ok(Storage(self.0.avg_pool2d_dyn(layout, kernel_size, stride)?))
    }

    pub fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        Ok(Storage(self.0.max_pool2d_dyn(layout, kernel_size, stride)?))
    }

    pub fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        Ok(Storage(self.0.upsample_nearest1d_dyn(layout, sz)?))
    }

    pub fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        Ok(Storage(self.0.upsample_nearest2d_dyn(layout, h, w)?))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        Ok(Storage(self.0.upsample_bilinear2d_dyn(layout, h, w, align_corners, scale_h, scale_w)?))
    }

    // -----------------------------------------------------------------------
    // Gather / Scatter / Index
    // -----------------------------------------------------------------------

    pub fn where_cond(
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
        Ok(Storage(self.0.where_cond_dyn(layout, &*t.0, layout_t, &*f.0, layout_f)?))
    }

    pub fn gather(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(indexes, "index-add")?;
        Ok(Storage(self.0.gather_dyn(l, &*indexes.0, indexes_l, d)?))
    }

    pub fn scatter_set(
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
        self.0.scatter_set_dyn(l, &*source.0, source_l, &*indexes.0, indexes_l, d)
    }

    pub fn scatter_add(
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
        self.0.scatter_add_set_dyn(l, &*source.0, source_l, &*indexes.0, indexes_l, d)
    }

    pub fn index_add(
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
        Ok(Storage(self.0.index_add_dyn(l, &*indexes.0, indexes_l, &*source.0, source_l, d)?))
    }

    pub fn index_select(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(rhs, "index-select")?;
        Ok(Storage(self.0.index_select_dyn(&*rhs.0, lhs_l, rhs_l, d)?))
    }

    // -----------------------------------------------------------------------
    // Matmul and copy
    // -----------------------------------------------------------------------

    pub fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, "matmul")?;
        self.same_dtype(rhs, "matmul")?;
        Ok(Storage(self.0.matmul_dyn(&*rhs.0, bmnk, lhs_layout, rhs_layout)?))
    }

    /// `self`, the source, can be strided whereas `dst` is contiguous.
    pub fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()> {
        self.0.copy_strided_src_dyn(&mut *dst.0, dst_offset, src_l)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        self.0.copy2d_dyn(&mut *dst.0, d1, d2, src_s, dst_s, src_o, dst_o)
    }
}
