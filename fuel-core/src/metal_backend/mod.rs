//! Thin delegation layer to the `fuel-metal` crate.
//!
//! All Metal types, kernel implementations, and device management live in
//! `fuel_metal`. This module re-exports them and provides the
//! [`BackendStorage`] and [`BackendDevice`] trait implementations that
//! fuel-core requires.

// Re-export everything from fuel-metal so that existing code using
// `crate::metal_backend::MetalDevice`, `MetalStorage`, etc. continues to work.
pub use fuel_metal::*;

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};

// ---------------------------------------------------------------------------
// BackendStorage delegation
// ---------------------------------------------------------------------------
//
// Every method delegates to the identically-named *inherent* method on
// `MetalStorage` (defined in fuel-metal).  Rust's method resolution prefers
// inherent methods over trait methods, so `self.foo()` inside this trait impl
// always resolves to the fuel-metal inherent impl — no infinite recursion.

impl BackendStorage for MetalStorage {
    type Device = MetalDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        self.try_clone(layout)
    }

    fn dtype(&self) -> DType {
        self.dtype()
    }

    fn device(&self) -> &MetalDevice {
        self.device()
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.to_cpu_storage()
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        self.affine(layout, mul, add)
    }

    fn powf(&self, layout: &Layout, exp: f64) -> Result<Self> {
        self.powf(layout, exp)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        self.elu(layout, alpha)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, axes: &[usize]) -> Result<Self> {
        self.reduce_op(op, layout, axes)
    }

    fn cmp(
        &self,
        op: CmpOp,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.cmp(op, rhs, lhs_layout, rhs_layout)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        self.to_dtype(layout, dtype)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        self.unary_impl::<B>(layout)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.binary_impl::<B>(rhs, lhs_layout, rhs_layout)
    }

    fn where_cond(
        &self,
        cond_layout: &Layout,
        on_true: &Self,
        on_true_layout: &Layout,
        on_false: &Self,
        on_false_layout: &Layout,
    ) -> Result<Self> {
        self.where_cond(cond_layout, on_true, on_true_layout, on_false, on_false_layout)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.conv1d(l, kernel, kernel_l, params)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.conv_transpose1d(l, kernel, kernel_l, params)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.conv2d(l, kernel, kernel_l, params)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.conv_transpose2d(l, kernel, kernel_l, params)
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.avg_pool2d(layout, kernel, stride)
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.max_pool2d(layout, kernel, stride)
    }

    fn upsample_nearest1d(&self, layout: &Layout, target_size: usize) -> Result<Self> {
        self.upsample_nearest1d(layout, target_size)
    }

    fn upsample_nearest2d(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
    ) -> Result<Self> {
        self.upsample_nearest2d(layout, target_h, target_w)
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        self.upsample_bilinear2d(layout, target_h, target_w, align_corners, scale_h, scale_w)
    }

    fn gather(
        &self,
        src_layout: &Layout,
        ids: &Self,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        self.gather(src_layout, ids, ids_layout, dim)
    }

    fn scatter_set(
        &mut self,
        self_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        ids: &Self,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.scatter_set(self_layout, src, src_layout, ids, ids_layout, dim)
    }

    fn scatter_add_set(
        &mut self,
        self_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        ids: &Self,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.scatter_add_set(self_layout, src, src_layout, ids, ids_layout, dim)
    }

    fn index_select(
        &self,
        ids: &Self,
        src_layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        self.index_select(ids, src_layout, ids_layout, dim)
    }

    fn index_add(
        &self,
        self_layout: &Layout,
        ids: &Self,
        ids_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        self.index_add(self_layout, ids, ids_layout, src, src_layout, dim)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.matmul(rhs, bmnk, lhs_layout, rhs_layout)
    }

    fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()> {
        self.copy_strided_src(dst, dst_offset, src_layout)
    }

    #[allow(clippy::too_many_arguments)]
    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        self.copy2d(dst, d1, d2, src_stride1, dst_stride1, src_offset, dst_offset)
    }

    fn const_set(&mut self, value: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        self.const_set(value, layout)
    }
}

// ---------------------------------------------------------------------------
// BackendDevice delegation
// ---------------------------------------------------------------------------

impl BackendDevice for MetalDevice {
    type Storage = MetalStorage;

    fn new(ordinal: usize) -> Result<Self> {
        MetalDevice::new(ordinal)
    }

    fn location(&self) -> crate::DeviceLocation {
        self.location()
    }

    fn same_device(&self, other: &Self) -> bool {
        self.same_device(other)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<MetalStorage> {
        self.zeros_impl(shape, dtype)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<MetalStorage> {
        unsafe { self.alloc_uninit(shape, dtype) }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<MetalStorage> {
        self.storage_from_slice(data)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<MetalStorage> {
        self.storage_from_cpu_storage(storage)
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<MetalStorage> {
        self.storage_from_cpu_storage_owned(storage)
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        hi: f64,
    ) -> Result<MetalStorage> {
        self.rand_uniform(shape, dtype, lo, hi)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<MetalStorage> {
        self.rand_normal(shape, dtype, mean, std)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        self.set_seed(seed)
    }

    fn get_current_seed(&self) -> Result<u64> {
        self.get_current_seed()
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize()
    }
}
