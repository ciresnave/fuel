//! `DynBackendStorage` and `DynBackendDevice` implementations for the Metal backend.
//!
//! This module defines newtype wrappers `MetalBackendStorage` and `MetalBackendDevice`
//! that implement the object-safe `DynBackend*` traits from `fuel-core-types`.
//!
//! Same orphan-rule motivation as `fuel-cuda`'s `CudaBackendStorage`:
//! both the traits and inner types live in `fuel-core-types`, so the impl must
//! use newtypes defined in *this* crate.
//!
//! For unary/binary ops, Metal dispatch is purely kernel-name-driven. We map
//! `UnaryOp`/`BinaryOp` enum variants to kernel name strings and call the
//! non-generic `MetalStorage::unary()` / `MetalStorage::binary()` methods.

use fuel_core_types::conv::{
    ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D,
};
use fuel_core_types::dyn_backend::{DynBackendDevice, DynBackendStorage};
use fuel_core_types::op::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use fuel_core_types::{CpuStorage, DType, DeviceLocation, Layout, Result, Scalar, Shape};
use std::any::Any;
use std::sync::Arc;

use crate::device::MetalDevice;
use crate::storage::MetalStorage;

// ---------------------------------------------------------------------------
// MetalBackendStorage — newtype wrapper
// ---------------------------------------------------------------------------

/// Newtype wrapper around [`MetalStorage`] implementing [`DynBackendStorage`].
#[derive(Debug)]
pub struct MetalBackendStorage {
    pub storage: MetalStorage,
    device_wrapper: MetalBackendDevice,
}

impl MetalBackendStorage {
    pub fn new(storage: MetalStorage) -> Self {
        let device_wrapper = MetalBackendDevice(storage.device().clone());
        Self {
            storage,
            device_wrapper,
        }
    }

    pub fn into_inner(self) -> MetalStorage {
        self.storage
    }

    pub fn inner(&self) -> &MetalStorage {
        &self.storage
    }

    pub fn inner_mut(&mut self) -> &mut MetalStorage {
        &mut self.storage
    }
}

impl From<MetalStorage> for MetalBackendStorage {
    fn from(s: MetalStorage) -> Self {
        Self::new(s)
    }
}

impl From<MetalBackendStorage> for MetalStorage {
    fn from(s: MetalBackendStorage) -> Self {
        s.storage
    }
}

// ---------------------------------------------------------------------------
// MetalBackendDevice — newtype wrapper
// ---------------------------------------------------------------------------

/// Newtype wrapper around [`MetalDevice`] implementing [`DynBackendDevice`].
#[derive(Debug, Clone)]
pub struct MetalBackendDevice(pub MetalDevice);

impl From<MetalDevice> for MetalBackendDevice {
    fn from(d: MetalDevice) -> Self {
        Self(d)
    }
}

impl From<MetalBackendDevice> for MetalDevice {
    fn from(d: MetalBackendDevice) -> Self {
        d.0
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn downcast(s: &dyn DynBackendStorage) -> Result<&MetalBackendStorage> {
    s.as_any()
        .downcast_ref::<MetalBackendStorage>()
        .ok_or_else(|| {
            fuel_core_types::Error::DeviceMismatchBinaryOp {
                lhs: DeviceLocation::Metal { gpu_id: 0 },
                rhs: s.device_dyn().location_dyn(),
                op: "metal_dyn_backend",
            }
            .bt()
        })
}

fn downcast_mut(s: &mut dyn DynBackendStorage) -> Result<&mut MetalBackendStorage> {
    let loc = s.device_dyn().location_dyn();
    s.as_any_mut()
        .downcast_mut::<MetalBackendStorage>()
        .ok_or_else(|| {
            fuel_core_types::Error::DeviceMismatchBinaryOp {
                lhs: DeviceLocation::Metal { gpu_id: 0 },
                rhs: loc,
                op: "metal_dyn_backend",
            }
            .bt()
        })
}

fn wrap(s: MetalStorage) -> Box<dyn DynBackendStorage> {
    Box::new(MetalBackendStorage::new(s))
}

// ---------------------------------------------------------------------------
// Kernel-name mappings
// ---------------------------------------------------------------------------

/// Maps a `UnaryOp` enum variant to its Metal kernel name.
fn unary_kernel_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Exp => "uexp",
        UnaryOp::Log => "ulog",
        UnaryOp::Sin => "usin",
        UnaryOp::Cos => "ucos",
        UnaryOp::Abs => "uabs",
        UnaryOp::Neg => "uneg",
        UnaryOp::Recip => "urecip",
        UnaryOp::Sqr => "usqr",
        UnaryOp::Sqrt => "usqrt",
        UnaryOp::Gelu => "ugelu",
        UnaryOp::GeluErf => "ugelu_erf",
        UnaryOp::Erf => "uerf",
        UnaryOp::Relu => "urelu",
        UnaryOp::Silu => "usilu",
        UnaryOp::Tanh => "utanh",
        UnaryOp::Floor => "ufloor",
        UnaryOp::Ceil => "uceil",
        UnaryOp::Round => "uround",
        UnaryOp::Sign => "usign",
    }
}

/// Maps a `BinaryOp` enum variant to its Metal kernel name.
fn binary_kernel_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "badd",
        BinaryOp::Sub => "bsub",
        BinaryOp::Mul => "bmul",
        BinaryOp::Div => "bdiv",
        BinaryOp::Minimum => "bminimum",
        BinaryOp::Maximum => "bmaximum",
    }
}

// ---------------------------------------------------------------------------
// impl DynBackendStorage for MetalBackendStorage
// ---------------------------------------------------------------------------

impl DynBackendStorage for MetalBackendStorage {
    fn try_clone_dyn(&self, layout: &Layout) -> Result<Box<dyn DynBackendStorage>> {
        self.storage.try_clone(layout).map(wrap)
    }

    fn dtype_dyn(&self) -> DType {
        self.storage.dtype()
    }

    fn device_dyn(&self) -> &dyn DynBackendDevice {
        &self.device_wrapper
    }

    fn device_arc_dyn(&self) -> Arc<dyn DynBackendDevice> {
        Arc::new(self.device_wrapper.clone())
    }

    fn to_cpu_storage_dyn(&self) -> Result<CpuStorage> {
        self.storage.to_cpu_storage()
    }

    fn affine_dyn(
        &self,
        layout: &Layout,
        mul: f64,
        add: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.storage.affine(layout, mul, add).map(wrap)
    }

    fn powf_dyn(&self, layout: &Layout, e: f64) -> Result<Box<dyn DynBackendStorage>> {
        self.storage.powf(layout, e).map(wrap)
    }

    fn elu_dyn(&self, layout: &Layout, alpha: f64) -> Result<Box<dyn DynBackendStorage>> {
        self.storage.elu(layout, alpha).map(wrap)
    }

    fn reduce_op_dyn(
        &self,
        op: ReduceOp,
        layout: &Layout,
        axes: &[usize],
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.storage.reduce_op(op, layout, axes).map(wrap)
    }

    fn cmp_dyn(
        &self,
        op: CmpOp,
        rhs: &dyn DynBackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let rhs = downcast(rhs)?;
        self.storage
            .cmp(op, &rhs.storage, lhs_layout, rhs_layout)
            .map(wrap)
    }

    fn to_dtype_dyn(&self, layout: &Layout, dtype: DType) -> Result<Box<dyn DynBackendStorage>> {
        self.storage.to_dtype(layout, dtype).map(wrap)
    }

    fn unary_op_dyn(&self, layout: &Layout, op: UnaryOp) -> Result<Box<dyn DynBackendStorage>> {
        let kname = unary_kernel_name(op);
        self.storage.unary(kname, layout).map(wrap)
    }

    fn binary_op_dyn(
        &self,
        rhs: &dyn DynBackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: BinaryOp,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let rhs = downcast(rhs)?;
        let kname = binary_kernel_name(op);
        self.storage
            .binary(kname, &rhs.storage, lhs_layout, rhs_layout)
            .map(wrap)
    }

    fn where_cond_dyn(
        &self,
        cond_layout: &Layout,
        on_true: &dyn DynBackendStorage,
        on_true_layout: &Layout,
        on_false: &dyn DynBackendStorage,
        on_false_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let t = downcast(on_true)?;
        let f = downcast(on_false)?;
        self.storage
            .where_cond(
                cond_layout,
                &t.storage,
                on_true_layout,
                &f.storage,
                on_false_layout,
            )
            .map(wrap)
    }

    fn conv1d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let kernel = downcast(kernel)?;
        self.storage
            .conv1d(l, &kernel.storage, kernel_l, params)
            .map(wrap)
    }

    fn conv_transpose1d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let kernel = downcast(kernel)?;
        self.storage
            .conv_transpose1d(l, &kernel.storage, kernel_l, params)
            .map(wrap)
    }

    fn conv2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let kernel = downcast(kernel)?;
        self.storage
            .conv2d(l, &kernel.storage, kernel_l, params)
            .map(wrap)
    }

    fn conv_transpose2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let kernel = downcast(kernel)?;
        self.storage
            .conv_transpose2d(l, &kernel.storage, kernel_l, params)
            .map(wrap)
    }

    fn avg_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.storage.avg_pool2d(layout, kernel, stride).map(wrap)
    }

    fn max_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.storage.max_pool2d(layout, kernel, stride).map(wrap)
    }

    fn upsample_nearest1d_dyn(
        &self,
        layout: &Layout,
        target_size: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.storage
            .upsample_nearest1d(layout, target_size)
            .map(wrap)
    }

    fn upsample_nearest2d_dyn(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.storage
            .upsample_nearest2d(layout, target_h, target_w)
            .map(wrap)
    }

    fn upsample_bilinear2d_dyn(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.storage
            .upsample_bilinear2d(layout, target_h, target_w, align_corners, scale_h, scale_w)
            .map(wrap)
    }

    fn gather_dyn(
        &self,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let ids = downcast(ids)?;
        self.storage
            .gather(src_layout, &ids.storage, ids_layout, dim)
            .map(wrap)
    }

    fn scatter_set_dyn(
        &mut self,
        self_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        let src = downcast(src)?;
        let ids = downcast(ids)?;
        // MetalStorage::scatter_set takes (l, ids, ids_l, src, src_l, dim)
        self.storage.scatter_set(
            self_layout,
            &ids.storage,
            ids_layout,
            &src.storage,
            src_layout,
            dim,
        )
    }

    fn scatter_add_set_dyn(
        &mut self,
        self_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        let src = downcast(src)?;
        let ids = downcast(ids)?;
        self.storage.scatter_add_set(
            self_layout,
            &ids.storage,
            ids_layout,
            &src.storage,
            src_layout,
            dim,
        )
    }

    fn index_select_dyn(
        &self,
        ids: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let ids = downcast(ids)?;
        self.storage
            .index_select(&ids.storage, src_layout, ids_layout, dim)
            .map(wrap)
    }

    fn index_add_dyn(
        &self,
        self_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let ids = downcast(ids)?;
        let src = downcast(src)?;
        self.storage
            .index_add(
                self_layout,
                &ids.storage,
                ids_layout,
                &src.storage,
                src_layout,
                dim,
            )
            .map(wrap)
    }

    fn matmul_dyn(
        &self,
        rhs: &dyn DynBackendStorage,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let rhs = downcast(rhs)?;
        self.storage
            .matmul(&rhs.storage, bmnk, lhs_layout, rhs_layout)
            .map(wrap)
    }

    fn copy_strided_src_dyn(
        &self,
        dst: &mut dyn DynBackendStorage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()> {
        let dst = downcast_mut(dst)?;
        self.storage
            .copy_strided_src(&mut dst.storage, dst_offset, src_layout)
    }

    fn copy2d_dyn(
        &self,
        dst: &mut dyn DynBackendStorage,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        let dst = downcast_mut(dst)?;
        self.storage.copy2d(
            &mut dst.storage,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        )
    }

    fn const_set_dyn(&mut self, value: Scalar, layout: &Layout) -> Result<()> {
        self.storage.const_set(value, layout)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// impl DynBackendDevice for MetalBackendDevice
// ---------------------------------------------------------------------------

impl DynBackendDevice for MetalBackendDevice {
    fn location_dyn(&self) -> DeviceLocation {
        self.0.location()
    }

    fn same_device_dyn(&self, other: &dyn DynBackendDevice) -> bool {
        other
            .as_any()
            .downcast_ref::<MetalBackendDevice>()
            .is_some_and(|o| self.0.same_device(&o.0))
    }

    fn supports_bf16(&self) -> bool {
        true
    }

    fn zeros_impl_dyn(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn DynBackendStorage>> {
        self.0.zeros_impl(shape, dtype).map(wrap)
    }

    unsafe fn alloc_uninit_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn DynBackendStorage>> {
        unsafe { self.0.alloc_uninit(shape, dtype) }.map(wrap)
    }

    fn storage_from_cpu_storage_dyn(&self, cpu: &CpuStorage) -> Result<Box<dyn DynBackendStorage>> {
        self.0.storage_from_cpu_storage(cpu).map(wrap)
    }

    fn storage_from_cpu_storage_owned_dyn(
        &self,
        cpu: CpuStorage,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.0.storage_from_cpu_storage_owned(cpu).map(wrap)
    }

    fn rand_uniform_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        hi: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.0.rand_uniform(shape, dtype, lo, hi).map(wrap)
    }

    fn rand_normal_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.0.rand_normal(shape, dtype, mean, std).map(wrap)
    }

    fn set_seed_dyn(&self, seed: u64) -> Result<()> {
        self.0.set_seed(seed)
    }

    fn get_current_seed_dyn(&self) -> Result<u64> {
        self.0.get_current_seed()
    }

    fn synchronize_dyn(&self) -> Result<()> {
        self.0.synchronize()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
