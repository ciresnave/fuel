//! `DynBackendStorage` and `DynBackendDevice` implementations for the CUDA backend.
//!
//! `DynBackendStorage` is implemented directly on `CudaStorage`, and
//! `DynBackendDevice` directly on `CudaDevice`. No newtype wrappers are needed:
//! both the trait (`fuel-core-types`) and the concrete type (`fuel-cuda-backend`)
//! live in crates we own, so the orphan rule is satisfied.
//!
//! `CudaBackendStorage` and `CudaBackendDevice` are kept as type aliases so
//! existing downstream code continues to compile unchanged.
//!
//! For unary/binary ops, CUDA dispatch is purely kernel-name-driven. We map
//! `UnaryOp`/`BinaryOp` enum variants to kernel name strings and reuse the
//! exact same CUDA kernel launch infrastructure.

use fuel_core_types::conv::{
    ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D,
};
use fuel_core_types::dyn_backend::{DynBackendDevice, DynBackendStorage};
use fuel_core_types::op::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use fuel_core_types::{HostBuffer, DType, DeviceLocation, Error, Layout, Result, Scalar, Shape};
use baracuda_driver::DeviceBuffer as CudaSlice;
use baracuda_types::{DeviceRepr, KernelArg as PushKernelArg, ValidAsZeroBits};
use crate::device::LaunchConfig;
use std::any::Any;
use std::sync::Arc;

use crate::utils::{Map1, Map2};
use crate::{CudaDevice, CudaStorage, SlicePtrOrNull, WrapErr, kernel_name, kernels};

// ---------------------------------------------------------------------------
// Backward-compat type aliases — downstream code can keep using these names.
// ---------------------------------------------------------------------------

/// Type alias for backward compatibility. `CudaStorage` now implements
/// `DynBackendStorage` directly; this alias lets existing `use` statements
/// compile unchanged.
pub type CudaBackendStorage = CudaStorage;

/// Type alias for backward compatibility. `CudaDevice` now implements
/// `DynBackendDevice` directly; this alias lets existing `use` statements
/// compile unchanged.
pub type CudaBackendDevice = CudaDevice;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn downcast(s: &dyn DynBackendStorage) -> Result<&CudaStorage> {
    s.as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            Error::DeviceMismatchBinaryOp {
                lhs: DeviceLocation::Cuda { gpu_id: 0 },
                rhs: s.device_dyn().location_dyn(),
                op: "cuda_dyn_backend",
            }
            .bt()
        })
}

fn downcast_mut(s: &mut dyn DynBackendStorage) -> Result<&mut CudaStorage> {
    let loc = s.device_dyn().location_dyn();
    s.as_any_mut()
        .downcast_mut::<CudaStorage>()
        .ok_or_else(|| {
            Error::DeviceMismatchBinaryOp {
                lhs: DeviceLocation::Cuda { gpu_id: 0 },
                rhs: loc,
                op: "cuda_dyn_backend",
            }
            .bt()
        })
}

fn wrap(s: CudaStorage) -> Box<dyn DynBackendStorage> {
    Box::new(s)
}

// ---------------------------------------------------------------------------
// Kernel-name-based helpers for unary/binary dispatch
// ---------------------------------------------------------------------------

/// Maps a `UnaryOp` enum variant to its CUDA kernel name.
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

/// Maps a `BinaryOp` enum variant to its CUDA kernel name.
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

/// A `Map1` implementation that dispatches a unary CUDA kernel by name.
///
/// This replicates the logic from `impl<U: UnaryOpT> Map1 for U` but uses a
/// runtime kernel name instead of a compile-time `U::KERNEL` constant.
pub(crate) struct UnaryKernel(pub(crate) &'static str);

impl Map1 for UnaryKernel {
    fn f<T: DeviceRepr + fuel_core_types::WithDType + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
    ) -> Result<CudaSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src = &src.slice(layout.start_offset()..src.len());
        let func = dev.get_or_load_func(&kernel_name::<T>(self.0), &kernels::UNARY)?;
        let mut out = unsafe { dev.alloc::<T>(el_count)? };
        let mut builder = func.builder();
        crate::builder_arg!(builder, el_count);
        crate::builder_arg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        builder.arg(src);
        builder.arg(&mut out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

/// A `Map2` implementation that dispatches a binary CUDA kernel by name.
///
/// Replicates `impl<U: BinaryOpT> Map2 for U` with a runtime kernel name.
pub(crate) struct BinaryKernel(pub(crate) &'static str);

impl Map2 for BinaryKernel {
    fn f<T: DeviceRepr + fuel_core_types::WithDType + ValidAsZeroBits>(
        &self,
        lhs: &CudaSlice<T>,
        lhs_l: &Layout,
        rhs: &CudaSlice<T>,
        rhs_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<CudaSlice<T>> {
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let dims_and_strides = if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
            SlicePtrOrNull::Null
        } else {
            SlicePtrOrNull::Ptr(dev.clone_htod(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
        };
        let lhs = &lhs.slice(lhs_l.start_offset()..lhs.len());
        let rhs = &rhs.slice(rhs_l.start_offset()..rhs.len());
        let func = dev.get_or_load_func(&kernel_name::<T>(self.0), &kernels::BINARY)?;
        let out = unsafe { dev.alloc::<T>(elem_count)? };
        let mut builder = func.builder();
        crate::builder_arg!(builder, elem_count);
        crate::builder_arg!(builder, dims.len());
        dims_and_strides.builder_arg(&mut builder);
        builder.arg(lhs);
        builder.arg(rhs);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// impl DynBackendStorage for CudaStorage
// ---------------------------------------------------------------------------

impl DynBackendStorage for CudaStorage {
    fn try_clone_dyn(&self, layout: &Layout) -> Result<Box<dyn DynBackendStorage>> {
        self.try_clone(layout).map(wrap)
    }

    fn dtype_dyn(&self) -> DType {
        self.dtype()
    }

    fn device_dyn(&self) -> &dyn DynBackendDevice {
        &self.device
    }

    fn device_arc_dyn(&self) -> Arc<dyn DynBackendDevice> {
        Arc::new(self.device.clone())
    }

    fn to_host_buffer_dyn(&self) -> Result<HostBuffer> {
        self.to_cpu_storage()
    }

    fn affine_dyn(
        &self,
        layout: &Layout,
        mul: f64,
        add: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.affine(layout, mul, add).map(wrap)
    }

    fn powf_dyn(&self, layout: &Layout, e: f64) -> Result<Box<dyn DynBackendStorage>> {
        self.powf(layout, e).map(wrap)
    }

    fn elu_dyn(&self, layout: &Layout, alpha: f64) -> Result<Box<dyn DynBackendStorage>> {
        self.elu(layout, alpha).map(wrap)
    }

    fn reduce_op_dyn(
        &self,
        op: ReduceOp,
        layout: &Layout,
        axes: &[usize],
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.reduce_op(op, layout, axes).map(wrap)
    }

    fn cmp_dyn(
        &self,
        op: CmpOp,
        rhs: &dyn DynBackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let rhs = downcast(rhs)?;
        self.cmp(op, rhs, lhs_layout, rhs_layout).map(wrap)
    }

    fn to_dtype_dyn(&self, layout: &Layout, dtype: DType) -> Result<Box<dyn DynBackendStorage>> {
        self.to_dtype(layout, dtype).map(wrap)
    }

    fn unary_op_dyn(&self, layout: &Layout, op: UnaryOp) -> Result<Box<dyn DynBackendStorage>> {
        let kname = unary_kernel_name(op);
        let slice = UnaryKernel(kname).map(&self.slice, &self.device, layout)?;
        Ok(wrap(CudaStorage { slice, device: self.device.clone() }))
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
        let slice = BinaryKernel(kname).map(
            &self.slice,
            lhs_layout,
            &rhs.slice,
            rhs_layout,
            &self.device,
        )?;
        Ok(wrap(CudaStorage { slice, device: self.device.clone() }))
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
        self.where_cond(cond_layout, t, on_true_layout, f, on_false_layout)
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
        self.conv1d(l, kernel, kernel_l, params).map(wrap)
    }

    fn conv_transpose1d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let kernel = downcast(kernel)?;
        self.conv_transpose1d(l, kernel, kernel_l, params).map(wrap)
    }

    fn conv2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let kernel = downcast(kernel)?;
        self.conv2d(l, kernel, kernel_l, params).map(wrap)
    }

    fn conv_transpose2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let kernel = downcast(kernel)?;
        self.conv_transpose2d(l, kernel, kernel_l, params).map(wrap)
    }

    fn avg_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.avg_pool2d(layout, kernel, stride).map(wrap)
    }

    fn max_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.max_pool2d(layout, kernel, stride).map(wrap)
    }

    fn upsample_nearest1d_dyn(
        &self,
        layout: &Layout,
        target_size: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.upsample_nearest1d(layout, target_size).map(wrap)
    }

    fn upsample_nearest2d_dyn(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.upsample_nearest2d(layout, target_h, target_w).map(wrap)
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
        self.upsample_bilinear2d(layout, target_h, target_w, align_corners, scale_h, scale_w)
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
        self.gather(src_layout, ids, ids_layout, dim).map(wrap)
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
        // CudaStorage::scatter_set takes (l, ids, ids_l, src, src_l, dim)
        // while DynBackendStorage takes (self_layout, src, src_layout, ids, ids_layout, dim).
        self.scatter_set(self_layout, ids, ids_layout, src, src_layout, dim)
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
        self.scatter_add_set(self_layout, ids, ids_layout, src, src_layout, dim)
    }

    fn index_select_dyn(
        &self,
        ids: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        let ids = downcast(ids)?;
        self.index_select(ids, src_layout, ids_layout, dim).map(wrap)
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
        self.index_add(self_layout, ids, ids_layout, src, src_layout, dim)
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
        self.matmul(rhs, bmnk, lhs_layout, rhs_layout).map(wrap)
    }

    fn copy_strided_src_dyn(
        &self,
        dst: &mut dyn DynBackendStorage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()> {
        let dst = downcast_mut(dst)?;
        self.copy_strided_src(dst, dst_offset, src_layout)
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
        self.copy2d(dst, d1, d2, src_stride1, dst_stride1, src_offset, dst_offset)
    }

    fn const_set_dyn(&mut self, value: Scalar, layout: &Layout) -> Result<()> {
        self.const_set(value, layout)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// impl DynBackendDevice for CudaDevice
// ---------------------------------------------------------------------------

impl DynBackendDevice for CudaDevice {
    fn location_dyn(&self) -> DeviceLocation {
        self.location()
    }

    fn same_device_dyn(&self, other: &dyn DynBackendDevice) -> bool {
        other
            .as_any()
            .downcast_ref::<CudaDevice>()
            .is_some_and(|o| self.same_device(o))
    }

    fn supports_bf16(&self) -> bool {
        true
    }

    fn zeros_impl_dyn(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn DynBackendStorage>> {
        self.zeros_impl(shape, dtype).map(wrap)
    }

    unsafe fn alloc_uninit_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn DynBackendStorage>> {
        unsafe { self.alloc_uninit(shape, dtype) }.map(wrap)
    }

    fn storage_from_host_buffer_dyn(&self, buf: &HostBuffer) -> Result<Box<dyn DynBackendStorage>> {
        self.storage_from_cpu_storage(buf).map(wrap)
    }

    fn storage_from_host_buffer_owned_dyn(
        &self,
        buf: HostBuffer,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.storage_from_cpu_storage_owned(buf).map(wrap)
    }

    fn rand_uniform_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        hi: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.rand_uniform(shape, dtype, lo, hi).map(wrap)
    }

    fn rand_normal_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.rand_normal(shape, dtype, mean, std).map(wrap)
    }

    fn set_seed_dyn(&self, seed: u64) -> Result<()> {
        self.set_seed(seed)
    }

    fn get_current_seed_dyn(&self) -> Result<u64> {
        self.get_current_seed()
    }

    fn synchronize_dyn(&self) -> Result<()> {
        self.synchronize()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_quantized_kernels(
        &self,
    ) -> Option<&dyn fuel_core_types::quantized::QuantizedDeviceKernels> {
        Some(self)
    }
}
