//! Implementation of Backend traits for CUDA device
//!

use fuel_core_types::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use fuel_core_types::dtype::WithDType;
use fuel_core_types::{HostBuffer, DType, Layout, Result};
use crate::builder_arg as barg;
use fuel_cuda_kernels as kernels;

use baracuda_driver::{DeviceBuffer as CudaSlice, DevicePtr};
use baracuda_types::{DeviceRepr, KernelArg as PushKernelArg, ValidAsZeroBits};
use crate::device::{LaunchArgs, LaunchConfig};
use half::{bf16, f16};


use crate::device::CudaDevice;
use crate::error::{CudaError, WrapErr};
use crate::utils::{Map1, Map1Any, Map2, Map2Any, Map2InPlace, S};

// cudarc-shaped GEMM config structs, replicated locally so Fuel's matmul
// call sites don't need rewriting. The BLAS call is now
// `baracuda_cublas::gemm_strided_batched_ex` — fields flow through
// positionally.
#[allow(dead_code)]
pub(crate) struct GemmConfig<T> {
    pub alpha: T,
    pub beta: T,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub transa: baracuda_cublas::Op,
    pub transb: baracuda_cublas::Op,
}

#[allow(dead_code)]
pub(crate) struct StridedBatchedConfig<T> {
    pub gemm: GemmConfig<T>,
    pub stride_a: i64,
    pub stride_b: i64,
    pub stride_c: i64,
    pub batch_size: i32,
}

pub enum SlicePtrOrNull<T: DeviceRepr> {
    Ptr(CudaSlice<T>),
    Null,
}

impl<T: DeviceRepr> SlicePtrOrNull<T> {
    pub fn builder_arg<'a, 'b: 'a>(&'b self, builder: &mut crate::device::LaunchArgs<'a>) {
        match self {
            SlicePtrOrNull::Ptr(slice) => builder.arg(slice),
            SlicePtrOrNull::Null => builder.arg(&0usize),
        };
    }
}

/// Pack a Layout's dims+strides into a single `Vec<usize>` ready to upload to a CUDA kernel
/// that expects `usize` strides. Casts signed strides through `stride_unsigned()` (which
/// debug-asserts non-negative).
pub(crate) fn dims_strides_usize(l: &Layout) -> Vec<usize> {
    let dims = l.dims();
    let stride = l.stride_unsigned();
    let mut v = Vec::with_capacity(dims.len() + stride.len());
    v.extend_from_slice(dims);
    v.extend_from_slice(&stride);
    v
}

/// Like `dims_strides_usize` but for two layouts sharing one dims slice (binary-op style):
/// `[dims, lhs.stride(), rhs.stride()]`.
pub(crate) fn dims_strides_strides_usize(dims: &[usize], a: &Layout, b: &Layout) -> Vec<usize> {
    let sa = a.stride_unsigned();
    let sb = b.stride_unsigned();
    let mut v = Vec::with_capacity(dims.len() + sa.len() + sb.len());
    v.extend_from_slice(dims);
    v.extend_from_slice(&sa);
    v.extend_from_slice(&sb);
    v
}

// `conv_dims_strides_usize` retired in Phase 5b alongside the PTX
// Conv*/ConvTranspose* structs that consumed its packed-strides arg.

fn push_scalar_arg<'a>(scalar: &'a fuel_core_types::scalar::Scalar, builder: &mut crate::device::LaunchArgs<'a>) {
    use fuel_core_types::scalar::Scalar;
    match scalar {
        Scalar::U8(v) => builder.arg(v),
        Scalar::I8(v) => builder.arg(v),
        Scalar::U32(v) => builder.arg(v),
        Scalar::I16(v) => builder.arg(v),
        Scalar::I32(v) => builder.arg(v),
        Scalar::I64(v) => builder.arg(v),
        Scalar::F32(v) => builder.arg(v),
        Scalar::F64(v) => builder.arg(v),
        Scalar::F16(v) => builder.arg(v),
        Scalar::BF16(v) => builder.arg(v),
        Scalar::F8E4M3(v) => builder.arg(v),
    };
}

impl SlicePtrOrNull<usize> {
    pub fn params_from_layout(dev: &CudaDevice, l: &Layout) -> Result<Self> {
        let ds = if l.is_contiguous() {
            SlicePtrOrNull::Null
        } else {
            SlicePtrOrNull::Ptr(dev.clone_htod(&dims_strides_usize(l))?)
        };
        Ok(ds)
    }
}

#[derive(Debug)]
pub enum CudaStorageSlice {
    U8(CudaSlice<u8>),
    I8(CudaSlice<i8>),
    U32(CudaSlice<u32>),
    I16(CudaSlice<i16>),
    I32(CudaSlice<i32>),
    I64(CudaSlice<i64>),
    BF16(CudaSlice<bf16>),
    F16(CudaSlice<f16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
    F8E4M3(CudaSlice<float8::F8E4M3>),
    // Dummy types that store raw bytes
    F6E2M3(CudaSlice<u8>),
    F6E3M2(CudaSlice<u8>),
    F4(CudaSlice<u8>),
    F8E8M0(CudaSlice<u8>),
}

struct Clone;
impl Map1 for Clone {
    fn f<T: DeviceRepr + WithDType + baracuda_types::ValidAsZeroBits>(
        &self,
        s: &CudaSlice<T>,
        dev: &CudaDevice,
        _: &Layout,
    ) -> Result<CudaSlice<T>> {
        dev.clone_dtod(s)
    }
}

pub fn kernel_name<T: WithDType>(root: &str) -> String {
    let dtype = T::DTYPE.as_str();
    format!("{root}_{dtype}")
}

pub(crate) struct Affine(pub(crate) f64, pub(crate) f64);
impl Map1 for Affine {
    fn f<T: DeviceRepr + WithDType>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
    ) -> Result<CudaSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src = &src.slice(layout.start_offset()..src.len());
        let func = dev.get_or_load_func(&kernel_name::<T>("affine"), &kernels::AFFINE)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        builder.arg(src);
        builder.arg(&out);
        barg!(builder, T::from_f64(self.0));
        barg!(builder, T::from_f64(self.1));
        // SAFETY: ffi.
        unsafe { builder.launch(cfg).w() }?;
        Ok(out)
    }
}

struct Elu(f64);
impl Map1 for Elu {
    fn f<T: DeviceRepr + WithDType>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
    ) -> Result<CudaSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src = &src.slice(layout.start_offset()..src.len());
        let func = dev.get_or_load_func(&kernel_name::<T>("uelu"), &kernels::UNARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        barg!(builder, T::from_f64(self.0));
        builder.arg(src);
        builder.arg(&out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

// Im2Col1D + Im2Col PTX structs retired in Phase 5b — the im2col +
// matmul conv fallback they backed is gone now that baracuda's
// `conv_{1,2}d_*_run` FFI handles conv directly via cuDNN.

struct Powf(f64);
impl Map1 for Powf {
    fn f<T: DeviceRepr + WithDType>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
    ) -> Result<CudaSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src = &src.slice(layout.start_offset()..src.len());
        let func = dev.get_or_load_func(&kernel_name::<T>("upowf"), &kernels::UNARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        barg!(builder, T::from_f64(self.0));
        builder.arg(src);
        builder.arg(&out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct FastReduce<'a>(&'a [usize], ReduceOp);
impl Map1Any for FastReduce<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
        wrap: W,
    ) -> Result<S> {
        let src_stride = layout.stride_unsigned();
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();
        // Source dims and strides with the sum dims at the end.
        let mut dims: Vec<usize> = vec![];
        let mut stride: Vec<usize> = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !self.0.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }
        for &dim_idx in self.0.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }
        let el_to_sum_per_block = src_el / dst_el;
        // The reduction loop requires the shared array to be properly initialized and for
        // this we want the number of threads to be a power of two.
        let block_dim = usize::min(1024, el_to_sum_per_block).next_power_of_two();
        let cfg = LaunchConfig {
            // TODO: Maybe use grid_y if the output is too large?
            // TODO: Specialized implementation when reducing on no or all dimensions or when
            // reducing only aggregate a small number of elements together.
            grid_dim: (dst_el as u32, 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let ds = dev.clone_htod(&[dims.as_slice(), stride.as_slice()].concat())?;
        let src = &src.slice(layout.start_offset()..src.len());
        let (name, check_empty, return_index) = match self.1 {
            ReduceOp::Sum => ("fast_sum", false, false),
            ReduceOp::Min => ("fast_min", true, false),
            ReduceOp::Max => ("fast_max", true, false),
            ReduceOp::ArgMin => ("fast_argmin", true, true),
            ReduceOp::ArgMax => ("fast_argmax", true, true),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::REDUCE)?;
        if return_index {
            // SAFETY: filled in by the follow up kernel.
            let out = unsafe { dev.alloc::<u32>(dst_el)? };
            let mut builder = func.builder();
            barg!(builder, src_el);
            barg!(builder, el_to_sum_per_block);
            barg!(builder, src_dims.len());
            builder.arg(&ds);
            builder.arg(src);
            builder.arg(&out);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(S::U32(out))
        } else {
            // SAFETY: filled in by the follow up kernel.
            let out = unsafe { dev.alloc::<T>(dst_el)? };
            let mut builder = func.builder();
            barg!(builder, src_el);
            barg!(builder, el_to_sum_per_block);
            barg!(builder, src_dims.len());
            builder.arg(&ds);
            builder.arg(src);
            builder.arg(&out);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(wrap(out))
        }
    }
}

impl<U: UnaryOpT> Map1 for U {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
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
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), &kernels::UNARY)?;
        // SAFETY: Set later by running the kernel.
        let mut out = unsafe { dev.alloc::<T>(el_count)? };
        let mut builder = func.builder();
        barg!(builder, el_count);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        builder.arg(src);
        builder.arg(&mut out);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

/// Lifetime-phantom stand-in for cudarc's `SyncOnDrop`. Baracuda's
/// `DeviceBuffer` manages its own lifetime via Arc refcounting, so no
/// explicit sync-on-drop guard is needed — but callers bound their
/// pointer usage to `_guard` drop order, so this PhantomData preserves
/// the call-site pattern without changing every destructure.
#[doc(hidden)]
pub struct SliceGuard<'a>(std::marker::PhantomData<&'a ()>);

fn slice_ptr<T: DeviceRepr>(v: &CudaSlice<T>, lo: usize) -> (u64, SliceGuard<'_>) {
    // Base pointer + byte-offset by `lo` elements.
    let base = v.as_raw().0 as u64;
    let offset_bytes = (lo * std::mem::size_of::<T>()) as u64;
    (base + offset_bytes, SliceGuard(std::marker::PhantomData))
}

struct IndexSelect<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map1 for IndexSelect<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        src_l: &Layout,
    ) -> Result<CudaSlice<T>> {
        let ids_l = &self.1;
        let (name, (ids, _guard)) = match &self.0.slice {
            CudaStorageSlice::U32(slice) => ("is_u32", slice_ptr(slice, ids_l.start_offset())),
            CudaStorageSlice::U8(slice) => ("is_u8", slice_ptr(slice, ids_l.start_offset())),
            CudaStorageSlice::I64(slice) => ("is_i64", slice_ptr(slice, ids_l.start_offset())),
            _ => Err(CudaError::UnexpectedDType {
                msg: "index_select ids should be u8, u32, or i64",
                expected: DType::U32,
                got: self.0.dtype(),
            })
            .w()?,
        };
        let ids_shape = ids_l.shape();
        let ids_dims = ids_shape.dims();
        let ds = dev.clone_htod(&dims_strides_usize(ids_l))?;
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-select" }.bt())?,
        };
        let left_size: usize = src_l.dims()[..self.2].iter().product();
        let right_size: usize = src_l.dims()[self.2 + 1..].iter().product();
        let src_dim_size = src_l.dims()[self.2];
        let ids_dim_size = ids_shape.elem_count();
        let dst_el = ids_shape.elem_count() * left_size * right_size;
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::INDEXING)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let mut builder = func.builder();
        barg!(builder, dst_el);
        barg!(builder, ids_dims.len());
        builder.arg(&ds);
        barg!(builder, ids);
        builder.arg(&src);
        builder.arg(&out);
        barg!(builder, left_size);
        barg!(builder, src_dim_size);
        barg!(builder, ids_dim_size);
        barg!(builder, right_size);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct Gather<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map1 for Gather<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        src_l: &Layout,
    ) -> Result<CudaSlice<T>> {
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let (name, (ids, _guard)) = match &ids.slice {
            CudaStorageSlice::U32(slice) => ("gather_u32", slice_ptr(slice, ids_o1)),
            CudaStorageSlice::U8(slice) => ("gather_u8", slice_ptr(slice, ids_o1)),
            CudaStorageSlice::I64(slice) => ("gather_i64", slice_ptr(slice, ids_o1)),
            _ => Err(CudaError::UnexpectedDType {
                msg: "gather ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let el = ids_l.shape().elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[dim];
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::INDEXING)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, ids);
        builder.arg(&src);
        builder.arg(&out);
        barg!(builder, left_sz);
        barg!(builder, src_dim_sz);
        barg!(builder, ids_dim_sz);
        barg!(builder, right_sz);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct IndexAdd<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map2InPlace for IndexAdd<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut CudaSlice<T>,
        dst_l: &Layout,
        src: &CudaSlice<T>,
        src_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<()> {
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let (name, (ids, _guard)) = match &ids.slice {
            CudaStorageSlice::U32(slice) => ("ia_u32", slice_ptr(slice, ids_o1)),
            CudaStorageSlice::I64(slice) => ("ia_i64", slice_ptr(slice, ids_o1)),
            CudaStorageSlice::U8(slice) => ("ia_u8", slice_ptr(slice, ids_o1)),
            _ => Err(CudaError::UnexpectedDType {
                msg: "index-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let dst = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[0];
        let cfg = LaunchConfig::for_num_elems((left_sz * right_sz) as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::INDEXING)?;
        let mut builder = func.builder();
        barg!(builder, ids);
        barg!(builder, ids_dim_sz);
        builder.arg(&src);
        builder.arg(&dst);
        barg!(builder, left_sz, src_dim_sz, dst_dim_sz, right_sz);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

struct Scatter<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map2InPlace for Scatter<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut CudaSlice<T>,
        dst_l: &Layout,
        src: &CudaSlice<T>,
        src_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<()> {
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };
        let (name, (ids, _guard)) = match &ids.slice {
            CudaStorageSlice::U32(slice) => ("s_u32", slice_ptr(slice, ids_o1)),
            CudaStorageSlice::I64(slice) => ("s_i64", slice_ptr(slice, ids_o1)),
            CudaStorageSlice::U8(slice) => ("s_u8", slice_ptr(slice, ids_o1)),
            _ => Err(CudaError::UnexpectedDType {
                msg: "scatter ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let dst = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];
        let cfg = LaunchConfig::for_num_elems((left_sz * right_sz) as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::INDEXING)?;
        let mut builder = func.builder();
        barg!(builder, ids);
        builder.arg(&src);
        builder.arg(&dst);
        barg!(builder, left_sz, src_dim_sz, dst_dim_sz, right_sz);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

struct ScatterAdd<'a>(&'a CudaStorage, &'a Layout, usize);
impl Map2InPlace for ScatterAdd<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut CudaSlice<T>,
        dst_l: &Layout,
        src: &CudaSlice<T>,
        src_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<()> {
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let (name, (ids, _guard)) = match &ids.slice {
            CudaStorageSlice::U32(slice) => ("sa_u32", slice_ptr(slice, ids_o1)),
            CudaStorageSlice::I64(slice) => ("sa_i64", slice_ptr(slice, ids_o1)),
            CudaStorageSlice::U8(slice) => ("sa_u8", slice_ptr(slice, ids_o1)),
            _ => Err(CudaError::UnexpectedDType {
                msg: "scatter-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let dst = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];
        let cfg = LaunchConfig::for_num_elems((left_sz * right_sz) as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::INDEXING)?;
        let mut builder = func.builder();
        barg!(builder, ids);
        builder.arg(&src);
        builder.arg(&dst);
        barg!(builder, left_sz, src_dim_sz, dst_dim_sz, right_sz);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

// Conv1D / Conv2D dispatch — baracuda alpha.38 cuDNN-backed conv FFI.
// PTX `Conv1D`/`Conv2D` Map2 impls + Fuel's internal `crate::cudnn::
// launch_conv*` wrappers retired in Phase 5b of the fuel-cuda-kernels
// retirement. `crate::baracuda::scratch::Workspace` (alloc-per-call) is
// used for the per-launch cuDNN workspace — sized to `0` lets baracuda
// auto-pick the workspace (it heap-allocates the needed bytes internally
// per the cuDNN contract).

fn baracuda_conv2d_fw<T: DeviceRepr + WithDType + ValidAsZeroBits>(
    inp: &CudaSlice<T>,
    inp_l: &Layout,
    k: &CudaSlice<T>,
    k_l: &Layout,
    params: &fuel_core_types::conv::ParamsConv2D,
    dev: &CudaDevice,
) -> Result<CudaSlice<T>> {
    use baracuda_kernels_sys as sys;
    let dt = T::DTYPE;
    if !inp_l.is_contiguous() || inp_l.start_offset() != 0 {
        fuel_core_types::bail!("baracuda conv_2d: expected contiguous NCHW input (start_offset=0)");
    }
    if !k_l.is_contiguous() || k_l.start_offset() != 0 {
        fuel_core_types::bail!("baracuda conv_2d: expected contiguous filter (start_offset=0)");
    }
    let inp_ptr = inp.as_raw().0 as *const std::ffi::c_void;
    let k_ptr = k.as_raw().0 as *const std::ffi::c_void;
    let (out_h, out_w) = (params.out_h(), params.out_w());
    let dst_el = params.c_out * out_h * out_w * params.b_size;
    let out = unsafe { dev.alloc::<T>(dst_el)? };
    let stream = dev.stream().as_raw() as *mut std::ffi::c_void;
    let out_ptr = out.as_raw().0 as *mut std::ffi::c_void;
    let (
        batch, c_in, c_out, h_in, w_in, kh, kw,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    ) = (
        params.b_size as i32,
        params.c_in as i32,
        params.c_out as i32,
        params.i_h as i32,
        params.i_w as i32,
        params.k_h as i32,
        params.k_w as i32,
        params.stride as i32, params.stride as i32,
        params.padding as i32, params.padding as i32,
        params.dilation as i32, params.dilation as i32,
        params.groups.max(1) as i32,
    );
    let (h_out, w_out) = (out_h as i32, out_w as i32);
    let status = match dt {
        // SAFETY: device-resident pointers + valid stream. Workspace is
        // null/0 — baracuda's wrapper internally heap-allocates the
        // cuDNN-reported workspace bytes.
        DType::F32 => unsafe {
            sys::baracuda_kernels_conv_2d_fw_f32_run(
                batch, c_in, c_out, h_in, w_in, h_out, w_out, kh, kw,
                stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::F64 => unsafe {
            sys::baracuda_kernels_conv_2d_fw_f64_run(
                batch, c_in, c_out, h_in, w_in, h_out, w_out, kh, kw,
                stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::F16 => unsafe {
            sys::baracuda_kernels_conv_2d_fw_f16_run(
                batch, c_in, c_out, h_in, w_in, h_out, w_out, kh, kw,
                stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::BF16 => unsafe {
            sys::baracuda_kernels_conv_2d_fw_bf16_run(
                batch, c_in, c_out, h_in, w_in, h_out, w_out, kh, kw,
                stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        other => fuel_core_types::bail!("baracuda conv_2d: unsupported dtype {other:?}"),
    };
    crate::baracuda::status::check(status, "conv_2d_fw")?;
    dev.synchronize()?;
    Ok(out)
}

fn baracuda_conv1d_fw<T: DeviceRepr + WithDType + ValidAsZeroBits>(
    inp: &CudaSlice<T>,
    inp_l: &Layout,
    k: &CudaSlice<T>,
    k_l: &Layout,
    params: &fuel_core_types::conv::ParamsConv1D,
    dev: &CudaDevice,
) -> Result<CudaSlice<T>> {
    use baracuda_kernels_sys as sys;
    let dt = T::DTYPE;
    if !inp_l.is_contiguous() || inp_l.start_offset() != 0 {
        fuel_core_types::bail!("baracuda conv_1d: expected contiguous NCL input (start_offset=0)");
    }
    if !k_l.is_contiguous() || k_l.start_offset() != 0 {
        fuel_core_types::bail!("baracuda conv_1d: expected contiguous filter (start_offset=0)");
    }
    let inp_ptr = inp.as_raw().0 as *const std::ffi::c_void;
    let k_ptr = k.as_raw().0 as *const std::ffi::c_void;
    let l_out = params.l_out();
    let dst_el = params.c_out * l_out * params.b_size;
    let out = unsafe { dev.alloc::<T>(dst_el)? };
    let stream = dev.stream().as_raw() as *mut std::ffi::c_void;
    let out_ptr = out.as_raw().0 as *mut std::ffi::c_void;
    let (batch, c_in, c_out, l_in, l_filt, stride_l, pad_l, dilation_l, groups) = (
        params.b_size as i32,
        params.c_in as i32,
        params.c_out as i32,
        params.l_in as i32,
        params.k_size as i32,
        params.stride as i32,
        params.padding as i32,
        params.dilation as i32,
        1_i32,
    );
    let l_out_i = l_out as i32;
    let status = match dt {
        DType::F32 => unsafe {
            sys::baracuda_kernels_conv_1d_fw_f32_run(
                batch, c_in, c_out, l_in, l_out_i, l_filt,
                stride_l, pad_l, dilation_l, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::F64 => unsafe {
            sys::baracuda_kernels_conv_1d_fw_f64_run(
                batch, c_in, c_out, l_in, l_out_i, l_filt,
                stride_l, pad_l, dilation_l, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::F16 => unsafe {
            sys::baracuda_kernels_conv_1d_fw_f16_run(
                batch, c_in, c_out, l_in, l_out_i, l_filt,
                stride_l, pad_l, dilation_l, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::BF16 => unsafe {
            sys::baracuda_kernels_conv_1d_fw_bf16_run(
                batch, c_in, c_out, l_in, l_out_i, l_filt,
                stride_l, pad_l, dilation_l, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        other => fuel_core_types::bail!("baracuda conv_1d: unsupported dtype {other:?}"),
    };
    crate::baracuda::status::check(status, "conv_1d_fw")?;
    dev.synchronize()?;
    Ok(out)
}

fn conv2d_dispatch(
    slice: &CudaStorageSlice,
    inp_l: &Layout,
    kernel: &CudaStorageSlice,
    k_l: &Layout,
    params: &fuel_core_types::conv::ParamsConv2D,
    dev: &CudaDevice,
) -> Result<CudaStorageSlice> {
    Ok(match (slice, kernel) {
        (CudaStorageSlice::F32(i), CudaStorageSlice::F32(k)) => {
            CudaStorageSlice::F32(baracuda_conv2d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::F64(i), CudaStorageSlice::F64(k)) => {
            CudaStorageSlice::F64(baracuda_conv2d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::F16(i), CudaStorageSlice::F16(k)) => {
            CudaStorageSlice::F16(baracuda_conv2d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::BF16(i), CudaStorageSlice::BF16(k)) => {
            CudaStorageSlice::BF16(baracuda_conv2d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::U8(_), CudaStorageSlice::U8(_)) => {
            Err(CudaError::InternalError("conv2d does not support u8 (cuDNN INT8 is signed)"))?
        }
        _ => Err(CudaError::InternalError("conv2d: dtype mismatch / unsupported"))?,
    })
}

fn baracuda_conv_transpose2d_fw<T: DeviceRepr + WithDType + ValidAsZeroBits>(
    inp: &CudaSlice<T>,
    inp_l: &Layout,
    k: &CudaSlice<T>,
    k_l: &Layout,
    params: &fuel_core_types::conv::ParamsConvTranspose2D,
    dev: &CudaDevice,
) -> Result<CudaSlice<T>> {
    use baracuda_kernels_sys as sys;
    let dt = T::DTYPE;
    if !inp_l.is_contiguous() || inp_l.start_offset() != 0 {
        fuel_core_types::bail!("baracuda conv_transpose_2d: expected contiguous NCHW input");
    }
    if !k_l.is_contiguous() || k_l.start_offset() != 0 {
        fuel_core_types::bail!("baracuda conv_transpose_2d: expected contiguous filter");
    }
    let (out_h, out_w) = (params.out_h(), params.out_w());
    let dst_el = params.c_out * out_h * out_w * params.b_size;
    let out = unsafe { dev.alloc::<T>(dst_el)? };
    let stream = dev.stream().as_raw() as *mut std::ffi::c_void;
    let inp_ptr = inp.as_raw().0 as *const std::ffi::c_void;
    let k_ptr = k.as_raw().0 as *const std::ffi::c_void;
    let out_ptr = out.as_raw().0 as *mut std::ffi::c_void;
    let (
        batch, c_in, c_out, h_in, w_in, kh, kw,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        opad_h, opad_w, groups,
    ) = (
        params.b_size as i32,
        params.c_in as i32,
        params.c_out as i32,
        params.i_h as i32,
        params.i_w as i32,
        params.k_h as i32,
        params.k_w as i32,
        params.stride as i32, params.stride as i32,
        params.padding as i32, params.padding as i32,
        params.dilation as i32, params.dilation as i32,
        params.output_padding as i32, params.output_padding as i32,
        1_i32,
    );
    let (h_out_i, w_out_i) = (out_h as i32, out_w as i32);
    let status = match dt {
        DType::F32 => unsafe {
            sys::baracuda_kernels_conv_transpose_2d_fw_f32_run(
                batch, c_in, c_out, h_in, w_in, h_out_i, w_out_i, kh, kw,
                stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                opad_h, opad_w, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::F64 => unsafe {
            sys::baracuda_kernels_conv_transpose_2d_fw_f64_run(
                batch, c_in, c_out, h_in, w_in, h_out_i, w_out_i, kh, kw,
                stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                opad_h, opad_w, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::F16 => unsafe {
            sys::baracuda_kernels_conv_transpose_2d_fw_f16_run(
                batch, c_in, c_out, h_in, w_in, h_out_i, w_out_i, kh, kw,
                stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                opad_h, opad_w, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::BF16 => unsafe {
            sys::baracuda_kernels_conv_transpose_2d_fw_bf16_run(
                batch, c_in, c_out, h_in, w_in, h_out_i, w_out_i, kh, kw,
                stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                opad_h, opad_w, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        other => fuel_core_types::bail!("baracuda conv_transpose_2d: unsupported dtype {other:?}"),
    };
    crate::baracuda::status::check(status, "conv_transpose_2d_fw")?;
    dev.synchronize()?;
    Ok(out)
}

fn baracuda_conv_transpose1d_fw<T: DeviceRepr + WithDType + ValidAsZeroBits>(
    inp: &CudaSlice<T>,
    inp_l: &Layout,
    k: &CudaSlice<T>,
    k_l: &Layout,
    params: &fuel_core_types::conv::ParamsConvTranspose1D,
    dev: &CudaDevice,
) -> Result<CudaSlice<T>> {
    use baracuda_kernels_sys as sys;
    let dt = T::DTYPE;
    if !inp_l.is_contiguous() || inp_l.start_offset() != 0 {
        fuel_core_types::bail!("baracuda conv_transpose_1d: expected contiguous NCL input");
    }
    if !k_l.is_contiguous() || k_l.start_offset() != 0 {
        fuel_core_types::bail!("baracuda conv_transpose_1d: expected contiguous filter");
    }
    let l_out = params.l_out();
    let dst_el = params.c_out * l_out * params.b_size;
    let out = unsafe { dev.alloc::<T>(dst_el)? };
    let stream = dev.stream().as_raw() as *mut std::ffi::c_void;
    let inp_ptr = inp.as_raw().0 as *const std::ffi::c_void;
    let k_ptr = k.as_raw().0 as *const std::ffi::c_void;
    let out_ptr = out.as_raw().0 as *mut std::ffi::c_void;
    let (
        batch, c_in, c_out, l_in, l_filt,
        stride_l, pad_l, dilation_l, output_pad_l, groups,
    ) = (
        params.b_size as i32,
        params.c_in as i32,
        params.c_out as i32,
        params.l_in as i32,
        params.k_size as i32,
        params.stride as i32,
        params.padding as i32,
        params.dilation as i32,
        params.output_padding as i32,
        1_i32,
    );
    let l_out_i = l_out as i32;
    let status = match dt {
        DType::F32 => unsafe {
            sys::baracuda_kernels_conv_transpose_1d_fw_f32_run(
                batch, c_in, c_out, l_in, l_out_i, l_filt,
                stride_l, pad_l, dilation_l, output_pad_l, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::F64 => unsafe {
            sys::baracuda_kernels_conv_transpose_1d_fw_f64_run(
                batch, c_in, c_out, l_in, l_out_i, l_filt,
                stride_l, pad_l, dilation_l, output_pad_l, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::F16 => unsafe {
            sys::baracuda_kernels_conv_transpose_1d_fw_f16_run(
                batch, c_in, c_out, l_in, l_out_i, l_filt,
                stride_l, pad_l, dilation_l, output_pad_l, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        DType::BF16 => unsafe {
            sys::baracuda_kernels_conv_transpose_1d_fw_bf16_run(
                batch, c_in, c_out, l_in, l_out_i, l_filt,
                stride_l, pad_l, dilation_l, output_pad_l, groups,
                inp_ptr, k_ptr, out_ptr,
                std::ptr::null_mut(), 0, stream,
            )
        },
        other => fuel_core_types::bail!("baracuda conv_transpose_1d: unsupported dtype {other:?}"),
    };
    crate::baracuda::status::check(status, "conv_transpose_1d_fw")?;
    dev.synchronize()?;
    Ok(out)
}

fn conv_transpose2d_dispatch(
    slice: &CudaStorageSlice,
    inp_l: &Layout,
    kernel: &CudaStorageSlice,
    k_l: &Layout,
    params: &fuel_core_types::conv::ParamsConvTranspose2D,
    dev: &CudaDevice,
) -> Result<CudaStorageSlice> {
    Ok(match (slice, kernel) {
        (CudaStorageSlice::F32(i), CudaStorageSlice::F32(k)) => {
            CudaStorageSlice::F32(baracuda_conv_transpose2d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::F64(i), CudaStorageSlice::F64(k)) => {
            CudaStorageSlice::F64(baracuda_conv_transpose2d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::F16(i), CudaStorageSlice::F16(k)) => {
            CudaStorageSlice::F16(baracuda_conv_transpose2d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::BF16(i), CudaStorageSlice::BF16(k)) => {
            CudaStorageSlice::BF16(baracuda_conv_transpose2d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        _ => Err(CudaError::InternalError("conv_transpose2d: dtype mismatch / unsupported"))?,
    })
}

fn conv_transpose1d_dispatch(
    slice: &CudaStorageSlice,
    inp_l: &Layout,
    kernel: &CudaStorageSlice,
    k_l: &Layout,
    params: &fuel_core_types::conv::ParamsConvTranspose1D,
    dev: &CudaDevice,
) -> Result<CudaStorageSlice> {
    Ok(match (slice, kernel) {
        (CudaStorageSlice::F32(i), CudaStorageSlice::F32(k)) => {
            CudaStorageSlice::F32(baracuda_conv_transpose1d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::F64(i), CudaStorageSlice::F64(k)) => {
            CudaStorageSlice::F64(baracuda_conv_transpose1d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::F16(i), CudaStorageSlice::F16(k)) => {
            CudaStorageSlice::F16(baracuda_conv_transpose1d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::BF16(i), CudaStorageSlice::BF16(k)) => {
            CudaStorageSlice::BF16(baracuda_conv_transpose1d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        _ => Err(CudaError::InternalError("conv_transpose1d: dtype mismatch / unsupported"))?,
    })
}

fn conv1d_dispatch(
    slice: &CudaStorageSlice,
    inp_l: &Layout,
    kernel: &CudaStorageSlice,
    k_l: &Layout,
    params: &fuel_core_types::conv::ParamsConv1D,
    dev: &CudaDevice,
) -> Result<CudaStorageSlice> {
    Ok(match (slice, kernel) {
        (CudaStorageSlice::F32(i), CudaStorageSlice::F32(k)) => {
            CudaStorageSlice::F32(baracuda_conv1d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::F64(i), CudaStorageSlice::F64(k)) => {
            CudaStorageSlice::F64(baracuda_conv1d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::F16(i), CudaStorageSlice::F16(k)) => {
            CudaStorageSlice::F16(baracuda_conv1d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        (CudaStorageSlice::BF16(i), CudaStorageSlice::BF16(k)) => {
            CudaStorageSlice::BF16(baracuda_conv1d_fw(i, inp_l, k, k_l, params, dev)?)
        }
        _ => Err(CudaError::InternalError("conv1d: dtype mismatch / unsupported"))?,
    })
}

// Col2Im1D + ConvTranspose1D + ConvTranspose2D PTX structs retired in
// Phase 5b — baracuda's `conv_transpose_{1,2}d_*_run` FFI subsumes both
// the Col2Im1D fast path and the general PTX transpose paths.

// Pool2D dispatch — baracuda alpha.36 cuDNN-backed Max/Avg pool 2D.
// The PTX-based Pool2D Map1 impl retired with the Phase 1 cleanup of
// the fuel-cuda-kernels retirement (audit doc at
// `docs/fuel-cuda-kernels-retirement-audit.md`). The baracuda symbols
// live behind the `cudnn` feature on `baracuda-kernels-sys`.

#[derive(Copy, Clone)]
enum PoolOp {
    Max,
    Avg,
}

fn pool2d_dispatch(
    slice: &CudaStorageSlice,
    dev: &CudaDevice,
    l: &Layout,
    k: (usize, usize),
    stride: (usize, usize),
    op: PoolOp,
) -> Result<CudaStorageSlice> {
    Ok(match slice {
        CudaStorageSlice::F32(s) => {
            CudaStorageSlice::F32(pool2d_baracuda(s, dev, l, k, stride, op)?)
        }
        CudaStorageSlice::F64(s) => {
            CudaStorageSlice::F64(pool2d_baracuda(s, dev, l, k, stride, op)?)
        }
        CudaStorageSlice::F16(s) => {
            CudaStorageSlice::F16(pool2d_baracuda(s, dev, l, k, stride, op)?)
        }
        CudaStorageSlice::BF16(s) => {
            CudaStorageSlice::BF16(pool2d_baracuda(s, dev, l, k, stride, op)?)
        }
        other => fuel_core_types::bail!("pool_2d: unsupported storage variant {other:?}"),
    })
}

fn pool2d_baracuda<T: DeviceRepr + WithDType + ValidAsZeroBits>(
    inp: &CudaSlice<T>,
    dev: &CudaDevice,
    inp_l: &Layout,
    kernel: (usize, usize),
    stride: (usize, usize),
    op: PoolOp,
) -> Result<CudaSlice<T>> {
    use baracuda_kernels_sys as sys;
    let dt = T::DTYPE;
    let dims = inp_l.shape().dims();
    if dims.len() != 4 {
        fuel_core_types::bail!("unexpected input shape for pool {dims:?}")
    }
    if !inp_l.is_contiguous() || inp_l.start_offset() != 0 {
        fuel_core_types::bail!("baracuda pool_2d: expected contiguous NCHW input")
    }
    let (batch, channels, h_in, w_in) = (dims[0], dims[1], dims[2], dims[3]);
    let (kh, kw) = (kernel.1, kernel.0);
    let (sh, sw) = (stride.1, stride.0);
    let h_out = (h_in - kh) / sh + 1;
    let w_out = (w_in - kw) / sw + 1;
    let dst_el = batch * channels * h_out * w_out;
    // SAFETY: output written by the cuDNN kernel below.
    let out = unsafe { dev.alloc::<T>(dst_el)? };
    let stream = dev.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = inp.as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out.as_raw().0 as *mut std::ffi::c_void;
    let (b, c, hi, wi, ho, wo) = (
        batch as i32, channels as i32, h_in as i32, w_in as i32, h_out as i32, w_out as i32,
    );
    let (kh, kw, sh, sw) = (kh as i32, kw as i32, sh as i32, sw as i32);
    let status = match (op, dt) {
        // SAFETY: device-resident input/output buffers + live stream.
        (PoolOp::Max, DType::F32) => unsafe {
            sys::baracuda_kernels_max_pool_2d_fw_f32_run(
                b, c, hi, wi, ho, wo, kh, kw, sh, sw, 0, 0, x_ptr, y_ptr, stream,
            )
        },
        (PoolOp::Max, DType::F64) => unsafe {
            sys::baracuda_kernels_max_pool_2d_fw_f64_run(
                b, c, hi, wi, ho, wo, kh, kw, sh, sw, 0, 0, x_ptr, y_ptr, stream,
            )
        },
        (PoolOp::Max, DType::F16) => unsafe {
            sys::baracuda_kernels_max_pool_2d_fw_f16_run(
                b, c, hi, wi, ho, wo, kh, kw, sh, sw, 0, 0, x_ptr, y_ptr, stream,
            )
        },
        (PoolOp::Max, DType::BF16) => unsafe {
            sys::baracuda_kernels_max_pool_2d_fw_bf16_run(
                b, c, hi, wi, ho, wo, kh, kw, sh, sw, 0, 0, x_ptr, y_ptr, stream,
            )
        },
        (PoolOp::Avg, DType::F32) => unsafe {
            sys::baracuda_kernels_avg_pool_2d_fw_f32_run(
                b, c, hi, wi, ho, wo, kh, kw, sh, sw, 0, 0, 0, x_ptr, y_ptr, stream,
            )
        },
        (PoolOp::Avg, DType::F64) => unsafe {
            sys::baracuda_kernels_avg_pool_2d_fw_f64_run(
                b, c, hi, wi, ho, wo, kh, kw, sh, sw, 0, 0, 0, x_ptr, y_ptr, stream,
            )
        },
        (PoolOp::Avg, DType::F16) => unsafe {
            sys::baracuda_kernels_avg_pool_2d_fw_f16_run(
                b, c, hi, wi, ho, wo, kh, kw, sh, sw, 0, 0, 0, x_ptr, y_ptr, stream,
            )
        },
        (PoolOp::Avg, DType::BF16) => unsafe {
            sys::baracuda_kernels_avg_pool_2d_fw_bf16_run(
                b, c, hi, wi, ho, wo, kh, kw, sh, sw, 0, 0, 0, x_ptr, y_ptr, stream,
            )
        },
        (_, other) => fuel_core_types::bail!("baracuda pool_2d: unsupported dtype {other:?}"),
    };
    crate::baracuda::status::check(status, "pool_2d_fw")?;
    dev.synchronize()?;
    Ok(out)
}

// UpsampleNearest2D dispatch — baracuda alpha.36
// `baracuda_kernels_upsample_nearest_2d_fw_<dtype>_run`. Bilinear
// stays on PTX for now because baracuda's interpolate-bilinear FFI
// hard-codes `align_corners = false`; Fuel exercises both modes via
// `bilinear_pytorch_align_corners_true_gpu`. Tracked as a Phase 5
// follow-up in `docs/fuel-cuda-kernels-retirement-audit.md`.

fn upsample_nearest2d_baracuda<T: DeviceRepr + WithDType + ValidAsZeroBits>(
    inp: &CudaSlice<T>,
    dev: &CudaDevice,
    inp_l: &Layout,
    out_w: usize, // legacy param name; semantically controls output-H (see below)
    out_h: usize, // legacy param name; semantically controls output-W (see below)
) -> Result<CudaSlice<T>> {
    use baracuda_kernels_sys as sys;
    let dt = T::DTYPE;
    let dims = inp_l.shape().dims();
    if dims.len() != 4 {
        fuel_core_types::bail!("unexpected input shape for upsample {dims:?}")
    }
    if !inp_l.is_contiguous() || inp_l.start_offset() != 0 {
        fuel_core_types::bail!("baracuda upsample_nearest_2d: expected contiguous NCHW input")
    }
    let (batch, channels, h_in, w_in) = (dims[0], dims[1], dims[2], dims[3]);
    // Fuel's tensor-level upsample_nearest2d(target_h, target_w) reaches
    // storage as (out_w, out_h), but the prior PTX kernel
    // (`upsample_nearest2d` in fuel-cuda-kernels) interpreted the first
    // parameter as the H-axis target and the second as the W-axis target
    // (see `scale_w = dims[2] / out_w` in the historical implementation,
    // where dims[2] is the H axis of an NCHW layout). Baracuda's FFI
    // uses unambiguous NCHW (OH, OW), so we map the legacy
    // (out_w → OH, out_h → OW).
    let (target_h, target_w) = (out_w, out_h);
    let dst_el = batch * channels * target_h * target_w;
    // SAFETY: filled by the kernel below.
    let out = unsafe { dev.alloc::<T>(dst_el)? };
    let scratch = crate::baracuda::scratch::Workspace::alloc(dev, 0)?;
    let stream = dev.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = inp.as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out.as_raw().0 as *mut std::ffi::c_void;
    let (n, c, ih, iw, oh, ow) = (
        batch as i32, channels as i32, h_in as i32, w_in as i32,
        target_h as i32, target_w as i32,
    );
    let status = match dt {
        // SAFETY: device-resident pointers + live stream.
        DType::F32 => unsafe {
            sys::baracuda_kernels_upsample_nearest_2d_fw_f32_run(
                n, c, ih, iw, oh, ow, x_ptr, y_ptr,
                scratch.as_raw(), scratch.bytes(), stream,
            )
        },
        DType::F64 => unsafe {
            sys::baracuda_kernels_upsample_nearest_2d_fw_f64_run(
                n, c, ih, iw, oh, ow, x_ptr, y_ptr,
                scratch.as_raw(), scratch.bytes(), stream,
            )
        },
        DType::F16 => unsafe {
            sys::baracuda_kernels_upsample_nearest_2d_fw_f16_run(
                n, c, ih, iw, oh, ow, x_ptr, y_ptr,
                scratch.as_raw(), scratch.bytes(), stream,
            )
        },
        DType::BF16 => unsafe {
            sys::baracuda_kernels_upsample_nearest_2d_fw_bf16_run(
                n, c, ih, iw, oh, ow, x_ptr, y_ptr,
                scratch.as_raw(), scratch.bytes(), stream,
            )
        },
        other => fuel_core_types::bail!("baracuda upsample_nearest_2d: unsupported dtype {other:?}"),
    };
    crate::baracuda::status::check(status, "upsample_nearest_2d_fw")?;
    dev.synchronize()?;
    Ok(out)
}

fn upsample_nearest2d_dispatch(
    slice: &CudaStorageSlice,
    dev: &CudaDevice,
    l: &Layout,
    out_w: usize,
    out_h: usize,
) -> Result<CudaStorageSlice> {
    Ok(match slice {
        CudaStorageSlice::F32(s) => {
            CudaStorageSlice::F32(upsample_nearest2d_baracuda(s, dev, l, out_w, out_h)?)
        }
        CudaStorageSlice::F64(s) => {
            CudaStorageSlice::F64(upsample_nearest2d_baracuda(s, dev, l, out_w, out_h)?)
        }
        CudaStorageSlice::F16(s) => {
            CudaStorageSlice::F16(upsample_nearest2d_baracuda(s, dev, l, out_w, out_h)?)
        }
        CudaStorageSlice::BF16(s) => {
            CudaStorageSlice::BF16(upsample_nearest2d_baracuda(s, dev, l, out_w, out_h)?)
        }
        other => fuel_core_types::bail!("upsample_nearest_2d: unsupported storage variant {other:?}"),
    })
}

// UpsampleBilinear2D dispatch — baracuda alpha.38 (Phase 19.2 +
// align_corners follow-up). The PTX `upsample_bilinear2d` Map1
// retired in Phase 5a-bilinear. Baracuda's
// `baracuda_kernels_interpolate_bilinear_2d_<dtype>_run` now takes
// align_corners + per-axis scale factor overrides (0.0 = derive),
// closing the parity gap with PyTorch's
// `nn.functional.interpolate(mode='bilinear', align_corners=...)`.

#[allow(clippy::too_many_arguments)]
fn upsample_bilinear2d_baracuda<T: DeviceRepr + WithDType + ValidAsZeroBits>(
    inp: &CudaSlice<T>,
    dev: &CudaDevice,
    inp_l: &Layout,
    out_w: usize,
    out_h: usize,
    align_corners: bool,
    scale_h_factor: Option<f64>,
    scale_w_factor: Option<f64>,
) -> Result<CudaSlice<T>> {
    use baracuda_kernels_sys as sys;
    let dt = T::DTYPE;
    let dims = inp_l.shape().dims();
    if dims.len() != 4 {
        fuel_core_types::bail!("unexpected input shape for upsample_bilinear2d {dims:?}")
    }
    if !inp_l.is_contiguous() || inp_l.start_offset() != 0 {
        fuel_core_types::bail!("baracuda upsample_bilinear_2d: expected contiguous NCHW input")
    }
    let (batch, channels, h_in, w_in) = (dims[0], dims[1], dims[2], dims[3]);
    let dst_el = batch * channels * out_w * out_h;
    // SAFETY: filled by the kernel below.
    let out = unsafe { dev.alloc::<T>(dst_el)? };
    let scratch = crate::baracuda::scratch::Workspace::alloc(dev, 0)?;
    let stream = dev.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = inp.as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out.as_raw().0 as *mut std::ffi::c_void;
    let (n, c, ih, iw, oh, ow) = (
        batch as i32, channels as i32, h_in as i32, w_in as i32, out_h as i32, out_w as i32,
    );
    let align = if align_corners { 1 } else { 0 };
    // Baracuda's override convention: 0.0 = derive from (in, out); nonzero
    // = explicit per-coord step. Fuel's `Option<f64>` maps cleanly.
    let sh = scale_h_factor.unwrap_or(0.0);
    let sw = scale_w_factor.unwrap_or(0.0);
    let status = match dt {
        // SAFETY: device-resident pointers + live stream.
        DType::F32 => unsafe {
            sys::baracuda_kernels_interpolate_bilinear_2d_f32_run(
                n, c, ih, iw, oh, ow, x_ptr, y_ptr,
                scratch.as_raw(), scratch.bytes(),
                align, sh, sw, stream,
            )
        },
        DType::F64 => unsafe {
            sys::baracuda_kernels_interpolate_bilinear_2d_f64_run(
                n, c, ih, iw, oh, ow, x_ptr, y_ptr,
                scratch.as_raw(), scratch.bytes(),
                align, sh, sw, stream,
            )
        },
        DType::F16 => unsafe {
            sys::baracuda_kernels_interpolate_bilinear_2d_f16_run(
                n, c, ih, iw, oh, ow, x_ptr, y_ptr,
                scratch.as_raw(), scratch.bytes(),
                align, sh, sw, stream,
            )
        },
        DType::BF16 => unsafe {
            sys::baracuda_kernels_interpolate_bilinear_2d_bf16_run(
                n, c, ih, iw, oh, ow, x_ptr, y_ptr,
                scratch.as_raw(), scratch.bytes(),
                align, sh, sw, stream,
            )
        },
        other => fuel_core_types::bail!("baracuda upsample_bilinear_2d: unsupported dtype {other:?}"),
    };
    crate::baracuda::status::check(status, "upsample_bilinear_2d_fw")?;
    dev.synchronize()?;
    Ok(out)
}

fn upsample_bilinear2d_dispatch(
    slice: &CudaStorageSlice,
    dev: &CudaDevice,
    l: &Layout,
    out_w: usize,
    out_h: usize,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
) -> Result<CudaStorageSlice> {
    Ok(match slice {
        CudaStorageSlice::F32(s) => CudaStorageSlice::F32(upsample_bilinear2d_baracuda(
            s, dev, l, out_w, out_h, align_corners, scale_h, scale_w,
        )?),
        CudaStorageSlice::F64(s) => CudaStorageSlice::F64(upsample_bilinear2d_baracuda(
            s, dev, l, out_w, out_h, align_corners, scale_h, scale_w,
        )?),
        CudaStorageSlice::F16(s) => CudaStorageSlice::F16(upsample_bilinear2d_baracuda(
            s, dev, l, out_w, out_h, align_corners, scale_h, scale_w,
        )?),
        CudaStorageSlice::BF16(s) => CudaStorageSlice::BF16(upsample_bilinear2d_baracuda(
            s, dev, l, out_w, out_h, align_corners, scale_h, scale_w,
        )?),
        other => fuel_core_types::bail!("upsample_bilinear_2d: unsupported storage variant {other:?}"),
    })
}

struct WhereCond<'a>(&'a CudaStorage, &'a Layout);
impl Map2 for WhereCond<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        t: &CudaSlice<T>,
        layout_t: &Layout,
        f: &CudaSlice<T>,
        layout_f: &Layout,
        dev: &CudaDevice,
    ) -> Result<CudaSlice<T>> {
        let ids_l = &self.1;
        let ((ids, _guard), name) = match &self.0.slice {
            CudaStorageSlice::U8(slice) => {
                let ptr = slice_ptr(slice, ids_l.start_offset());
                (ptr, "where_u8")
            }
            CudaStorageSlice::U32(slice) => {
                let ptr = slice_ptr(slice, ids_l.start_offset());
                (ptr, "where_u32")
            }
            CudaStorageSlice::I64(slice) => {
                let ptr = slice_ptr(slice, ids_l.start_offset());
                (ptr, "where_i64")
            }
            _ => Err(CudaError::UnexpectedDType {
                msg: "where conditions should be u8/u32/i64",
                expected: DType::U32,
                got: self.0.dtype(),
            })
            .w()?,
        };
        let shape = ids_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let ds = {
            let s_ids = ids_l.stride_unsigned();
            let s_t = layout_t.stride_unsigned();
            let s_f = layout_f.stride_unsigned();
            let mut v = Vec::with_capacity(dims.len() + s_ids.len() + s_t.len() + s_f.len());
            v.extend_from_slice(dims);
            v.extend_from_slice(&s_ids);
            v.extend_from_slice(&s_t);
            v.extend_from_slice(&s_f);
            dev.clone_htod(&v)?
        };
        let t = &t.slice(layout_t.start_offset()..t.len());
        let f = &f.slice(layout_f.start_offset()..f.len());
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::TERNARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, dims.len());
        builder.arg(&ds);
        barg!(builder, ids);
        builder.arg(t);
        builder.arg(f);
        builder.arg(&out);
        // SAFETY: ffi
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

impl<U: fuel_core_types::op::BinaryOpT> Map2 for U {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
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
            SlicePtrOrNull::Ptr(dev.clone_htod(&dims_strides_strides_usize(dims, lhs_l, rhs_l))?)
        };
        let lhs = &lhs.slice(lhs_l.start_offset()..lhs.len());
        let rhs = &rhs.slice(rhs_l.start_offset()..rhs.len());
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), &kernels::BINARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<T>(elem_count)? };
        let mut builder = func.builder();
        barg!(builder, elem_count);
        barg!(builder, dims.len());
        dims_and_strides.builder_arg(&mut builder);
        builder.arg(lhs);
        builder.arg(rhs);
        builder.arg(&out);
        // SAFETY: ffi
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

struct Cmp(CmpOp);
impl Map2Any for Cmp {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        lhs: &CudaSlice<T>,
        lhs_l: &Layout,
        rhs: &CudaSlice<T>,
        rhs_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<S> {
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let dims_and_strides = if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
            SlicePtrOrNull::Null
        } else {
            SlicePtrOrNull::Ptr(dev.clone_htod(&dims_strides_strides_usize(dims, lhs_l, rhs_l))?)
        };
        let lhs = &lhs.slice(lhs_l.start_offset()..lhs.len());
        let rhs = &rhs.slice(rhs_l.start_offset()..rhs.len());
        let name = match self.0 {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
        };
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::BINARY)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { dev.alloc::<u8>(elem_count)? };
        let mut builder = func.builder();
        barg!(builder, elem_count);
        barg!(builder, dims.len());
        dims_and_strides.builder_arg(&mut builder);
        builder.arg(lhs);
        builder.arg(rhs);
        builder.arg(&out);
        // SAFETY: ffi
        unsafe { builder.launch(cfg) }.w()?;
        Ok(S::U8(out))
    }
}

fn slice_src_and_dst<'a, T: DeviceRepr>(
    src: &'a CudaSlice<T>,
    src_l: &Layout,
    dst: &'a mut CudaSlice<T>,
    dst_offset: usize,
) -> (
    baracuda_driver::DeviceSlice<'a, T>,
    baracuda_driver::DeviceSliceMut<'a, T>,
) {
    let src_offset = src_l.start_offset();
    let to_copy = dst
        .len()
        .saturating_sub(dst_offset)
        .min(src.len().saturating_sub(src_offset));
    let src = src.slice(src_offset..src_offset + to_copy);
    let dst = dst.slice_mut(dst_offset..dst_offset + to_copy);
    (src, dst)
}

#[derive(Debug)]
pub struct CudaStorage {
    pub slice: CudaStorageSlice,
    pub device: CudaDevice,
}

pub trait CudaDType: Sized + DeviceRepr {
    fn as_cuda_slice(s: &CudaStorage) -> Result<&CudaSlice<Self>>;
    fn as_cuda_slice_mut(s: &mut CudaStorage) -> Result<&mut CudaSlice<Self>>;
    fn wrap_cuda_slice(s: CudaSlice<Self>, dev: CudaDevice) -> CudaStorage;
}

macro_rules! cuda_dtype {
    ($ty:ty, $dtype:ident) => {
        impl CudaDType for $ty {
            fn as_cuda_slice(s: &CudaStorage) -> Result<&CudaSlice<Self>> {
                match &s.slice {
                    CudaStorageSlice::$dtype(data) => Ok(&data),
                    _ => Err(crate::Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn as_cuda_slice_mut(s: &mut CudaStorage) -> Result<&mut CudaSlice<Self>> {
                match s.slice {
                    CudaStorageSlice::$dtype(ref mut data) => Ok(data),
                    _ => Err(crate::Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn wrap_cuda_slice(slice: CudaSlice<Self>, device: CudaDevice) -> CudaStorage {
                let slice = CudaStorageSlice::$dtype(slice);
                CudaStorage { slice, device }
            }
        }
    };
}
cuda_dtype!(u8, U8);
cuda_dtype!(i8, I8);
cuda_dtype!(u32, U32);
cuda_dtype!(i16, I16);
cuda_dtype!(i32, I32);
cuda_dtype!(i64, I64);
cuda_dtype!(f16, F16);
cuda_dtype!(bf16, BF16);
cuda_dtype!(f32, F32);
cuda_dtype!(f64, F64);
cuda_dtype!(float8::F8E4M3, F8E4M3);

impl CudaStorage {
    pub fn wrap_cuda_slice<T: CudaDType>(slice: CudaSlice<T>, device: CudaDevice) -> CudaStorage {
        T::wrap_cuda_slice(slice, device)
    }

    pub fn as_cuda_slice<T: CudaDType>(&self) -> Result<&CudaSlice<T>> {
        T::as_cuda_slice(self)
    }

    pub fn as_cuda_slice_mut<T: CudaDType>(&mut self) -> Result<&mut CudaSlice<T>> {
        T::as_cuda_slice_mut(self)
    }

    pub fn transfer_to_device(&self, dst: &CudaDevice) -> Result<Self> {
        let storage_slice = match self.dtype() {
            DType::U8 => {
                let cuda_slice = self.as_cuda_slice::<u8>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::U8(result)
            }
            DType::I8 => {
                let cuda_slice = self.as_cuda_slice::<i8>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::I8(result)
            }
            DType::U32 => {
                let cuda_slice = self.as_cuda_slice::<u32>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::U32(result)
            }
            DType::I16 => {
                let cuda_slice = self.as_cuda_slice::<i16>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::I16(result)
            }
            DType::I32 => {
                let cuda_slice = self.as_cuda_slice::<i32>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::I32(result)
            }
            DType::I64 => {
                let cuda_slice = self.as_cuda_slice::<i64>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::I64(result)
            }
            DType::BF16 => {
                let cuda_slice = self.as_cuda_slice::<bf16>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::BF16(result)
            }
            DType::F16 => {
                let cuda_slice = self.as_cuda_slice::<f16>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::F16(result)
            }
            DType::F32 => {
                let cuda_slice = self.as_cuda_slice::<f32>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::F32(result)
            }
            DType::F64 => {
                let cuda_slice = self.as_cuda_slice::<f64>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::F64(result)
            }
            DType::F8E4M3 => {
                let cuda_slice = self.as_cuda_slice::<float8::F8E4M3>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::F8E4M3(result)
            }
            DType::F6E2M3 => {
                let cuda_slice = self.as_cuda_slice::<u8>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::F6E2M3(result)
            }
            DType::F6E3M2 => {
                let cuda_slice = self.as_cuda_slice::<u8>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::F6E3M2(result)
            }
            DType::F4 => {
                let cuda_slice = self.as_cuda_slice::<u8>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::F4(result)
            }
            DType::F8E8M0 => {
                let cuda_slice = self.as_cuda_slice::<u8>()?;
                let result = dst.clone_dtod(cuda_slice)?;
                CudaStorageSlice::F8E8M0(result)
            }
        };

        Ok(Self {
            slice: storage_slice,
            device: dst.clone(),
        })
    }
}

fn gemm_config<T>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<StridedBatchedConfig<T>> {
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
    use baracuda_cublas::Op;

    let lhs_stride = lhs_l.stride_unsigned();
    let rhs_stride = rhs_l.stride_unsigned();
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    // The "A" tensor in cuBLAS terms is our `rhs` (we do the row-
    // major↔col-major swap by passing rhs as cuBLAS A and lhs as
    // cuBLAS B). Logical shape of rhs: (k, n) row-major.
    //
    // Two stride patterns are supported, each mapping to one of
    // cuBLAS's transpose flags:
    //
    //   Op::N  — row-major-like: stride[-1] == 1 and stride[-2] >= n.
    //            The natural contig case is stride[-2] == n; the
    //            relaxed case stride[-2] > n covers strided views
    //            that arise from `permute()` of a larger tensor
    //            (rows are contiguous internally but the matrix is
    //            embedded in a parent buffer with row spacing > n).
    //            cuBLAS lda IS the row stride, so we pass stride[-2].
    //
    //   Op::T  — col-major-like / transposed: stride[-2] == 1 and
    //            stride[-1] >= k. The natural case is stride[-1] == k
    //            (a clean transpose); the relaxed case is the same
    //            permute-of-larger-tensor pattern but on the
    //            transposed axis. lda = stride[-1].
    //
    // Degenerate dims (n==1 or k==1) make the corresponding stride
    // irrelevant — we still need to emit a positive lda, so clamp
    // by the logical dim size.
    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 >= n || k == 1) {
        let ld = rhs_m2.max(n).max(1) as i32;
        (ld, Op::N)
    } else if (rhs_m2 == 1 || k == 1) && (rhs_m1 >= k || n == 1) {
        let ld = rhs_m1.max(k).max(1) as i32;
        (ld, Op::T)
    } else {
        Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?
    };
    // The "B" tensor in cuBLAS terms is our `lhs`. Logical shape: (m, k)
    // row-major. Same two patterns apply, swapping (n, k) ↔ (k, m).
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 >= k || m == 1) {
        let ld = lhs_m2.max(k).max(1) as i32;
        (ld, Op::N)
    } else if (lhs_m2 == 1 || m == 1) && (lhs_m1 >= m || k == 1) {
        let ld = lhs_m1.max(m).max(1) as i32;
        (ld, Op::T)
    } else {
        Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?
    };
    // The setup below was copied from:
    // https://github.com/lebedov/scikit-cuda/blob/7e7300474286019c917a6c8a4bca59405c64fbce/tests/test_cublas.py#L531
    let gemm = GemmConfig {
        alpha,
        beta,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: n as i32,
        transa,
        transb,
    };

    let stride_b: usize = match lhs_stride[..lhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
        [_, stride] if lhs_l.dims()[0] == 1 => stride,
        [stride, _] if lhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => m * k,
        _ => Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?,
    };
    let stride_a: usize = match rhs_stride[..rhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
        [_, stride] if rhs_l.dims()[0] == 1 => stride,
        [stride, _] if rhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => n * k,
        _ => Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        })?,
    };
    Ok(StridedBatchedConfig {
        batch_size: b as i32,
        gemm,
        stride_a: stride_a as i64,
        stride_b: stride_b as i64,
        stride_c: (m * n) as i64,
    })
}

impl CudaStorage {
    pub fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let slice = Clone.map(&self.slice, self.device(), layout)?;
        let device = self.device.clone();
        Ok(Self { slice, device })
    }

    pub fn dtype(&self) -> DType {
        match self.slice {
            CudaStorageSlice::U8(_) => DType::U8,
            CudaStorageSlice::I8(_) => DType::I8,
            CudaStorageSlice::U32(_) => DType::U32,
            CudaStorageSlice::I16(_) => DType::I16,
            CudaStorageSlice::I32(_) => DType::I32,
            CudaStorageSlice::I64(_) => DType::I64,
            CudaStorageSlice::BF16(_) => DType::BF16,
            CudaStorageSlice::F16(_) => DType::F16,
            CudaStorageSlice::F32(_) => DType::F32,
            CudaStorageSlice::F64(_) => DType::F64,
            CudaStorageSlice::F8E4M3(_) => DType::F8E4M3,
            CudaStorageSlice::F6E2M3(_) => DType::F6E2M3,
            CudaStorageSlice::F6E3M2(_) => DType::F6E3M2,
            CudaStorageSlice::F4(_) => DType::F4,
            CudaStorageSlice::F8E8M0(_) => DType::F8E8M0,
        }
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn const_set(&mut self, s: fuel_core_types::scalar::Scalar, layout: &Layout) -> Result<()> {
        let dev = &self.device;
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src_o = layout.start_offset();
        let ((src, _guard_src), kernel_name) = match &mut self.slice {
            S::U8(s) => (slice_ptr(s, src_o), "const_set_u8"),
            S::I8(s) => (slice_ptr(s, src_o), "const_set_i8"),
            S::U32(s) => (slice_ptr(s, src_o), "const_set_u32"),
            S::I16(s) => (slice_ptr(s, src_o), "const_set_i16"),
            S::I32(s) => (slice_ptr(s, src_o), "const_set_i32"),
            S::I64(s) => (slice_ptr(s, src_o), "const_set_i64"),
            S::BF16(s) => (slice_ptr(s, src_o), "const_set_bf16"),
            S::F16(s) => (slice_ptr(s, src_o), "const_set_f16"),
            S::F32(s) => (slice_ptr(s, src_o), "const_set_f32"),
            S::F64(s) => (slice_ptr(s, src_o), "const_set_f64"),
            S::F8E4M3(s) => (slice_ptr(s, src_o), "const_set_f8_e4m3"),
            S::F4(_) | S::F6E2M3(_) | S::F6E3M2(_) | S::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: self.dtype(),
                    op: "const_set",
                }
                .into());
            }
        };

        let func = dev.get_or_load_func(kernel_name, &kernels::FILL)?;
        let mut builder = func.builder();
        barg!(builder, el_count);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        push_scalar_arg(&s, &mut builder);
        barg!(builder, src);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }

    pub fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let dev = self.device();
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let start_o = layout.start_offset();
        // This returns an i64 rather than a &i64, this is useful to get around some temporary
        // lifetime issue and is safe as long as self.slice does not go out of scope before inp
        // is used.
        let (inp, _guard) = match &self.slice {
            CudaStorageSlice::U8(inp) => slice_ptr(inp, start_o),
            CudaStorageSlice::I8(inp) => slice_ptr(inp, start_o),
            CudaStorageSlice::U32(inp) => slice_ptr(inp, start_o),
            CudaStorageSlice::I16(inp) => slice_ptr(inp, start_o),
            CudaStorageSlice::I32(inp) => slice_ptr(inp, start_o),
            CudaStorageSlice::I64(inp) => slice_ptr(inp, start_o),
            CudaStorageSlice::BF16(inp) => slice_ptr(inp, start_o),
            CudaStorageSlice::F16(inp) => slice_ptr(inp, start_o),
            CudaStorageSlice::F32(inp) => slice_ptr(inp, start_o),
            CudaStorageSlice::F64(inp) => slice_ptr(inp, start_o),
            CudaStorageSlice::F8E4M3(inp) => slice_ptr(inp, start_o),
            CudaStorageSlice::F4(_)
            | CudaStorageSlice::F6E2M3(_)
            | CudaStorageSlice::F6E3M2(_)
            | CudaStorageSlice::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: self.dtype(),
                    op: "to_dtype",
                }
                .into());
            }
        };
        let inp = &inp;

        let kernel_name = format!("cast_{}_{}", self.dtype().as_str(), dtype.as_str());
        let func = dev.get_or_load_func(&kernel_name, &kernels::CAST)?;
        let slice = match dtype {
            DType::U8 => {
                let out = unsafe { dev.alloc::<u8>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::U8(out)
            }
            DType::U32 => {
                let out = unsafe { dev.alloc::<u32>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::U32(out)
            }
            DType::I64 => {
                let out = unsafe { dev.alloc::<i64>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::I64(out)
            }
            DType::BF16 => {
                let out = unsafe { dev.alloc::<bf16>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(out)
            }
            DType::F16 => {
                let out = unsafe { dev.alloc::<f16>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(out)
            }
            DType::F32 => {
                let out = unsafe { dev.alloc::<f32>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(out)
            }
            DType::F64 => {
                let out = unsafe { dev.alloc::<f64>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F64(out)
            }
            DType::F8E4M3 => {
                let out = unsafe { dev.alloc::<float8::F8E4M3>(el)? };
                let mut builder = func.builder();
                barg!(builder, el);
                barg!(builder, dims.len());
                ds.builder_arg(&mut builder);
                barg!(builder, *inp);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F8E4M3(out)
            }
            DType::I8 | DType::I16 | DType::I32 => {
                return Err(CudaError::InternalError("i8,i16,i32 dtypes are not supported").into())
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(Self {
            slice,
            device: dev.clone(),
        })
    }

    pub fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Affine(mul, add).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    pub fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Powf(e).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    pub fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Elu(alpha).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    pub fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device().clone();
        let slice = FastReduce(sum_dims, op).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    pub fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let slice = Cmp(op).map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
        Ok(Self { slice, device })
    }

    pub fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let slice = U::V.map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    /// Run a unary CUDA kernel by name (e.g. `"uneg"`, `"uexp"`,
    /// `"usilu"`). This bypasses the `UnaryOpT` type-parameter
    /// dispatch and is the entry point for executors that can't
    /// depend on the concrete op-type structs in fuel-core.
    pub fn unary_by_name(&self, kernel: &'static str, layout: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let slice = crate::dyn_impl::UnaryKernel(kernel)
            .map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    pub fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = B::V.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
        Ok(Self { slice, device })
    }

    /// Run a binary CUDA kernel by name (e.g. `"badd"`, `"bmul"`).
    /// Same rationale as [`unary_by_name`].
    pub fn binary_by_name(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
        kernel: &'static str,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = crate::dyn_impl::BinaryKernel(kernel)
            .map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
        Ok(Self { slice, device })
    }

    /// Softmax along the last dimension. Uses the fused CUDA kernel
    /// from `reduce.cu` (one thread-block per row).
    ///
    /// `layout` must be contiguous. `n_rows` × `n_cols` must equal
    /// the total element count described by the layout's shape.
    pub fn softmax_last_dim(&self, layout: &Layout) -> Result<Self> {
        let shape = layout.shape();
        let dims = shape.dims();
        let n_cols = *dims.last().expect("softmax: empty shape");
        let n_rows = shape.elem_count() / n_cols;
        let device = self.device().clone();

        macro_rules! launch_softmax {
            ($slice:expr, $kname:expr, $ty:ty) => {{
                use crate::device::LaunchConfig;
                let src = &$slice.slice(layout.start_offset()..slice.len());
                let func = device.get_or_load_func($kname, &kernels::REDUCE)?;
                let mut out = unsafe { device.alloc::<$ty>(n_rows * n_cols)? };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = func.builder();
                builder.arg(src);
                builder.arg(&mut out);
                crate::builder_arg!(builder, n_cols as i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::$variant(out)
            }};
        }

        let slice = match &self.slice {
            CudaStorageSlice::F32(s) => {
                use crate::device::LaunchConfig;
                let src = &s.slice(layout.start_offset()..s.len());
                let func = device.get_or_load_func("softmax_f32", &kernels::REDUCE)?;
                let mut out = unsafe { device.alloc::<f32>(n_rows * n_cols)? };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = func.builder();
                builder.arg(src);
                builder.arg(&mut out);
                crate::builder_arg!(builder, n_cols as i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(out)
            }
            CudaStorageSlice::F64(s) => {
                use crate::device::LaunchConfig;
                let src = &s.slice(layout.start_offset()..s.len());
                let func = device.get_or_load_func("softmax_f64", &kernels::REDUCE)?;
                let mut out = unsafe { device.alloc::<f64>(n_rows * n_cols)? };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = func.builder();
                builder.arg(src);
                builder.arg(&mut out);
                crate::builder_arg!(builder, n_cols as i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F64(out)
            }
            _ => {
                return fuel_core_types::bail!("softmax_last_dim: unsupported dtype {:?}", self.dtype());
            }
        };
        Ok(Self { slice, device })
    }

    /// Q4_0 matmul: `out = a @ dequant_q4_0(w_q_bytes)`.
    /// - `a`: F32 activations, shape `[..., m, k]`, contiguous.
    /// - `w_q_bytes`: Q4_0-packed weights stored as U32 chunks. Logical
    ///    weight shape is `[n, k]`; blob size is `n * (k/32) * 18` bytes.
    /// - Output: F32, shape `[..., m, n]`.
    ///
    /// **First-cut limitation:** only `m = 1` (the decode path) is
    /// supported on CUDA today. Prefill (m > 1) bails; the caller
    /// should route to a different backend (e.g. Vulkan has a tiled
    /// M>1 kernel). Lifting this is mechanical — either loop the gemv
    /// kernel N rows or port the tiled dequant+matmul path — but
    /// blocked on time for this sprint.
    pub fn matmul_q4_0(
        &self,
        w_q_bytes: &Self,
        k: usize,
        n: usize,
        a_layout: &Layout,
    ) -> Result<Self> {
        use crate::device::LaunchConfig;
        if self.dtype() != DType::F32 {
            return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_0: A must be F32, got {:?}", self.dtype());
        }
        if !a_layout.is_contiguous() {
            return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_0: requires contiguous A");
        }
        let a_dims = a_layout.shape().dims();
        let rank = a_dims.len();
        if rank < 2 {
            return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_0: A must be rank >= 2");
        }
        let m = a_dims[rank - 2];
        let batch: usize = a_dims[..rank - 2].iter().product::<usize>().max(1);
        let total_rows = batch * m;
        if total_rows != 1 {
            return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_0: only M=1 supported on CUDA today; \
                 got total_rows={total_rows}. Route prefill to Vulkan.");
        }
        if k % 32 != 0 {
            return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_0: k must be multiple of 32 (Q4_0 block size), got {k}");
        }
        let device = self.device().clone();

        let a_src = match &self.slice {
            CudaStorageSlice::F32(s) => s.slice(a_layout.start_offset()..s.len()),
            _ => return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_0: A must be F32"),
        };
        let w_src = match &w_q_bytes.slice {
            CudaStorageSlice::U32(s) => s.slice(0..s.len()),
            _ => return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_0: weight blob must be U32 storage"),
        };
        let mut out = unsafe { device.alloc::<f32>(n)? };

        // ggml-style launch: WARP_SIZE=32 threads per row for the warp
        // reduce; MMV_Y rows per block (pick 2 as the ggml default).
        const WARP_SIZE: u32 = 32;
        const MMV_Y: u32 = 2;
        let grid_x = ((n as u32) + MMV_Y - 1) / MMV_Y;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (WARP_SIZE, MMV_Y, 1),
            shared_mem_bytes: 0,
        };
        let func = device.get_or_load_func(
            "dequantize_mul_mat_vec_q4_0_cuda", &kernels::QUANTIZED)?;
        let mut builder = func.builder();
        builder.arg(&w_src);      // vx = weight blob
        builder.arg(&a_src);      // y = activation vector
        builder.arg(&mut out);    // dst = output
        crate::builder_arg!(builder, k as i32);
        crate::builder_arg!(builder, n as i32);
        unsafe { builder.launch(cfg) }.w()?;

        Ok(Self { slice: CudaStorageSlice::F32(out), device })
    }

    /// Q4_K_M matmul: `out = a @ dequant_q4_km(w_q_bytes)`.
    /// - `a`: F32 activations, shape `[..., m, k]`, contiguous.
    /// - `w_q_bytes`: Q4_K-packed weights stored as U32. Each
    ///    256-element super-block is 144 bytes.
    /// - Output: F32, shape `[..., m, n]`.
    ///
    /// First-cut limitation: M=1 only (decode). `k` must be a
    /// multiple of 256 (Q4_K super-block size).
    pub fn matmul_q4_km(
        &self,
        w_q_bytes: &Self,
        k: usize,
        n: usize,
        a_layout: &Layout,
    ) -> Result<Self> {
        use crate::device::LaunchConfig;
        if self.dtype() != DType::F32 {
            return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_km: A must be F32, got {:?}", self.dtype());
        }
        if !a_layout.is_contiguous() {
            return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_km: requires contiguous A");
        }
        let a_dims = a_layout.shape().dims();
        let rank = a_dims.len();
        if rank < 2 {
            return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_km: A must be rank >= 2");
        }
        let m = a_dims[rank - 2];
        let batch: usize = a_dims[..rank - 2].iter().product::<usize>().max(1);
        let total_rows = batch * m;
        if total_rows != 1 {
            return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_km: only M=1 supported on CUDA today; \
                 got total_rows={total_rows}. Route prefill to Vulkan.");
        }
        if k % 256 != 0 {
            return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_km: k must be multiple of 256 (Q4_K super-block), got {k}");
        }
        let device = self.device().clone();

        let a_src = match &self.slice {
            CudaStorageSlice::F32(s) => s.slice(a_layout.start_offset()..s.len()),
            _ => return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_km: A must be F32"),
        };
        let w_src = match &w_q_bytes.slice {
            CudaStorageSlice::U32(s) => s.slice(0..s.len()),
            _ => return fuel_core_types::bail!(
                "CudaStorage::matmul_q4_km: weight blob must be U32 storage"),
        };
        let mut out = unsafe { device.alloc::<f32>(n)? };

        // ggml launch for Q4_K: 1 warp (32 threads) per row, 1 row per
        // block. With K_QUANTS_PER_ITERATION=2, ny = 2/2 = 1.
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = device.get_or_load_func(
            "dequantize_mul_mat_vec_q4_k", &kernels::QUANTIZED)?;
        let mut builder = func.builder();
        builder.arg(&w_src);      // vx
        builder.arg(&a_src);      // yy
        builder.arg(&mut out);    // dst
        crate::builder_arg!(builder, k as i32);
        crate::builder_arg!(builder, n as i32);
        unsafe { builder.launch(cfg) }.w()?;

        Ok(Self { slice: CudaStorageSlice::F32(out), device })
    }

    /// Fused RMS normalization along the last dim:
    /// `out[r, c] = x[r, c] / sqrt(mean(x[r, :]^2) + eps)`.
    /// Mirrors VulkanBackend::rms_norm_last_dim semantics (no gain
    /// vector — the caller multiplies by gain in a separate op).
    /// F32 only today.
    pub fn rms_norm_last_dim(&self, layout: &Layout, eps: f64) -> Result<Self> {
        use crate::device::LaunchConfig;
        if self.dtype() != DType::F32 {
            return fuel_core_types::bail!(
                "CudaStorage: rms_norm_last_dim requires f32 input, got {:?}",
                self.dtype()
            );
        }
        let dims = layout.shape().dims();
        let n_cols = *dims.last().expect("rms_norm: empty shape");
        let n_rows = layout.shape().elem_count() / n_cols;
        let device = self.device().clone();

        let src = match &self.slice {
            CudaStorageSlice::F32(s) => s.slice(layout.start_offset()..s.len()),
            _ => return fuel_core_types::bail!(
                "CudaStorage::rms_norm_last_dim: expected F32"),
        };
        let mut out = unsafe { device.alloc::<f32>(n_rows * n_cols)? };

        // Block size: match the kernel's expectation (one block per row,
        // block_size threads cooperate on the reduction). Keep small
        // enough that the warp-reduce path is exercised.
        let block_size = 256i32;
        let func = device.get_or_load_func("rmsnorm_f32_noalpha", &kernels::REDUCE)?;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&src);
        builder.arg(&mut out);
        crate::builder_arg!(builder, n_cols as i32);
        crate::builder_arg!(builder, block_size);
        crate::builder_arg!(builder, eps as f32);
        unsafe { builder.launch(cfg) }.w()?;

        Ok(Self { slice: CudaStorageSlice::F32(out), device })
    }

    /// Apply RoPE rotation to the input, using the fused CUDA kernel
    /// from `reduce.cu`. Mirrors VulkanBackend::rope's semantics:
    /// pairs `(i, i + head_dim/2)` within the last dim, rotates via
    /// the precomputed `cos`/`sin` tables.
    ///
    /// Layout: input of rank ≥ 2 with last two dims `[seq, head_dim]`;
    /// `cos` and `sin` shape `[seq, head_dim/2]`. The input must be
    /// contiguous (first-cut limitation; stride-aware path can come
    /// later — the underlying kernel supports a single `stride_b`
    /// for per-batch strided inputs).
    ///
    /// F32 only today (mirrors Vulkan's constraint).
    pub fn rope(
        &self,
        cos: &Self,
        sin: &Self,
        x_layout: &Layout,
    ) -> Result<Self> {
        use crate::device::LaunchConfig;
        if self.dtype() != DType::F32 || cos.dtype() != DType::F32 || sin.dtype() != DType::F32 {
            return fuel_core_types::bail!("CudaStorage: rope requires f32 inputs");
        }
        let dims = x_layout.shape().dims();
        let rank = dims.len();
        if rank < 2 {
            return fuel_core_types::bail!("CudaStorage: rope requires rank >= 2, got {dims:?}");
        }
        if !x_layout.is_contiguous() {
            return fuel_core_types::bail!(
                "CudaStorage: rope first-cut requires contiguous x_layout"
            );
        }
        let seq = dims[rank - 2];
        let head_dim = dims[rank - 1];
        if head_dim % 2 != 0 {
            return fuel_core_types::bail!(
                "CudaStorage: rope head_dim must be even, got {head_dim}"
            );
        }
        let outer: usize = dims[..rank - 2].iter().product::<usize>().max(1);

        // Map to the kernel's (bh, td, d) terms:
        //   bh = outer (batch * heads)
        //   td = seq * head_dim (contiguous per-batch element count)
        //   d  = head_dim (pair (i, i+d/2) within each head_dim block)
        // Total threads = bh * td / 2 = outer * seq * head_dim / 2.
        let bh = outer as u32;
        let td = (seq * head_dim) as u32;
        let d = head_dim as u32;
        let stride_b: u32 = 0; // contiguous → single shared cos/sin table

        let total_threads = (bh as u64) * (td as u64) / 2;
        let device = self.device().clone();

        let (x_src, out) = match &self.slice {
            CudaStorageSlice::F32(s) => {
                let src = s.slice(x_layout.start_offset()..s.len());
                let out = unsafe { device.alloc::<f32>(outer * seq * head_dim)? };
                (src, out)
            }
            _ => return fuel_core_types::bail!(
                "CudaStorage::rope: expected F32, got {:?}", self.dtype()),
        };
        let cos_src = match &cos.slice {
            CudaStorageSlice::F32(s) => s.slice(0..s.len()),
            _ => return fuel_core_types::bail!("CudaStorage::rope: cos must be F32"),
        };
        let sin_src = match &sin.slice {
            CudaStorageSlice::F32(s) => s.slice(0..s.len()),
            _ => return fuel_core_types::bail!("CudaStorage::rope: sin must be F32"),
        };
        let mut out = out;

        let func = device.get_or_load_func("rope_f32", &kernels::REDUCE)?;
        let block_size = 256u32;
        let grid = ((total_threads + block_size as u64 - 1) / block_size as u64) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&x_src);
        builder.arg(&cos_src);
        builder.arg(&sin_src);
        builder.arg(&mut out);
        crate::builder_arg!(builder, bh);
        crate::builder_arg!(builder, td);
        crate::builder_arg!(builder, d);
        crate::builder_arg!(builder, stride_b);
        unsafe { builder.launch(cfg) }.w()?;

        Ok(Self { slice: CudaStorageSlice::F32(out), device })
    }

    pub fn to_cpu_storage(&self) -> Result<HostBuffer> {
        let device = &self.device;
        match &self.slice {
            CudaStorageSlice::U8(slice) => {
                let cpu_storage = device.clone_dtoh(&slice.as_slice())?;
                Ok(HostBuffer::U8(cpu_storage))
            }
            CudaStorageSlice::I8(slice) => {
                let cpu_storage = device.clone_dtoh(&slice.as_slice())?;
                Ok(HostBuffer::I8(cpu_storage))
            }
            CudaStorageSlice::U32(slice) => {
                let cpu_storage = device.clone_dtoh(&slice.as_slice())?;
                Ok(HostBuffer::U32(cpu_storage))
            }
            CudaStorageSlice::I16(slice) => {
                let cpu_storage = device.clone_dtoh(&slice.as_slice())?;
                Ok(HostBuffer::I16(cpu_storage))
            }
            CudaStorageSlice::I32(slice) => {
                let cpu_storage = device.clone_dtoh(&slice.as_slice())?;
                Ok(HostBuffer::I32(cpu_storage))
            }
            CudaStorageSlice::I64(slice) => {
                let cpu_storage = device.clone_dtoh(&slice.as_slice())?;
                Ok(HostBuffer::I64(cpu_storage))
            }
            CudaStorageSlice::BF16(slice) => {
                let cpu_storage = device.clone_dtoh(&slice.as_slice())?;
                Ok(HostBuffer::BF16(cpu_storage))
            }
            CudaStorageSlice::F16(slice) => {
                let cpu_storage = device.clone_dtoh(&slice.as_slice())?;
                Ok(HostBuffer::F16(cpu_storage))
            }
            CudaStorageSlice::F32(slice) => {
                let cpu_storage = device.clone_dtoh(&slice.as_slice())?;
                Ok(HostBuffer::F32(cpu_storage))
            }
            CudaStorageSlice::F64(slice) => {
                let cpu_storage = device.clone_dtoh(&slice.as_slice())?;
                Ok(HostBuffer::F64(cpu_storage))
            }
            CudaStorageSlice::F8E4M3(slice) => {
                let cpu_storage = device.clone_dtoh(&slice.as_slice())?;
                Ok(HostBuffer::F8E4M3(cpu_storage))
            }
            CudaStorageSlice::F4(_)
            | CudaStorageSlice::F6E2M3(_)
            | CudaStorageSlice::F6E3M2(_)
            | CudaStorageSlice::F8E8M0(_) => Err(CudaError::UnsupportedDtype {
                dtype: self.dtype(),
                op: "to_cpu_storage",
            }
            .into()),
        }
    }

    pub fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = WhereCond(self, layout).map(&t.slice, t_l, &f.slice, f_l, &device)?;
        Ok(Self { slice, device })
    }

    /// Conv1D forward — baracuda alpha.38 cuDNN-backed FFI.
    pub fn conv1d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &fuel_core_types::conv::ParamsConv1D,
    ) -> Result<Self> {
        let device = self.device().clone();
        // Contiguize inputs if needed (baracuda's conv FFI takes plain
        // NCL contig pointers, unlike the prior `crate::cudnn` path which
        // supported strided descriptors).
        let (inp_storage, inp_l_owned, _inp_keep);
        let (inp_ref, inp_l_ref): (&Self, &Layout);
        if inp_l.is_contiguous() && inp_l.start_offset() == 0 {
            inp_ref = self;
            inp_l_ref = inp_l;
        } else {
            let mut t = unsafe { device.alloc_uninit(inp_l.shape(), self.dtype())? };
            self.copy_strided_src(&mut t, 0, inp_l)?;
            inp_l_owned = Layout::contiguous(inp_l.shape().clone());
            _inp_keep = ();
            inp_storage = t;
            inp_ref = &inp_storage;
            inp_l_ref = &inp_l_owned;
        }
        let (k_storage, k_l_owned);
        let (k_ref, k_l_ref): (&Self, &Layout);
        if kernel_l.is_contiguous() && kernel_l.start_offset() == 0 {
            k_ref = kernel;
            k_l_ref = kernel_l;
        } else {
            let mut t = unsafe { device.alloc_uninit(kernel_l.shape(), kernel.dtype())? };
            kernel.copy_strided_src(&mut t, 0, kernel_l)?;
            k_l_owned = Layout::contiguous(kernel_l.shape().clone());
            k_storage = t;
            k_ref = &k_storage;
            k_l_ref = &k_l_owned;
        }
        let slice = conv1d_dispatch(&inp_ref.slice, inp_l_ref, &k_ref.slice, k_l_ref, params, &device)?;
        Ok(Self { slice, device })
    }

    /// ConvTranspose1D forward — baracuda alpha.38 cuDNN-backed FFI.
    /// The prior PTX `ConvTranspose1D` Map2 + the Col2Im1D fast path
    /// retired in Phase 5b — baracuda handles both regular and
    /// output_padding cases natively.
    pub fn conv_transpose1d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &fuel_core_types::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let inp_storage;
        let inp_l_owned;
        let (inp_ref, inp_l_ref): (&Self, &Layout);
        if inp_l.is_contiguous() && inp_l.start_offset() == 0 {
            inp_ref = self;
            inp_l_ref = inp_l;
        } else {
            inp_storage = unsafe {
                let mut t = device.alloc_uninit(inp_l.shape(), self.dtype())?;
                self.copy_strided_src(&mut t, 0, inp_l)?;
                t
            };
            inp_l_owned = Layout::contiguous(inp_l.shape().clone());
            inp_ref = &inp_storage;
            inp_l_ref = &inp_l_owned;
        }
        let k_storage;
        let k_l_owned;
        let (k_ref, k_l_ref): (&Self, &Layout);
        if kernel_l.is_contiguous() && kernel_l.start_offset() == 0 {
            k_ref = kernel;
            k_l_ref = kernel_l;
        } else {
            k_storage = unsafe {
                let mut t = device.alloc_uninit(kernel_l.shape(), kernel.dtype())?;
                kernel.copy_strided_src(&mut t, 0, kernel_l)?;
                t
            };
            k_l_owned = Layout::contiguous(kernel_l.shape().clone());
            k_ref = &k_storage;
            k_l_ref = &k_l_owned;
        }
        let slice = conv_transpose1d_dispatch(
            &inp_ref.slice, inp_l_ref, &k_ref.slice, k_l_ref, params, &device,
        )?;
        Ok(Self { slice, device })
    }

    /// Conv2D forward — baracuda alpha.38 cuDNN-backed FFI. The prior
    /// cfg-gated dual implementation (im2col + matmul fallback for no-cudnn
    /// builds, `crate::cudnn::launch_conv2d` for cudnn builds) collapsed
    /// into one path now that baracuda always ships the conv FFI under
    /// the `cudnn` feature on `baracuda-kernels-sys` (always enabled in
    /// Fuel's Cargo.toml).
    pub fn conv2d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &fuel_core_types::conv::ParamsConv2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        // Contiguize inputs if needed (baracuda's conv FFI takes plain
        // contig NCHW pointers, no stride descriptors).
        let inp_storage;
        let inp_l_owned;
        let (inp_ref, inp_l_ref): (&Self, &Layout);
        if inp_l.is_contiguous() && inp_l.start_offset() == 0 {
            inp_ref = self;
            inp_l_ref = inp_l;
        } else {
            inp_storage = unsafe {
                let mut t = device.alloc_uninit(inp_l.shape(), self.dtype())?;
                self.copy_strided_src(&mut t, 0, inp_l)?;
                t
            };
            inp_l_owned = Layout::contiguous(inp_l.shape().clone());
            inp_ref = &inp_storage;
            inp_l_ref = &inp_l_owned;
        }
        let k_storage;
        let k_l_owned;
        let (k_ref, k_l_ref): (&Self, &Layout);
        if kernel_l.is_contiguous() && kernel_l.start_offset() == 0 {
            k_ref = kernel;
            k_l_ref = kernel_l;
        } else {
            k_storage = unsafe {
                let mut t = device.alloc_uninit(kernel_l.shape(), kernel.dtype())?;
                kernel.copy_strided_src(&mut t, 0, kernel_l)?;
                t
            };
            k_l_owned = Layout::contiguous(kernel_l.shape().clone());
            k_ref = &k_storage;
            k_l_ref = &k_l_owned;
        }
        let slice = conv2d_dispatch(&inp_ref.slice, inp_l_ref, &k_ref.slice, k_l_ref, params, &device)?;
        Ok(Self { slice, device })
    }

    /// ConvTranspose2D forward — baracuda alpha.38 cuDNN-backed FFI.
    pub fn conv_transpose2d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &fuel_core_types::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let inp_storage;
        let inp_l_owned;
        let (inp_ref, inp_l_ref): (&Self, &Layout);
        if inp_l.is_contiguous() && inp_l.start_offset() == 0 {
            inp_ref = self;
            inp_l_ref = inp_l;
        } else {
            inp_storage = unsafe {
                let mut t = device.alloc_uninit(inp_l.shape(), self.dtype())?;
                self.copy_strided_src(&mut t, 0, inp_l)?;
                t
            };
            inp_l_owned = Layout::contiguous(inp_l.shape().clone());
            inp_ref = &inp_storage;
            inp_l_ref = &inp_l_owned;
        }
        let k_storage;
        let k_l_owned;
        let (k_ref, k_l_ref): (&Self, &Layout);
        if kernel_l.is_contiguous() && kernel_l.start_offset() == 0 {
            k_ref = kernel;
            k_l_ref = kernel_l;
        } else {
            k_storage = unsafe {
                let mut t = device.alloc_uninit(kernel_l.shape(), kernel.dtype())?;
                kernel.copy_strided_src(&mut t, 0, kernel_l)?;
                t
            };
            k_l_owned = Layout::contiguous(kernel_l.shape().clone());
            k_ref = &k_storage;
            k_l_ref = &k_l_owned;
        }
        let slice = conv_transpose2d_dispatch(
            &inp_ref.slice, inp_l_ref, &k_ref.slice, k_l_ref, params, &device,
        )?;
        Ok(Self { slice, device })
    }

    pub fn avg_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let slice = pool2d_dispatch(&self.slice, &device, l, k, stride, PoolOp::Avg)?;
        Ok(Self { slice, device })
    }

    pub fn max_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let slice = pool2d_dispatch(&self.slice, &device, l, k, stride, PoolOp::Max)?;
        Ok(Self { slice, device })
    }

    pub fn upsample_nearest1d(&self, _: &Layout, _out_sz: usize) -> Result<Self> {
        fuel_core_types::bail!("upsample-nearest1d is not supported on cuda")
    }

    pub fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = upsample_nearest2d_dispatch(&self.slice, &device, l, out_w, out_h)?;
        Ok(Self { slice, device })
    }

    pub fn upsample_bilinear2d(
        &self,
        l: &Layout,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = upsample_bilinear2d_dispatch(
            &self.slice, &device, l, out_w, out_h, align_corners, scale_h, scale_w,
        )?;
        Ok(Self { slice, device })
    }

    pub fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = IndexSelect(ids, ids_l, dim).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }
    pub fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = Gather(ids, ids_l, dim).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }
    pub fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let device = self.device().clone();
        Scatter(ids, ids_l, dim).map(&mut self.slice, l, &src.slice, src_l, &device)
    }
    pub fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let device = self.device().clone();
        ScatterAdd(ids, ids_l, dim).map(&mut self.slice, l, &src.slice, src_l, &device)
    }
    pub fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let device = self.device().clone();
        let mut acc = unsafe { device.alloc_uninit(l.shape(), self.dtype())? };
        self.copy_strided_src(&mut acc, 0, l)?;
        IndexAdd(ids, ids_l, dim).map(&mut acc.slice, l, &src.slice, src_l, &device)?;
        Ok(acc)
    }

    pub fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let elem_count = b * m * n;
        let dev = &self.device;
        let slice = match (&self.slice, &rhs.slice) {
            (CudaStorageSlice::BF16(lhs), CudaStorageSlice::BF16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..lhs.len());
                let rhs = &rhs.slice(rhs_l.start_offset()..rhs.len());
                let cfg = gemm_config(bf16::ONE, bf16::ZERO, (b, m, n, k), lhs_l, rhs_l)?;
                let mut out = unsafe { dev.alloc::<bf16>(elem_count)? };
                unsafe { gemm_strided_batched_bf16(&self.device.blas.0, cfg, rhs, lhs, &mut out) }
                    .w()?;
                CudaStorageSlice::BF16(out)
            }
            (CudaStorageSlice::F16(lhs), CudaStorageSlice::F16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..lhs.len());
                let rhs = &rhs.slice(rhs_l.start_offset()..rhs.len());
                let cfg = gemm_config(f16::ONE, f16::ZERO, (b, m, n, k), lhs_l, rhs_l)?;
                let mut out = unsafe { dev.alloc::<f16>(elem_count)? };
                unsafe { gemm_strided_batched_f16(&self.device.blas.0, cfg, rhs, lhs, &mut out) }
                    .w()?;
                CudaStorageSlice::F16(out)
            }
            (CudaStorageSlice::F32(lhs), CudaStorageSlice::F32(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..lhs.len());
                let rhs = &rhs.slice(rhs_l.start_offset()..rhs.len());
                let cfg = gemm_config(1., 0., (b, m, n, k), lhs_l, rhs_l)?;
                let mut out = unsafe { dev.alloc::<f32>(elem_count)? };
                unsafe { gemm_strided_batched_f32(&self.device.blas.0, cfg, rhs, lhs, &mut out) }
                    .w()?;
                CudaStorageSlice::F32(out)
            }
            (CudaStorageSlice::F64(lhs), CudaStorageSlice::F64(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..lhs.len());
                let rhs = &rhs.slice(rhs_l.start_offset()..rhs.len());
                let cfg = gemm_config(1., 0., (b, m, n, k), lhs_l, rhs_l)?;
                let mut out = unsafe { dev.alloc::<f64>(elem_count)? };
                unsafe { gemm_strided_batched_f64(&self.device.blas.0, cfg, rhs, lhs, &mut out) }
                    .w()?;
                CudaStorageSlice::F64(out)
            }
            _ => Err(CudaError::InternalError("dtype mismatch in matmul op"))?,
        };
        let device = dev.clone();
        Ok(Self { slice, device })
    }

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
        let dev = &self.device;
        let d1 = d1 as u32;
        let d2 = d2 as u32;
        // Nothing to copy so we exit early to avoid launching a kernel and some potential invalid
        // argument with a null pointer.
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let dst_s = dst_s as u32;
        let src_s = src_s as u32;
        let ((src, _guard_src), (dst, _guard_dst), kname) = match (&self.slice, &mut dst.slice) {
            (S::U8(s), S::U8(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_u8"),
            (S::U32(s), S::U32(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_u32"),
            (S::I16(s), S::I16(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_i16"),
            (S::I32(s), S::I32(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_i32"),
            (S::I64(s), S::I64(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_i64"),
            (S::BF16(s), S::BF16(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_bf16"),
            (S::F16(s), S::F16(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_f16"),
            (S::F32(s), S::F32(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_f32"),
            (S::F64(s), S::F64(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_f64"),
            (S::F8E4M3(s), S::F8E4M3(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_u8"),
            (S::F8E8M0(s), S::F8E8M0(d)) => (slice_ptr(s, src_o), slice_ptr(d, dst_o), "copy2d_u8"),
            _ => Err(CudaError::InternalError("dtype mismatch in copy2d"))?,
        };
        let func = dev.get_or_load_func(kname, &kernels::FILL)?;
        let cfg = LaunchConfig::for_num_elems(d1 * d2);
        let mut builder = func.builder();
        barg!(builder, src);
        barg!(builder, dst);
        barg!(builder, d1);
        barg!(builder, d2);
        builder.arg(&src_s);
        builder.arg(&dst_s);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }

    pub fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let dev = &self.device;
        let ds = SlicePtrOrNull::params_from_layout(dev, src_l)?;
        match (&self.slice, &mut dst.slice) {
            (CudaStorageSlice::BF16(src), CudaStorageSlice::BF16(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_bf16", &kernels::UNARY)?;
                    let mut builder = func.builder();
                    barg!(builder, el_count);
                    barg!(builder, dims.len());
                    ds.builder_arg(&mut builder);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    // SAFETY: ffi.
                    unsafe { builder.launch(cfg) }.w()?;
                }
            }
            (CudaStorageSlice::F16(src), CudaStorageSlice::F16(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_f16", &kernels::UNARY)?;
                    let mut builder = func.builder();
                    barg!(builder, el_count);
                    barg!(builder, dims.len());
                    ds.builder_arg(&mut builder);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    // SAFETY: ffi.
                    unsafe { builder.launch(cfg) }.w()?;
                }
            }
            (CudaStorageSlice::F32(src), CudaStorageSlice::F32(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_f32", &kernels::UNARY)?;
                    let mut builder = func.builder();
                    barg!(builder, el_count);
                    barg!(builder, dims.len());
                    ds.builder_arg(&mut builder);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    // SAFETY: ffi.
                    unsafe { builder.launch(cfg) }.w()?;
                }
            }
            (CudaStorageSlice::U8(src), CudaStorageSlice::U8(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_u8", &kernels::UNARY)?;
                    let mut builder = func.builder();
                    barg!(builder, el_count);
                    barg!(builder, dims.len());
                    ds.builder_arg(&mut builder);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    // SAFETY: ffi.
                    unsafe { builder.launch(cfg) }.w()?;
                }
            }
            (CudaStorageSlice::U32(src), CudaStorageSlice::U32(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_u32", &kernels::UNARY)?;
                    let mut builder = func.builder();
                    barg!(builder, el_count);
                    barg!(builder, dims.len());
                    ds.builder_arg(&mut builder);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    // SAFETY: ffi.
                    unsafe { builder.launch(cfg) }.w()?;
                }
            }
            (CudaStorageSlice::I16(src), CudaStorageSlice::I16(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_i16", &kernels::UNARY)?;
                    let mut builder = func.builder();
                    barg!(builder, el_count);
                    barg!(builder, dims.len());
                    ds.builder_arg(&mut builder);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    // SAFETY: ffi.
                    unsafe { builder.launch(cfg) }.w()?;
                }
            }
            (CudaStorageSlice::I32(src), CudaStorageSlice::I32(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_i32", &kernels::UNARY)?;
                    let mut builder = func.builder();
                    barg!(builder, el_count);
                    barg!(builder, dims.len());
                    ds.builder_arg(&mut builder);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    // SAFETY: ffi.
                    unsafe { builder.launch(cfg) }.w()?;
                }
            }
            (CudaStorageSlice::I64(src), CudaStorageSlice::I64(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_i64", &kernels::UNARY)?;
                    let mut builder = func.builder();
                    barg!(builder, el_count);
                    barg!(builder, dims.len());
                    ds.builder_arg(&mut builder);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    // SAFETY: ffi.
                    unsafe { builder.launch(cfg) }.w()?;
                }
            }
            (CudaStorageSlice::F64(src), CudaStorageSlice::F64(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_f64", &kernels::UNARY)?;
                    let mut builder = func.builder();
                    barg!(builder, el_count);
                    barg!(builder, dims.len());
                    ds.builder_arg(&mut builder);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    // SAFETY: ffi.
                    unsafe { builder.launch(cfg) }.w()?;
                }
            }
            (CudaStorageSlice::F8E4M3(src), CudaStorageSlice::F8E4M3(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_f8e4m3", &kernels::UNARY)?;
                    let mut builder = func.builder();
                    barg!(builder, el_count);
                    barg!(builder, dims.len());
                    ds.builder_arg(&mut builder);
                    builder.arg(&src);
                    builder.arg(&mut dst);
                    // SAFETY: ffi.
                    unsafe { builder.launch(cfg) }.w()?;
                }
            }
            _ => Err(CudaError::InternalError(
                "dtype mismatch in copy_strided op",
            ))?,
        }
        Ok(())
    }
}

// Default for the reduced precision setting is false, similar to pytorch.
// https://github.com/pytorch/pytorch/issues/123157
static MM_F16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_BF16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_F32_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
/// allowed with f32 GEMMs.
pub fn gemm_reduced_precision_f32() -> bool {
    MM_F32_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
/// allowed with f32 GEMMs.
pub fn set_gemm_reduced_precision_f32(b: bool) {
    MM_F32_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn gemm_reduced_precision_f16() -> bool {
    MM_F16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn set_gemm_reduced_precision_f16(b: bool) {
    MM_F16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn gemm_reduced_precision_bf16() -> bool {
    MM_BF16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn set_gemm_reduced_precision_bf16(b: bool) {
    MM_BF16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

unsafe fn gemm_strided_batched_f32(
    cublas: &baracuda_cublas::Handle,
    cfg: StridedBatchedConfig<f32>,
    a: &baracuda_driver::DeviceSlice<f32>,
    b: &baracuda_driver::DeviceSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> std::result::Result<(), baracuda_cublas::Error> {
    use baracuda_cublas::{cublasComputeType_t, cudaDataType_t};

    let compute_type = if gemm_reduced_precision_f32() {
        cublasComputeType_t::Compute32FFastTF32
    } else {
        cublasComputeType_t::Compute32F
    };
    let alpha = &cfg.gemm.alpha as *const f32 as *const _;
    let beta = &cfg.gemm.beta as *const f32 as *const _;

    let a_ptr = a.as_raw().0 as *const _;
    let b_ptr = b.as_raw().0 as *const _;
    let c_ptr = c.as_raw().0 as *mut _;

    unsafe {
        baracuda_cublas::gemm_strided_batched_ex(
            cublas,
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            alpha,
            a_ptr,
            cudaDataType_t::R_32F,
            cfg.gemm.lda,
            cfg.stride_a,
            b_ptr,
            cudaDataType_t::R_32F,
            cfg.gemm.ldb,
            cfg.stride_b,
            beta,
            c_ptr,
            cudaDataType_t::R_32F,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            compute_type,
            99_i32,
        )
    }
}

unsafe fn gemm_strided_batched_f16(
    cublas: &baracuda_cublas::Handle,
    cfg: StridedBatchedConfig<f16>,
    a: &baracuda_driver::DeviceSlice<f16>,
    b: &baracuda_driver::DeviceSlice<f16>,
    c: &mut CudaSlice<f16>,
) -> std::result::Result<(), baracuda_cublas::Error> {
    use baracuda_cublas::{cublasComputeType_t, cudaDataType_t};

    let alpha = cfg.gemm.alpha;
    let beta = cfg.gemm.beta;
    let alpha_f32: f32 = cfg.gemm.alpha.to_f32();
    let beta_f32: f32 = cfg.gemm.beta.to_f32();
    let (compute_type, alpha, beta) = if gemm_reduced_precision_f16() {
        (
            cublasComputeType_t::Compute16F,
            (&alpha) as *const f16 as *const _,
            (&beta) as *const f16 as *const _,
        )
    } else {
        (
            cublasComputeType_t::Compute32F,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    };

    let a_ptr = a.as_raw().0 as *const _;
    let b_ptr = b.as_raw().0 as *const _;
    let c_ptr = c.as_raw().0 as *mut _;
    unsafe {
        baracuda_cublas::gemm_strided_batched_ex(
            cublas,
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            alpha,
            a_ptr,
            cudaDataType_t::R_16F,
            cfg.gemm.lda,
            cfg.stride_a,
            b_ptr,
            cudaDataType_t::R_16F,
            cfg.gemm.ldb,
            cfg.stride_b,
            beta,
            c_ptr,
            cudaDataType_t::R_16F,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            compute_type,
            99_i32,
        )
    }
}

unsafe fn gemm_strided_batched_bf16(
    cublas: &baracuda_cublas::Handle,
    cfg: StridedBatchedConfig<bf16>,
    a: &baracuda_driver::DeviceSlice<bf16>,
    b: &baracuda_driver::DeviceSlice<bf16>,
    c: &mut CudaSlice<bf16>,
) -> std::result::Result<(), baracuda_cublas::Error> {
    use baracuda_cublas::{cublasComputeType_t, cudaDataType_t};

    let alpha_f32: f32 = cfg.gemm.alpha.to_f32();
    let beta_f32: f32 = cfg.gemm.beta.to_f32();
    // The type for alpha and beta depends on the computeType.
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex
    let (compute_type, alpha, beta) = if gemm_reduced_precision_bf16() {
        (
            cublasComputeType_t::Compute32FFast16BF,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    } else {
        (
            cublasComputeType_t::Compute32F,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    };

    let a_ptr = a.as_raw().0 as *const _;
    let b_ptr = b.as_raw().0 as *const _;
    let c_ptr = c.as_raw().0 as *mut _;
    unsafe {
        baracuda_cublas::gemm_strided_batched_ex(
            cublas,
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            alpha,
            a_ptr,
            cudaDataType_t::R_16BF,
            cfg.gemm.lda,
            cfg.stride_a,
            b_ptr,
            cudaDataType_t::R_16BF,
            cfg.gemm.ldb,
            cfg.stride_b,
            beta,
            c_ptr,
            cudaDataType_t::R_16BF,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            compute_type,
            99_i32,
        )
    }
}

unsafe fn gemm_strided_batched_f64(
    cublas: &baracuda_cublas::Handle,
    cfg: StridedBatchedConfig<f64>,
    a: &baracuda_driver::DeviceSlice<f64>,
    b: &baracuda_driver::DeviceSlice<f64>,
    c: &mut CudaSlice<f64>,
) -> std::result::Result<(), baracuda_cublas::Error> {
    use baracuda_cublas::{cublasComputeType_t, cudaDataType_t};

    let alpha = &cfg.gemm.alpha as *const f64 as *const _;
    let beta = &cfg.gemm.beta as *const f64 as *const _;

    let a_ptr = a.as_raw().0 as *const _;
    let b_ptr = b.as_raw().0 as *const _;
    let c_ptr = c.as_raw().0 as *mut _;

    unsafe {
        baracuda_cublas::gemm_strided_batched_ex(
            cublas,
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            alpha,
            a_ptr,
            cudaDataType_t::R_64F,
            cfg.gemm.lda,
            cfg.stride_a,
            b_ptr,
            cudaDataType_t::R_64F,
            cfg.gemm.ldb,
            cfg.stride_b,
            beta,
            c_ptr,
            cudaDataType_t::R_64F,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            cublasComputeType_t::Compute64F,
            99_i32,
        )
    }
}
