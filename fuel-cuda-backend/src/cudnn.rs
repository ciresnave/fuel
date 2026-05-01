//! cuDNN-backed convolution launchers (forward only).
//!
//! Uses the safe baracuda-cudnn wrappers for handle / descriptor / workspace
//! management, but calls `cudnnConvolutionForward` directly via
//! `baracuda-cudnn-sys`. The safe wrapper requires `&DeviceBuffer<T>` for
//! the input + filter operands; Fuel's call sites work with
//! `DeviceSlice<T>` views (offset into a parent allocation), so we go
//! through the raw FFI to pass `CUdeviceptr` from either type.

use fuel_core_types::dtype::WithDType;
use baracuda_cudnn::{
    convolution_forward_workspace_size, ConvMode, ConvolutionDescriptor, CudnnDataType,
    FilterDescriptor, FwdAlgo, Handle as Cudnn, TensorDescriptor, TensorFormat,
};
use baracuda_driver::{DeviceBuffer as CudaSlice, DeviceSlice as CudaView};
use baracuda_types::{DeviceRepr, ValidAsZeroBits};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::WrapErr;

// The cudnn handles are stored per thread here rather than on the CudaDevice as they are neither
// send nor sync.
thread_local! {
    static CUDNN: RefCell<HashMap<crate::DeviceId, Arc<Cudnn>>> = HashMap::new().into();
}

fn get_or_init_handle(dev: &crate::CudaDevice) -> fuel_core_types::Result<Arc<Cudnn>> {
    let id = dev.id();
    CUDNN.with(|cell| -> fuel_core_types::Result<Arc<Cudnn>> {
        if let Some(h) = cell.borrow().get(&id) {
            return Ok(h.clone());
        }
        let h = Cudnn::new().w()?;
        h.set_stream(&dev.cuda_stream()).w()?;
        let arc = Arc::new(h);
        cell.borrow_mut().insert(id, arc.clone());
        Ok(arc)
    })
}

fn map_fwd_algo(algo: fuel_core_types::conv::CudnnFwdAlgo) -> FwdAlgo {
    use fuel_core_types::conv::CudnnFwdAlgo as FuelAlgo;
    match algo {
        FuelAlgo::ImplicitGemm => FwdAlgo::ImplicitGemm,
        FuelAlgo::ImplicitPrecompGemm => FwdAlgo::ImplicitPrecompGemm,
        FuelAlgo::Gemm => FwdAlgo::Gemm,
        FuelAlgo::Direct => FwdAlgo::Direct,
        FuelAlgo::Fft => FwdAlgo::Fft,
        FuelAlgo::FftTiling => FwdAlgo::FftTiling,
        FuelAlgo::Winograd => FwdAlgo::Winograd,
        FuelAlgo::WinogradNonFused => FwdAlgo::WinogradNonfused,
        // No `Count` in the alpha.4 enum (it was a marker, never an algorithm).
        FuelAlgo::Count => FwdAlgo::ImplicitGemm,
    }
}

/// `Y = α · conv(X, W) + β · Y` via raw FFI. Accepts any source pointer
/// type (`DeviceBuffer`, `DeviceSlice`, `DeviceSliceMut`) by going through
/// `as_raw()` → `CUdeviceptr`. Mirrors `baracuda_cudnn::convolution_forward`
/// but without the `&DeviceBuffer<T>` constraint.
unsafe fn convolution_forward_raw<C>(
    handle: &Cudnn,
    alpha: &C,
    x_desc: &TensorDescriptor,
    x_ptr: baracuda_cuda_sys::CUdeviceptr,
    w_desc: &FilterDescriptor,
    w_ptr: baracuda_cuda_sys::CUdeviceptr,
    conv: &ConvolutionDescriptor,
    algo: FwdAlgo,
    workspace: &mut CudaSlice<u8>,
    beta: &C,
    y_desc: &TensorDescriptor,
    y_ptr: baracuda_cuda_sys::CUdeviceptr,
) -> fuel_core_types::Result<()> {
    use baracuda_cudnn_sys::cudnnConvolutionFwdAlgo_t as Algo;
    let loader = baracuda_cudnn_sys::cudnn().w()?;
    let func = loader.cudnn_convolution_forward().w()?;
    let raw_algo = match algo {
        FwdAlgo::ImplicitGemm => Algo::ImplicitGemm,
        FwdAlgo::ImplicitPrecompGemm => Algo::ImplicitPrecompGemm,
        FwdAlgo::Gemm => Algo::Gemm,
        FwdAlgo::Direct => Algo::Direct,
        FwdAlgo::Fft => Algo::Fft,
        FwdAlgo::FftTiling => Algo::FftTiling,
        FwdAlgo::Winograd => Algo::Winograd,
        FwdAlgo::WinogradNonfused => Algo::WinogradNonfused,
    };
    let status = unsafe {
        func(
            handle.as_raw(),
            alpha as *const C as *const core::ffi::c_void,
            x_desc.as_raw(),
            x_ptr.0 as *const core::ffi::c_void,
            w_desc.as_raw(),
            w_ptr.0 as *const core::ffi::c_void,
            conv.as_raw(),
            raw_algo,
            workspace.as_raw().0 as *mut core::ffi::c_void,
            workspace.byte_size(),
            beta as *const C as *const core::ffi::c_void,
            y_desc.as_raw(),
            y_ptr.0 as *mut core::ffi::c_void,
        )
    };
    baracuda_cudnn::Error::check(status).w()?;
    Ok(())
}

pub(crate) fn launch_conv2d<T, Y>(
    src: &CudaView<T>,
    src_l: &fuel_core_types::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &fuel_core_types::conv::ParamsConv2D,
    dev: &crate::CudaDevice,
) -> fuel_core_types::Result<()>
where
    T: DeviceRepr + WithDType + ValidAsZeroBits + CudnnDataType,
    Y: DeviceRepr + WithDType + CudnnDataType,
{
    let cudnn = get_or_init_handle(dev)?;

    let groups = params.groups.max(1) as i32;
    let conv = ConvolutionDescriptor::new_2d(
        params.padding as i32, params.padding as i32,
        params.stride as i32, params.stride as i32,
        params.dilation as i32, params.dilation as i32,
        ConvMode::CrossCorrelation,
        <Y as CudnnDataType>::DTYPE,
    ).w()?;
    if groups != 1 {
        conv.set_group_count(groups).w()?;
    }

    // cuDNN sees the full input (per-group channels × groups). Filter dim 1
    // stays per-group; output channels stay total.
    let n = params.b_size as i32;
    let c_in_total = (params.c_in * params.groups.max(1)) as i32;
    let h = params.i_h as i32;
    let w = params.i_w as i32;
    let x_desc = if src_l.is_contiguous() {
        TensorDescriptor::new_4d(TensorFormat::Nchw, <T as CudnnDataType>::DTYPE, n, c_in_total, h, w).w()?
    } else {
        let s = src_l.stride();
        TensorDescriptor::new_4d_ex(
            <T as CudnnDataType>::DTYPE, n, c_in_total, h, w,
            s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32,
        ).w()?
    };

    let w_desc = FilterDescriptor::new_4d(
        TensorFormat::Nchw, <T as CudnnDataType>::DTYPE,
        params.c_out as i32, params.c_in as i32,
        params.k_h as i32, params.k_w as i32,
    ).w()?;

    let (h_out, w_out) = (params.out_h() as i32, params.out_w() as i32);
    let y_desc = TensorDescriptor::new_4d(
        TensorFormat::Nchw, <T as CudnnDataType>::DTYPE,
        n, params.c_out as i32, h_out, w_out,
    ).w()?;

    let algo = params
        .cudnn_fwd_algo
        .map(map_fwd_algo)
        .unwrap_or(FwdAlgo::ImplicitGemm);

    let workspace_size =
        convolution_forward_workspace_size(&cudnn, &x_desc, &w_desc, &conv, &y_desc, algo).w()?;
    let mut workspace =
        CudaSlice::<u8>::zeros(dev.context_ref(), workspace_size.max(1)).w()?;

    let alpha: Y = Y::one();
    let beta: Y = Y::zero();
    unsafe {
        convolution_forward_raw(
            &cudnn, &alpha, &x_desc, src.as_raw(), &w_desc, filter.as_raw(),
            &conv, algo, &mut workspace, &beta, &y_desc, dst.as_raw(),
        )?;
    }
    Ok(())
}

pub(crate) fn launch_conv1d<T, Y>(
    src: &CudaView<T>,
    src_l: &fuel_core_types::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &fuel_core_types::conv::ParamsConv1D,
    dev: &crate::CudaDevice,
) -> fuel_core_types::Result<()>
where
    T: DeviceRepr + WithDType + ValidAsZeroBits + CudnnDataType,
    Y: DeviceRepr + WithDType + CudnnDataType,
{
    let cudnn = get_or_init_handle(dev)?;

    let conv = ConvolutionDescriptor::new_2d(
        params.padding as i32, 0,
        params.stride as i32, 1,
        params.dilation as i32, 1,
        ConvMode::CrossCorrelation,
        <Y as CudnnDataType>::DTYPE,
    ).w()?;

    // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html
    // > Tensors are restricted to having at least 4 dimensions; create a 4D
    // > tensor with the unused dimension set to 1.
    let n = params.b_size as i32;
    let c = params.c_in as i32;
    let l = params.l_in as i32;
    let x_desc = if src_l.is_contiguous() {
        TensorDescriptor::new_4d(TensorFormat::Nchw, <T as CudnnDataType>::DTYPE, n, c, l, 1).w()?
    } else {
        let s = src_l.stride();
        TensorDescriptor::new_4d_ex(
            <T as CudnnDataType>::DTYPE, n, c, l, 1,
            s[0] as i32, s[1] as i32, s[2] as i32, 1,
        ).w()?
    };
    let w_desc = FilterDescriptor::new_4d(
        TensorFormat::Nchw, <T as CudnnDataType>::DTYPE,
        params.c_out as i32, params.c_in as i32, params.k_size as i32, 1,
    ).w()?;
    let l_out = params.l_out() as i32;
    let y_desc = TensorDescriptor::new_4d(
        TensorFormat::Nchw, <T as CudnnDataType>::DTYPE,
        n, params.c_out as i32, l_out, 1,
    ).w()?;

    let algo = params
        .cudnn_fwd_algo
        .map(map_fwd_algo)
        .unwrap_or(FwdAlgo::ImplicitGemm);

    let workspace_size =
        convolution_forward_workspace_size(&cudnn, &x_desc, &w_desc, &conv, &y_desc, algo).w()?;
    let mut workspace =
        CudaSlice::<u8>::zeros(dev.context_ref(), workspace_size.max(1)).w()?;

    let alpha: Y = Y::one();
    let beta: Y = Y::zero();
    unsafe {
        convolution_forward_raw(
            &cudnn, &alpha, &x_desc, src.as_raw(), &w_desc, filter.as_raw(),
            &conv, algo, &mut workspace, &beta, &y_desc, dst.as_raw(),
        )?;
    }
    Ok(())
}
