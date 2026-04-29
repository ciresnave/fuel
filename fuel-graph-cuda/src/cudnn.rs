use fuel_core_types::dtype::WithDType;
use baracuda_cudnn::{ConvolutionDescriptor as ConvForward, Handle as Cudnn};
use baracuda_driver::{DeviceBuffer as CudaSlice, DeviceSlice as CudaView};
use baracuda_types::{DeviceRepr, ValidAsZeroBits};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// The cudnn handles are stored per thread here rather than on the CudaDevice as they are neither
// send nor sync.
thread_local! {
    static CUDNN: RefCell<HashMap<crate::DeviceId, Arc<Cudnn>>> = HashMap::new().into();
}

pub(crate) fn launch_conv2d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + baracuda_cudnn::CudnnDataType,
    Y: baracuda_cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &fuel_core_types::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &fuel_core_types::conv::ParamsConv2D,
    dev: &crate::CudaDevice,
) -> fuel_core_types::Result<()> {
    use fuel_core_types::conv::CudnnFwdAlgo as FuelAlgo;
    use baracuda_cudnn_sys::types::cudnnConvolutionFwdAlgo_t as A;

    let device_id = dev.id();
    let cudnn = CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?;
    let conv = cudnn.create_conv2d::<Y>(
        /* pad */ [params.padding as i32, params.padding as i32],
        /* stride */ [params.stride as i32, params.stride as i32],
        /* dilation */ [params.dilation as i32, params.dilation as i32],
        baracuda_cudnn_sys::types::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.i_h as i32,
        params.i_w as i32,
    ];
    // Note that `src` already starts at the proper offset.
    let x = if src_l.is_contiguous() {
        cudnn.create_4d_tensor::<T>(
            baracuda_cudnn_sys::types::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            x_shape,
        )?
    } else {
        let s = src_l.stride();
        cudnn.create_4d_tensor_ex::<T>(
            x_shape,
            [s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32],
        )?
    };
    let w = cudnn.create_4d_filter::<T>(
        baracuda_cudnn_sys::types::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_out as i32,
            params.c_in as i32,
            params.k_h as i32,
            params.k_w as i32,
        ],
    )?;
    let (w_out, h_out) = (params.out_w() as i32, params.out_h() as i32);
    let y = cudnn.create_4d_tensor::<T>(
        baracuda_cudnn_sys::types::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [params.b_size as i32, params.c_out as i32, h_out, w_out],
    )?;
    let conv2d = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let alg = match params.cudnn_fwd_algo {
        None => conv2d.pick_algorithm()?,
        Some(FuelAlgo::ImplicitGemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        Some(FuelAlgo::ImplicitPrecompGemm) => {
            A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        Some(FuelAlgo::Gemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        Some(FuelAlgo::Direct) => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        Some(FuelAlgo::Fft) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        Some(FuelAlgo::FftTiling) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        Some(FuelAlgo::Winograd) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        Some(FuelAlgo::WinogradNonFused) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        Some(FuelAlgo::Count) => A::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
    };
    let workspace_size = conv2d.get_workspace_size(alg)?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        conv2d.launch::<CudaSlice<u8>, _, _, _>(
            alg,
            Some(&mut workspace),
            (T::one(), T::zero()),
            src,
            filter,
            dst,
        )?;
    }
    Ok(())
}

pub(crate) fn launch_conv1d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + baracuda_cudnn::CudnnDataType,
    Y: baracuda_cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &fuel_core_types::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &fuel_core_types::conv::ParamsConv1D,
    dev: &crate::CudaDevice,
) -> fuel_core_types::Result<()> {
    use fuel_core_types::conv::CudnnFwdAlgo as FuelAlgo;
    use baracuda_cudnn_sys::types::cudnnConvolutionFwdAlgo_t as A;

    let device_id = dev.id();
    let cudnn = CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?;
    let conv = cudnn.create_conv2d::<Y>(
        /* pad */ [params.padding as i32, 0],
        /* stride */ [params.stride as i32, 1],
        /* dilation */ [params.dilation as i32, 1],
        baracuda_cudnn_sys::types::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnsettensornddescriptor
    // > Tensors are restricted to having at least 4 dimensions, and at most CUDNN_DIM_MAX
    // > dimensions (defined in cudnn.h). When working with lower dimensional data, it is
    // > recommended that the user create a 4D tensor, and set the size along unused dimensions
    // > to 1.
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.l_in as i32,
        1,
    ];
    // Note that `src` already starts at the proper offset.
    let x = if src_l.is_contiguous() {
        cudnn.create_4d_tensor::<T>(
            baracuda_cudnn_sys::types::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            x_shape,
        )?
    } else {
        let s = src_l.stride();
        cudnn.create_4d_tensor_ex::<T>(x_shape, [s[0] as i32, s[1] as i32, s[2] as i32, 1i32])?
    };
    let w = cudnn.create_4d_filter::<T>(
        baracuda_cudnn_sys::types::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_out as i32,
            params.c_in as i32,
            params.k_size as i32,
            1,
        ],
    )?;
    let l_out = params.l_out() as i32;
    let y = cudnn.create_4d_tensor::<T>(
        baracuda_cudnn_sys::types::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [params.b_size as i32, params.c_out as i32, l_out, 1],
    )?;
    let conv1d = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let alg = match params.cudnn_fwd_algo {
        None => conv1d.pick_algorithm()?,
        Some(FuelAlgo::ImplicitGemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        Some(FuelAlgo::ImplicitPrecompGemm) => {
            A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        Some(FuelAlgo::Gemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        Some(FuelAlgo::Direct) => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        Some(FuelAlgo::Fft) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        Some(FuelAlgo::FftTiling) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        Some(FuelAlgo::Winograd) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        Some(FuelAlgo::WinogradNonFused) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        Some(FuelAlgo::Count) => A::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
    };
    let workspace_size = conv1d.get_workspace_size(alg)?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        conv1d.launch::<CudaSlice<u8>, _, _, _>(
            alg,
            Some(&mut workspace),
            (T::one(), T::zero()),
            src,
            filter,
            dst,
        )?;
    }
    Ok(())
}
