// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
#[cfg(feature = "cuda")]
use fuel::cuda_backend::kernels::ffi;
#[allow(unused_imports)]
use fuel::quantized::{self, QTensor};
use fuel::{Result, Tensor};

/// Dispatches token-to-expert GEMM using CUDA WMMA kernels.
///
/// Only available with the `cuda` feature; on other backends it always returns an error.
///
/// # Example
///
/// ```no_run
/// // Requires the `cuda` feature and CUDA-resident tensors.
/// // use fuel_nn::moe_gemm;
/// // let out = moe_gemm(&input, &weights, &None, &sorted_ids, &expert_ids, topk, false)?;
/// ```
#[cfg(feature = "cuda")]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    
    use fuel::DType;
    use half::{bf16, f16};

    fn cuda_inner<
        T: fuel_graph_cuda::storage::CudaDType + baracuda_types::DeviceRepr + baracuda_types::ValidAsZeroBits,
    >(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (mut size_m, size_k1) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k1,
            size_k
        );
        let dev = fuel::cuda_backend::as_device(input.device())?;
        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => {
                fuel::bail!("moe_gemm_wmma only accepts f16/bf16 inputs")
            }
        };

        let (input, _) = input.storage_and_layout();
        let input = input.downcast_ref::<fuel::CudaStorage>().ok_or_else(|| fuel::Error::Msg("input must be a cuda tensor".to_string()).bt())?.as_cuda_slice::<T>()?;

        let (weights, _) = weights.storage_and_layout();
        let weights = weights.downcast_ref::<fuel::CudaStorage>().ok_or_else(|| fuel::Error::Msg("weight must be a cuda tensor".to_string()).bt())?.as_cuda_slice::<T>()?;

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = sorted_token_ids.downcast_ref::<fuel::CudaStorage>().ok_or_else(|| fuel::Error::Msg("sorted_token_ids must be a cuda tensor".to_string()).bt())?.as_cuda_slice::<u32>()?;

        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = experts_ids.downcast_ref::<fuel::CudaStorage>().ok_or_else(|| fuel::Error::Msg("experts_ids must be a cuda tensor".to_string()).bt())?.as_cuda_slice::<u32>()?;

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
        let topk_weights = topk_weights.downcast_ref::<fuel::CudaStorage>().ok_or_else(|| fuel::Error::Msg("topk_weights must be a cuda tensor".to_string()).bt())?.as_cuda_slice::<f32>()?;
            let weights_ptr = topk_weights.as_raw().0 as *const f32;
            weights_ptr
        } else {
            std::ptr::null()
        };

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;
        let expert_counts = unsafe { dev.alloc::<u32>(num_experts) }?;
        let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;

        let stream = dev.cuda_stream().as_raw() as i64;
        use core::ffi::c_void;

        unsafe {
            ffi::moe_gemm_wmma(
                input.as_raw().0 as *const c_void, // [size_m, size_k]
                weights.as_raw().0 as *const c_void, // [num_experts, size_n, size_k]
                sorted_token_ids.as_raw().0 as *const i32,
                experts_ids.as_raw().0 as *const i32,
                topk_weights_ptr,
                output.as_raw().0 as *mut c_void, // [size_m, size_n]
                expert_counts.as_raw().0 as *mut i32, // pre-allocated buffer [num_experts]
                expert_offsets.as_raw().0 as *mut i32, // pre-allocated buffer [num_experts + 1]
                num_experts as i32,
                topk as i32,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                data_type as i32, // 0=float16, 1=bf16 (for input/output)
                is_prefill,
                stream,
            );
        }

        use fuel::op::BackpropOp;
        let output = fuel::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            fuel::Storage::new(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );

        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_inner::<f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        DType::BF16 => cuda_inner::<bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        _ => {
            fuel::bail!("moe_gemm only accepts f16/bf16 inputs")
        }
    }
}

#[cfg(not(feature = "cuda"))]
/// Mixture-of-Experts GEMM: dispatches tokens to experts and collects results.
///
/// On non-CUDA builds this always returns an error. Requires the `cuda` feature.
///
/// # Example
///
/// ```text
/// // Requires the `cuda` feature and CUDA-resident tensors.
/// // use fuel_nn::moe_gemm;
/// // let out = moe_gemm(&input, &weights, &None, &sorted_ids, &expert_ids, topk, false)?;
/// ```
pub fn moe_gemm(
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    fuel::bail!("moe_gemm is only implemented for the cuda backend")
}

/// Dispatches token-to-expert quantized GEMM (GGUF weights) using CUDA kernels.
///
/// Only available with the `cuda` feature; on other backends it always returns an error.
///
/// # Example
///
/// ```text
/// // Requires the `cuda` feature and CUDA-resident tensors.
/// // use fuel_nn::moe_gemm_gguf;
/// // let out = moe_gemm_gguf(&input, &weights, &None, &sorted_ids, &expert_ids, topk, false, fuel::DType::F32)?;
/// ```
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: fuel::DType,
) -> Result<Tensor> {
    
    use fuel::quantized::GgmlDType;
    use fuel::DType;
    use half::{bf16, f16};

    #[allow(clippy::too_many_arguments)]
    fn cuda_inner(
        input: &Tensor,
        weights: &QTensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let (mut size_m, size_k) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k,
            size_k1,
        );
        let dev = fuel::cuda_backend::as_device(input.device())?;

        // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5
        let gguf_dtype = match weights.dtype() {
            GgmlDType::Q8_0 => 0,
            GgmlDType::Q4K => 1,
            GgmlDType::Q2K => 2,
            GgmlDType::Q3K => 3,
            GgmlDType::Q5K => 4,
            GgmlDType::Q6K => 5,
            _ => {
                fuel::bail!(
                    "moe_gemm_gguf `ISQ` only accept q2k, q3k, q4k, q5k, q6k or q8_0 weights!"
                )
            }
        };

        let weight_ptr = weights.device_ptr()?;

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
        let topk_weights = topk_weights.downcast_ref::<fuel::CudaStorage>().ok_or_else(|| fuel::Error::Msg("topk_weights must be a cuda tensor".to_string()).bt())?.as_cuda_slice::<f32>()?;
            let w_ptr = topk_weights.as_raw().0 as *const f32;
            w_ptr
        } else {
            std::ptr::null()
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = sorted_token_ids.downcast_ref::<fuel::CudaStorage>().ok_or_else(|| fuel::Error::Msg("sorted_token_ids must be a cuda tensor".to_string()).bt())?.as_cuda_slice::<u32>()?;
        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = experts_ids.downcast_ref::<fuel::CudaStorage>().ok_or_else(|| fuel::Error::Msg("experts_ids must be a cuda tensor".to_string()).bt())?.as_cuda_slice::<u32>()?;

        let output = unsafe { dev.alloc::<f32>(size_m * size_n) }?;
        let stream = dev.cuda_stream().as_raw() as i64;
        use fuel::op::BackpropOp;
        use core::ffi::c_void;

        assert!(size_k % 8 == 0, "size_k must divisible by 8");
        unsafe {
            if is_prefill {
                let input = input.to_dtype(dtype)?;
                let (input, _) = input.storage_and_layout();
                let input_cuda = input
                    .downcast_ref::<fuel::CudaStorage>()
                    .ok_or_else(|| fuel::Error::Msg("input must be a cuda tensor".into()).bt())?;
                let (input_ptr, input_dtype) = if dtype == DType::F16 {
                    let c = input_cuda.as_cuda_slice::<f16>()?;
                    (c.as_raw().0 as *const c_void, 0)
                } else {
                    let c = input_cuda.as_cuda_slice::<bf16>()?;
                    (c.as_raw().0 as *const c_void, 1)
                };
                ffi::moe_gemm_gguf_prefill(
                    input_ptr,  // [size_m or size_m/topk, size_k]
                    weight_ptr, // [num_experts, size_n, size_k]
                    sorted_token_ids.as_raw().0 as *const i32,
                    experts_ids.as_raw().0 as *const i32,
                    topk_weights_ptr,
                    output.as_raw().0 as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    input_dtype,
                    gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                    stream,
                );
            } else {
                let (input, _) = input.storage_and_layout();
        let input = input.downcast_ref::<fuel::CudaStorage>().ok_or_else(|| fuel::Error::Msg("input must be a cuda tensor".to_string()).bt())?.as_cuda_slice::<f32>()?;

                ffi::moe_gemm_gguf(
                    input.as_raw().0 as *const f32, // [size_m or size_m/topk, size_k]
                    weight_ptr as *const c_void, // [num_experts, size_n, size_k]
                    sorted_token_ids.as_raw().0 as *const i32,
                    experts_ids.as_raw().0 as *const i32,
                    topk_weights_ptr,
                    output.as_raw().0 as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                    stream,
                );
            }
        }

        let output = fuel::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            fuel::Storage::new(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );

        Ok(output)
    }

    match input.dtype() {
        DType::F32 => cuda_inner(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
            dtype,
        ),
        _ => {
            fuel::bail!("moe_gemm_gguf only accepts f32 inputs")
        }
    }
}

/// Dispatches token-to-expert quantized GEMM (GGUF weights) using CUDA kernels.
///
/// On non-CUDA builds this always returns an error. Requires the `cuda` feature.
///
/// # Example
///
/// ```text
/// // Requires the `cuda` feature and CUDA-resident tensors.
/// // use fuel_nn::moe_gemm_gguf;
/// // let out = moe_gemm_gguf(&input, &weights, &None, &sorted_ids, &expert_ids, topk, false, fuel::DType::F32)?;
/// ```
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    _: &Tensor,
    _: &QTensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
    _: fuel::DType,
) -> Result<Tensor> {
    fuel::bail!("moe_gemm_gguf is only implemented for the cuda backend")
}
