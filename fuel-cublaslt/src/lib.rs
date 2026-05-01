//! CUBLASLt fused GEMM operations for the Fuel ML framework.
//!
//! Provides hardware-accelerated fused matrix multiplication with optional bias
//! addition and activation functions (GELU/ReLU) using NVIDIA cuBLASLt. Exposed
//! as Fuel [`CustomOp2`](fuel::CustomOp2) and [`CustomOp3`](fuel::CustomOp3) so
//! they integrate with Fuel's autograd and device management.
//!
//! Built on baracuda's low-level cuBLASLt wrappers
//! ([`baracuda_cublas::lt`]). Exposes the same public surface as the
//! cudarc-based predecessor (`CublasLt` handle, `CublasLTMatmul`/
//! `CublasLTBatchMatmul` ops, `fused_matmul` / `fused_batch_matmul`).

pub use baracuda_cublas::lt::Activation;

use baracuda_cublas::lt::{self as lt, LtHandle, MatmulDesc, MatrixLayout};
use baracuda_cublas_sys::functions::{cublasComputeType_t, cudaDataType_t};
use baracuda_cublas_sys::types::cublasOperation_t;
use fuel::cuda::WrapErr;
use fuel::dyn_backend::DynBackendStorage;
use fuel::{CudaStorage, DType, Device, Error, Layout, Result, Shape, Tensor};
use half::{bf16, f16};
use std::ffi::{c_int, c_void};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// LtHandle wrapper + Sync/Send
// ---------------------------------------------------------------------------
//
// Matches fuel-graph-cuda's `CublasHandle` shim: baracuda marks cuBLASLt
// handles `Send` but `!Sync` (per-handle single-thread use). Fuel's graph
// executor serialises GPU dispatch onto one thread, so the promise holds at
// the application level; unsafe-impl Sync to let the handle flow through
// `Arc` in CustomOp impls.
struct SyncLtHandle(LtHandle);
unsafe impl Send for SyncLtHandle {}
unsafe impl Sync for SyncLtHandle {}
impl std::ops::Deref for SyncLtHandle {
    type Target = LtHandle;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A handle to the cuBLASLt library context.
///
/// Cheaply clones via `Arc`. Bind one per CUDA device.
#[derive(Clone)]
pub struct CublasLt(Arc<SyncLtHandle>);

impl std::fmt::Debug for CublasLt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CublasLt").finish_non_exhaustive()
    }
}

impl CublasLt {
    /// Create a new `CublasLt` handle for the given CUDA device.
    pub fn new(device: &Device) -> Result<Self> {
        if !device.is_cuda() {
            fuel::bail!("`device` must be a `cuda` device");
        }
        let handle = LtHandle::new().w()?;
        Ok(Self(Arc::new(SyncLtHandle(handle))))
    }
}

// ---------------------------------------------------------------------------
// LtDType — per-element-type cuBLASLt config
// ---------------------------------------------------------------------------

trait LtDType:
    baracuda_types::DeviceRepr + baracuda_types::ValidAsZeroBits + fuel_graph_cuda::storage::CudaDType + Copy
{
    const CUDA_DATA_TYPE: cudaDataType_t;
}

impl LtDType for f16 {
    const CUDA_DATA_TYPE: cudaDataType_t = cudaDataType_t::R_16F;
}
impl LtDType for bf16 {
    const CUDA_DATA_TYPE: cudaDataType_t = cudaDataType_t::R_16BF;
}
impl LtDType for f32 {
    const CUDA_DATA_TYPE: cudaDataType_t = cudaDataType_t::R_32F;
}

// ---------------------------------------------------------------------------
// Generic lt_matmul helper — the one place that talks to cuBLASLt.
// Mirrors what cudarc's `CudaBlasLT::matmul` did on our behalf previously.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
unsafe fn lt_matmul<T: LtDType>(
    handle: &LtHandle,
    stream: &baracuda_driver::Stream,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    a_ptr: *const c_void,
    lda: i64,
    stride_a: Option<i64>,
    b_ptr: *const c_void,
    ldb: i64,
    stride_b: Option<i64>,
    c_ptr: *mut c_void,
    ldc: i64,
    stride_c: Option<i64>,
    bias_ptr: Option<*const c_void>,
    activation: Option<Activation>,
    batch_count: Option<i32>,
) -> Result<()> {
    // Compute type is always Compute32F (the scale type / alpha-beta are f32
    // regardless of A/B/C element type — matches cudarc's previous behaviour).
    let desc =
        MatmulDesc::new(cublasComputeType_t::Compute32F, cudaDataType_t::R_32F).w()?;
    // A is transposed (TN layout — matches the old cudarc-era fwd), B isn't.
    desc.set_transa(cublasOperation_t::T).w()?;
    desc.set_transb(cublasOperation_t::N).w()?;
    // Bias + activation fuse into a single epilogue code. The caller supplies
    // the Activation variant; if a bias pointer is present we upgrade to the
    // Bias variant when none was explicitly requested.
    let epilogue: i32 = match (activation, bias_ptr) {
        (Some(act), _) => act as i32,
        (None, Some(_)) => Activation::Bias as i32,
        (None, None) => Activation::Identity as i32,
    };
    desc.set_epilogue(epilogue).w()?;
    if let Some(bptr) = bias_ptr {
        desc.set_bias_pointer(bptr).w()?;
    }

    // Layouts. A is MxK stored as (rows=K, cols=M) because transa=T.
    let a_layout = MatrixLayout::new(T::CUDA_DATA_TYPE, k as u64, m as u64, lda).w()?;
    let b_layout = MatrixLayout::new(T::CUDA_DATA_TYPE, k as u64, n as u64, ldb).w()?;
    let c_layout = MatrixLayout::new(T::CUDA_DATA_TYPE, m as u64, n as u64, ldc).w()?;
    let d_layout = MatrixLayout::new(T::CUDA_DATA_TYPE, m as u64, n as u64, ldc).w()?;

    if let Some(b) = batch_count {
        a_layout.set_batch_count(b).w()?;
        b_layout.set_batch_count(b).w()?;
        c_layout.set_batch_count(b).w()?;
        d_layout.set_batch_count(b).w()?;
        if let Some(s) = stride_a {
            a_layout.set_strided_batch_offset(s).w()?;
        }
        if let Some(s) = stride_b {
            b_layout.set_strided_batch_offset(s).w()?;
        }
        if let Some(s) = stride_c {
            c_layout.set_strided_batch_offset(s).w()?;
            d_layout.set_strided_batch_offset(s).w()?;
        }
    }

    let alpha_f32 = alpha;
    let beta_f32 = beta;

    unsafe {
        lt::matmul(
            handle,
            &desc,
            &alpha_f32 as *const f32 as *const c_void,
            a_ptr,
            &a_layout,
            b_ptr,
            &b_layout,
            &beta_f32 as *const f32 as *const c_void,
            c_ptr,
            &c_layout,
            c_ptr,
            &d_layout,
            None,
            core::ptr::null_mut(),
            0,
            Some(stream),
        )
        .w()?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pointer extractors — collapse the 3 dtype branches via LtDType.
// ---------------------------------------------------------------------------

fn slice_ptr<T: LtDType>(
    storage: &CudaStorage,
    layout: &Layout,
) -> Result<*const c_void> {
    let slice = storage.as_cuda_slice::<T>()?;
    let slice = slice.slice(layout.start_offset()..slice.len());
    Ok(slice.as_raw().0 as *const c_void)
}

fn bias_ptr<T: LtDType>(
    bias: Option<&CudaStorage>,
    bias_l: Option<&Layout>,
    expected_len: usize,
) -> Result<Option<*const c_void>> {
    match (bias, bias_l) {
        (Some(bias), Some(bias_l)) => {
            if bias_l.shape().dims1()? != expected_len {
                fuel::bail!("Bias does not have the correct shape");
            }
            let slice = bias.as_cuda_slice::<T>()?;
            let slice = slice.slice(bias_l.start_offset()..slice.len());
            Ok(Some(slice.as_raw().0 as *const c_void))
        }
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// CublasLTMatmul (2-D) — MxK × NxK → NxM
// ---------------------------------------------------------------------------

pub struct CublasLTMatmul {
    pub cublaslt: CublasLt,
    pub act: Option<Activation>,
    pub c: Option<Tensor>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,
}

impl CublasLTMatmul {
    fn run<T: LtDType>(
        &self,
        a: &CudaStorage,
        a_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
        bias: Option<&CudaStorage>,
        bias_l: Option<&Layout>,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = a.device();
        let (m, k) = a_l.shape().dims2()?;
        let (n, b_1) = b_l.shape().dims2()?;
        if b_1 != k {
            fuel::bail!("This layer only supports TN layout");
        }
        let lda = k as i64;
        let ldb = k as i64;
        let ldc = m as i64;
        let out_shape = Shape::from((n, m));

        let a_ptr = slice_ptr::<T>(a, a_l)?;
        let b_ptr = slice_ptr::<T>(b, b_l)?;
        let b_opt = bias_ptr::<T>(bias, bias_l, m)?;

        let out: baracuda_driver::DeviceBuffer<T> = match &self.c {
            Some(c) => {
                let (c_storage, c_l) = c.storage_and_layout();
                let c = c_storage
                    .downcast_ref::<fuel::CudaStorage>()
                    .ok_or_else(|| Error::Msg("`c` must be a cuda tensor".into()).bt())?
                    .as_cuda_slice::<T>()?;
                match c_l.contiguous_offsets() {
                    Some((o1, o2)) => {
                        if o1 != 0 {
                            fuel::bail!("`c` start offset must be 0");
                        }
                        if o2 != out_shape.elem_count() {
                            fuel::bail!("`c` end offset must be {}", out_shape.elem_count())
                        }
                    }
                    None => fuel::bail!("`c` has to be contiguous"),
                };
                if c_l.shape().dims2()? != (n, m) {
                    fuel::bail!("`c` does not have the correct shape");
                }
                dev.clone_dtod(c)?
            }
            None => unsafe { dev.alloc::<T>(out_shape.elem_count())? },
        };
        let out_ptr = out.as_raw().0 as *mut c_void;

        unsafe {
            lt_matmul::<T>(
                &self.cublaslt.0,
                &dev.cuda_stream(),
                m,
                n,
                k,
                self.alpha.unwrap_or(1.0),
                self.beta.unwrap_or(0.0),
                a_ptr,
                lda,
                None,
                b_ptr,
                ldb,
                None,
                out_ptr,
                ldc,
                None,
                b_opt,
                self.act,
                None,
            )?;
        }

        Ok((CudaStorage::wrap_cuda_slice(out, dev.clone()), out_shape))
    }

    fn dispatch(
        &self,
        a: &CudaStorage,
        a_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
        bias: Option<&CudaStorage>,
        bias_l: Option<&Layout>,
    ) -> Result<(CudaStorage, Shape)> {
        match a.dtype() {
            DType::F16 => self.run::<f16>(a, a_l, b, b_l, bias, bias_l),
            DType::BF16 => self.run::<bf16>(a, a_l, b, b_l, bias, bias_l),
            DType::F32 => self.run::<f32>(a, a_l, b, b_l, bias, bias_l),
            dt => fuel::bail!("cublaslt-matmul is only supported for f16/bf16/f32 ({dt:?})"),
        }
    }
}

fn wrap_cuda(out: CudaStorage) -> Box<dyn DynBackendStorage> {
    Box::new(out)
}

fn downcast_cuda<'a>(s: &'a dyn DynBackendStorage) -> Result<&'a CudaStorage> {
    s.as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| Error::Msg("cublaslt ops require a CUDA storage".into()).bt())
}

impl fuel::CustomOp2 for CublasLTMatmul {
    fn name(&self) -> &'static str {
        "cublaslt-matmul"
    }

    fn fwd(
        &self,
        s1: &dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)> {
        let a = downcast_cuda(s1)?;
        let b = downcast_cuda(s2)?;
        let (out, shape) = self.dispatch(a, l1, b, l2, None, None)?;
        Ok((wrap_cuda(out), shape))
    }
}

impl fuel::CustomOp3 for CublasLTMatmul {
    fn name(&self) -> &'static str {
        "cublaslt-matmul-add"
    }

    fn fwd(
        &self,
        s1: &dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
        s3: &dyn DynBackendStorage,
        l3: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)> {
        let a = downcast_cuda(s1)?;
        let b = downcast_cuda(s2)?;
        let bias = downcast_cuda(s3)?;
        let (out, shape) = self.dispatch(a, l1, b, l2, Some(bias), Some(l3))?;
        Ok((wrap_cuda(out), shape))
    }
}

/// Fused matmul + add + ReLU/GELU activation using cuBLASLt.
///
/// Computes `result = act(alpha * A * B^T + beta * C + bias)` where `A` is
/// `MxK` and `B` is `NxK`, producing output of shape `NxM`.
pub fn fused_matmul(
    a: &Tensor,
    b: &Tensor,
    out: Option<&Tensor>,
    alpha: Option<f32>,
    beta: Option<f32>,
    bias: Option<&Tensor>,
    act: Option<Activation>,
    cublaslt: CublasLt,
) -> Result<Tensor> {
    let op = CublasLTMatmul {
        act,
        cublaslt,
        c: out.cloned(),
        alpha,
        beta,
    };

    if let Some(bias) = bias {
        a.apply_op3(b, bias, op)
    } else {
        a.apply_op2(b, op)
    }
}

// ---------------------------------------------------------------------------
// CublasLTBatchMatmul (3-D) — BxMxK × BxNxK → BxNxM
// ---------------------------------------------------------------------------

pub struct CublasLTBatchMatmul {
    pub cublaslt: CublasLt,
    pub act: Option<Activation>,
    pub c: Option<Tensor>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,
}

impl CublasLTBatchMatmul {
    fn run<T: LtDType>(
        &self,
        a: &CudaStorage,
        a_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
        bias: Option<&CudaStorage>,
        bias_l: Option<&Layout>,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = a.device();
        let (batch_size, m, k) = a_l.shape().dims3()?;
        let (b_0, n, b_2) = b_l.shape().dims3()?;
        if b_2 != k {
            fuel::bail!("This layer only supports TN layout");
        }
        if b_0 != batch_size {
            fuel::bail!("`b` must have the same batch size as `a`");
        }
        let lda = k as i64;
        let ldb = k as i64;
        let ldc = m as i64;
        let out_shape = Shape::from((batch_size, n, m));

        let a_ptr = slice_ptr::<T>(a, a_l)?;
        let b_ptr = slice_ptr::<T>(b, b_l)?;
        let b_opt = bias_ptr::<T>(bias, bias_l, m)?;

        let (out, stride_c): (baracuda_driver::DeviceBuffer<T>, usize) = match &self.c {
            Some(c) => {
                let (c_storage, c_l) = c.storage_and_layout();
                let c = c_storage
                    .downcast_ref::<fuel::CudaStorage>()
                    .ok_or_else(|| Error::Msg("`c` must be a cuda tensor".into()).bt())?
                    .as_cuda_slice::<T>()?;
                match c_l.contiguous_offsets() {
                    Some((o1, o2)) => {
                        if o1 != 0 {
                            fuel::bail!("`c` start offset must be 0");
                        }
                        if o2 != out_shape.elem_count() {
                            fuel::bail!("`c` end offset must be {}", out_shape.elem_count())
                        }
                    }
                    None => fuel::bail!("`c` has to be contiguous"),
                };
                if c_l.shape().dims3()? != (batch_size, n, m) {
                    fuel::bail!("`c` does not have the correct shape");
                }
                (dev.clone_dtod(c)?, c_l.stride()[0])
            }
            None => (
                unsafe { dev.alloc::<T>(out_shape.elem_count())? },
                n * m,
            ),
        };
        let out_ptr = out.as_raw().0 as *mut c_void;

        unsafe {
            lt_matmul::<T>(
                &self.cublaslt.0,
                &dev.cuda_stream(),
                m,
                n,
                k,
                self.alpha.unwrap_or(1.0),
                self.beta.unwrap_or(0.0),
                a_ptr,
                lda,
                Some(a_l.stride()[0] as i64),
                b_ptr,
                ldb,
                Some(b_l.stride()[0] as i64),
                out_ptr,
                ldc,
                Some(stride_c as i64),
                b_opt,
                self.act,
                Some(batch_size as c_int),
            )?;
        }

        Ok((CudaStorage::wrap_cuda_slice(out, dev.clone()), out_shape))
    }

    fn dispatch(
        &self,
        a: &CudaStorage,
        a_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
        bias: Option<&CudaStorage>,
        bias_l: Option<&Layout>,
    ) -> Result<(CudaStorage, Shape)> {
        match a.dtype() {
            DType::F16 => self.run::<f16>(a, a_l, b, b_l, bias, bias_l),
            DType::BF16 => self.run::<bf16>(a, a_l, b, b_l, bias, bias_l),
            DType::F32 => self.run::<f32>(a, a_l, b, b_l, bias, bias_l),
            dt => {
                fuel::bail!("cublaslt-batch-matmul is only supported for f16/bf16/f32 ({dt:?})")
            }
        }
    }
}

impl fuel::CustomOp2 for CublasLTBatchMatmul {
    fn name(&self) -> &'static str {
        "cublaslt-batch-matmul"
    }

    fn fwd(
        &self,
        s1: &dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)> {
        let a = downcast_cuda(s1)?;
        let b = downcast_cuda(s2)?;
        let (out, shape) = self.dispatch(a, l1, b, l2, None, None)?;
        Ok((wrap_cuda(out), shape))
    }
}

impl fuel::CustomOp3 for CublasLTBatchMatmul {
    fn name(&self) -> &'static str {
        "cublaslt-batch-matmul-add"
    }

    fn fwd(
        &self,
        s1: &dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
        s3: &dyn DynBackendStorage,
        l3: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)> {
        let a = downcast_cuda(s1)?;
        let b = downcast_cuda(s2)?;
        let bias = downcast_cuda(s3)?;
        let (out, shape) = self.dispatch(a, l1, b, l2, Some(bias), Some(l3))?;
        Ok((wrap_cuda(out), shape))
    }
}

/// Fused batch matmul + add + ReLU/GELU activation using cuBLASLt.
///
/// Computes `result = act(alpha * A * B^T + beta * C + bias)` where `A` is
/// `BxMxK` and `B` is `BxNxK`, producing output of shape `BxNxM`.
pub fn fused_batch_matmul(
    a: &Tensor,
    b: &Tensor,
    out: Option<&Tensor>,
    alpha: Option<f32>,
    beta: Option<f32>,
    bias: Option<&Tensor>,
    act: Option<Activation>,
    cublaslt: CublasLt,
) -> Result<Tensor> {
    let op = CublasLTBatchMatmul {
        act,
        cublaslt,
        c: out.cloned(),
        alpha,
        beta,
    };

    if let Some(bias) = bias {
        a.apply_op3(b, bias, op)
    } else {
        a.apply_op2(b, op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel::{DType, Device};

    fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
        let b = 10f32.powi(digits);
        let t = t.to_vec3::<f32>()?;
        let t = t
            .iter()
            .map(|t| {
                t.iter()
                    .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                    .collect()
            })
            .collect();
        Ok(t)
    }

    #[test]
    fn test_fused_matmul() -> Result<()> {
        let device = fuel::cuda_backend::new_device(0)?;

        let a = Tensor::randn(0., 1., (8, 4), &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., (2, 4), &device)?.to_dtype(DType::F32)?;
        let bias = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let cublaslt = CublasLt::new(&device)?;

        let res = fused_matmul(&a, &b, None, None, None, Some(&bias), None, cublaslt)?;
        let expected = (b.matmul(&a.t()?)? + bias.broadcast_left(2)?)?;

        let diff = ((res.clone() - expected.clone())?.abs()? / expected.clone())?
            .sum_all()?
            .to_vec0::<f32>()?;

        assert!(diff < 2e-3, "{res} != {expected}, Diff {diff}");

        Ok(())
    }

    #[test]
    fn test_fused_batch_matmul() -> Result<()> {
        let device = fuel::cuda_backend::new_device(0)?;

        let a = Tensor::randn(0., 1., (3, 8, 4), &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., (3, 2, 4), &device)?.to_dtype(DType::F32)?;
        let c = Tensor::randn(0., 1., (3, 2, 8), &device)?.to_dtype(DType::F32)?;
        let bias = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let cublaslt = CublasLt::new(&device)?;

        let res = fused_batch_matmul(
            &a,
            &b,
            Some(&c),
            None,
            Some(1.0),
            Some(&bias),
            None,
            cublaslt,
        )?;
        let expected = (b.matmul(&a.t()?)?.add(&c)? + bias.broadcast_left((3, 2))?)?;

        assert_eq!(
            to_vec3_round(res.to_dtype(DType::F32)?, 4)?,
            to_vec3_round(expected.to_dtype(DType::F32)?, 4)?
        );
        Ok(())
    }
}
