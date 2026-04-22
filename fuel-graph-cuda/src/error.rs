use fuel_core_types::{DType, Layout};

/// Errors from the CUDA backend — driver, NVRTC, cuBLAS, and curand
/// failures, plus Fuel-local variants for missing kernels / dtype
/// mismatches / non-contiguous matmul.
#[derive(thiserror::Error, Debug)]
pub enum CudaError {
    #[error(transparent)]
    Cuda(#[from] baracuda_driver::Error),

    #[error(transparent)]
    Compiler(#[from] baracuda_nvrtc::Error),

    #[error(transparent)]
    Cublas(#[from] baracuda_cublas::Error),

    #[error(transparent)]
    Curand(#[from] baracuda_curand::Error),

    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: String },

    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDtype { dtype: DType, op: &'static str },

    #[error("internal error '{0}'")]
    InternalError(&'static str),

    #[error(
        "matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}"
    )]
    MatMulNonContiguous {
        lhs_stride: Layout,
        rhs_stride: Layout,
        mnk: (usize, usize, usize),
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("{cuda} when loading {module_name}")]
    Load {
        cuda: baracuda_driver::Error,
        module_name: String,
    },
}

impl From<CudaError> for fuel_core_types::Error {
    fn from(val: CudaError) -> Self {
        fuel_core_types::Error::cuda(val)
    }
}

pub trait WrapErr<O> {
    fn w(self) -> std::result::Result<O, fuel_core_types::Error>;
}

impl<O, E: Into<CudaError>> WrapErr<O> for std::result::Result<O, E> {
    fn w(self) -> std::result::Result<O, fuel_core_types::Error> {
        self.map_err(|e| fuel_core_types::Error::cuda(e.into()))
    }
}
