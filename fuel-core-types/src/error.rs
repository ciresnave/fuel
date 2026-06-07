//! Fuel-specific Error and Result types.
use std::{convert::Infallible, fmt::Display};

use crate::{DType, DeviceLocation, Layout, Shape};

/// Diagnostic information for a matrix multiplication that encountered unexpected striding.
#[derive(Debug, Clone)]
pub struct MatMulUnexpectedStriding {
    pub lhs_l: Layout,
    pub rhs_l: Layout,
    pub bmnk: (usize, usize, usize, usize),
    pub msg: &'static str,
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// The main error type for the fuel library.
#[derive(thiserror::Error)]
pub enum Error {
    // === DType Errors ===
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("dtype mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DTypeMismatchBinaryOp {
        lhs: DType,
        rhs: DType,
        op: &'static str,
    },

    #[error("unsupported dtype {0:?} for op {1}")]
    UnsupportedDTypeForOp(DType, &'static str),

    // === Dimension Index Errors ===
    #[error("{op}: dimension index {dim} out of range for shape {shape:?}")]
    DimOutOfRange {
        shape: Shape,
        dim: i32,
        op: &'static str,
    },

    #[error("{op}: duplicate dim index {dims:?} for shape {shape:?}")]
    DuplicateDimIndex {
        shape: Shape,
        dims: Vec<usize>,
        op: &'static str,
    },

    // === Shape Errors ===
    #[error("unexpected rank, expected: {expected}, got: {got} ({shape:?})")]
    UnexpectedNumberOfDims {
        expected: usize,
        got: usize,
        shape: Shape,
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedShape {
        msg: String,
        expected: Shape,
        got: Shape,
    },

    #[error(
        "Shape mismatch, got buffer of size {buffer_size} which is compatible with shape {shape:?}"
    )]
    ShapeMismatch { buffer_size: usize, shape: Shape },

    #[error("shape mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    ShapeMismatchBinaryOp {
        lhs: Shape,
        rhs: Shape,
        op: &'static str,
    },

    #[error(
        "shape mismatch in cat for dim {dim}, shape for arg 1: {first_shape:?} shape for arg {n}: {nth_shape:?}"
    )]
    ShapeMismatchCat {
        dim: usize,
        first_shape: Shape,
        n: usize,
        nth_shape: Shape,
    },

    #[error("Cannot divide tensor of shape {shape:?} equally along dim {dim} into {n_parts}")]
    ShapeMismatchSplit {
        shape: Shape,
        dim: usize,
        n_parts: usize,
    },

    #[error("{op} can only be performed on a single dimension")]
    OnlySingleDimension { op: &'static str, dims: Vec<usize> },

    #[error("empty tensor for {op}")]
    EmptyTensor { op: &'static str },

    // === Device Errors ===
    #[error("device mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DeviceMismatchBinaryOp {
        lhs: DeviceLocation,
        rhs: DeviceLocation,
        op: &'static str,
    },

    // === Op Specific Errors ===
    #[error("narrow invalid args {msg}: {shape:?}, dim: {dim}, start: {start}, len:{len}")]
    NarrowInvalidArgs {
        shape: Shape,
        dim: usize,
        start: usize,
        len: usize,
        msg: &'static str,
    },

    #[error(
        "conv1d invalid args {msg}: inp: {inp_shape:?}, k: {k_shape:?}, pad: {padding}, stride: {stride}"
    )]
    Conv1dInvalidArgs {
        inp_shape: Shape,
        k_shape: Shape,
        padding: usize,
        stride: usize,
        msg: &'static str,
    },

    #[error("{op} invalid index {index} with dim size {size}")]
    InvalidIndex {
        op: &'static str,
        index: usize,
        size: usize,
    },

    #[error("cannot broadcast {src_shape:?} to {dst_shape:?}")]
    BroadcastIncompatibleShapes { src_shape: Shape, dst_shape: Shape },

    #[error("cannot set variable {msg}")]
    CannotSetVar { msg: &'static str },

    #[error("{0:?}")]
    MatMulUnexpectedStriding(Box<MatMulUnexpectedStriding>),

    #[error("{op} only supports contiguous tensors")]
    RequiresContiguous { op: &'static str },

    #[error("{op} expects at least one tensor")]
    OpRequiresAtLeastOneTensor { op: &'static str },

    #[error("{op} expects at least two tensors")]
    OpRequiresAtLeastTwoTensors { op: &'static str },

    #[error("backward is not supported for {op}")]
    BackwardNotSupported { op: &'static str },

    // === Phase 7.5 storage-unification dispatch errors ===
    /// No registered backend supports `(op, dtypes)` for the given
    /// input residency. Fires at DAG construction (planning time),
    /// not at execution.
    ///
    /// `dtypes` is the per-operand dtype list (inputs in order, then
    /// outputs) used as the binding-table lookup key. For uniform-
    /// precision ops the list contains a single repeated dtype; for
    /// mixed-precision ops (e.g. Cast: src→dst) the list distinguishes
    /// the operands.
    #[error(
        "no backend supports {op} on {dtypes:?}; available backends: \
         {available_backends:?}; supported (op, dtypes) by backend: \
         {supported_combinations:?}"
    )]
    NoBackendForOp {
        op: crate::dispatch::OpKind,
        dtypes: Vec<DType>,
        available_backends: Vec<crate::probe::BackendId>,
        supported_combinations: Vec<(
            crate::probe::BackendId,
            crate::dispatch::OpKind,
            Vec<DType>,
        )>,
    },

    /// No transfer path connects `from` and `to` directly, and
    /// host-staging fallback isn't available either (e.g. neither
    /// device can copy_to_host). Fires at DAG construction.
    #[error(
        "unsupported transfer from {from:?} to {to:?}; available paths: {available_paths:?}"
    )]
    UnsupportedTransfer {
        from: DeviceLocation,
        to: DeviceLocation,
        available_paths: Vec<crate::backend::TransferPath>,
    },

    /// Source storage's alignment doesn't meet the destination
    /// backend's required alignment, and Router has no way to
    /// repack on this path. Fires at DAG construction.
    #[error(
        "alignment mismatch on {backend:?}: required {required} bytes, got {actual} bytes"
    )]
    AlignmentMismatch {
        backend: crate::probe::BackendId,
        required: usize,
        actual: usize,
    },

    /// The `ExecutionPlan` consumed by the executor was built against
    /// an older `SystemTopology` generation than the one observable
    /// at dispatch time — backends were added, removed, or had their
    /// capabilities re-probed since the plan was committed. Caught
    /// at dispatch-chunk boundaries by Phase 4.3 of the picker-work
    /// arc; the realize layer catches this and rebuilds the plan
    /// against the fresh topology.
    ///
    /// `plan_generation` is the counter the plan was stamped with at
    /// `compile_plan` time. `current_generation` is the value
    /// `fuel_dispatch::dispatch::topology_generation()` observed when
    /// the executor crossed the chunk boundary.
    #[error(
        "execution plan was built against topology generation {plan_generation}, \
         but generation {current_generation} is now current; rebuild and retry"
    )]
    TopologyChanged {
        plan_generation: u64,
        current_generation: u64,
    },

    // === Other Errors ===
    #[error("the fuel crate has not been built with cuda support")]
    NotCompiledWithCudaSupport,

    #[error("the fuel crate has not been built with metal support")]
    NotCompiledWithMetalSupport,

    #[error("the fuel crate has not been built with vulkan support")]
    NotCompiledWithVulkanSupport,

    #[error("cannot find tensor {path}")]
    CannotFindTensor { path: String },

    // === Wrapped Errors ===
    #[error(transparent)]
    Cuda(Box<dyn std::error::Error + Send + Sync>),

    #[error(transparent)]
    Metal(Box<dyn std::error::Error + Send + Sync>),

    #[error(transparent)]
    Vulkan(Box<dyn std::error::Error + Send + Sync>),

    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios"), feature = "ug"))]
    #[error(transparent)]
    Ug(#[from] fuel_ug::Error),

    #[error(transparent)]
    TryFromIntError(#[from] core::num::TryFromIntError),

    #[error("npy/npz error {0}")]
    Npy(String),

    /// Zip file format error.
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    /// Utf8 parse error.
    #[error(transparent)]
    FromUtf8(#[from] std::string::FromUtf8Error),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// SafeTensor error.
    #[error(transparent)]
    SafeTensor(#[from] safetensors::SafeTensorError),

    #[error("unsupported safetensor dtype {0:?}")]
    UnsupportedSafeTensorDtype(safetensors::Dtype),

    /// Arbitrary errors wrapping.
    #[error("{0}")]
    Wrapped(Box<dyn std::fmt::Display + Send + Sync>),

    /// Arbitrary errors wrapping with context.
    #[error("{wrapped:?}\n{context:?}")]
    WrappedContext {
        wrapped: Box<dyn std::error::Error + Send + Sync>,
        context: String,
    },

    #[error("{context}\n{inner}")]
    Context {
        inner: Box<Self>,
        context: Box<dyn std::fmt::Display + Send + Sync>,
    },

    /// Adding path information to an error.
    #[error("path: {path:?} {inner}")]
    WithPath {
        inner: Box<Self>,
        path: std::path::PathBuf,
    },

    #[error("{inner}\n{backtrace}")]
    WithBacktrace {
        inner: Box<Self>,
        backtrace: Box<std::backtrace::Backtrace>,
    },

    /// User generated error message, typically created via `bail!`.
    #[error("{0}")]
    Msg(String),

    #[error("unwrap none")]
    UnwrapNone,

    /// A hard filter in the optimizer ranker's filter chain rejected
    /// every candidate at this decision point. Phase 1.1 of the
    /// picker-work arc. The user asked for something the binding-table
    /// can't deliver — typically a precision floor or tolerance budget
    /// no registered kernel meets — and the ranker surfaces it rather
    /// than silently substituting a non-admissible alternative.
    ///
    /// Soft filters (caps preferences, empirical refinements) never
    /// raise this error; only filters classified `FilterClass::Hard`
    /// fail loudly when they would leave the candidate set empty.
    #[error(
        "ranker hard filter `{filter}` rejected all {available_alternatives} candidate(s); \
         context: {ctx_summary}"
    )]
    FilterRejected {
        /// Name of the filter that triggered the rejection.
        filter: &'static str,
        /// Short diagnostic summary of the decision point's context
        /// (op kind, dtypes, device).
        ctx_summary: String,
        /// How many candidates entered the filter (zero on the way
        /// in is also possible — the candidate enumerator may have
        /// produced an empty set).
        available_alternatives: usize,
    },
}

/// A specialized [`Result`](std::result::Result) type for fuel operations.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Wraps an arbitrary displayable error.
    pub fn wrap(err: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        Self::Wrapped(Box::new(err)).bt()
    }

    /// Creates an error message from any displayable value.
    pub fn msg(err: impl std::fmt::Display) -> Self {
        Self::Msg(err.to_string()).bt()
    }

    /// Creates an error message from the debug representation of a value.
    pub fn debug(err: impl std::fmt::Debug) -> Self {
        Self::Msg(format!("{err:?}")).bt()
    }

    /// Wraps a CUDA backend error.
    pub fn cuda(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Cuda(Box::new(err)).bt()
    }

    /// Wraps a Metal backend error.
    pub fn metal(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Metal(Box::new(err)).bt()
    }

    /// Wraps a Vulkan backend error.
    pub fn vulkan(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Vulkan(Box::new(err)).bt()
    }

    /// Captures a backtrace and attaches it to this error.
    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled
            | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace {
                inner: Box::new(self),
                backtrace: Box::new(backtrace),
            },
        }
    }

    /// Attaches a filesystem path to this error for better diagnostics.
    pub fn with_path<P: AsRef<std::path::Path>>(self, p: P) -> Self {
        Self::WithPath {
            inner: Box::new(self),
            path: p.as_ref().to_path_buf(),
        }
    }

    /// Wraps this error with additional context.
    pub fn context(self, c: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        Self::Context {
            inner: Box::new(self),
            context: Box::new(c),
        }
    }
}

/// Returns early from a function with a formatted error message.
#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()).bt())
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()).bt())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()).bt())
    };
}

/// Combines two results into a single result containing a tuple.
pub fn zip<T, U>(r1: Result<T>, r2: Result<U>) -> Result<(T, U)> {
    match (r1, r2) {
        (Ok(r1), Ok(r2)) => Ok((r1, r2)),
        (Err(e), _) => Err(e),
        (_, Err(e)) => Err(e),
    }
}

#[doc(hidden)]
pub mod private {
    pub trait Sealed {}

    impl<T, E> Sealed for std::result::Result<T, E> where E: std::error::Error {}
    impl<T> Sealed for Option<T> {}
}

/// Attach more context to an error.
pub trait Context<T, E>: private::Sealed {
    /// Wrap the error value with additional context.
    fn context<C>(self, context: C) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static;

    /// Wrap the error value with additional context evaluated lazily.
    fn with_context<C, F>(self, f: F) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C;
}

impl<T, E> Context<T, E> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context<C>(self, context: C) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
    {
        self.map_err(|e| Error::WrappedContext {
            wrapped: Box::new(e),
            context: context.to_string(),
        })
    }

    fn with_context<C, F>(self, context: F) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(Error::WrappedContext {
                wrapped: Box::new(error),
                context: context().to_string(),
            }
            .bt()),
        }
    }
}

impl<T> Context<T, Infallible> for Option<T> {
    fn context<C>(self, context: C) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
    {
        // Not using ok_or_else to save 2 useless frames off the captured
        // backtrace.
        match self {
            Some(ok) => Ok(ok),
            None => Err(Error::msg(context).bt()),
        }
    }

    fn with_context<C, F>(self, context: F) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        match self {
            Some(v) => Ok(v),
            None => Err(Error::UnwrapNone.context(context()).bt()),
        }
    }
}
