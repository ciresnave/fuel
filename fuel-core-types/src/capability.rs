//! Backend capabilities.
//!
//! Each concrete backend (CPU, Vulkan, CUDA, Metal, ...) declares
//! which operations it implements natively via a slice of
//! [`Capability`] values. The multi-backend router / scheduler reads
//! these slices to decide:
//!
//! - whether a graph can run on a given device without CPU fallback,
//! - which backend to route an op to when several are attached,
//! - what the per-op cost is (later Phase 4: `None` = ∞).
//!
//! Variants are intentionally flat — separate capability tokens per
//! quantization format (`MatMulQ4_0`, `MatMulQ4KM`, ...) rather than
//! one parameterized `MatMul(QuantType)` variant, because the
//! underlying kernels are already specialized per quant format
//! (smaller inlined dequant, better register usage). Flat variants
//! also keep this enum free of dependencies on the graph IR crate.
//!
//! Adding a new kernel means (1) adding a variant here, (2) returning
//! it from the backend's `capabilities()` slice, (3) implementing the
//! corresponding `GraphBackend` method. Drift between declared and
//! implemented capabilities is caught by the
//! `backend_capabilities_match_implementation` test in each backend
//! crate.

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum Capability {
    // -- memory --
    /// Allocate a zero-initialized tensor on the device.
    Alloc,
    /// Upload a host buffer to the device.
    Upload,
    /// Download a device storage back to a host buffer.
    Download,
    /// Clone a contiguous (or strided) region described by a Layout.
    TryClone,
    /// Copy a strided region between two device-resident storages.
    CopyStridedSrc,
    /// Cross-device copy (Router-only; concrete backends bail).
    /// Source stays resident on its original device.
    CopyTo,

    // -- compute: elementwise / linalg --
    MatMul,
    Unary,
    Binary,
    Affine,
    Powf,
    Cast,
    Reduce,
    SoftmaxLastDim,
    IndexSelect,
    Gather,

    // -- compute: fused --
    RmsNormLastDim,
    ConcatAlongDim,
    Rope,
    AddAssignScaled,

    // -- backward (training-time) --
    RmsNormLastDimBackward,
    LayerNormLastDimBackward,
    SoftmaxLastDimBackward,

    // -- quantized matmul (one per quant format) --
    MatMulQ4_0,
    MatMulQ4KM,
    MatMulQ8_0,

    // -- quantization helpers (one per quant format) --
    QuantizeQ8_0,
    DequantizeQ8_0,
    DequantizeQ4_0,
    DequantizeQ4KM,
}
