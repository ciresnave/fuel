//! Backend-agnostic quantized kernel registration trait.
//!
//! Per-backend quantized fast paths (CUDA, Metal, future Vulkan/CPU-SIMD)
//! implement [`DynQuantizedStorage`] for their concrete storage type and
//! [`QuantizedDeviceKernels`] for their device handle. fuel-core dispatches
//! through these traits without naming concrete backend types.
//!
//! The CPU "fast path" is the bare ggml block math (avx/neon/simd128
//! vec_dots inside `k_quants.rs`); it stays in fuel-core because
//! `BlockQX` types and the file-format readers (gguf/ggml/imatrix) are a
//! single cohesive numerics unit. The trait still covers the CPU case via
//! a fuel-core-side adapter (see `fuel-core/src/quantized/mod.rs`).

use crate::dyn_backend::{DynBackendDevice, DynBackendStorage};
use crate::error::Result;
use crate::layout::Layout;
use crate::shape::Shape;
use std::any::Any;
use std::borrow::Cow;

/// The ggml block-format dtype tag. Mirrors llama.cpp's `ggml_type` for
/// the subset fuel supports; lives here (rather than in `quantized/mod.rs`)
/// because per-backend kernel crates need to name it without depending on
/// fuel-core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlDType {
    pub fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            _ => return Err(crate::Error::Msg(format!("unknown dtype for tensor {u}")).bt()),
        };
        Ok(dtype)
    }

    pub fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::BF16 => 30,
        }
    }

    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            // ggml block sizes (must match k_quants::BlockQX struct sizes)
            Self::Q4_0 => 18,  // 2 + 16
            Self::Q4_1 => 20,  // 4 + 16
            Self::Q5_0 => 22,  // 2 + 4 + 16
            Self::Q5_1 => 24,  // 4 + 4 + 16
            Self::Q8_0 => 34,  // 2 + 32
            Self::Q8_1 => 36,  // 4 + 32
            Self::Q2K => 84,   // QK_K/16 + QK_K/4 + 2 + 2
            Self::Q3K => 110,  // QK_K/8 + QK_K/4 + 12 + 2
            Self::Q4K => 144,  // 2 + 2 + 12 + QK_K/2
            Self::Q5K => 176,  // 2 + 2 + 12 + QK_K/8 + QK_K/2
            Self::Q6K => 210,  // QK_K/2 + QK_K/4 + QK_K/16 + 2
            Self::Q8K => 292,  // 4 + QK_K + QK_K/16 * 2
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }
}

/// Object-safe per-backend quantized storage. Each backend (CPU, CUDA,
/// Metal, ...) supplies a concrete type that implements this trait;
/// fuel-core holds them as `Box<dyn DynQuantizedStorage>` and dispatches
/// without naming backends.
///
/// The `_src` arguments to `quantize*` are typed as `&dyn DynBackendStorage`
/// so the implementor can downcast to its own concrete storage; the
/// `_onto` variants take a `HostBuffer`-shaped CPU source.
pub trait DynQuantizedStorage: Send + Sync + std::fmt::Debug {
    fn dtype(&self) -> GgmlDType;
    fn block_size(&self) -> usize;
    fn storage_size_in_bytes(&self) -> usize;

    /// Quantize an in-device source storage onto self.
    fn quantize(&mut self, src: &dyn DynBackendStorage) -> Result<()>;

    /// Quantize with importance-matrix weighting.
    fn quantize_imatrix(
        &mut self,
        src: &dyn DynBackendStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()>;

    /// Quantize a CPU source onto self (cross-device).
    fn quantize_onto(&mut self, src: &dyn DynBackendStorage) -> Result<()>;

    /// Quantize with importance matrix from a CPU source.
    fn quantize_imatrix_onto(
        &mut self,
        src: &dyn DynBackendStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()>;

    /// Dequantize to backend-native f32 storage.
    fn dequantize(&self, elem_count: usize) -> Result<Box<dyn DynBackendStorage>>;

    /// Dequantize to f16 (CUDA fast path; default impl rejects).
    fn dequantize_f16(&self, _elem_count: usize) -> Result<Box<dyn DynBackendStorage>> {
        Err(crate::Error::Msg("dequantize_f16 not supported on this backend".into()).bt())
    }

    /// Raw bytes of the quantized data (host-readable copy).
    fn data(&self) -> Result<Cow<'_, [u8]>>;

    /// Device pointer for callers that need raw addressing (CUDA only today).
    fn device_ptr(&self) -> Result<*const u8> {
        Err(crate::Error::Msg("device_ptr not supported on this backend".into()).bt())
    }

    /// QMatMul forward against an in-device input storage. Returns a fresh
    /// device storage and its output shape. `self_shape` is the weight shape
    /// (the QTensor's own shape).
    fn fwd(
        &self,
        self_shape: &Shape,
        input: &dyn DynBackendStorage,
        layout: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)>;

    /// Indexed MoE forward (CUDA-only today; default rejects).
    fn indexed_moe_forward(
        &self,
        _self_shape: &Shape,
        _input: &dyn DynBackendStorage,
        _input_layout: &Layout,
        _ids: &dyn DynBackendStorage,
        _ids_layout: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)> {
        Err(crate::Error::Msg(
            "indexed_moe_forward not supported on this backend".into(),
        )
        .bt())
    }

    /// Identity returns the host device tag this storage is on.
    fn as_any(&self) -> &dyn Any;

    /// Owning device handle. Lets fuel-core recover a `Device` from a
    /// `QTensor` without keeping a parallel `device` field on QTensor.
    fn device_arc_dyn(&self) -> std::sync::Arc<dyn DynBackendDevice>;
}

/// Backend device → quantized-kernel constructors. fuel-core looks this up
/// per Device to allocate fresh QStorage of a given dtype, or to load
/// pre-quantized bytes.
pub trait QuantizedDeviceKernels: DynBackendDevice {
    /// Allocate a zero-initialized quantized storage on this device.
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<Box<dyn DynQuantizedStorage>>;

    /// Load pre-quantized block-format bytes onto this device. `dtype`
    /// describes the block format; `data` is interpreted as a flat byte
    /// slice of those blocks.
    fn load_quantized(
        &self,
        dtype: GgmlDType,
        data: Cow<'_, [u8]>,
    ) -> Result<Box<dyn DynQuantizedStorage>>;
}
