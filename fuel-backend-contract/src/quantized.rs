//! Backend-agnostic quantized kernel registration traits.
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
//!
//! The `GgmlDType` block-format *data* tag lives in [`fuel_ir::quantized`].

use crate::dyn_backend::{DynBackendDevice, DynBackendStorage};
use fuel_ir::{Error, GgmlDType, Layout, Result, Shape};
use std::any::Any;
use std::borrow::Cow;

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
        Err(Error::Msg("dequantize_f16 not supported on this backend".into()).bt())
    }

    /// Raw bytes of the quantized data (host-readable copy).
    fn data(&self) -> Result<Cow<'_, [u8]>>;

    /// Device pointer for callers that need raw addressing (CUDA only today).
    fn device_ptr(&self) -> Result<*const u8> {
        Err(Error::Msg("device_ptr not supported on this backend".into()).bt())
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
        Err(Error::Msg(
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
