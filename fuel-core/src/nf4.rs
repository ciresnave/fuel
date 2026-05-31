//! bitsandbytes NF4 weight materialization helpers.
//!
//! Companion to [`fuel_graph::Tensor::nf4_matmul`] (registered as
//! [`fuel_graph::registry::FusedOps::NF4_MATMUL`], CPU dispatch
//! shipped in `fuel-cpu-backend::byte_kernels::nf4_matmul_{f32,f16,bf16}`).
//! The fused op takes 3 inputs: activations `[..., M, K]` + a packed
//! weight tensor `[N, K/2]` U8 + a per-block absmax scale tensor
//! `[N, K/block_size]` F32. This module takes a caller-supplied
//! pair of byte buffers (typically loaded via the user's preferred
//! safetensors parser) and constructs the two `LazyTensor` inputs in
//! the layout the fused op expects.
//!
//! ## bitsandbytes on-disk layout primer
//!
//! bnb 0.43+ stores a 4-bit-quantized linear layer as three
//! safetensors entries per `Linear4bit`:
//!
//! 1. **`<prefix>.weight`**: U8 tensor of shape `[N * K / 2]` flat.
//!    Two 4-bit NF4 codes per byte, K-fastest packing. Lower nibble
//!    at byte position `(n * K + k) / 2` (for even k) holds code for
//!    `(n, k)`; upper nibble holds the next k position.
//! 2. **`<prefix>.weight.absmax`**: F32 tensor of shape
//!    `[N * K / block_size]` flat. Per-output-row, per-block
//!    (block_size typically 64) scale.
//! 3. **`<prefix>.weight.quant_state.bitsandbytes__nf4`**: JSON blob
//!    in the safetensors `__metadata__` carrying `blocksize`, the
//!    original `shape: [N, K]`, and the `quant_type: "nf4"` discriminant.
//!
//! Some checkpoints additionally double-quantize the absmax itself
//! (`nested_absmax`, `nested_quant_map`). This module does NOT
//! handle nested quantization — the caller must dequantize the
//! absmax to F32 first. A follow-up that wires nested-quant through
//! a separate registry entry is the path forward when a checkpoint
//! that uses it materializes.
//!
//! ## What this module does NOT do
//!
//! - Parse safetensors files (use [`crate::safetensors`] or your
//!   preferred loader).
//! - Parse bnb `quant_state` JSON metadata (read `blocksize`, `shape`,
//!   etc. from your own loader and pass them as args).
//! - Handle nested-absmax dequantization.
//! - Wire baracuda CUDA dispatch (`nf4_gemv_m{1,2,4,8}_{f16,bf16}`)
//!   — that's a separate session blocked on the parallel dispatch
//!   migration.
//!
//! What it DOES: take the three concrete pieces (packed bytes,
//! absmax scales, geometry) and build the two `LazyTensor` inputs
//! the fused op expects, including the reshape from bnb's flat
//! layouts to Fuel's 2D ones.

use crate::lazy::LazyTensor;
use crate::Device;
use fuel_core_types::{Result, Shape};

/// A bitsandbytes-style NF4-quantized weight, materialized as the two
/// `LazyTensor` inputs that [`fuel_graph::Tensor::nf4_matmul`]
/// expects, plus the cached `(n, k, block_size)` geometry for the
/// matmul builder.
///
/// The two inputs are kept as separate `LazyTensor` fields rather
/// than bundled into a single tensor — that matches the fused op's
/// 3-input signature exactly. [`Self::matmul`] is the convenience
/// builder for the common case where the caller has an `activations:
/// LazyTensor` and just wants the linear-layer output.
pub struct Nf4Weight {
    /// Packed weight bytes, shape `[N, K/2]` U8.
    pub w_packed: LazyTensor,
    /// Per-block absmax scales, shape `[N, K/block_size]` F32.
    pub absmax: LazyTensor,
    /// Output features (rows of the weight matrix).
    pub n: usize,
    /// Input features (the K in `(M, K) @ (K, N) → (M, N)`).
    pub k: usize,
    /// Quantization block size (typically 64 in bitsandbytes).
    pub block_size: usize,
}

/// Build an [`Nf4Weight`] from caller-supplied byte/scale buffers.
///
/// The caller is responsible for loading `w_packed` and `absmax`
/// from their checkpoint source (safetensors, GGUF, raw files, …)
/// and supplying the geometry. This function:
///
/// 1. Validates the buffer sizes against the geometry.
/// 2. Constructs the two `LazyTensor`s on `device` with the 2D
///    shapes the fused op expects (`[n, k/2]` and `[n, k/block_size]`,
///    reshaping bnb's flat layouts).
/// 3. Returns the `Nf4Weight` ready for use.
///
/// Requirements:
/// - `k` must be even (NF4 packs 2 codes per byte along K).
/// - `k` must be a multiple of `block_size`.
/// - `w_packed.len() == n * k / 2`.
/// - `absmax.len() == n * (k / block_size)`.
pub fn nf4_from_bytes(
    w_packed: impl Into<Vec<u8>>,
    absmax: impl Into<Vec<f32>>,
    n: usize,
    k: usize,
    block_size: usize,
    device: &Device,
) -> Result<Nf4Weight> {
    let w_packed: Vec<u8> = w_packed.into();
    let absmax: Vec<f32> = absmax.into();
    if k == 0 || k % 2 != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "nf4_from_bytes: k={k} must be even and non-zero",
        )).bt());
    }
    if block_size == 0 || k % block_size != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "nf4_from_bytes: k={k} must be a positive multiple of block_size={block_size}",
        )).bt());
    }
    let w_expected = n.saturating_mul(k / 2);
    if w_packed.len() != w_expected {
        return Err(fuel_core_types::Error::Msg(format!(
            "nf4_from_bytes: w_packed has {} bytes, expected {w_expected} (n={n} × k/2={})",
            w_packed.len(), k / 2,
        )).bt());
    }
    let abs_expected = n.saturating_mul(k / block_size);
    if absmax.len() != abs_expected {
        return Err(fuel_core_types::Error::Msg(format!(
            "nf4_from_bytes: absmax has {} f32 elements, expected {abs_expected} \
             (n={n} × k/block_size={})",
            absmax.len(), k / block_size,
        )).bt());
    }
    // Build the LazyTensors. const_u8_like / const_f32_like need a
    // host tensor as the "graph anchor"; we anchor on a tiny f32
    // scalar built directly from `device`. Once both are pushed onto
    // the same graph, they're ready to feed nf4_matmul.
    let anchor = LazyTensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), device);
    let w_packed_tensor = anchor.graph_tensor().const_u8_like(
        w_packed,
        Shape::from_dims(&[n, k / 2]),
    );
    let absmax_tensor = anchor.graph_tensor().const_f32_like(
        absmax,
        Shape::from_dims(&[n, k / block_size]),
    );
    Ok(Nf4Weight {
        w_packed: LazyTensor::from_graph_tensor(w_packed_tensor),
        absmax: LazyTensor::from_graph_tensor(absmax_tensor),
        n,
        k,
        block_size,
    })
}

impl Nf4Weight {
    /// Convenience: run `activations @ dequant(weight)` via the
    /// fused [`fuel_graph::Tensor::nf4_matmul`] op. `activations`
    /// must live on the same graph as `self.w_packed` and `self.absmax`;
    /// the caller threads this by building `activations` from
    /// `self.w_packed.graph_tensor()` (or any other tensor on the
    /// same graph).
    ///
    /// Output dtype matches `activations`' dtype (F32 / F16 / BF16
    /// — see [`fuel_graph::Tensor::nf4_matmul`] for the full
    /// contract).
    pub fn matmul(&self, activations: &LazyTensor) -> LazyTensor {
        activations.nf4_matmul(&self.w_packed, &self.absmax, self.block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: build an Nf4Weight from bytes, run matmul, verify
    /// against the hand-computed two-outputs-two-blocks case from the
    /// byte-kernel test.
    #[test]
    fn nf4_weight_matmul_round_trip() {
        let device = Device::cpu();
        // Same numbers as fuel-cpu-backend's nf4_matmul_f32_two_outputs_two_blocks:
        //   activations [1, 2, 2, 4] (m=1, k=4)
        //   w_packed bytes [247, 247, 127, 127] (n=2, k/2=2)
        //   absmax [1.0, 2.0, 10.0, 20.0] (n=2, k/block_size=2)
        //   block_size = 2
        //   expected output [10.0, 50.0]
        let weight = nf4_from_bytes(
            vec![247_u8, 247, 127, 127],
            vec![1.0_f32, 2.0, 10.0, 20.0],
            /* n */ 2, /* k */ 4, /* block_size */ 2,
            &device,
        )
        .expect("nf4_from_bytes");
        let activations = LazyTensor::from_f32(
            vec![1.0_f32, 2.0, 2.0, 4.0],
            Shape::from_dims(&[1, 4]),
            &device,
        );
        // Must be on the same graph as the weight tensors — go
        // through the weight's graph anchor.
        let activations_t = weight.w_packed.graph_tensor().const_f32_like(
            vec![1.0_f32, 2.0, 2.0, 4.0],
            Shape::from_dims(&[1, 4]),
        );
        let _ = activations; // keep the original visible for symmetry in the docs
        let act = LazyTensor::from_graph_tensor(activations_t);
        let y = weight.matmul(&act).realize_f32();
        assert_eq!(y.len(), 2);
        assert!((y[0] - 10.0).abs() < 1e-5, "out 0: {}", y[0]);
        assert!((y[1] - 50.0).abs() < 1e-5, "out 1: {}", y[1]);
    }

    /// Validation: k must be even.
    #[test]
    fn nf4_from_bytes_rejects_odd_k() {
        let r = nf4_from_bytes(
            vec![0_u8; 6],
            vec![0.0_f32; 2],
            /* n */ 2, /* k */ 3, /* block_size */ 1,
            &Device::cpu(),
        );
        assert!(r.is_err());
    }

    /// Validation: k must be a multiple of block_size.
    #[test]
    fn nf4_from_bytes_rejects_block_size_mismatch() {
        let r = nf4_from_bytes(
            vec![0_u8; 4],
            vec![0.0_f32; 2],
            /* n */ 2, /* k */ 4, /* block_size */ 3,
            &Device::cpu(),
        );
        assert!(r.is_err());
    }

    /// Validation: w_packed byte count must match `n × k/2`.
    #[test]
    fn nf4_from_bytes_rejects_wrong_packed_size() {
        let r = nf4_from_bytes(
            vec![0_u8; 5], // wrong: should be 4
            vec![0.0_f32; 2],
            /* n */ 2, /* k */ 4, /* block_size */ 2,
            &Device::cpu(),
        );
        assert!(r.is_err());
    }
}
