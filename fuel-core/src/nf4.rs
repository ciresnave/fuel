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

// =============================================================================
// bitsandbytes quant_state JSON parser + safetensors-view loader
// =============================================================================

/// The geometry + format discriminators extracted from a
/// bitsandbytes `quant_state` JSON blob. Parsed via
/// [`parse_bnb_quant_state`]; consumed by [`load_nf4_layer`] to
/// reconstruct the `[N, K/block_size]` absmax layout from bnb's
/// flat `[N * K / block_size]` storage.
///
/// Only the fields needed to drive [`Nf4Weight`] construction are
/// extracted; the parser ignores unknown fields (forward-compat
/// with future bnb metadata additions).
#[derive(Debug, Clone, PartialEq)]
pub struct BnbQuantState {
    /// Number of output features (rows of the weight matrix).
    pub n: usize,
    /// Number of input features (the K in `(M, K) @ (K, N) → (M, N)`).
    pub k: usize,
    /// First-level (un-nested) quantization block size — typically 64.
    pub block_size: usize,
}

/// Parse the JSON `quant_state` blob that bitsandbytes 0.43+ stores
/// alongside an NF4-quantized weight. Format documented at
/// <https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py>
/// (search for `quant_state` serialization).
///
/// **Validated fields:**
/// - `quant_type == "nf4"` — refuses other formats (fp4 / nf4_double /
///   etc.) with a clear error. v1 only handles standalone NF4.
/// - `shape` is `[N, K]` two integers.
/// - `blocksize` is a positive usize that divides K.
///
/// **Rejected with a clear error:**
/// - Nested-absmax double quantization (`nested_absmax` /
///   `nested_quant_map` keys present). The dequant of the absmax
///   itself isn't implemented in this module — handle it in the
///   caller, dequantize the absmax to F32 first, then pass the
///   resulting F32 absmax tensor to [`nf4_from_bytes`].
///
/// **Ignored fields:** `dtype` (original pre-quantization dtype —
/// the loader uses the activations' dtype at matmul time, not the
/// original weight dtype), `device`, and any future bnb additions.
pub fn parse_bnb_quant_state(json_bytes: &[u8]) -> fuel_core_types::Result<BnbQuantState> {
    let v: serde_json::Value = serde_json::from_slice(json_bytes).map_err(|e| {
        fuel_core_types::Error::Msg(format!(
            "parse_bnb_quant_state: failed to parse JSON: {e}",
        ))
        .bt()
    })?;
    let obj = v.as_object().ok_or_else(|| {
        fuel_core_types::Error::Msg(
            "parse_bnb_quant_state: expected a JSON object at the top level".to_string(),
        )
        .bt()
    })?;
    // Discriminator: must be "nf4". bnb also has "fp4" + variants;
    // we explicitly reject those so the caller knows v1 scope.
    let quant_type = obj.get("quant_type").and_then(|v| v.as_str()).ok_or_else(|| {
        fuel_core_types::Error::Msg(
            "parse_bnb_quant_state: missing or non-string 'quant_type' field".to_string(),
        )
        .bt()
    })?;
    if quant_type != "nf4" {
        return Err(fuel_core_types::Error::Msg(format!(
            "parse_bnb_quant_state: only quant_type=\"nf4\" is supported, \
             got {quant_type:?}. Other bnb formats (fp4, etc.) need their \
             own loader path.",
        ))
        .bt());
    }
    // Reject nested-absmax (double-quantization). Caller must
    // dequantize first.
    if obj.contains_key("nested_absmax") || obj.contains_key("nested_quant_map") {
        return Err(fuel_core_types::Error::Msg(
            "parse_bnb_quant_state: this weight uses double-quantization \
             (nested_absmax / nested_quant_map present). v1 only handles \
             single-level absmax. Workaround: dequantize the absmax to F32 \
             in your loader before calling nf4_from_bytes; pass the resulting \
             F32 absmax directly."
                .to_string(),
        )
        .bt());
    }
    // shape: [N, K]
    let shape = obj.get("shape").and_then(|v| v.as_array()).ok_or_else(|| {
        fuel_core_types::Error::Msg(
            "parse_bnb_quant_state: missing or non-array 'shape' field".to_string(),
        )
        .bt()
    })?;
    if shape.len() != 2 {
        return Err(fuel_core_types::Error::Msg(format!(
            "parse_bnb_quant_state: 'shape' must be 2-element [N, K], got len={}",
            shape.len(),
        ))
        .bt());
    }
    let n = shape[0].as_u64().ok_or_else(|| {
        fuel_core_types::Error::Msg("parse_bnb_quant_state: shape[0] (N) must be a non-negative integer".to_string()).bt()
    })? as usize;
    let k = shape[1].as_u64().ok_or_else(|| {
        fuel_core_types::Error::Msg("parse_bnb_quant_state: shape[1] (K) must be a non-negative integer".to_string()).bt()
    })? as usize;
    // blocksize
    let block_size = obj.get("blocksize").and_then(|v| v.as_u64()).ok_or_else(|| {
        fuel_core_types::Error::Msg(
            "parse_bnb_quant_state: missing or non-integer 'blocksize' field".to_string(),
        )
        .bt()
    })? as usize;
    if block_size == 0 {
        return Err(fuel_core_types::Error::Msg(
            "parse_bnb_quant_state: blocksize must be positive".to_string(),
        )
        .bt());
    }
    if k % block_size != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "parse_bnb_quant_state: blocksize={block_size} doesn't divide K={k} \
             (this would leave a partial block at the end of each row, which \
             bnb's standard format doesn't produce — your checkpoint may be \
             corrupted or use a non-standard variant)",
        ))
        .bt());
    }
    if k % 2 != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "parse_bnb_quant_state: K={k} must be even (NF4 packs 2 codes per byte)",
        ))
        .bt());
    }
    Ok(BnbQuantState { n, k, block_size })
}

/// Load a single NF4-quantized layer from a parsed safetensors view
/// (typically obtained via the `safetensors` crate's
/// `SafeTensors::deserialize` or fuel-core's
/// [`crate::safetensors::MmapedSafetensors`]).
///
/// `prefix` is the layer's name in the safetensors file
/// (e.g. `"model.layers.0.self_attn.q_proj"`). The function expects
/// three tensors at:
/// - `{prefix}.weight` — U8, flat `[N * K / 2]`.
/// - `{prefix}.weight.absmax` — F32, flat `[N * K / block_size]`.
/// - `{prefix}.weight.quant_state.bitsandbytes__nf4` — U8 bytes
///   holding the JSON metadata blob (see [`parse_bnb_quant_state`]).
///
/// All three tensors are read into host memory, validated against
/// the geometry from the JSON blob, and reshaped into the 2D
/// layouts that [`Nf4Weight`] / [`fuel_graph::Tensor::nf4_matmul`]
/// expect.
///
/// **Limitations** (see `parse_bnb_quant_state` for the full list):
/// - Double-quantization (nested_absmax) is rejected.
/// - Non-NF4 bnb formats (fp4, etc.) are rejected.
pub fn load_nf4_layer(
    st: &safetensors::SafeTensors,
    prefix: &str,
    device: &Device,
) -> fuel_core_types::Result<Nf4Weight> {
    let weight_key = format!("{prefix}.weight");
    let absmax_key = format!("{prefix}.weight.absmax");
    let quant_state_key = format!("{prefix}.weight.quant_state.bitsandbytes__nf4");

    let weight_view = st.tensor(&weight_key).map_err(|e| {
        fuel_core_types::Error::Msg(format!(
            "load_nf4_layer: tensor {weight_key:?} not found in safetensors: {e}",
        ))
        .bt()
    })?;
    let absmax_view = st.tensor(&absmax_key).map_err(|e| {
        fuel_core_types::Error::Msg(format!(
            "load_nf4_layer: tensor {absmax_key:?} not found in safetensors: {e}",
        ))
        .bt()
    })?;
    let quant_state_view = st.tensor(&quant_state_key).map_err(|e| {
        fuel_core_types::Error::Msg(format!(
            "load_nf4_layer: tensor {quant_state_key:?} not found in safetensors: {e}",
        ))
        .bt()
    })?;

    // quant_state is a U8 byte stream holding UTF-8 JSON. Parse it
    // first so we have the geometry to validate the other two
    // tensors' shapes against.
    let qs = parse_bnb_quant_state(quant_state_view.data())?;

    // weight: U8, flat [N*K/2].
    if weight_view.dtype() != safetensors::Dtype::U8 {
        return Err(fuel_core_types::Error::Msg(format!(
            "load_nf4_layer: {weight_key:?} dtype={:?}, expected U8",
            weight_view.dtype(),
        ))
        .bt());
    }
    let w_bytes: Vec<u8> = weight_view.data().to_vec();

    // absmax: F32, flat [N*K/block_size]. Read as raw bytes and
    // bytemuck-cast to f32s (the safetensors crate exposes the
    // bytes as &[u8] regardless of dtype).
    if absmax_view.dtype() != safetensors::Dtype::F32 {
        return Err(fuel_core_types::Error::Msg(format!(
            "load_nf4_layer: {absmax_key:?} dtype={:?}, expected F32. \
             If this is a double-quantized absmax stored as U8 with its own \
             quant_state, dequantize it before calling this loader.",
            absmax_view.dtype(),
        ))
        .bt());
    }
    let absmax_bytes = absmax_view.data();
    if absmax_bytes.len() % 4 != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "load_nf4_layer: {absmax_key:?} has {} bytes (not a multiple of 4 — corrupt F32 data)",
            absmax_bytes.len(),
        ))
        .bt());
    }
    let absmax_f32: Vec<f32> = absmax_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Hand off to the bytes-in-hand helper for layout validation +
    // LazyTensor construction. The byte/scale counts get checked
    // against (n, k, block_size) there.
    nf4_from_bytes(w_bytes, absmax_f32, qs.n, qs.k, qs.block_size, device)
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

    /// Quant-state parser: well-formed minimal NF4 metadata.
    #[test]
    fn parse_bnb_quant_state_minimal_nf4() {
        let json = br#"{"quant_type":"nf4","shape":[256,128],"blocksize":64,"dtype":"float16"}"#;
        let qs = parse_bnb_quant_state(json).expect("parse_bnb_quant_state");
        assert_eq!(qs, BnbQuantState { n: 256, k: 128, block_size: 64 });
    }

    /// Quant-state parser: rejects non-nf4 quant_type with a clear error.
    #[test]
    fn parse_bnb_quant_state_rejects_fp4() {
        let json = br#"{"quant_type":"fp4","shape":[256,128],"blocksize":64}"#;
        let r = parse_bnb_quant_state(json);
        let err = r.unwrap_err().to_string();
        assert!(err.contains("nf4"), "error should mention nf4: {err}");
    }

    /// Quant-state parser: rejects double-quantized weights with a
    /// pointer to the workaround.
    #[test]
    fn parse_bnb_quant_state_rejects_nested_absmax() {
        let json = br#"{"quant_type":"nf4","shape":[256,128],"blocksize":64,"nested_absmax":[1.0,2.0]}"#;
        let r = parse_bnb_quant_state(json);
        let err = r.unwrap_err().to_string();
        assert!(err.contains("double-quantization"), "error should mention double-quantization: {err}");
    }

    /// Quant-state parser: rejects blocksize that doesn't divide K.
    #[test]
    fn parse_bnb_quant_state_rejects_misaligned_blocksize() {
        let json = br#"{"quant_type":"nf4","shape":[256,130],"blocksize":64}"#;
        let r = parse_bnb_quant_state(json);
        let err = r.unwrap_err().to_string();
        assert!(err.contains("doesn't divide K") || err.contains("must be even"),
            "error should flag misalignment: {err}");
    }

    /// Quant-state parser: rejects odd K (NF4 packs 2 codes/byte).
    #[test]
    fn parse_bnb_quant_state_rejects_odd_k() {
        let json = br#"{"quant_type":"nf4","shape":[256,127],"blocksize":1}"#;
        let r = parse_bnb_quant_state(json);
        let err = r.unwrap_err().to_string();
        assert!(err.contains("must be even"), "error should flag odd K: {err}");
    }

    /// Quant-state parser: gracefully errors on malformed JSON.
    #[test]
    fn parse_bnb_quant_state_rejects_malformed_json() {
        let r = parse_bnb_quant_state(b"not json at all");
        assert!(r.is_err());
    }

    /// Quant-state parser: rejects missing required fields.
    #[test]
    fn parse_bnb_quant_state_rejects_missing_shape() {
        let json = br#"{"quant_type":"nf4","blocksize":64}"#;
        let r = parse_bnb_quant_state(json);
        let err = r.unwrap_err().to_string();
        assert!(err.contains("shape"), "error should mention shape: {err}");
    }

    /// End-to-end: synthesize a minimal in-memory safetensors file
    /// containing the 3 expected entries (weight U8 + absmax F32 +
    /// quant_state JSON-as-bytes), then load it via load_nf4_layer.
    ///
    /// Uses the same hand-computed numbers as
    /// `nf4_weight_matmul_round_trip` so we know the loaded weight
    /// produces the expected matmul output.
    #[test]
    fn load_nf4_layer_round_trip_synthetic_safetensors() {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;

        let n: usize = 2;
        let k: usize = 4;
        let block_size: usize = 2;

        // weight bytes [N * K / 2] = 4 bytes (same as hand-computed test).
        let weight_bytes = vec![247_u8, 247, 127, 127];

        // absmax bytes: [N * K / block_size] = 4 f32 = 16 bytes.
        let absmax_floats: Vec<f32> = vec![1.0, 2.0, 10.0, 20.0];
        let absmax_bytes: Vec<u8> = absmax_floats
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // quant_state: UTF-8 JSON blob.
        let quant_state_json = format!(
            r#"{{"quant_type":"nf4","shape":[{n},{k}],"blocksize":{block_size},"dtype":"float16"}}"#,
        );
        let quant_state_bytes = quant_state_json.as_bytes().to_vec();

        // Build TensorViews. safetensors::TensorView::new takes
        // (dtype, shape, data) — shape is just metadata, data is
        // the raw bytes.
        let weight_view = TensorView::new(
            safetensors::Dtype::U8,
            vec![n * k / 2],
            &weight_bytes,
        )
        .expect("weight TensorView");
        let absmax_view = TensorView::new(
            safetensors::Dtype::F32,
            vec![n * k / block_size],
            &absmax_bytes,
        )
        .expect("absmax TensorView");
        let quant_state_view = TensorView::new(
            safetensors::Dtype::U8,
            vec![quant_state_bytes.len()],
            &quant_state_bytes,
        )
        .expect("quant_state TensorView");

        // Serialize to in-memory safetensors bytes.
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        tensors.insert("layer.weight".to_string(), weight_view);
        tensors.insert("layer.weight.absmax".to_string(), absmax_view);
        tensors.insert(
            "layer.weight.quant_state.bitsandbytes__nf4".to_string(),
            quant_state_view,
        );
        let metadata: Option<std::collections::HashMap<String, String>> = None;
        let serialized = safetensors::serialize(&tensors, metadata.clone())
            .expect("safetensors::serialize");

        // Deserialize + load via the new loader.
        let st = safetensors::SafeTensors::deserialize(&serialized)
            .expect("SafeTensors::deserialize");
        let weight = load_nf4_layer(&st, "layer", &Device::cpu())
            .expect("load_nf4_layer");
        assert_eq!(weight.n, n);
        assert_eq!(weight.k, k);
        assert_eq!(weight.block_size, block_size);

        // Sanity matmul: same expected output as nf4_weight_matmul_round_trip.
        let act = LazyTensor::from_graph_tensor(
            weight.w_packed.graph_tensor().const_f32_like(
                vec![1.0_f32, 2.0, 2.0, 4.0],
                Shape::from_dims(&[1, 4]),
            ),
        );
        let y = weight.matmul(&act).realize_f32();
        assert!((y[0] - 10.0).abs() < 1e-5, "out 0: {}", y[0]);
        assert!((y[1] - 50.0).abs() < 1e-5, "out 1: {}", y[1]);
    }

    /// load_nf4_layer: missing weight tensor produces a clear error
    /// pointing at the missing key.
    #[test]
    fn load_nf4_layer_missing_weight_errors() {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;
        // Only the quant_state and absmax — no weight.
        let quant_state_bytes = br#"{"quant_type":"nf4","shape":[2,4],"blocksize":2}"#.to_vec();
        let absmax_bytes: Vec<u8> = [1.0_f32, 2.0, 10.0, 20.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        tensors.insert(
            "layer.weight.absmax".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![4], &absmax_bytes).unwrap(),
        );
        tensors.insert(
            "layer.weight.quant_state.bitsandbytes__nf4".to_string(),
            TensorView::new(safetensors::Dtype::U8, vec![quant_state_bytes.len()], &quant_state_bytes).unwrap(),
        );
        let metadata: Option<HashMap<String, String>> = None;
        let serialized = safetensors::serialize(&tensors, metadata.clone()).unwrap();
        let st = safetensors::SafeTensors::deserialize(&serialized).unwrap();
        // `unwrap_err` would require Nf4Weight: Debug; match the Err
        // arm explicitly so we don't pull in that bound.
        let err = match load_nf4_layer(&st, "layer", &Device::cpu()) {
            Ok(_) => panic!("expected an error from load_nf4_layer with missing weight tensor"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("layer.weight"), "error should mention the missing tensor: {err}");
    }
}
