// SPDX-License-Identifier: MIT OR Apache-2.0
//! bitsandbytes NF4 weight materialization helpers.
//!
//! Companion to [`fuel_graph::NodeHandle::nf4_matmul`] (registered as
//! [`fuel_graph::registry::FusedOps::NF4_MATMUL`], CPU dispatch
//! shipped in `fuel-cpu-backend::byte_kernels::nf4_matmul_{f32,f16,bf16}`).
//! The fused op takes 3 inputs: activations `[..., M, K]` + a packed
//! weight tensor `[N, K/2]` U8 + a per-block absmax scale tensor
//! `[N, K/block_size]` F32. This module takes a caller-supplied
//! pair of byte buffers (typically loaded via the user's preferred
//! safetensors parser) and constructs the two `Tensor` inputs in
//! the layout the fused op expects.
//!
//! ## bitsandbytes on-disk layout primer
//!
//! bnb 0.43+ stores a 4-bit-quantized linear layer as three
//! safetensors entries per `Linear4bit`:
//!
//! 1. **`<prefix>.weight`**: U8 tensor of shape `[N * K / 2]` flat.
//!    Two 4-bit NF4 codes per byte, K-fastest packing. **The UPPER
//!    nibble of byte `(n * K + k) / 2` (for even k) holds the code for
//!    `(n, k)`; the lower nibble holds the next k position** (bnb packs
//!    `(q[::2] << 4) | q[1::2]`; verified against a real checkpoint, see
//!    `tests/fixtures/bnb_double_quant/PROVENANCE.md`). Fuel's fused NF4
//!    op uses the opposite order (lower nibble = even k), so
//!    [`load_nf4_layer`] swaps the nibbles of every byte at load;
//!    [`nf4_from_bytes`] takes bytes already in Fuel's order.
//! 2. **`<prefix>.weight.absmax`**: F32 tensor of shape
//!    `[N * K / block_size]` flat. Per-output-row, per-block
//!    (block_size typically 64) scale.
//! 3. **`<prefix>.weight.quant_state.bitsandbytes__nf4`**: JSON blob
//!    in the safetensors `__metadata__` carrying `blocksize`, the
//!    original `shape: [N, K]`, and the `quant_type: "nf4"` discriminant.
//!
//! Checkpoints saved with `bnb_4bit_use_double_quant=True` additionally
//! quantize the absmax itself: `<prefix>.weight.absmax` is then **U8**,
//! with the 256-entry code map in `<prefix>.weight.nested_quant_map`
//! (F32), the second-level scales in `<prefix>.weight.nested_absmax`
//! (F32, one per `nested_blocksize` codes), and `nested_blocksize`,
//! `nested_dtype`, `nested_offset` in the JSON blob. [`load_nf4_layer`]
//! decodes it once at load with [`dequantize_nested_absmax`]
//! (`absmax[i] = nested_quant_map[q[i]] * nested_absmax[i / nested_blocksize]`
//! plus `nested_offset`, all in f32, as bitsandbytes' `dequantize_blockwise`
//! does) and then runs the ordinary single-level path.
//!
//! ## What this module does NOT do
//!
//! - Parse safetensors files (use [`fuel_loaders::safetensors`] or your
//!   preferred loader).
//! - Parse bnb `quant_state` JSON metadata (read `blocksize`, `shape`,
//!   etc. from your own loader and pass them as args).
//! - Wire baracuda CUDA dispatch (`nf4_gemv_m{1,2,4,8}_{f16,bf16}`)
//!   — that's a separate session blocked on the parallel dispatch
//!   migration.
//!
//! What it DOES: take the three concrete pieces (packed bytes,
//! absmax scales, geometry) and build the two `Tensor` inputs
//! the fused op expects, including the reshape from bnb's flat
//! layouts to Fuel's 2D ones.

use crate::Device;
use crate::lazy::Tensor;
use fuel_ir::{Result, Shape};

/// A bitsandbytes-style NF4-quantized weight, materialized as the two
/// `Tensor` inputs that [`fuel_graph::NodeHandle::nf4_matmul`]
/// expects, plus the cached `(n, k, block_size)` geometry for the
/// matmul builder.
///
/// The two inputs are kept as separate `Tensor` fields rather
/// than bundled into a single tensor — that matches the fused op's
/// 3-input signature exactly. [`Self::matmul`] is the convenience
/// builder for the common case where the caller has an `activations:
/// Tensor` and just wants the linear-layer output.
pub struct Nf4Weight {
    /// Packed weight bytes, shape `[N, K/2]` U8.
    pub w_packed: Tensor,
    /// Per-block absmax scales, shape `[N, K/block_size]` F32.
    pub absmax: Tensor,
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
/// 2. Constructs the two `Tensor`s on `device` with the 2D
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
    if k == 0 || !k.is_multiple_of(2) {
        return Err(fuel_ir::Error::Msg(format!(
            "nf4_from_bytes: k={k} must be even and non-zero",
        ))
        .bt());
    }
    if block_size == 0 || !k.is_multiple_of(block_size) {
        return Err(fuel_ir::Error::Msg(format!(
            "nf4_from_bytes: k={k} must be a positive multiple of block_size={block_size}",
        ))
        .bt());
    }
    let w_expected = n.saturating_mul(k / 2);
    if w_packed.len() != w_expected {
        return Err(fuel_ir::Error::Msg(format!(
            "nf4_from_bytes: w_packed has {} bytes, expected {w_expected} (n={n} × k/2={})",
            w_packed.len(),
            k / 2,
        ))
        .bt());
    }
    let abs_expected = n.saturating_mul(k / block_size);
    if absmax.len() != abs_expected {
        return Err(fuel_ir::Error::Msg(format!(
            "nf4_from_bytes: absmax has {} f32 elements, expected {abs_expected} \
             (n={n} × k/block_size={})",
            absmax.len(),
            k / block_size,
        ))
        .bt());
    }
    // Build the Tensors. const_u8_like / const_f32_like need a
    // host tensor as the "graph anchor"; we anchor on a tiny f32
    // scalar built directly from `device`. Once both are pushed onto
    // the same graph, they're ready to feed nf4_matmul.
    let anchor = Tensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), device)?;
    let w_packed_tensor = anchor
        .graph_tensor()
        .const_u8_like(w_packed, Shape::from_dims(&[n, k / 2]))?;
    let absmax_tensor = anchor
        .graph_tensor()
        .const_f32_like(absmax, Shape::from_dims(&[n, k / block_size]))?;
    Ok(Nf4Weight {
        w_packed: Tensor::from_graph_tensor(w_packed_tensor),
        absmax: Tensor::from_graph_tensor(absmax_tensor),
        n,
        k,
        block_size,
    })
}

impl Nf4Weight {
    /// Convenience: run `activations @ dequant(weight)` via the
    /// fused [`fuel_graph::NodeHandle::nf4_matmul`] op. `activations`
    /// must live on the same graph as `self.w_packed` and `self.absmax`;
    /// the caller threads this by building `activations` from
    /// `self.w_packed.graph_tensor()` (or any other tensor on the
    /// same graph).
    ///
    /// Output dtype matches `activations`' dtype (F32 / F16 / BF16
    /// — see [`fuel_graph::NodeHandle::nf4_matmul`] for the full
    /// contract).
    pub fn matmul(&self, activations: &Tensor) -> Tensor {
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
    /// Second-level (double-quantization) parameters when the absmax is
    /// itself quantized; `None` for a plain single-level checkpoint.
    pub nested: Option<BnbNestedState>,
}

/// The JSON-carried half of bitsandbytes double quantization. The other
/// half (`nested_absmax`, `nested_quant_map`) is stored as separate
/// safetensors tensors, not in the JSON blob.
#[derive(Debug, Clone, PartialEq)]
pub struct BnbNestedState {
    /// Number of first-level absmax codes sharing one second-level scale
    /// (bnb's default is 256).
    pub block_size: usize,
    /// Mean of the first-level absmax, added back after dequantization.
    pub offset: f32,
}

/// Parse the JSON `quant_state` blob that bitsandbytes 0.43+ stores
/// alongside an NF4-quantized weight. Format documented at
/// <https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py>
/// (search for `quant_state` serialization).
///
/// **Validated fields:**
/// - `quant_type == "nf4"` — refuses other formats (fp4, etc.) with a
///   clear error: FP4 needs a different 16-entry codebook than the fused
///   NF4 op carries (tracked as a follow-up).
/// - `shape` is `[N, K]` two integers.
/// - `blocksize` is a positive usize that divides K.
///
/// **Double quantization:** when `nested_blocksize` is present the
/// returned state carries a [`BnbNestedState`] (`nested_blocksize`,
/// `nested_offset`; `nested_dtype` must be `"float32"`). Supplying only
/// some of the three is an error.
///
/// **Rejected with a clear error:**
/// - `nested_absmax` / `nested_quant_map` inside the JSON. bitsandbytes
///   stores those as tensors, so seeing them here means the blob is not
///   in the format this parser understands.
///
/// **Ignored fields:** `dtype` (original pre-quantization dtype —
/// the loader uses the activations' dtype at matmul time, not the
/// original weight dtype), `device`, and any future bnb additions.
pub fn parse_bnb_quant_state(json_bytes: &[u8]) -> fuel_ir::Result<BnbQuantState> {
    let v: serde_json::Value = serde_json::from_slice(json_bytes).map_err(|e| {
        fuel_ir::Error::Msg(format!("parse_bnb_quant_state: failed to parse JSON: {e}",)).bt()
    })?;
    let obj = v.as_object().ok_or_else(|| {
        fuel_ir::Error::Msg(
            "parse_bnb_quant_state: expected a JSON object at the top level".to_string(),
        )
        .bt()
    })?;
    // Discriminator: must be "nf4". bnb also has "fp4" + variants;
    // we explicitly reject those so the caller knows v1 scope.
    let quant_type = obj
        .get("quant_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            fuel_ir::Error::Msg(
                "parse_bnb_quant_state: missing or non-string 'quant_type' field".to_string(),
            )
            .bt()
        })?;
    if quant_type != "nf4" {
        return Err(fuel_ir::Error::Msg(format!(
            "parse_bnb_quant_state: only quant_type=\"nf4\" is supported, \
             got {quant_type:?}. Other bnb formats (fp4, etc.) need their \
             own loader path.",
        ))
        .bt());
    }
    // bitsandbytes keeps nested_absmax / nested_quant_map as tensors; they
    // never appear in the JSON blob.
    if obj.contains_key("nested_absmax") || obj.contains_key("nested_quant_map") {
        return Err(fuel_ir::Error::Msg(
            "parse_bnb_quant_state: nested_absmax / nested_quant_map are tensors in \
             bitsandbytes checkpoints, not JSON fields; this blob is not in the \
             packed quant_state format"
                .to_string(),
        )
        .bt());
    }
    let nested = parse_nested(obj)?;
    // shape: [N, K]
    let shape = obj.get("shape").and_then(|v| v.as_array()).ok_or_else(|| {
        fuel_ir::Error::Msg("parse_bnb_quant_state: missing or non-array 'shape' field".to_string())
            .bt()
    })?;
    if shape.len() != 2 {
        return Err(fuel_ir::Error::Msg(format!(
            "parse_bnb_quant_state: 'shape' must be 2-element [N, K], got len={}",
            shape.len(),
        ))
        .bt());
    }
    let n = shape[0].as_u64().ok_or_else(|| {
        fuel_ir::Error::Msg(
            "parse_bnb_quant_state: shape[0] (N) must be a non-negative integer".to_string(),
        )
        .bt()
    })? as usize;
    let k = shape[1].as_u64().ok_or_else(|| {
        fuel_ir::Error::Msg(
            "parse_bnb_quant_state: shape[1] (K) must be a non-negative integer".to_string(),
        )
        .bt()
    })? as usize;
    // blocksize
    let block_size = obj
        .get("blocksize")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            fuel_ir::Error::Msg(
                "parse_bnb_quant_state: missing or non-integer 'blocksize' field".to_string(),
            )
            .bt()
        })? as usize;
    if block_size == 0 {
        return Err(fuel_ir::Error::Msg(
            "parse_bnb_quant_state: blocksize must be positive".to_string(),
        )
        .bt());
    }
    if !k.is_multiple_of(block_size) {
        return Err(fuel_ir::Error::Msg(format!(
            "parse_bnb_quant_state: blocksize={block_size} doesn't divide K={k} \
             (this would leave a partial block at the end of each row, which \
             bnb's standard format doesn't produce — your checkpoint may be \
             corrupted or use a non-standard variant)",
        ))
        .bt());
    }
    if !k.is_multiple_of(2) {
        return Err(fuel_ir::Error::Msg(format!(
            "parse_bnb_quant_state: K={k} must be even (NF4 packs 2 codes per byte)",
        ))
        .bt());
    }
    Ok(BnbQuantState {
        n,
        k,
        block_size,
        nested,
    })
}

/// Read the optional `nested_*` JSON fields: all of `nested_blocksize`,
/// `nested_offset` and `nested_dtype` (= `"float32"`), or none.
fn parse_nested(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> fuel_ir::Result<Option<BnbNestedState>> {
    let bs = obj.get("nested_blocksize");
    let off = obj.get("nested_offset");
    let dt = obj.get("nested_dtype");
    if bs.is_none() && off.is_none() && dt.is_none() {
        return Ok(None);
    }
    let err = |m: &str| fuel_ir::Error::Msg(format!("parse_bnb_quant_state: {m}")).bt();
    let (Some(bs), Some(off), Some(dt)) = (bs, off, dt) else {
        return Err(err(
            "double quantization needs all of nested_blocksize, nested_offset and nested_dtype",
        ));
    };
    let block_size =
        bs.as_u64()
            .filter(|b| *b > 0)
            .ok_or_else(|| err("nested_blocksize must be a positive integer"))? as usize;
    let offset = off
        .as_f64()
        .ok_or_else(|| err("nested_offset must be a number"))? as f32;
    if dt.as_str() != Some("float32") {
        return Err(err(&format!(
            "only nested_dtype \"float32\" is supported, got {dt}"
        )));
    }
    Ok(Some(BnbNestedState { block_size, offset }))
}

/// Decode a double-quantized absmax exactly as bitsandbytes does:
/// `absmax[i] = nested_quant_map[q[i]] * nested_absmax[i / nested_block_size] + offset`,
/// every operation in f32. `q` is the U8 `<prefix>.weight.absmax` tensor.
///
/// Errors (never panics) when `nested_quant_map` is not 256 entries, the
/// block size is zero, or `nested_absmax` does not hold exactly one scale
/// per block of `q`.
pub fn dequantize_nested_absmax(
    q: &[u8],
    nested_absmax: &[f32],
    nested_quant_map: &[f32],
    nested_block_size: usize,
    offset: f32,
) -> fuel_ir::Result<Vec<f32>> {
    let err = |m: String| fuel_ir::Error::Msg(format!("dequantize_nested_absmax: {m}")).bt();
    if nested_quant_map.len() != 256 {
        return Err(err(format!(
            "nested_quant_map has {} entries, expected 256",
            nested_quant_map.len()
        )));
    }
    if nested_block_size == 0 {
        return Err(err("nested block size must be positive".to_string()));
    }
    let want = q.len().div_ceil(nested_block_size);
    if nested_absmax.len() != want {
        return Err(err(format!(
            "nested_absmax has {} scales, expected {want} ({} codes / block {nested_block_size})",
            nested_absmax.len(),
            q.len()
        )));
    }
    Ok(q.iter()
        .enumerate()
        .map(|(i, &code)| {
            nested_quant_map[code as usize] * nested_absmax[i / nested_block_size] + offset
        })
        .collect())
}

/// Little-endian f32 values from a raw tensor payload.
fn f32_from_le_bytes(bytes: &[u8], key: &str) -> fuel_ir::Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(fuel_ir::Error::Msg(format!(
            "load_nf4_layer: {key:?} has {} bytes (not a multiple of 4 — corrupt F32 data)",
            bytes.len(),
        ))
        .bt());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// Load a single NF4-quantized layer from a parsed safetensors view
/// (typically obtained via the `safetensors` crate's
/// `SafeTensors::deserialize` or fuel-core's
/// [`fuel_loaders::safetensors::MmapedSafetensors`]).
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
/// layouts that [`Nf4Weight`] / [`fuel_graph::NodeHandle::nf4_matmul`]
/// expect.
///
/// **Double quantization** (`bnb_4bit_use_double_quant=True`): when the JSON
/// carries `nested_*` fields, `{prefix}.weight.absmax` must be U8 and the
/// loader additionally reads `{prefix}.weight.nested_absmax` (F32) and
/// `{prefix}.weight.nested_quant_map` (F32, 256 entries), decoding the
/// absmax with [`dequantize_nested_absmax`]. A U8 absmax without nested
/// parameters, or an F32 absmax with them, is rejected.
///
/// **Nibble order:** bitsandbytes stores the even-k code in the upper
/// nibble; Fuel's fused op reads it from the lower one, so every packed
/// byte is nibble-swapped here before it reaches [`nf4_from_bytes`].
///
/// **Limitations** (see `parse_bnb_quant_state` for the full list):
/// - Non-NF4 bnb formats (fp4, etc.) are rejected: FP4 needs its own
///   16-entry codebook, which the fused NF4 op does not carry.
/// - `{prefix}.weight.quant_map` is not cross-checked against Fuel's NF4
///   table.
pub fn load_nf4_layer(
    st: &safetensors::SafeTensors,
    prefix: &str,
    device: &Device,
) -> fuel_ir::Result<Nf4Weight> {
    let weight_key = format!("{prefix}.weight");
    let absmax_key = format!("{prefix}.weight.absmax");
    let quant_state_key = format!("{prefix}.weight.quant_state.bitsandbytes__nf4");

    let weight_view = st.tensor(&weight_key).map_err(|e| {
        fuel_ir::Error::Msg(format!(
            "load_nf4_layer: tensor {weight_key:?} not found in safetensors: {e}",
        ))
        .bt()
    })?;
    let absmax_view = st.tensor(&absmax_key).map_err(|e| {
        fuel_ir::Error::Msg(format!(
            "load_nf4_layer: tensor {absmax_key:?} not found in safetensors: {e}",
        ))
        .bt()
    })?;
    let quant_state_view = st.tensor(&quant_state_key).map_err(|e| {
        fuel_ir::Error::Msg(format!(
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
        return Err(fuel_ir::Error::Msg(format!(
            "load_nf4_layer: {weight_key:?} dtype={:?}, expected U8",
            weight_view.dtype(),
        ))
        .bt());
    }
    // bnb: even-k code in the UPPER nibble. Fuel's op: LOWER nibble.
    let w_bytes: Vec<u8> = weight_view
        .data()
        .iter()
        .map(|b| b.rotate_left(4))
        .collect();

    // absmax: flat [N*K/block_size]. F32 for a plain checkpoint; U8 codes
    // plus nested tensors for a double-quantized one.
    let absmax_f32: Vec<f32> = match &qs.nested {
        None => {
            if absmax_view.dtype() != safetensors::Dtype::F32 {
                return Err(fuel_ir::Error::Msg(format!(
                    "load_nf4_layer: {absmax_key:?} dtype={:?}, expected F32 (the quant_state \
                     blob has no nested_* fields, so the absmax is not double-quantized)",
                    absmax_view.dtype(),
                ))
                .bt());
            }
            f32_from_le_bytes(absmax_view.data(), &absmax_key)?
        }
        Some(nested) => {
            if absmax_view.dtype() != safetensors::Dtype::U8 {
                return Err(fuel_ir::Error::Msg(format!(
                    "load_nf4_layer: {absmax_key:?} dtype={:?}, expected U8 (the quant_state \
                     blob is double-quantized)",
                    absmax_view.dtype(),
                ))
                .bt());
            }
            let read_f32 = |suffix: &str| -> fuel_ir::Result<Vec<f32>> {
                let key = format!("{prefix}.weight.{suffix}");
                let view = st.tensor(&key).map_err(|e| {
                    fuel_ir::Error::Msg(format!(
                        "load_nf4_layer: tensor {key:?} not found in safetensors: {e}",
                    ))
                    .bt()
                })?;
                if view.dtype() != safetensors::Dtype::F32 {
                    return Err(fuel_ir::Error::Msg(format!(
                        "load_nf4_layer: {key:?} dtype={:?}, expected F32",
                        view.dtype(),
                    ))
                    .bt());
                }
                f32_from_le_bytes(view.data(), &key)
            };
            let nested_absmax = read_f32("nested_absmax")?;
            let nested_quant_map = read_f32("nested_quant_map")?;
            dequantize_nested_absmax(
                absmax_view.data(),
                &nested_absmax,
                &nested_quant_map,
                nested.block_size,
                nested.offset,
            )?
        }
    };

    // Hand off to the bytes-in-hand helper for layout validation +
    // Tensor construction. The byte/scale counts get checked
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
            /* n */ 2,
            /* k */ 4,
            /* block_size */ 2,
            &device,
        )
        .expect("nf4_from_bytes");
        let activations = Tensor::from_f32(
            vec![1.0_f32, 2.0, 2.0, 4.0],
            Shape::from_dims(&[1, 4]),
            &device,
        )
        .unwrap();
        // Must be on the same graph as the weight tensors — go
        // through the weight's graph anchor.
        let activations_t = weight
            .w_packed
            .graph_tensor()
            .const_f32_like(vec![1.0_f32, 2.0, 2.0, 4.0], Shape::from_dims(&[1, 4]))
            .unwrap();
        let _ = activations; // keep the original visible for symmetry in the docs
        let act = Tensor::from_graph_tensor(activations_t);
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
            /* n */ 2,
            /* k */ 3,
            /* block_size */ 1,
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
            /* n */ 2,
            /* k */ 4,
            /* block_size */ 3,
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
            /* n */ 2,
            /* k */ 4,
            /* block_size */ 2,
            &Device::cpu(),
        );
        assert!(r.is_err());
    }

    /// Quant-state parser: well-formed minimal NF4 metadata.
    #[test]
    fn parse_bnb_quant_state_minimal_nf4() {
        let json = br#"{"quant_type":"nf4","shape":[256,128],"blocksize":64,"dtype":"float16"}"#;
        let qs = parse_bnb_quant_state(json).expect("parse_bnb_quant_state");
        assert_eq!(
            qs,
            BnbQuantState {
                n: 256,
                k: 128,
                block_size: 64,
                nested: None,
            }
        );
    }

    /// Quant-state parser: rejects non-nf4 quant_type with a clear error.
    #[test]
    fn parse_bnb_quant_state_rejects_fp4() {
        let json = br#"{"quant_type":"fp4","shape":[256,128],"blocksize":64}"#;
        let r = parse_bnb_quant_state(json);
        let err = r.unwrap_err().to_string();
        assert!(err.contains("nf4"), "error should mention nf4: {err}");
    }

    /// Quant-state parser: `nested_absmax` / `nested_quant_map` are tensors in
    /// real checkpoints, so finding them in the JSON is a malformed blob.
    #[test]
    fn parse_bnb_quant_state_rejects_nested_absmax() {
        let json =
            br#"{"quant_type":"nf4","shape":[256,128],"blocksize":64,"nested_absmax":[1.0,2.0]}"#;
        let r = parse_bnb_quant_state(json);
        let err = r.unwrap_err().to_string();
        assert!(
            err.contains("nested_absmax"),
            "error should name the offending field: {err}"
        );
    }

    /// Quant-state parser: rejects blocksize that doesn't divide K.
    #[test]
    fn parse_bnb_quant_state_rejects_misaligned_blocksize() {
        let json = br#"{"quant_type":"nf4","shape":[256,130],"blocksize":64}"#;
        let r = parse_bnb_quant_state(json);
        let err = r.unwrap_err().to_string();
        assert!(
            err.contains("doesn't divide K") || err.contains("must be even"),
            "error should flag misalignment: {err}"
        );
    }

    /// Quant-state parser: rejects odd K (NF4 packs 2 codes/byte).
    #[test]
    fn parse_bnb_quant_state_rejects_odd_k() {
        let json = br#"{"quant_type":"nf4","shape":[256,127],"blocksize":1}"#;
        let r = parse_bnb_quant_state(json);
        let err = r.unwrap_err().to_string();
        assert!(
            err.contains("must be even"),
            "error should flag odd K: {err}"
        );
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

        // weight bytes [N * K / 2] = 4 bytes: the hand-computed test's bytes in
        // Fuel's order (247, 247, 127, 127) nibble-swapped into bitsandbytes'
        // order (even-k code in the upper nibble), which the loader converts.
        let weight_bytes = vec![0x7F_u8, 0x7F, 0xF7, 0xF7];

        // absmax bytes: [N * K / block_size] = 4 f32 = 16 bytes.
        let absmax_floats: Vec<f32> = vec![1.0, 2.0, 10.0, 20.0];
        let absmax_bytes: Vec<u8> = absmax_floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        // quant_state: UTF-8 JSON blob.
        let quant_state_json = format!(
            r#"{{"quant_type":"nf4","shape":[{n},{k}],"blocksize":{block_size},"dtype":"float16"}}"#,
        );
        let quant_state_bytes = quant_state_json.as_bytes().to_vec();

        // Build TensorViews. safetensors::TensorView::new takes
        // (dtype, shape, data) — shape is just metadata, data is
        // the raw bytes.
        let weight_view = TensorView::new(safetensors::Dtype::U8, vec![n * k / 2], &weight_bytes)
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
        let serialized =
            safetensors::serialize(&tensors, metadata.clone()).expect("safetensors::serialize");

        // Deserialize + load via the new loader.
        let st =
            safetensors::SafeTensors::deserialize(&serialized).expect("SafeTensors::deserialize");
        let weight = load_nf4_layer(&st, "layer", &Device::cpu()).expect("load_nf4_layer");
        assert_eq!(weight.n, n);
        assert_eq!(weight.k, k);
        assert_eq!(weight.block_size, block_size);

        // Sanity matmul: same expected output as nf4_weight_matmul_round_trip.
        let act = Tensor::from_graph_tensor(
            weight
                .w_packed
                .graph_tensor()
                .const_f32_like(vec![1.0_f32, 2.0, 2.0, 4.0], Shape::from_dims(&[1, 4]))
                .unwrap(),
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
            TensorView::new(
                safetensors::Dtype::U8,
                vec![quant_state_bytes.len()],
                &quant_state_bytes,
            )
            .unwrap(),
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
        assert!(
            err.contains("layer.weight"),
            "error should mention the missing tensor: {err}"
        );
    }

    // -----------------------------------------------------------------
    // Double quantization (bitsandbytes `bnb_4bit_use_double_quant=True`)
    // -----------------------------------------------------------------

    /// First 16 rows of `model.layers.1.self_attn.k_proj` from
    /// `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` (NF4, double-quantized), with
    /// the JSON `shape` patched to `[16, 1024]`; see the fixture's
    /// PROVENANCE.md. Real bitsandbytes output, not synthetic.
    const DQ_FIXTURE: &[u8] = include_bytes!(
        "../tests/fixtures/bnb_double_quant/bnb_double_quant_k_proj_16rows.safetensors"
    );
    /// The same 16 rows of the unquantized `Qwen/Qwen3-0.6B` weight, bf16.
    const DQ_ORIGINAL_BF16: &[u8] =
        include_bytes!("../tests/fixtures/bnb_double_quant/orig_k_proj_16rows_bf16.bin");
    const DQ_PREFIX: &str = "model.layers.1.self_attn.k_proj";
    const DQ_N: usize = 16;
    const DQ_K: usize = 1024;

    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// Relative RMS error of `got[n*K + k]` against the original bf16 rows.
    fn rel_rms_vs_original(got: impl Fn(usize, usize) -> f32) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for n in 0..DQ_N {
            for k in 0..DQ_K {
                let i = n * DQ_K + k;
                let bits =
                    u16::from_le_bytes([DQ_ORIGINAL_BF16[2 * i], DQ_ORIGINAL_BF16[2 * i + 1]]);
                let want = f32::from_bits((bits as u32) << 16) as f64;
                let d = got(n, k) as f64 - want;
                num += d * d;
                den += want * want;
            }
        }
        (num / den).sqrt()
    }

    /// Dequantized weight `W[n][k]` through Fuel's own fused NF4 op: with an
    /// identity activation matrix the matmul output is `W^T`, so
    /// `y[m * N + n] = W[n][m]`. This also pins the nibble order against
    /// real bitsandbytes data.
    fn dequant_through_fuel(w: &Nf4Weight) -> Vec<f32> {
        let mut eye = vec![0.0f32; DQ_K * DQ_K];
        for i in 0..DQ_K {
            eye[i * DQ_K + i] = 1.0;
        }
        let act = Tensor::from_graph_tensor(
            w.w_packed
                .graph_tensor()
                .const_f32_like(eye, Shape::from_dims(&[DQ_K, DQ_K]))
                .unwrap(),
        );
        w.matmul(&act).realize_f32()
    }

    #[test]
    fn double_quant_fixture_json_parses_with_nested_state() {
        let st = safetensors::SafeTensors::deserialize(DQ_FIXTURE).unwrap();
        let blob = st
            .tensor(&format!("{DQ_PREFIX}.weight.quant_state.bitsandbytes__nf4"))
            .unwrap();
        let qs = parse_bnb_quant_state(blob.data()).expect("real bnb blob parses");
        assert_eq!((qs.n, qs.k, qs.block_size), (DQ_N, DQ_K, 64));
        let nested = qs.nested.expect("double-quant blob carries nested state");
        assert_eq!(nested.block_size, 256);
        assert!(
            (nested.offset - 0.072_918_33).abs() < 1e-7,
            "{}",
            nested.offset
        );
    }

    #[test]
    fn dequantize_nested_absmax_hand_computed() {
        // map[q] * nested[i / 2] + offset, in f32.
        let mut map = vec![0.0f32; 256];
        map[0] = -1.0;
        map[1] = 0.5;
        map[255] = 1.0;
        let out =
            dequantize_nested_absmax(&[0, 1, 255, 1, 0], &[2.0, 4.0, 8.0], &map, 2, 0.25).unwrap();
        assert_eq!(out, vec![-1.75, 1.25, 4.25, 2.25, -7.75]);
    }

    #[test]
    fn dequantize_nested_absmax_rejects_bad_geometry() {
        let map = vec![0.0f32; 256];
        assert!(dequantize_nested_absmax(&[0; 4], &[1.0; 2], &map[..255], 2, 0.0).is_err());
        assert!(dequantize_nested_absmax(&[0; 4], &[1.0; 2], &map, 0, 0.0).is_err());
        // 4 codes at block 2 need exactly 2 scales.
        assert!(dequantize_nested_absmax(&[0; 4], &[1.0; 3], &map, 2, 0.0).is_err());
        assert!(dequantize_nested_absmax(&[0; 4], &[1.0; 1], &map, 2, 0.0).is_err());
    }

    #[test]
    fn parse_nested_requires_all_three_fields() {
        let ok = br#"{"quant_type":"nf4","shape":[16,128],"blocksize":64,
            "nested_blocksize":256,"nested_dtype":"float32","nested_offset":0.5}"#;
        let qs = parse_bnb_quant_state(ok).unwrap();
        assert_eq!(
            qs.nested,
            Some(BnbNestedState {
                block_size: 256,
                offset: 0.5
            })
        );
        let missing_offset = br#"{"quant_type":"nf4","shape":[16,128],"blocksize":64,
            "nested_blocksize":256,"nested_dtype":"float32"}"#;
        assert!(parse_bnb_quant_state(missing_offset).is_err());
        let wrong_dtype = br#"{"quant_type":"nf4","shape":[16,128],"blocksize":64,
            "nested_blocksize":256,"nested_dtype":"float16","nested_offset":0.5}"#;
        assert!(parse_bnb_quant_state(wrong_dtype).is_err());
    }

    /// The headline gate. A real double-quantized bitsandbytes layer loads,
    /// and its dequantized weights match the ORIGINAL bf16 weights to NF4's
    /// ordinary 4-bit error. A wrong absmax decode (wrong block index,
    /// missing scale or offset) lands far outside that band; the sabotage
    /// sibling below proves the band can tell.
    #[test]
    fn load_nf4_layer_double_quant_matches_original_weights() {
        let st = safetensors::SafeTensors::deserialize(DQ_FIXTURE).unwrap();
        let w = match load_nf4_layer(&st, DQ_PREFIX, &Device::cpu()) {
            Ok(w) => w,
            Err(e) => panic!("double-quant load failed: {e}"),
        };
        assert_eq!((w.n, w.k, w.block_size), (DQ_N, DQ_K, 64));
        let y = dequant_through_fuel(&w);
        let err = rel_rms_vs_original(|n, k| y[k * DQ_N + n]);
        assert!(err < 0.16, "relative RMS error vs original weights {err}");
    }

    /// Retained sabotage sibling: decode the same real checkpoint but drop
    /// `nested_offset` (and, separately, the second-level scale). Each must
    /// land well outside the band the gate above accepts.
    #[test]
    fn double_quant_error_band_rejects_a_wrong_absmax_decode() {
        let st = safetensors::SafeTensors::deserialize(DQ_FIXTURE).unwrap();
        let get = |suffix: &str| st.tensor(&format!("{DQ_PREFIX}.weight{suffix}")).unwrap();
        let q = get(".absmax").data().to_vec();
        let nested_absmax = f32s(get(".nested_absmax").data());
        let map = f32s(get(".nested_quant_map").data());
        let raw_packed = get("").data().to_vec();
        // Fuel's order, as `load_nf4_layer` produces it.
        let packed: Vec<u8> = raw_packed.iter().map(|b| b.rotate_left(4)).collect();
        let build = |absmax: Vec<f32>| {
            let w = nf4_from_bytes(packed.clone(), absmax, DQ_N, DQ_K, 64, &Device::cpu()).unwrap();
            let y = dequant_through_fuel(&w);
            rel_rms_vs_original(|n, k| y[k * DQ_N + n])
        };
        let good =
            build(dequantize_nested_absmax(&q, &nested_absmax, &map, 256, 0.072_918_33).unwrap());
        // Un-swapped (raw bnb) nibbles are as wrong as a bad scale decode.
        let raw_nibbles = {
            let absmax =
                dequantize_nested_absmax(&q, &nested_absmax, &map, 256, 0.072_918_33).unwrap();
            let w =
                nf4_from_bytes(raw_packed.clone(), absmax, DQ_N, DQ_K, 64, &Device::cpu()).unwrap();
            let y = dequant_through_fuel(&w);
            rel_rms_vs_original(|n, k| y[k * DQ_N + n])
        };
        assert!(
            raw_nibbles > 2.0 * good,
            "raw-nibble {raw_nibbles} vs good {good}"
        );
        let no_offset =
            build(dequantize_nested_absmax(&q, &nested_absmax, &map, 256, 0.0).unwrap());
        let no_scale =
            build(dequantize_nested_absmax(&q, &[1.0], &map, 256, 0.072_918_33).unwrap());
        assert!(good < 0.16, "good decode {good}");
        assert!(
            no_offset > 2.0 * good,
            "offset-less {no_offset} vs good {good}"
        );
        assert!(
            no_scale > 2.0 * good,
            "scale-less {no_scale} vs good {good}"
        );
    }

    /// A U8 absmax without nested parameters is a malformed checkpoint, not
    /// something to reinterpret.
    #[test]
    fn load_nf4_layer_rejects_u8_absmax_without_nested_state() {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;
        let w = vec![0u8; 4];
        let am = vec![0u8; 2];
        let qs = br#"{"quant_type":"nf4","shape":[2,4],"blocksize":2}"#.to_vec();
        let mut t: HashMap<String, TensorView<'_>> = HashMap::new();
        t.insert(
            "l.weight".into(),
            TensorView::new(safetensors::Dtype::U8, vec![4], &w).unwrap(),
        );
        t.insert(
            "l.weight.absmax".into(),
            TensorView::new(safetensors::Dtype::U8, vec![2], &am).unwrap(),
        );
        t.insert(
            "l.weight.quant_state.bitsandbytes__nf4".into(),
            TensorView::new(safetensors::Dtype::U8, vec![qs.len()], &qs).unwrap(),
        );
        let ser = safetensors::serialize(&t, None::<HashMap<String, String>>).unwrap();
        let st = safetensors::SafeTensors::deserialize(&ser).unwrap();
        assert!(load_nf4_layer(&st, "l", &Device::cpu()).is_err());
    }
}
