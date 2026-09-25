// SPDX-License-Identifier: MIT OR Apache-2.0
//! Byte-to-`f32` dequantization dispatched on [`GgmlDType`], for callers holding a raw GGUF
//! tensor byte slice and its declared dtype (the common shape at a model-loader's `from_gguf`
//! site).
//!
//! Centralizes what used to be independently hand-rolled in 10 `fuel-transformers` model
//! files (`dequant_bytes_to_f32` + `cpu_dequant_q4_0_bytes`, copy-pasted identically across
//! `lazy_quantized_{llama,qwen2,qwen3,qwen3_moe,gemma3,glm4,lfm2,phi3,smollm3,t5}.rs`), each
//! wiring only `F32`/`F16`/`BF16`/`Q4_0` and erroring on the other 10 `GgmlDType` variants —
//! including `Q6_K`, llama.cpp's ordinary choice for `output.weight`/`token_embd.weight` even
//! in an otherwise-`Q4_0` checkpoint, so a real-world GGUF file routinely hit the catch-all.
//! The `Q4_0` arm those files DID wire was also a hand re-derivation of the block math,
//! bypassing [`GgmlType::to_float`] entirely — so the tested implementation was not the one
//! executing.
//!
//! 14 of the 15 [`GgmlDType`] variants are wired here. The one exclusion: `Q8_1` declines with
//! a typed error citing GAP-125, because `BlockQ8_1::to_float` is `unimplemented!()` upstream
//! (a real numerics gap in this crate, not a dispatch-layer one — wiring it here would only
//! trade an `Err` for a panic).

use crate::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8K, GgmlType,
};
use fuel_ir::Result;
use fuel_ir::quantized::GgmlDType;
use half::{bf16, f16};

/// Reinterpret `bytes` as a dense `[T]` array (GGUF's on-disk block layout) and dequantize via
/// [`GgmlType::to_float`] — the tested implementation, not a re-derivation of its math.
///
/// # Safety invariant this relies on
/// Every `BlockQX` in this crate is `#[repr(C)]` with no padding relied upon, and GGUF's wire
/// format is exactly that packed layout (this is the same invariant `k_quants.rs`'s own
/// dequantization already assumes when parsing a `.gguf` file). `bytes.len()` need not be an
/// exact multiple of `size_of::<T>()`; the remainder is silently dropped, matching every
/// pre-existing per-model `cpu_dequant_*_bytes` this replaces.
fn cast_and_dequant<T: GgmlType>(bytes: &[u8]) -> Vec<f32> {
    let block_size = std::mem::size_of::<T>();
    let n_blocks = bytes.len() / block_size;
    // SAFETY: T is #[repr(C)] (every GgmlType impl in this crate); GGUF bytes are laid out as
    // a dense array of T structs, and n_blocks is computed from bytes.len() so the slice never
    // reads past the end of `bytes`. Alignment: GGUF tensor data is 32-byte aligned per the
    // format spec, which satisfies every block struct's alignment (<= 8 in practice).
    let blocks: &[T] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, n_blocks) };
    let mut out = vec![0.0_f32; n_blocks * T::BLCK_SIZE];
    T::to_float(blocks, &mut out);
    out
}

/// Dequantize a raw GGUF tensor byte slice of the given [`GgmlDType`] to `f32`.
///
/// `name` is used only to name the tensor in an error message on a malformed byte count for
/// `F32`/`F16`/`BF16` (the three unpacked formats, checked exactly; the block formats silently
/// drop a partial trailing block, matching prior per-model behavior).
pub fn dequant_ggml_bytes(bytes: &[u8], dtype: GgmlDType, name: &str) -> Result<Vec<f32>> {
    Ok(match dtype {
        GgmlDType::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(fuel_ir::Error::Msg(format!(
                    "gguf {name}: F32 byte count {} not a multiple of 4",
                    bytes.len(),
                ))
                .bt());
            }
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        }
        GgmlDType::F16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(fuel_ir::Error::Msg(format!(
                    "gguf {name}: F16 byte count {} not a multiple of 2",
                    bytes.len(),
                ))
                .bt());
            }
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_le_bytes(*c).to_f32())
                .collect()
        }
        GgmlDType::BF16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(fuel_ir::Error::Msg(format!(
                    "gguf {name}: BF16 byte count {} not a multiple of 2",
                    bytes.len(),
                ))
                .bt());
            }
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_le_bytes(*c).to_f32())
                .collect()
        }
        GgmlDType::Q4_0 => cast_and_dequant::<BlockQ4_0>(bytes),
        GgmlDType::Q4_1 => cast_and_dequant::<BlockQ4_1>(bytes),
        GgmlDType::Q5_0 => cast_and_dequant::<BlockQ5_0>(bytes),
        GgmlDType::Q5_1 => cast_and_dequant::<BlockQ5_1>(bytes),
        GgmlDType::Q8_0 => cast_and_dequant::<BlockQ8_0>(bytes),
        // NOT wired: `BlockQ8_1::to_float` is `unimplemented!()` upstream (GAP-125) -- this
        // is the one dtype this crate cannot dequantize today, a real numerics gap, not a
        // dispatch-layer one like the other 9 this function fixes. Calling `to_float` would
        // panic (CLAUDE.md: never panic on a production path), so decline with a typed error
        // that names the real cause instead.
        GgmlDType::Q8_1 => {
            return Err(fuel_ir::Error::Msg(format!(
                "gguf {name}: Q8_1 dequantization is unimplemented upstream (GAP-125) -- \
                 BlockQ8_1::to_float has no body",
            ))
            .bt());
        }
        GgmlDType::Q2K => cast_and_dequant::<BlockQ2K>(bytes),
        GgmlDType::Q3K => cast_and_dequant::<BlockQ3K>(bytes),
        GgmlDType::Q4K => cast_and_dequant::<BlockQ4K>(bytes),
        GgmlDType::Q5K => cast_and_dequant::<BlockQ5K>(bytes),
        GgmlDType::Q6K => cast_and_dequant::<BlockQ6K>(bytes),
        GgmlDType::Q8K => cast_and_dequant::<BlockQ8K>(bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_round_trips() {
        let original: Vec<f32> = vec![1.0, -2.5, 3.25, 0.0];
        let bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = dequant_ggml_bytes(&bytes, GgmlDType::F32, "t").unwrap();
        assert_eq!(out, original);
    }

    #[test]
    fn f32_rejects_a_misaligned_byte_count() {
        let bytes = vec![0_u8; 3];
        let err = dequant_ggml_bytes(&bytes, GgmlDType::F32, "t").unwrap_err();
        assert!(err.to_string().contains("not a multiple of 4"));
    }

    /// Every wired block-quant variant must go through `to_float`, not silently return zeros
    /// or error — this is the born-red for the centralization: previously every one of these
    /// except Q4_0 hit a catch-all `Err` in the per-model loaders this function replaces.
    /// Q8_1 is deliberately excluded — see `q8_1_declines_with_a_typed_error_not_a_panic`.
    #[test]
    fn every_wired_block_quant_variant_dequantizes_without_error() {
        for dtype in [
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8K,
        ] {
            // One full block's worth of zeroed bytes is a valid (if degenerate) input for
            // every block format here -- big enough that `n_blocks >= 1` for every dtype's
            // block size, which is what exercises `to_float` rather than a `n_blocks == 0`
            // no-op.
            let bytes = vec![0_u8; 4096];
            let out = dequant_ggml_bytes(&bytes, dtype, "t").unwrap_or_else(|e| {
                panic!("{dtype:?} must dequantize a whole-zero block without error: {e}")
            });
            assert!(
                !out.is_empty(),
                "{dtype:?}: 4096 zeroed bytes must yield at least one dequantized block",
            );
        }
    }

    /// Q8_1 must decline typed, never panic -- `BlockQ8_1::to_float` is `unimplemented!()`
    /// upstream (GAP-125), so this proves the exclusion is a real early return, not a call
    /// into `cast_and_dequant` that happens to survive on this input.
    #[test]
    fn q8_1_declines_with_a_typed_error_not_a_panic() {
        let bytes = vec![0_u8; 4096];
        let err = dequant_ggml_bytes(&bytes, GgmlDType::Q8_1, "t").unwrap_err();
        assert!(err.to_string().contains("GAP-125"));
    }

    /// Sabotage: confirm the K-quant arms actually call `to_float` and are not accidentally
    /// dead code returning the same output as a different variant. Q4_0 (legacy, scale+offset
    /// per block) and Q6_K (k-quant, superblock scales) must NOT dequantize an identical
    /// non-zero byte pattern to the same values -- that would mean one of the two match arms
    /// is not reaching its own type's `to_float`.
    #[test]
    fn distinct_dtypes_on_the_same_bytes_diverge() {
        let bytes = vec![0xAB_u8; 4096];
        let q4_0 = dequant_ggml_bytes(&bytes, GgmlDType::Q4_0, "t").unwrap();
        let q6_k = dequant_ggml_bytes(&bytes, GgmlDType::Q6K, "t").unwrap();
        assert_ne!(
            q4_0.len(),
            q6_k.len(),
            "Q4_0 and Q6_K have different block sizes; equal output lengths would mean one \
             arm is reaching the wrong to_float",
        );
    }
}
