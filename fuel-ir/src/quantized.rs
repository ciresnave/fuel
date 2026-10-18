// SPDX-License-Identifier: MIT OR Apache-2.0
//! The ggml block-format dtype tag (`GgmlDType`).
//!
//! As of B0.3 the backend-agnostic quantized *traits* (`DynQuantizedStorage`,
//! `QuantizedDeviceKernels`) moved to the `fuel-backend-contract` crate. What
//! stays here is the `GgmlDType` **data** tag — per-backend kernel crates and
//! the contract traits both need to name it, so it lives in this bottom
//! vocabulary crate.

use crate::error::Result;

/// The ggml block-format dtype tag. Mirrors llama.cpp's `ggml_type`
/// **@`9d57ce456c`** for the subset fuel supports (the classic 15 — F32/F16/BF16
/// plus Q4_0..Q8_1 and Q2K..Q8K, plus every IQ family: IQ4_NL/IQ4_XS (GAP-211 B1)
/// and the decode-only IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S (GAP-211 B2);
/// block geometry and the dequant formula verified against `ggml-common.h` and
/// `ggml-quants.c` at that commit), plus the decode-only TQ1_0/TQ2_0/MXFP4/NVFP4 (GAP-211 C);
/// NOT the other ggml types, which an unsupported-tensor GGUF load reports by name instead).
///
/// Geometry checked against that commit (GAP-248 instance 5), and the referee
/// differs by column:
/// - `type_size` (bytes/block) is **ADJUDICATED** — upstream's own `static_assert`
///   on each block struct is the referee; 12/12 for the quant types here.
/// - `block_size` (elems/block) and the numeric codes are **CORROBORATED** — both
///   are transcribed upstream (`type_traits[].blck_size`, `enum ggml_type`) and
///   agree 12/12 with MLMF's INDEPENDENT transcription; no referee exists.
/// - per-block **alignment** is UNANCHORED by both projects; fuel tracks none, so
///   an upstream alignment change is undetectable here.
///
/// NO DETECTOR: this pin makes drift ANSWERABLE, not detected — it ages silently.
/// Re-run the comparison on any llama.cpp bump that touches `ggml_type` or
/// `type_traits[]`.
///
/// Lives here (rather than in `quantized/mod.rs`) because per-backend kernel
/// crates need to name it without depending on fuel-core.
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
    /// llama.cpp `GGML_TYPE_IQ4_NL` (20): 32 elems/block, non-linear 4-bit codebook.
    Iq4Nl,
    /// llama.cpp `GGML_TYPE_IQ4_XS` (23): 256 elems/block, 4-bit codebook, 6-bit sub-block scales.
    Iq4Xs,
    /// llama.cpp `GGML_TYPE_IQ2_XXS` (16): 256 elems/block, grid-table quantization; DECODE-ONLY (GAP-211 B2).
    Iq2Xxs,
    /// llama.cpp `GGML_TYPE_IQ2_XS` (17): 256 elems/block, grid-table quantization; DECODE-ONLY (GAP-211 B2).
    Iq2Xs,
    /// llama.cpp `GGML_TYPE_IQ3_XXS` (18): 256 elems/block, grid-table quantization; DECODE-ONLY (GAP-211 B2).
    Iq3Xxs,
    /// llama.cpp `GGML_TYPE_IQ1_S` (19): 256 elems/block, grid-table quantization; DECODE-ONLY (GAP-211 B2).
    Iq1S,
    /// llama.cpp `GGML_TYPE_IQ3_S` (21): 256 elems/block, grid-table quantization; DECODE-ONLY (GAP-211 B2).
    Iq3S,
    /// llama.cpp `GGML_TYPE_IQ2_S` (22): 256 elems/block, grid-table quantization; DECODE-ONLY (GAP-211 B2).
    Iq2S,
    /// llama.cpp `GGML_TYPE_IQ1_M` (29): 256 elems/block, grid-table quantization; DECODE-ONLY (GAP-211 B2).
    Iq1M,
    /// llama.cpp `GGML_TYPE_TQ1_0` (34): 256 elems/block, ternary (5 trits/byte); DECODE-ONLY (GAP-211 C).
    Tq1_0,
    /// llama.cpp `GGML_TYPE_TQ2_0` (35): 256 elems/block, ternary (2 bits/elem); DECODE-ONLY (GAP-211 C).
    Tq2_0,
    /// llama.cpp `GGML_TYPE_MXFP4` (39): 32 elems/block, E2M1 with an E8M0 block scale; DECODE-ONLY (GAP-211 C).
    Mxfp4,
    /// llama.cpp `GGML_TYPE_NVFP4` (40): 64 elems/block, E2M1 with four UE4M3 sub-block scales; DECODE-ONLY (GAP-211 C).
    Nvfp4,
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
            20 => Self::Iq4Nl,
            23 => Self::Iq4Xs,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            29 => Self::Iq1M,
            34 => Self::Tq1_0,
            35 => Self::Tq2_0,
            39 => Self::Mxfp4,
            40 => Self::Nvfp4,
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
            Self::Iq4Nl => 20,
            Self::Iq4Xs => 23,
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq1S => 19,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq1M => 29,
            Self::Tq1_0 => 34,
            Self::Tq2_0 => 35,
            Self::Mxfp4 => 39,
            Self::Nvfp4 => 40,
            Self::BF16 => 30,
        }
    }

    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            // ggml block sizes (must match k_quants::BlockQX struct sizes)
            Self::Q4_0 => 18,   // 2 + 16
            Self::Q4_1 => 20,   // 4 + 16
            Self::Q5_0 => 22,   // 2 + 4 + 16
            Self::Q5_1 => 24,   // 4 + 4 + 16
            Self::Q8_0 => 34,   // 2 + 32
            Self::Q8_1 => 36,   // 4 + 32
            Self::Q2K => 84,    // QK_K/16 + QK_K/4 + 2 + 2
            Self::Q3K => 110,   // QK_K/8 + QK_K/4 + 12 + 2
            Self::Q4K => 144,   // 2 + 2 + 12 + QK_K/2
            Self::Q5K => 176,   // 2 + 2 + 12 + QK_K/8 + QK_K/2
            Self::Q6K => 210,   // QK_K/2 + QK_K/4 + QK_K/16 + 2
            Self::Q8K => 292,   // 4 + QK_K + QK_K/16 * 2
            Self::Iq4Nl => 18,  // 2 + QK4_NL/2
            Self::Iq4Xs => 136, // 2 + 2 + QK_K/64 + QK_K/2
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq3Xxs => 98,
            Self::Iq1S => 50,
            Self::Iq3S => 110,
            Self::Iq2S => 82,
            Self::Iq1M => 56,
            Self::Tq1_0 => 54, // QK_K/64 + (QK_K - 4*QK_K/64)/5 + 2
            Self::Tq2_0 => 66, // QK_K/4 + 2
            Self::Mxfp4 => 17, // 1 + QK_MXFP4/2
            Self::Nvfp4 => 36, // QK_NVFP4/16 + QK_NVFP4/2
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::Iq4Nl => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::Iq4Xs
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3Xxs
            | Self::Iq1S
            | Self::Iq3S
            | Self::Iq2S
            | Self::Iq1M
            | Self::Tq1_0
            | Self::Tq2_0 => 256,
            Self::Mxfp4 => 32,
            Self::Nvfp4 => 64,
        }
    }

    /// `false` for the grid-table IQ formats (GAP-211 B2) and the TQ/MXFP4/NVFP4
    /// formats (GAP-211 C): Fuel decodes them but has no quantizer for them.
    pub fn supports_quantize(self) -> bool {
        !matches!(
            self,
            Self::Iq2Xxs
                | Self::Iq2Xs
                | Self::Iq2S
                | Self::Iq3Xxs
                | Self::Iq3S
                | Self::Iq1S
                | Self::Iq1M
                | Self::Tq1_0
                | Self::Tq2_0
                | Self::Mxfp4
                | Self::Nvfp4
        )
    }

    /// Declines quantizing to a decode-only format with a typed error that names it,
    /// instead of reaching `GgmlType::from_float`, which has nothing to run.
    pub fn check_quantize(self) -> Result<()> {
        if self.supports_quantize() {
            Ok(())
        } else {
            Err(crate::Error::Msg(format!(
                "cannot quantize to {self:?}: Fuel decodes this format but has no quantizer for it (GAP-211 B2/C)"
            ))
            .bt())
        }
    }

    /// GAP-333: `numel` dequantized elements are read from `numel / block_size`
    /// whole blocks of `type_size` bytes. Declines a count that is not a whole
    /// number of blocks, or whose blocks do not fit in `byte_len`.
    ///
    /// The one copy every backend's dequantize calls before it reads, so the
    /// predicate cannot drift between them.
    pub fn check_dequant_count(self, op_label: &str, numel: usize, byte_len: usize) -> Result<()> {
        let block = self.block_size();
        if !numel.is_multiple_of(block) {
            return Err(crate::Error::Msg(format!(
                "{op_label}: element count {numel} is not a multiple of {self:?}'s block size {block}",
            ))
            .bt());
        }
        match (numel / block).checked_mul(self.type_size()) {
            Some(need) if need <= byte_len => Ok(()),
            Some(need) => Err(crate::Error::Msg(format!(
                "{op_label}: {numel} elements need {need} bytes of {self:?} blocks, but the \
                 buffer holds {byte_len}",
            ))
            .bt()),
            None => Err(crate::Error::Msg(format!(
                "{op_label}: {numel} {self:?} elements overflow a byte count"
            ))
            .bt()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GgmlDType;

    #[track_caller]
    fn declined(dtype: GgmlDType, numel: usize, byte_len: usize, needle: &str) {
        let msg = dtype
            .check_dequant_count("t", numel, byte_len)
            .expect_err("expected a decline")
            .to_string();
        assert!(msg.contains(needle), "wrong decline: {msg}");
    }

    #[test]
    fn whole_blocks_that_fit_are_accepted() {
        // 2 Q4_0 blocks = 64 elements = 36 bytes; 1 Q4_K block = 256 = 144 bytes.
        GgmlDType::Q4_0.check_dequant_count("t", 64, 36).unwrap();
        GgmlDType::Q4K.check_dequant_count("t", 256, 144).unwrap();
        GgmlDType::F32.check_dequant_count("t", 3, 12).unwrap();
        GgmlDType::Q4_0.check_dequant_count("t", 0, 0).unwrap();
    }

    #[test]
    fn a_partial_block_is_declined() {
        declined(
            GgmlDType::Q4_0,
            65,
            1_000,
            "element count 65 is not a multiple of Q4_0's block size 32",
        );
        declined(
            GgmlDType::Q4K,
            128,
            1_000,
            "element count 128 is not a multiple of Q4K's block size 256",
        );
    }

    #[test]
    fn blocks_one_byte_past_the_buffer_are_declined() {
        declined(
            GgmlDType::Q4_0,
            64,
            35,
            "64 elements need 36 bytes of Q4_0 blocks, but the buffer holds 35",
        );
        declined(
            GgmlDType::F16,
            3,
            5,
            "3 elements need 6 bytes of F16 blocks",
        );
        GgmlDType::F16.check_dequant_count("t", 3, 6).unwrap();
    }

    #[test]
    fn an_overflowing_count_is_a_typed_error() {
        declined(
            GgmlDType::F32,
            usize::MAX,
            usize::MAX,
            "overflow a byte count",
        );
    }
}
