//! The ggml block-format dtype tag (`GgmlDType`).
//!
//! As of B0.3 the backend-agnostic quantized *traits* (`DynQuantizedStorage`,
//! `QuantizedDeviceKernels`) moved to the `fuel-backend-contract` crate. What
//! stays here is the `GgmlDType` **data** tag — per-backend kernel crates and
//! the contract traits both need to name it, so it lives in this bottom
//! vocabulary crate.

use crate::error::Result;

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
