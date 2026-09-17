// SPDX-License-Identifier: MIT OR Apache-2.0
//! Format-agnostic CPU helpers for quantized storage.
//!
//! Holds the dyn-boxable `QuantizedType` trait (per-block-format scalar
//! ops) plus `cpu_zeros` / `cpu_from_data` constructors used by the
//! `fuel-cpu-backend`-side `CpuQStorage` adapter and by the file-format
//! readers in `fuel-core/src/quantized/`.

use crate::k_quants::{
    self, BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K,
    BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, GgmlType,
};
use fuel_ir::quantized::GgmlDType;
use fuel_ir::{HostBuffer, Result};
use half::{bf16, f16};
use std::borrow::Cow;

pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<HostBuffer>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]);
    #[allow(clippy::wrong_self_convention)]
    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize);
    fn size(&self) -> usize;
}

impl<T: GgmlType + Send + Sync> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        k_quants::matmul_f16(mkn, lhs, self.as_slice(), dst)
    }
    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }
    fn from_float(&mut self, xs: &[f32]) {
        T::from_float(xs, self)
    }
    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize) {
        T::from_float_imatrix(xs, self, imatrix_weights, n_per_row)
    }
    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }
    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }
    fn dequantize(&self, elem_count: usize) -> Result<HostBuffer> {
        // GAP-333: without this, too many elements panicked on a block index
        // and a partial block left a silently-zero tail (`to_float` only
        // `debug_assert`s the block multiple).
        T::DTYPE.check_dequant_count("dequantize", elem_count, self.storage_size_in_bytes())?;
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys);
        Ok(HostBuffer::F32(ys))
    }
    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }
}

/// GAP-336: copies `data` into owned blocks, one `read_unaligned` per block.
///
/// This replaces `as_t_slice`, which took the `Cow` by value and returned a
/// slice into it. For `Cow::Owned` that slice outlived the buffer it pointed
/// into (a use-after-free, even with valid input), and it also panicked on a
/// byte buffer that was not aligned for `T`. Nothing is borrowed here, so the
/// result depends on neither the input's lifetime nor its alignment.
fn blocks_from_bytes<T: GgmlType>(data: &[u8]) -> Result<Vec<T>> {
    let size = std::mem::size_of::<T>();
    if !data.len().is_multiple_of(size) {
        return Err(fuel_ir::Error::Msg(format!(
            "cpu_from_data: {} bytes is not a whole number of {:?} blocks ({size} bytes each)",
            data.len(),
            T::DTYPE,
        ))
        .bt());
    }
    Ok(data
        .chunks_exact(size)
        // SAFETY: each chunk is exactly `size_of::<T>()` bytes, and every `T`
        // this is called with (in `cpu_from_data`) is plain data: `f32`,
        // `f16`, `bf16`, or a block of `f16`/`f32` and integer arrays, so any
        // byte pattern is a valid `T`.
        .map(|c| unsafe { std::ptr::read_unaligned(c.as_ptr().cast::<T>()) })
        .collect())
}

pub fn cpu_zeros(dtype: GgmlDType, elem_count: usize) -> Box<dyn QuantizedType> {
    match dtype {
        GgmlDType::F32 => Box::new(vec![f32::zeros(); elem_count]),
        GgmlDType::F16 => Box::new(vec![f16::zeros(); elem_count]),
        GgmlDType::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
        GgmlDType::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
        GgmlDType::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
        GgmlDType::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
        GgmlDType::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
        GgmlDType::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
        GgmlDType::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
        GgmlDType::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
        GgmlDType::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
        GgmlDType::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
        GgmlDType::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
        GgmlDType::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
        GgmlDType::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
    }
}

/// Builds CPU quantized storage from raw block bytes. The bytes are copied,
/// so `data` may be owned or borrowed and need not be aligned for `dtype`.
/// Returns `Err` if `data` is not a whole number of blocks.
pub fn cpu_from_data(dtype: GgmlDType, data: Cow<'_, [u8]>) -> Result<Box<dyn QuantizedType>> {
    let data: &[u8] = &data;
    Ok(match dtype {
        GgmlDType::F32 => Box::new(blocks_from_bytes::<f32>(data)?),
        GgmlDType::F16 => Box::new(blocks_from_bytes::<f16>(data)?),
        GgmlDType::BF16 => Box::new(blocks_from_bytes::<bf16>(data)?),
        GgmlDType::Q4_0 => Box::new(blocks_from_bytes::<BlockQ4_0>(data)?),
        GgmlDType::Q4_1 => Box::new(blocks_from_bytes::<BlockQ4_1>(data)?),
        GgmlDType::Q5_0 => Box::new(blocks_from_bytes::<BlockQ5_0>(data)?),
        GgmlDType::Q5_1 => Box::new(blocks_from_bytes::<BlockQ5_1>(data)?),
        GgmlDType::Q8_0 => Box::new(blocks_from_bytes::<BlockQ8_0>(data)?),
        GgmlDType::Q8_1 => Box::new(blocks_from_bytes::<BlockQ8_1>(data)?),
        GgmlDType::Q2K => Box::new(blocks_from_bytes::<BlockQ2K>(data)?),
        GgmlDType::Q3K => Box::new(blocks_from_bytes::<BlockQ3K>(data)?),
        GgmlDType::Q4K => Box::new(blocks_from_bytes::<BlockQ4K>(data)?),
        GgmlDType::Q5K => Box::new(blocks_from_bytes::<BlockQ5K>(data)?),
        GgmlDType::Q6K => Box::new(blocks_from_bytes::<BlockQ6K>(data)?),
        GgmlDType::Q8K => Box::new(blocks_from_bytes::<BlockQ8K>(data)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(xs: &[f32]) -> Vec<u8> {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn dequant(q: &dyn QuantizedType, n: usize) -> Vec<f32> {
        q.dequantize(n).unwrap().as_slice::<f32>().unwrap().to_vec()
    }

    /// `Cow::Owned` is the input `as_t_slice` turned into a dangling slice.
    /// This pins the values; the use-after-free itself is not observable from
    /// a plain test (it needs Miri), so the evidence for it is the lifetime
    /// argument in `blocks_from_bytes`'s doc.
    #[test]
    fn owned_f32_bytes_round_trip() {
        let xs = [1.5_f32, -2.0, 3.25];
        let q = cpu_from_data(GgmlDType::F32, Cow::Owned(f32_bytes(&xs))).unwrap();
        assert_eq!(dequant(q.as_ref(), 3), xs);
    }

    #[test]
    fn owned_q4_0_blocks_round_trip() {
        // Two hand-built blocks, checked against the format's own rule
        // (`value = (nibble - 8) * d`, low nibbles first), so the test trusts
        // neither the quantizer nor a raw-pointer read.
        let mut bytes = Vec::new();
        for (d, q) in [(1.0_f32, 0x3A_u8), (0.5, 0x81)] {
            bytes.extend_from_slice(&f16::from_f32(d).to_le_bytes());
            bytes.extend(std::iter::repeat_n(q, 16));
        }
        let back = cpu_from_data(GgmlDType::Q4_0, Cow::Owned(bytes)).unwrap();
        // Block 0: low 0xA -> 2, high 0x3 -> -5, scale 1. Block 1: low 0x1 -> -7,
        // high 0x8 -> 0, scale 0.5.
        let want: Vec<f32> = [2.0, -5.0, -3.5, 0.0]
            .iter()
            .flat_map(|&v| std::iter::repeat_n(v, 16))
            .collect();
        assert_eq!(dequant(back.as_ref(), 64), want);
    }

    /// GAP-333: `dequantize` checks the count against the stored blocks.
    #[test]
    fn dequantize_declines_a_count_the_blocks_cannot_supply() {
        let q = cpu_zeros(GgmlDType::Q4_0, 64); // 2 blocks, 36 bytes
        assert_eq!(dequant(q.as_ref(), 64), vec![0.0; 64]);
        let too_many = q.dequantize(96).expect_err("3 blocks from 2").to_string();
        assert!(
            too_many.contains("96 elements need 54 bytes of Q4_0 blocks, but the buffer holds 36"),
            "{too_many}"
        );
        let partial = q.dequantize(40).expect_err("a partial block").to_string();
        assert!(
            partial.contains("element count 40 is not a multiple of Q4_0's block size 32"),
            "{partial}"
        );
    }

    #[test]
    fn a_partial_trailing_block_is_declined() {
        let err = cpu_from_data(GgmlDType::Q4_0, Cow::Owned(vec![0_u8; 18 + 5]))
            .err()
            .expect("a partial block must be declined")
            .to_string();
        assert!(
            err.contains("23 bytes is not a whole number of Q4_0 blocks (18 bytes each)"),
            "wrong decline: {err}"
        );
    }

    /// `as_t_slice` panicked on this input ("Data pointer must be aligned").
    #[test]
    fn misaligned_bytes_are_accepted() {
        let xs = [0.5_f32, 7.0, -1.25, 2.0];
        let body = f32_bytes(&xs);
        let mut buf = vec![0_u8; 3 + body.len()];
        let off = (0..3)
            .find(|o| !(buf.as_ptr() as usize + o).is_multiple_of(std::mem::align_of::<f32>()))
            .expect("one of three offsets is misaligned for f32");
        buf[off..off + body.len()].copy_from_slice(&body);
        let bytes = &buf[off..off + body.len()];
        assert!(!(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()));
        let q = cpu_from_data(GgmlDType::F32, Cow::Borrowed(bytes)).unwrap();
        assert_eq!(dequant(q.as_ref(), 4), xs);
    }
}
