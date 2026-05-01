//! Format-agnostic CPU helpers for quantized storage.
//!
//! Holds the dyn-boxable `QuantizedType` trait (per-block-format scalar
//! ops) plus `cpu_zeros` / `cpu_from_data` constructors used by the
//! `fuel-cpu-backend`-side `CpuQStorage` adapter and by the file-format
//! readers in `fuel-core/src/quantized/`.

use crate::k_quants::{
    self, BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1,
    BlockQ6K, BlockQ8K, BlockQ8_0, BlockQ8_1, GgmlType,
};
use fuel_core_types::quantized::GgmlDType;
use fuel_core_types::{HostBuffer, Result};
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

pub fn as_t_slice<T>(data: Cow<'_, [u8]>) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert_eq!(
        data.len() % size,
        0,
        "Data length must be a multiple of T's size"
    );
    let ptr = data.as_ptr();
    assert_eq!(
        (ptr as usize) % std::mem::align_of::<T>(),
        0,
        "Data pointer must be aligned to T's alignment"
    );
    unsafe { std::slice::from_raw_parts(ptr as *const T, data.len() / size) }
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

pub fn cpu_from_data(dtype: GgmlDType, data: Cow<'_, [u8]>) -> Box<dyn QuantizedType> {
    match dtype {
        GgmlDType::F32 => Box::new(as_t_slice::<f32>(data).to_vec()),
        GgmlDType::F16 => Box::new(as_t_slice::<f16>(data).to_vec()),
        GgmlDType::BF16 => Box::new(as_t_slice::<bf16>(data).to_vec()),
        GgmlDType::Q4_0 => Box::new(as_t_slice::<BlockQ4_0>(data).to_vec()),
        GgmlDType::Q4_1 => Box::new(as_t_slice::<BlockQ4_1>(data).to_vec()),
        GgmlDType::Q5_0 => Box::new(as_t_slice::<BlockQ5_0>(data).to_vec()),
        GgmlDType::Q5_1 => Box::new(as_t_slice::<BlockQ5_1>(data).to_vec()),
        GgmlDType::Q8_0 => Box::new(as_t_slice::<BlockQ8_0>(data).to_vec()),
        GgmlDType::Q8_1 => Box::new(as_t_slice::<BlockQ8_1>(data).to_vec()),
        GgmlDType::Q2K => Box::new(as_t_slice::<BlockQ2K>(data).to_vec()),
        GgmlDType::Q3K => Box::new(as_t_slice::<BlockQ3K>(data).to_vec()),
        GgmlDType::Q4K => Box::new(as_t_slice::<BlockQ4K>(data).to_vec()),
        GgmlDType::Q5K => Box::new(as_t_slice::<BlockQ5K>(data).to_vec()),
        GgmlDType::Q6K => Box::new(as_t_slice::<BlockQ6K>(data).to_vec()),
        GgmlDType::Q8K => Box::new(as_t_slice::<BlockQ8K>(data).to_vec()),
    }
}
