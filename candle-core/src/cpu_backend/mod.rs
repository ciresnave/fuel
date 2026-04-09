//! Implementation of Backend Fns for CPU
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{DType, Error, Layout, Result, Shape, WithDType};
// Import upstream WithDType for method resolution on concrete types (e.g. f32::from_f64).
use candle_core_types::dtype::WithDType as _;
use float8::F8E4M3;
use half::{bf16, f16};
use candle_cpu_backend::utils::{
    Map1 as CbMap1, Map1Any as CbMap1Any, Map2 as CbMap2, Map2InPlace as CbMap2IP,
    Map2U8 as CbMap2U8,
};
use rayon::prelude::*;

mod utils;
pub use utils::{
    binary_map, binary_map_vec, unary_map, unary_map_vec, Map1, Map1Any, Map2, Map2InPlace, Map2U8,
};

// CpuStorage, CpuStorageRef, and CpuDevice are now the same types as in
// candle-core-types (and candle-cpu-backend). No local enum copies needed.
pub use candle_core_types::{CpuDevice, CpuStorage, CpuStorageRef};

// Thin delegation macros: now that CpuStorage is unified across crates,
// these just forward to candle_cpu_backend's trait `.map()` methods with
// error conversion via `?`.
macro_rules! cpu_map1 {
    ($op:expr, $self_:expr, $layout:expr) => {{
        Ok::<_, crate::Error>(CbMap1::map(&{ $op }, $self_, $layout)?)
    }};
}

macro_rules! cpu_map1any {
    ($op:expr, $self_:expr, $layout:expr) => {{
        Ok::<_, crate::Error>(CbMap1Any::map(&{ $op }, $self_, $layout)?)
    }};
}

macro_rules! cpu_map2 {
    ($op:expr, $self_:expr, $l1:expr, $rhs:expr, $l2:expr) => {{
        Ok::<_, crate::Error>(CbMap2::map(&{ $op }, $self_, $l1, $rhs, $l2)?)
    }};
}

macro_rules! cpu_map2u8 {
    ($op:expr, $self_:expr, $l1:expr, $rhs:expr, $l2:expr) => {{
        Ok::<_, crate::Error>(CbMap2U8::map(&{ $op }, $self_, $l1, $rhs, $l2)?)
    }};
}

macro_rules! cpu_map2_in_place {
    ($op:expr, $self_:expr, $l1:expr, $rhs:expr, $l2:expr) => {{
        CbMap2IP::map(&{ $op }, $self_, $l1, $rhs, $l2)?;
        Ok::<_, crate::Error>(())
    }};
}

const USE_IM2COL_CONV1D: bool = true;
const USE_COL2IM_CONV1D_TR: bool = true;

#[allow(clippy::too_many_arguments)]
fn copy2d_<T: Copy>(
    src: &[T],
    dst: &mut [T],
    d1: usize,
    d2: usize,
    src_stride1: usize,
    dst_stride1: usize,
    src_offset: usize,
    dst_offset: usize,
) {
    for i1 in 0..d1 {
        let dst_idx = i1 * dst_stride1 + dst_offset;
        let src_idx = i1 * src_stride1 + src_offset;
        let dst = &mut dst[dst_idx..dst_idx + d2];
        let src = &src[src_idx..src_idx + d2];
        dst.copy_from_slice(src)
    }
}

fn copy_strided_src_<T: Copy>(src: &[T], dst: &mut [T], dst_offset: usize, src_l: &Layout) {
    match src_l.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => {
            let to_copy = (dst.len() - dst_offset).min(len);
            dst[dst_offset..dst_offset + to_copy]
                .copy_from_slice(&src[start_offset..start_offset + to_copy])
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len: 1,
        } => {
            for (dst_index, src_index) in block_start_index.enumerate() {
                let dst_index = dst_index + dst_offset;
                if dst_index >= dst.len() {
                    break;
                }
                dst[dst_index] = src[src_index]
            }
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let mut dst_index = dst_offset;
            for src_index in block_start_index {
                let next_dst_index = dst_index + block_len;
                if dst_index >= dst.len() {
                    break;
                }
                let to_copy = usize::min(block_len, dst.len() - dst_index);
                dst[dst_index..dst_index + to_copy]
                    .copy_from_slice(&src[src_index..src_index + to_copy]);
                dst_index = next_dst_index
            }
        }
    }
}


fn elu<T: num_traits::Float>(v: T, alpha: T) -> T {
    if v.is_sign_positive() {
        v
    } else {
        (v.exp() - T::one()) * alpha
    }
}

// as_slice() and concat() are already defined as inherent methods on CpuStorage
// in candle-core-types — no need to redefine them here.

impl BackendStorage for CpuStorage {
    type Device = CpuDevice;

    fn dtype(&self) -> DType {
        // Delegates to CpuStorage's inherent dtype() from candle-core-types.
        // DType is the same type via re-export.
        CpuStorage::dtype(self)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        // TODO: find a way around the quadratic number of cases below.
        match (self, dtype) {
            (Self::U8(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::U32(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::I64(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::BF16(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::BF16(data))
            }
            (Self::F16(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v.to_f32()));
                Ok(Self::BF16(data))
            }
            (Self::F32(storage), DType::BF16) => {
                let data = unary_map(storage, layout, bf16::from_f32);
                Ok(Self::BF16(data))
            }
            (Self::F64(storage), DType::BF16) => {
                let data = unary_map(storage, layout, bf16::from_f64);
                Ok(Self::BF16(data))
            }
            (Self::U8(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::U32(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::I64(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::BF16(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v.to_f32()));
                Ok(Self::F16(data))
            }
            (Self::F16(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F16(data))
            }
            (Self::F32(storage), DType::F16) => {
                let data = unary_map(storage, layout, f16::from_f32);
                Ok(Self::F16(data))
            }
            (Self::F64(storage), DType::F16) => {
                let data = unary_map(storage, layout, f16::from_f64);
                Ok(Self::F16(data))
            }
            (Self::U8(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::U32(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::I64(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::BF16(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::F16(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::F32(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F32(data))
            }
            (Self::F64(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::U8(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::U8(data))
            }
            (Self::BF16(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u8);
                Ok(Self::U8(data))
            }
            (Self::F16(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u8);
                Ok(Self::U8(data))
            }
            (Self::F32(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::F64(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::U32(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::I64(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::U8(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::U32(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::U32(data))
            }
            (Self::I64(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::BF16(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::F16(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::F32(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::F64(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::U8(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::U32(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::I64(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::I64(data))
            }
            (Self::BF16(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i64);
                Ok(Self::I64(data))
            }
            (Self::F16(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i64);
                Ok(Self::I64(data))
            }
            (Self::F32(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::F64(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::U8(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::U32(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::I64(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::BF16(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v.to_f64());
                Ok(Self::F64(data))
            }
            (Self::F16(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v.to_f64());
                Ok(Self::F64(data))
            }
            (Self::F32(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::F64(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F64(data))
            }
            // Conversions to F8E4M3
            (Self::U8(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from_f32(v as f32));
                Ok(Self::F8E4M3(data))
            }
            (Self::U32(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from_f32(v as f32));
                Ok(Self::F8E4M3(data))
            }
            (Self::I64(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from_f32(v as f32));
                Ok(Self::F8E4M3(data))
            }
            (Self::BF16(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from_f32(v.to_f32()));
                Ok(Self::F8E4M3(data))
            }
            (Self::F16(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from_f32(v.to_f32()));
                Ok(Self::F8E4M3(data))
            }
            (Self::F32(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, F8E4M3::from_f32);
                Ok(Self::F8E4M3(data))
            }
            (Self::F64(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, F8E4M3::from_f64);
                Ok(Self::F8E4M3(data))
            }
            (Self::F8E4M3(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F8E4M3(data))
            }
            // Conversions from F8E4M3
            (Self::F8E4M3(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u8);
                Ok(Self::U8(data))
            }
            (Self::F8E4M3(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::F8E4M3(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i64);
                Ok(Self::I64(data))
            }
            (Self::F8E4M3(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v.to_f32()));
                Ok(Self::BF16(data))
            }
            (Self::F8E4M3(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v.to_f32()));
                Ok(Self::F16(data))
            }
            (Self::F8E4M3(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::F8E4M3(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v.to_f64());
                Ok(Self::F64(data))
            }
            // Conversions to I16
            (Self::U8(storage), DType::I16) => {
                let data = unary_map(storage, layout, |v| v as i16);
                Ok(Self::I16(data))
            }
            (Self::U32(storage), DType::I16) => {
                let data = unary_map(storage, layout, |v| v as i16);
                Ok(Self::I16(data))
            }
            (Self::I16(storage), DType::I16) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::I16(data))
            }
            (Self::I32(storage), DType::I16) => {
                let data = unary_map(storage, layout, |v| v as i16);
                Ok(Self::I16(data))
            }
            (Self::I64(storage), DType::I16) => {
                let data = unary_map(storage, layout, |v| v as i16);
                Ok(Self::I16(data))
            }
            (Self::BF16(storage), DType::I16) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i16);
                Ok(Self::I16(data))
            }
            (Self::F16(storage), DType::I16) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i16);
                Ok(Self::I16(data))
            }
            (Self::F32(storage), DType::I16) => {
                let data = unary_map(storage, layout, |v| v as i16);
                Ok(Self::I16(data))
            }
            (Self::F64(storage), DType::I16) => {
                let data = unary_map(storage, layout, |v| v as i16);
                Ok(Self::I16(data))
            }
            (Self::F8E4M3(storage), DType::I16) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i16);
                Ok(Self::I16(data))
            }
            // Conversions to I32
            (Self::U8(storage), DType::I32) => {
                let data = unary_map(storage, layout, |v| v as i32);
                Ok(Self::I32(data))
            }
            (Self::U32(storage), DType::I32) => {
                let data = unary_map(storage, layout, |v| v as i32);
                Ok(Self::I32(data))
            }
            (Self::I16(storage), DType::I32) => {
                let data = unary_map(storage, layout, |v| v as i32);
                Ok(Self::I32(data))
            }
            (Self::I32(storage), DType::I32) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::I32(data))
            }
            (Self::I64(storage), DType::I32) => {
                let data = unary_map(storage, layout, |v| v as i32);
                Ok(Self::I32(data))
            }
            (Self::BF16(storage), DType::I32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i32);
                Ok(Self::I32(data))
            }
            (Self::F16(storage), DType::I32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i32);
                Ok(Self::I32(data))
            }
            (Self::F32(storage), DType::I32) => {
                let data = unary_map(storage, layout, |v| v as i32);
                Ok(Self::I32(data))
            }
            (Self::F64(storage), DType::I32) => {
                let data = unary_map(storage, layout, |v| v as i32);
                Ok(Self::I32(data))
            }
            (Self::F8E4M3(storage), DType::I32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i32);
                Ok(Self::I32(data))
            }
            // Conversions from I16
            (Self::I16(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::I16(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::I16(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::I16(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::I16(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::I16(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::I16(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::I16(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from_f32(v as f32));
                Ok(Self::F8E4M3(data))
            }
            // Conversions from I32
            (Self::I32(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::I32(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::I32(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::I32(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::I32(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::I32(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::I32(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::I32(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from_f32(v as f32));
                Ok(Self::F8E4M3(data))
            }
            // Dummy types - return error for all conversions to/from dummy types
            (_, DType::F6E2M3) | (_, DType::F6E3M2) | (_, DType::F4) | (_, DType::F8E8M0) => {
                Err(Error::UnsupportedDTypeForOp(dtype, "to_dtype").bt())
            }
            (Self::F6E2M3(_), _)
            | (Self::F6E3M2(_), _)
            | (Self::F4(_), _)
            | (Self::F8E8M0(_), _) => {
                Err(Error::UnsupportedDTypeForOp(self.dtype(), "to_dtype").bt())
            }
        }
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        match op {
            ReduceOp::Sum => {
                let src_dims = layout.dims();
                let mut dst_dims = src_dims.to_vec();
                for &dim in reduce_dims.iter() {
                    dst_dims[dim] = 1;
                }
                let dst_shape = Shape::from(dst_dims);
                let mut reduce_dims = reduce_dims.to_vec();
                reduce_dims.sort();
                let reduce_dims_and_stride: Vec<_> = reduce_dims
                    .iter()
                    .map(|&d| (src_dims[d], src_dims[d + 1..].iter().product::<usize>()))
                    .collect();
                cpu_map1!(
                    candle_cpu_backend::ops::ReduceSum {
                        dst_shape: &dst_shape,
                        reduce_dims: &reduce_dims,
                        reduce_dims_and_stride,
                    },
                    self,
                    layout
                )
            }
            ReduceOp::Min | ReduceOp::ArgMin | ReduceOp::Max | ReduceOp::ArgMax => {
                let reduce_dim_index = match reduce_dims {
                    [reduce_dim_index] => *reduce_dim_index,
                    _ => {
                        let op = match op {
                            ReduceOp::Min => "min",
                            ReduceOp::ArgMin => "argmin",
                            ReduceOp::Max => "max",
                            ReduceOp::ArgMax => "argmax",
                            _ => unreachable!(),
                        };
                        let dims = reduce_dims.to_vec();
                        Err(Error::OnlySingleDimension { op, dims })?
                    }
                };
                let (use_min, return_index) = match op {
                    ReduceOp::Min => (true, false),
                    ReduceOp::ArgMin => (true, true),
                    ReduceOp::Max => (false, false),
                    ReduceOp::ArgMax => (false, true),
                    _ => unreachable!(),
                };
                cpu_map1any!(
                    candle_cpu_backend::ops::ReduceIndex {
                        reduce_dim_index,
                        use_min,
                        return_index,
                    },
                    self,
                    layout
                )
            }
        }
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        cpu_map2u8!(candle_cpu_backend::ops::Cmp(op), self, lhs_l, rhs, rhs_l)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        cpu_map1!(candle_cpu_backend::ops::Affine(mul, add), self, layout)
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        cpu_map1!(candle_cpu_backend::ops::AvgPool2D(kernel_size, stride), self, layout)
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        cpu_map1!(candle_cpu_backend::ops::MaxPool2D(kernel_size, stride), self, layout)
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        cpu_map1!(candle_cpu_backend::ops::UpsampleNearest1D(sz), self, layout)
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        cpu_map1!(candle_cpu_backend::ops::UpsampleNearest2D(h, w), self, layout)
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        cpu_map1!(
            candle_cpu_backend::ops::UpsampleBilinear2D {
                target_h: h,
                target_w: w,
                align_corners,
                scale_h_factor: scale_h,
                scale_w_factor: scale_w,
            },
            self,
            layout
        )
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        use num_traits::Float;
        // TODO: Have some generic map for functions that apply on num_traits::Float elements.
        match self {
            Self::BF16(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(bf16::from_f64(e)));
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(f16::from_f64(e)));
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(e as f32));
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(e));
                Ok(Self::F64(data))
            }
            Self::F8E4M3(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(F8E4M3::from_f64(e)));
                Ok(Self::F8E4M3(data))
            }
            Self::U8(_) => Err(Error::UnsupportedDTypeForOp(DType::U8, "powf").bt()),
            Self::U32(_) => Err(Error::UnsupportedDTypeForOp(DType::U32, "powf").bt()),
            Self::I16(_) => Err(Error::UnsupportedDTypeForOp(DType::I16, "powf").bt()),
            Self::I32(_) => Err(Error::UnsupportedDTypeForOp(DType::I32, "powf").bt()),
            Self::I64(_) => Err(Error::UnsupportedDTypeForOp(DType::I64, "powf").bt()),
            Self::F6E2M3(_) => Err(Error::UnsupportedDTypeForOp(DType::F6E2M3, "powf").bt()),
            Self::F6E3M2(_) => Err(Error::UnsupportedDTypeForOp(DType::F6E3M2, "powf").bt()),
            Self::F4(_) => Err(Error::UnsupportedDTypeForOp(DType::F4, "powf").bt()),
            Self::F8E8M0(_) => Err(Error::UnsupportedDTypeForOp(DType::F8E8M0, "powf").bt()),
        }
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        // TODO: Have some generic map for functions that apply on num_traits::Float elements.
        match self {
            Self::BF16(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, bf16::from_f64(alpha)));
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, f16::from_f64(alpha)));
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, f32::from_f64(alpha)));
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, alpha));
                Ok(Self::F64(data))
            }
            Self::F8E4M3(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, F8E4M3::from_f64(alpha)));
                Ok(Self::F8E4M3(data))
            }
            Self::U8(_) => Err(Error::UnsupportedDTypeForOp(DType::U8, "elu").bt()),
            Self::U32(_) => Err(Error::UnsupportedDTypeForOp(DType::U32, "elu").bt()),
            Self::I16(_) => Err(Error::UnsupportedDTypeForOp(DType::I16, "elu").bt()),
            Self::I32(_) => Err(Error::UnsupportedDTypeForOp(DType::I32, "elu").bt()),
            Self::I64(_) => Err(Error::UnsupportedDTypeForOp(DType::I64, "elu").bt()),
            Self::F6E2M3(_) => Err(Error::UnsupportedDTypeForOp(DType::F6E2M3, "elu").bt()),
            Self::F6E3M2(_) => Err(Error::UnsupportedDTypeForOp(DType::F6E3M2, "elu").bt()),
            Self::F4(_) => Err(Error::UnsupportedDTypeForOp(DType::F4, "elu").bt()),
            Self::F8E8M0(_) => Err(Error::UnsupportedDTypeForOp(DType::F8E8M0, "elu").bt()),
        }
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        match self {
            Self::BF16(storage) => {
                if B::BF16_VEC {
                    let data = unary_map_vec(storage, layout, B::bf16, B::bf16_vec);
                    Ok(Self::BF16(data))
                } else {
                    let data = unary_map(storage, layout, B::bf16);
                    Ok(Self::BF16(data))
                }
            }
            Self::F16(storage) => {
                if B::F16_VEC {
                    let data = unary_map_vec(storage, layout, B::f16, B::f16_vec);
                    Ok(Self::F16(data))
                } else {
                    let data = unary_map(storage, layout, B::f16);
                    Ok(Self::F16(data))
                }
            }
            Self::F32(storage) => {
                if B::F32_VEC {
                    let data = unary_map_vec(storage, layout, B::f32, B::f32_vec);
                    Ok(Self::F32(data))
                } else {
                    let data = unary_map(storage, layout, B::f32);
                    Ok(Self::F32(data))
                }
            }
            Self::F64(storage) => {
                if B::F64_VEC {
                    let data = unary_map_vec(storage, layout, B::f64, B::f64_vec);
                    Ok(Self::F64(data))
                } else {
                    let data = unary_map(storage, layout, B::f64);
                    Ok(Self::F64(data))
                }
            }
            Self::U8(storage) => {
                let data = unary_map(storage, layout, B::u8);
                Ok(Self::U8(data))
            }
            Self::U32(storage) => {
                let data = unary_map(storage, layout, B::u32);
                Ok(Self::U32(data))
            }
            Self::I16(storage) => {
                let data = unary_map(storage, layout, B::i16);
                Ok(Self::I16(data))
            }
            Self::I32(storage) => {
                let data = unary_map(storage, layout, B::i32);
                Ok(Self::I32(data))
            }
            Self::I64(storage) => {
                let data = unary_map(storage, layout, B::i64);
                Ok(Self::I64(data))
            }
            Self::F8E4M3(storage) => {
                let data = unary_map(storage, layout, B::f8e4m3);
                Ok(Self::F8E4M3(data))
            }
            Self::F6E2M3(_) => Err(Error::UnsupportedDTypeForOp(DType::F6E2M3, "unary").bt()),
            Self::F6E3M2(_) => Err(Error::UnsupportedDTypeForOp(DType::F6E3M2, "unary").bt()),
            Self::F4(_) => Err(Error::UnsupportedDTypeForOp(DType::F4, "unary").bt()),
            Self::F8E8M0(_) => Err(Error::UnsupportedDTypeForOp(DType::F8E8M0, "unary").bt()),
        }
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        match (self, rhs) {
            (Self::BF16(lhs), Self::BF16(rhs)) => {
                let data = if B::BF16_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::bf16, B::bf16_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::bf16)
                };
                Ok(Self::BF16(data))
            }
            (Self::F16(lhs), Self::F16(rhs)) => {
                let data = if B::F16_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::f16, B::f16_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::f16)
                };
                Ok(Self::F16(data))
            }
            (Self::F32(lhs), Self::F32(rhs)) => {
                let data = if B::F32_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::f32, B::f32_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::f32)
                };
                Ok(Self::F32(data))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                let data = if B::F64_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::f64, B::f64_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::f64)
                };
                Ok(Self::F64(data))
            }
            (Self::U32(lhs), Self::U32(rhs)) => {
                let data = if B::U32_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::u32, B::u32_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::u32)
                };
                Ok(Self::U32(data))
            }
            (Self::I16(lhs), Self::I16(rhs)) => {
                let data = binary_map(lhs_l, rhs_l, lhs, rhs, B::i16);
                Ok(Self::I16(data))
            }
            (Self::I32(lhs), Self::I32(rhs)) => {
                let data = binary_map(lhs_l, rhs_l, lhs, rhs, B::i32);
                Ok(Self::I32(data))
            }
            (Self::I64(lhs), Self::I64(rhs)) => {
                let data = if B::I64_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::i64, B::i64_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::i64)
                };
                Ok(Self::I64(data))
            }
            (Self::U8(lhs), Self::U8(rhs)) => {
                let data = if B::U8_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::u8, B::u8_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::u8)
                };
                Ok(Self::U8(data))
            }
            (Self::F8E4M3(lhs), Self::F8E4M3(rhs)) => {
                let data = binary_map(lhs_l, rhs_l, lhs, rhs, B::f8e4m3);
                Ok(Self::F8E4M3(data))
            }
            _ => {
                // This should be covered by the dtype check above.
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: rhs.dtype(),
                    op: B::NAME,
                }
                .bt())
            }
        }
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        match (self, dst) {
            (Self::U8(src), Self::U8(dst)) => copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o),
            (Self::U32(src), Self::U32(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::I16(src), Self::I16(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::I32(src), Self::I32(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::I64(src), Self::I64(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::BF16(src), Self::BF16(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::F16(src), Self::F16(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::F32(src), Self::F32(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::F64(src), Self::F64(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::F8E4M3(src), Self::F8E4M3(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::F6E2M3(src), Self::F6E2M3(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::F6E3M2(src), Self::F6E3M2(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::F4(src), Self::F4(dst)) => copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o),
            (Self::F8E8M0(src), Self::F8E8M0(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (_, dst) => {
                return Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: dst.dtype(),
                    op: "copy2d",
                }
                .bt());
            }
        }
        Ok(())
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        match (self, dst) {
            (Self::U8(src), Self::U8(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::U32(src), Self::U32(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::I16(src), Self::I16(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::I32(src), Self::I32(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::I64(src), Self::I64(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::BF16(src), Self::BF16(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F16(src), Self::F16(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F32(src), Self::F32(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F64(src), Self::F64(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F8E4M3(src), Self::F8E4M3(dst)) => {
                copy_strided_src_(src, dst, dst_offset, src_l)
            }
            (Self::F6E2M3(src), Self::F6E2M3(dst)) => {
                copy_strided_src_(src, dst, dst_offset, src_l)
            }
            (Self::F6E3M2(src), Self::F6E3M2(dst)) => {
                copy_strided_src_(src, dst, dst_offset, src_l)
            }
            (Self::F4(src), Self::F4(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F8E8M0(src), Self::F8E8M0(dst)) => {
                copy_strided_src_(src, dst, dst_offset, src_l)
            }
            (_, dst) => {
                // This should be covered by the dtype check above.
                return Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: dst.dtype(),
                    op: "copy_strided",
                }
                .bt());
            }
        }
        Ok(())
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        match self {
            Self::U8(pred) => cpu_map2!(candle_cpu_backend::ops::WCond(pred, layout), t, t_l, f, f_l),
            Self::U32(pred) => cpu_map2!(candle_cpu_backend::ops::WCond(pred, layout), t, t_l, f, f_l),
            Self::I16(pred) => cpu_map2!(candle_cpu_backend::ops::WCond(pred, layout), t, t_l, f, f_l),
            Self::I32(pred) => cpu_map2!(candle_cpu_backend::ops::WCond(pred, layout), t, t_l, f, f_l),
            Self::I64(pred) => cpu_map2!(candle_cpu_backend::ops::WCond(pred, layout), t, t_l, f, f_l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "where-cond")),
        }
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        if !USE_IM2COL_CONV1D {
            return cpu_map2!(candle_cpu_backend::ops::Conv1D(params), self, l, kernel, kernel_l);
        }
        let l_k = params.k_size;
        let col = cpu_map1!(
            candle_cpu_backend::ops::Im2Col1D {
                l_k,
                padding: params.padding,
                stride: params.stride,
                dilation: params.dilation,
            },
            self,
            l
        )?;
        let b = params.b_size;
        let n = params.c_out;
        let l_out = params.l_out();
        let k = l_k * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b, m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = unsafe {
                self.device()
                    .alloc_uninit(kernel_l.shape(), kernel.dtype())?
            };
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, l_out, params.c_out)).transpose(1, 2)?;
        let mut res_t = unsafe { self.device().alloc_uninit(res_l.shape(), res.dtype())? };
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let can_use_col2im = kernel_l.is_contiguous()
            && params.dilation == 1
            && params.padding == 0
            && params.output_padding == 0;
        if USE_COL2IM_CONV1D_TR && can_use_col2im {
            let (b_size, c_in, l_in) = l.shape().dims3()?;
            let (c_in2, c_out, k_size) = kernel_l.shape().dims3()?;
            if !kernel_l.is_contiguous() {
                crate::bail!(
                    "convtr1d: the second argument (kernel) has to be contiguous {kernel_l:?}"
                )
            }
            if c_in != c_in2 {
                crate::bail!(
                    "convtr1d: shape mismatch on c_in {:?} {:?}",
                    l.shape(),
                    kernel_l.shape()
                )
            }
            let col = {
                // This merges the last two dimensions of the kernel together.
                let kernel_l_mm = Layout::new(
                    (b_size, c_in, k_size * c_out).into(),
                    vec![0, k_size * c_out, 1].into(),
                    kernel_l.start_offset(),
                );
                self.matmul(
                    kernel,
                    (
                        b_size,
                        /* m */ l_in,
                        /* n */ c_out * k_size,
                        /* k */ c_in,
                    ),
                    &l.transpose(1, 2)?,
                    &kernel_l_mm,
                )?
            };
            let col_l = Layout::contiguous((b_size, l_in, c_out, k_size));
            cpu_map1!(
                candle_cpu_backend::ops::Col2Im1D {
                    stride: params.stride,
                },
                &col,
                &col_l
            )
        } else {
            cpu_map2!(candle_cpu_backend::ops::ConvTranspose1D(params), self, l, kernel, kernel_l)
        }
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        cpu_map2!(candle_cpu_backend::conv2d::Conv2D(params), self, l, kernel, kernel_l)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        cpu_map2!(candle_cpu_backend::ops::ConvTranspose2D(params), self, l, kernel, kernel_l)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        match ids {
            Self::U8(ids) => cpu_map1!(candle_cpu_backend::ops::IndexSelect { ids, ids_l, dim }, self, l),
            Self::U32(ids) => cpu_map1!(candle_cpu_backend::ops::IndexSelect { ids, ids_l, dim }, self, l),
            Self::I64(ids) => cpu_map1!(candle_cpu_backend::ops::IndexSelect { ids, ids_l, dim }, self, l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "index-select").bt()),
        }
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        match ids {
            Self::U8(ids) => cpu_map1!(candle_cpu_backend::ops::Gather { ids, ids_l, dim }, self, l),
            Self::U32(ids) => cpu_map1!(candle_cpu_backend::ops::Gather { ids, ids_l, dim }, self, l),
            Self::I64(ids) => cpu_map1!(candle_cpu_backend::ops::Gather { ids, ids_l, dim }, self, l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "gather").bt()),
        }
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        match ids {
            Self::U8(ids) => cpu_map2_in_place!(candle_cpu_backend::ops::Scatter::<_, candle_cpu_backend::ops::Set>::new(ids, ids_l, dim), self, l, src, src_l),
            Self::U32(ids) => cpu_map2_in_place!(candle_cpu_backend::ops::Scatter::<_, candle_cpu_backend::ops::Set>::new(ids, ids_l, dim), self, l, src, src_l),
            Self::I64(ids) => cpu_map2_in_place!(candle_cpu_backend::ops::Scatter::<_, candle_cpu_backend::ops::Set>::new(ids, ids_l, dim), self, l, src, src_l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "scatter").bt()),
        }
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        match ids {
            Self::U8(ids) => cpu_map2_in_place!(candle_cpu_backend::ops::Scatter::<_, candle_cpu_backend::ops::Add>::new(ids, ids_l, dim), self, l, src, src_l),
            Self::U32(ids) => cpu_map2_in_place!(candle_cpu_backend::ops::Scatter::<_, candle_cpu_backend::ops::Add>::new(ids, ids_l, dim), self, l, src, src_l),
            Self::I16(ids) => cpu_map2_in_place!(candle_cpu_backend::ops::Scatter::<_, candle_cpu_backend::ops::Add>::new(ids, ids_l, dim), self, l, src, src_l),
            Self::I32(ids) => cpu_map2_in_place!(candle_cpu_backend::ops::Scatter::<_, candle_cpu_backend::ops::Add>::new(ids, ids_l, dim), self, l, src, src_l),
            Self::I64(ids) => cpu_map2_in_place!(candle_cpu_backend::ops::Scatter::<_, candle_cpu_backend::ops::Add>::new(ids, ids_l, dim), self, l, src, src_l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "scatter-add").bt()),
        }
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        match ids {
            Self::U8(ids) => {
                let ids = match ids_l.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                cpu_map2!(candle_cpu_backend::ops::IndexAdd { ids, dim }, self, l, src, src_l)
            }
            Self::U32(ids) => {
                let ids = match ids_l.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                cpu_map2!(candle_cpu_backend::ops::IndexAdd { ids, dim }, self, l, src, src_l)
            }
            Self::I16(ids) => {
                let ids = match ids_l.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                cpu_map2!(candle_cpu_backend::ops::IndexAdd { ids, dim }, self, l, src, src_l)
            }
            Self::I32(ids) => {
                let ids = match ids_l.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                cpu_map2!(candle_cpu_backend::ops::IndexAdd { ids, dim }, self, l, src, src_l)
            }
            Self::I64(ids) => {
                let ids = match ids_l.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                cpu_map2!(candle_cpu_backend::ops::IndexAdd { ids, dim }, self, l, src, src_l)
            }
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "index-add").bt()),
        }
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        cpu_map2!(candle_cpu_backend::ops::MatMul(bmnk), self, lhs_l, rhs, rhs_l)
    }

    fn device(&self) -> &Self::Device {
        &CpuDevice
    }

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Ok(self.clone())
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        use crate::scalar::Scalar;
        fn set<T: crate::WithDType>(src: &mut [T], l: &Layout, s: T) {
            match l.strided_blocks() {
                crate::StridedBlocks::SingleBlock { start_offset, len } => {
                    src[start_offset..start_offset + len].fill(s)
                }
                crate::StridedBlocks::MultipleBlocks {
                    block_start_index,
                    block_len: 1,
                } => {
                    for src_index in block_start_index {
                        src[src_index] = s
                    }
                }
                crate::StridedBlocks::MultipleBlocks {
                    block_start_index,
                    block_len,
                } => {
                    for src_index in block_start_index {
                        src[src_index..src_index + block_len].fill(s)
                    }
                }
            }
        }
        match (self, s) {
            (Self::BF16(storage), Scalar::BF16(v)) => set(storage, l, v),
            (Self::F16(storage), Scalar::F16(v)) => set(storage, l, v),
            (Self::F32(storage), Scalar::F32(v)) => set(storage, l, v),
            (Self::F64(storage), Scalar::F64(v)) => set(storage, l, v),
            (Self::U8(storage), Scalar::U8(v)) => set(storage, l, v),
            (Self::U32(storage), Scalar::U32(v)) => set(storage, l, v),
            (Self::I16(storage), Scalar::I16(v)) => set(storage, l, v),
            (Self::I32(storage), Scalar::I32(v)) => set(storage, l, v),
            (Self::I64(storage), Scalar::I64(v)) => set(storage, l, v),
            (Self::F8E4M3(storage), Scalar::F8E4M3(v)) => set(storage, l, v),
            // Dummy types don't support scalar operations
            (Self::F6E2M3(_), _) => {
                crate::bail!("const_set not supported for dummy type F6E2M3")
            }
            (Self::F6E3M2(_), _) => {
                crate::bail!("const_set not supported for dummy type F6E3M2")
            }
            (Self::F4(_), _) => {
                crate::bail!("const_set not supported for dummy type F4")
            }
            (Self::F8E8M0(_), _) => {
                crate::bail!("const_set not supported for dummy type F8E8M0")
            }
            (st, s) => crate::bail!(
                "const_set dtype mismatch, expected {:?} but got {:?}",
                st.dtype(),
                s
            ),
        }
        Ok(())
    }
}

impl BackendDevice for CpuDevice {
    type Storage = CpuStorage;

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cpu
    }

    fn same_device(&self, _: &Self) -> bool {
        true
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        Ok(T::to_cpu_storage(s))
    }

    fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<Self::Storage> {
        Ok(s.clone())
    }

    fn storage_from_cpu_storage_owned(&self, s: CpuStorage) -> Result<Self::Storage> {
        Ok(s)
    }

    fn new(_: usize) -> Result<Self> {
        Ok(Self)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        crate::bail!("cannot seed the CPU rng with set_seed")
    }

    fn get_current_seed(&self) -> Result<u64> {
        crate::bail!("cannot get the CPU rng seed with get_current_seed")
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<CpuStorage> {
        use rand::prelude::*;

        let elem_count = shape.elem_count();
        let mut rng = rand::rng();
        match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F6E2M3
            | DType::F6E3M2
            | DType::F4
            | DType::F8E8M0 => Err(Error::UnsupportedDTypeForOp(dtype, "rand_uniform").bt()),
            DType::BF16 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand::distr::Uniform::new(bf16::from_f64(min), bf16::from_f64(max))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(rng.sample::<bf16, _>(uniform))
                }
                Ok(CpuStorage::BF16(data))
            }
            DType::F16 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand::distr::Uniform::new(f16::from_f64(min), f16::from_f64(max))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(rng.sample::<f16, _>(uniform))
                }
                Ok(CpuStorage::F16(data))
            }
            DType::F8E4M3 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform =
                    rand::distr::Uniform::new(F8E4M3::from_f64(min), F8E4M3::from_f64(max))
                        .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(rng.sample::<F8E4M3, _>(uniform))
                }
                Ok(CpuStorage::F8E4M3(data))
            }
            DType::F32 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform =
                    rand::distr::Uniform::new(min as f32, max as f32).map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(rng.sample::<f32, _>(uniform))
                }
                Ok(CpuStorage::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand::distr::Uniform::new(min, max).map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(rng.sample::<f64, _>(uniform))
                }
                Ok(CpuStorage::F64(data))
            }
        }
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CpuStorage> {
        use rand::prelude::*;

        let elem_count = shape.elem_count();
        let mut rng = rand::rng();
        match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F6E2M3
            | DType::F6E3M2
            | DType::F4
            | DType::F8E8M0 => Err(Error::UnsupportedDTypeForOp(dtype, "rand_normal").bt()),
            DType::BF16 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(bf16::from_f64(mean), bf16::from_f64(std))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::BF16(data))
            }
            DType::F16 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(f16::from_f64(mean), f16::from_f64(std))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F16(data))
            }
            DType::F8E4M3 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(F8E4M3::from_f64(mean), F8E4M3::from_f64(std))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F8E4M3(data))
            }
            DType::F32 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal =
                    rand_distr::Normal::new(mean as f32, std as f32).map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(mean, std).map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F64(data))
            }
        }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<CpuStorage> {
        let elem_count = shape.elem_count();
        // The code below is highly unsafe but hopefully not directly unsound as we only consider
        // types that are Copy, not Drop, and for which all bit patterns are proper values.
        // It's still pretty risky, see the following for more details:
        // https://github.com/rust-lang/rust-clippy/issues/4483
        let storage = match dtype {
            DType::U8 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::U8(v)
            }
            DType::U32 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::U32(v)
            }
            DType::I16 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::I16(v)
            }
            DType::I32 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::I32(v)
            }
            DType::I64 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::I64(v)
            }
            DType::BF16 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::BF16(v)
            }
            DType::F16 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::F16(v)
            }
            DType::F32 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::F32(v)
            }
            DType::F64 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::F64(v)
            }
            DType::F8E4M3 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::F8E4M3(v)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(Error::UnsupportedDTypeForOp(dtype, "alloc_uninit").bt())
            }
        };
        Ok(storage)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CpuStorage> {
        let elem_count = shape.elem_count();
        let storage = match dtype {
            DType::U8 => CpuStorage::U8(vec![0u8; elem_count]),
            DType::U32 => CpuStorage::U32(vec![0u32; elem_count]),
            DType::I16 => CpuStorage::I16(vec![0i16; elem_count]),
            DType::I32 => CpuStorage::I32(vec![0i32; elem_count]),
            DType::I64 => CpuStorage::I64(vec![0i64; elem_count]),
            DType::BF16 => CpuStorage::BF16(vec![bf16::ZERO; elem_count]),
            DType::F16 => CpuStorage::F16(vec![f16::ZERO; elem_count]),
            DType::F32 => CpuStorage::F32(vec![0f32; elem_count]),
            DType::F64 => CpuStorage::F64(vec![0f64; elem_count]),
            DType::F8E4M3 => CpuStorage::F8E4M3(vec![F8E4M3::ZERO; elem_count]),
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(Error::UnsupportedDTypeForOp(dtype, "zeros").bt())
            }
        };
        Ok(storage)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

#[macro_export]
macro_rules! map_dtype {
    ($name:expr, $storage:ident, $fn:expr, ($($dtypes:ident),+)) => {
        match $storage {
            $(CpuStorage::$dtypes(__e) => CpuStorage::$dtypes($fn(__e)),)*
            s => Err(Error::UnsupportedDTypeForOp(s.dtype(), $name).bt())?,
        }
    };
}
