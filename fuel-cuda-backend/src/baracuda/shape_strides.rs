//! Layout → baracuda `(rank, shape, stride)` triples.
//!
//! Baracuda's strided kernels take `rank: i32` + `*const i32` shape +
//! `*const i64` stride (in elements). Fuel's [`Layout`] carries
//! `Vec<usize>` dims + `Vec<isize>` strides. This module bridges them.
//!
//! All conversions validate that each dim fits in `i32` and report
//! [`crate::CudaError::BaracudaShapeOverflow`] on overflow — far better
//! than letting a silent `as i32` cast wrap to a negative number that
//! baracuda's kernel would then reject as "invalid problem."

use fuel_ir::{Error, Layout, Result};

use crate::error::CudaError;

/// Per-tensor shape + stride buffers in baracuda's expected types.
///
/// Strides are in **elements** (not bytes) to match baracuda's
/// convention. Owned `Vec`s so the pointers stay live for the kernel
/// launch.
pub struct ShapeStridesI32 {
    pub rank: i32,
    pub shape: Vec<i32>,
    pub stride_x: Vec<i64>,
    pub stride_y: Vec<i64>,
}

impl ShapeStridesI32 {
    /// Build from one input layout `x_layout` and one output `y_layout`.
    /// Used by elementwise / softmax / norm kernels that take separate
    /// input and output stride buffers (so non-contiguous inputs can
    /// write into a contiguous output).
    ///
    /// Both layouts must share the same shape; `op_label` is folded
    /// into any overflow / mismatch error.
    pub fn from_two(
        x_layout: &Layout,
        y_layout: &Layout,
        op_label: &'static str,
    ) -> Result<Self> {
        let dims = x_layout.shape().dims();
        let y_dims = y_layout.shape().dims();
        if dims != y_dims {
            return Err(Error::Msg(format!(
                "{op_label}: x shape {dims:?} doesn't match y shape {y_dims:?}",
            ))
            .bt());
        }
        let mut shape = Vec::with_capacity(dims.len());
        for (i, &d) in dims.iter().enumerate() {
            shape.push(i32::try_from(d).map_err(|_| {
                Error::cuda(CudaError::BaracudaShapeOverflow {
                    op: op_label,
                    dim_index: i,
                    dim_value: d,
                })
            })?);
        }
        let stride_x: Vec<i64> = x_layout.stride().iter().map(|&s| s as i64).collect();
        let stride_y: Vec<i64> = y_layout.stride().iter().map(|&s| s as i64).collect();
        Ok(Self {
            rank: dims.len() as i32,
            shape,
            stride_x,
            stride_y,
        })
    }

    /// Total element count (`numel`). Convenience derived from the
    /// shape so the kernel-launch site doesn't recompute it.
    pub fn numel(&self) -> i64 {
        self.shape.iter().map(|&d| d as i64).product()
    }
}
