//! Lazy port of `fuel-nn`'s one-hot encoding helper.
//!
//! Given a `U32` [`LazyTensor`] of label indices of arbitrary shape
//! and a `num_classes` depth, produces a float [`LazyTensor`] of
//! shape `(...input_shape..., num_classes)` carrying `on_value` at
//! each label position and `off_value` everywhere else.
//!
//! Semantics: built entirely at lazy-graph-build time as a small
//! composition of existing primitives — a `[num_classes,
//! num_classes]` lookup table (each row `i` has `on_value` at column
//! `i` and `off_value` elsewhere) followed by an [`index_select`]
//! along dim 0 with the flattened label vector, then a final
//! [`reshape`] back to the trailing-class output shape. No new
//! kernel; output dtype is always `F32`.
//!
//! v1 scope:
//!   - Index dtype: `U32` only. PyTorch's `i64` / TensorFlow's `u8`
//!     entry points can be added later by widening the dtype gate;
//!     callers that hold `i64` labels can [`to_dtype`] to `U32`
//!     first.
//!   - Negative-sentinel rows ("`-1` ⇒ all-off") are not supported
//!     here. The eager fuel-nn helper accepts `-1` because it works
//!     on signed inputs; the lazy version stays U32-only and rejects
//!     out-of-range labels structurally (the lookup table has
//!     exactly `num_classes` rows, so any label `>= num_classes`
//!     would surface as a realize-time index-out-of-bounds from
//!     `index_select` rather than a build-time check).
//!   - Output dtype is `F32`. Other dtypes (`U8` one-cold etc.) can
//!     be obtained via [`to_dtype`] post-hoc.
//!
//! [`index_select`]: LazyTensor::index_select
//! [`reshape`]: LazyTensor::reshape
//! [`to_dtype`]: LazyTensor::to_dtype

use crate::Result;
use crate::lazy::LazyTensor;
use fuel_ir::{DType, Shape};
use std::sync::Arc;

/// Build a one-hot (or one-cold) encoding of `labels` along a new
/// trailing dimension of size `num_classes`.
///
/// `labels` must be `U32` with arbitrary rank (including rank-0 /
/// scalar labels). The output has shape
/// `(...labels.shape()..., num_classes)` and dtype `F32`. Each
/// element along the trailing dim is `on_value` at the label
/// position and `off_value` elsewhere.
///
/// Common parameter pairings:
///   - `on_value = 1.0, off_value = 0.0` — classic one-hot encoding.
///   - `on_value = 0.0, off_value = 1.0` — "one-cold" (every entry
///     hot except the label position).
///
/// Frequently consumed by label-based losses (cross-entropy with
/// label smoothing, label-mixed augmentations, soft-target
/// distillation) when a dense per-class target tensor is needed
/// downstream.
pub fn one_hot(
    labels: &LazyTensor,
    num_classes: usize,
    on_value: f32,
    off_value: f32,
) -> Result<LazyTensor> {
    if labels.dtype() != DType::U32 {
        return Err(crate::Error::Msg(format!(
            "one_hot: labels must be U32, got {:?}",
            labels.dtype(),
        ))
        .bt());
    }
    if num_classes == 0 {
        return Err(crate::Error::Msg(
            "one_hot: num_classes must be ≥ 1".into(),
        )
        .bt());
    }

    let labels_shape = labels.shape();
    let labels_dims: Vec<usize> = labels_shape.dims().to_vec();
    let n: usize = labels_shape.elem_count();

    // Build the `[num_classes, num_classes]` lookup table. Row `i`
    // has `on_value` at column `i` and `off_value` elsewhere — a
    // scaled identity matrix in disguise.
    let mut table: Vec<f32> = vec![off_value; num_classes * num_classes];
    for i in 0..num_classes {
        table[i * num_classes + i] = on_value;
    }
    let table_data: Arc<[f32]> = Arc::<[f32]>::from(table);
    let table_t = labels.const_f32_like(
        table_data,
        Shape::from_dims(&[num_classes, num_classes]),
    );

    // index_select wants a rank-1 U32 index tensor. Flatten the
    // label tensor; if it's already rank-1 the reshape is a
    // metadata no-op.
    let flat_labels = labels.reshape(Shape::from_dims(&[n]))?;
    let picked = table_t.index_select(0_usize, &flat_labels)?;
    // `picked` is `[n, num_classes]`; restore the leading dims.
    let mut out_dims = labels_dims;
    out_dims.push(num_classes);
    picked.reshape(Shape::from_dims(&out_dims))
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn one_hot_rank1_classic_1_0_pattern() {
        // labels = [0, 2, 1], num_classes = 3 → output shape [3, 3]:
        //   [[1, 0, 0],
        //    [0, 0, 1],
        //    [0, 1, 0]]
        let device = Device::cpu();
        let labels = LazyTensor::from_u32(
            vec![0_u32, 2, 1],
            Shape::from_dims(&[3]),
            &device,
        );
        let oh = one_hot(&labels, 3, 1.0, 0.0).unwrap();
        assert_eq!(oh.shape().dims(), &[3, 3]);
        let v = oh.realize_f32();
        let expected = [
            1.0_f32, 0.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
        ];
        assert_eq!(v.len(), expected.len());
        for (i, (g, e)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*g, *e, 1e-7),
                "one_hot mismatch at {i}: got {g} expected {e}",
            );
        }
    }

    #[test]
    fn one_hot_rank2_preserves_leading_dims_and_class_dim() {
        // labels = [[0, 1], [2, 0]], num_classes = 3 → output
        // shape [2, 2, 3]. Hand-computed:
        //   [[[1, 0, 0], [0, 1, 0]],
        //    [[0, 0, 1], [1, 0, 0]]]
        let device = Device::cpu();
        let labels = LazyTensor::from_u32(
            vec![0_u32, 1, 2, 0],
            Shape::from_dims(&[2, 2]),
            &device,
        );
        let oh = one_hot(&labels, 3, 1.0, 0.0).unwrap();
        assert_eq!(oh.shape().dims(), &[2, 2, 3]);
        let v = oh.realize_f32();
        let expected = [
            1.0_f32, 0.0, 0.0,  0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,      1.0, 0.0, 0.0,
        ];
        assert_eq!(v.len(), expected.len());
        for (i, (g, e)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*g, *e, 1e-7),
                "one_hot mismatch at {i}: got {g} expected {e}",
            );
        }
    }

    #[test]
    fn one_hot_one_cold_pattern_inverts_on_off() {
        // on_value = 0.0, off_value = 1.0 produces a complement of
        // the classic one-hot pattern.
        //   labels = [1, 0], num_classes = 3 → output [2, 3]:
        //     [[1, 0, 1],
        //      [0, 1, 1]]
        let device = Device::cpu();
        let labels = LazyTensor::from_u32(
            vec![1_u32, 0],
            Shape::from_dims(&[2]),
            &device,
        );
        let oc = one_hot(&labels, 3, 0.0, 1.0).unwrap();
        assert_eq!(oc.shape().dims(), &[2, 3]);
        let v = oc.realize_f32();
        let expected = [
            1.0_f32, 0.0, 1.0,
            0.0, 1.0, 1.0,
        ];
        assert_eq!(v.len(), expected.len());
        for (i, (g, e)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*g, *e, 1e-7),
                "one_cold mismatch at {i}: got {g} expected {e}",
            );
        }
    }

    #[test]
    fn one_hot_rejects_non_u32_labels() {
        let device = Device::cpu();
        // Build a labels tensor with the wrong dtype (I64) via
        // const_i64_like off a U32 source — exercises the
        // build-time dtype gate.
        let probe = LazyTensor::from_u32(
            vec![0_u32],
            Shape::from_dims(&[1]),
            &device,
        );
        let bad = probe.const_i64_like(vec![0_i64], Shape::from_dims(&[1]));
        let err = one_hot(&bad, 3, 1.0, 0.0);
        assert!(err.is_err(), "one_hot should reject non-U32 labels");
    }

    #[test]
    fn one_hot_rejects_zero_num_classes() {
        let device = Device::cpu();
        let labels = LazyTensor::from_u32(
            vec![0_u32],
            Shape::from_dims(&[1]),
            &device,
        );
        let err = one_hot(&labels, 0, 1.0, 0.0);
        assert!(err.is_err(), "one_hot should reject num_classes == 0");
    }
}
