//! Lazy port of `fuel-nn`'s loss functions.
//!
//! Each loss is a small composition of [`LazyTensor`] primitives and
//! returns a scalar (or per-sample) `LazyTensor` describing the loss.
//! Numerical conventions match the eager fuel-nn implementations
//! verbatim; the [`Reduction`] enum mirrors PyTorch's
//! `'mean' | 'sum' | 'none'` parameter shape.

use crate::Result;
use crate::lazy::LazyTensor;
use fuel_ir::{DType, Shape};

/// Reduction mode for losses with per-sample outputs. Matches
/// PyTorch's `reduction` parameter shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Reduction {
    /// Sum the per-sample losses and divide by the sample count.
    Mean,
    /// Sum the per-sample losses without dividing.
    Sum,
    /// No reduction — return the per-sample loss tensor.
    None,
}

impl Reduction {
    fn apply(self, per_sample: LazyTensor, n: usize) -> Result<LazyTensor> {
        match self {
            Reduction::None => Ok(per_sample),
            Reduction::Sum => Ok(per_sample.sum_all()),
            Reduction::Mean => {
                let denom = (n.max(1)) as f64;
                Ok(per_sample.sum_all().mul_scalar(1.0 / denom))
            }
        }
    }
}

/// Negative log likelihood loss.
///
/// `inp` is `[N, C]` log-probabilities; `target` is `[N]` `U32`
/// class labels. Picks `inp[i, target[i]]` for every row via
/// gather, then negates and reduces.
pub fn nll(
    inp: &LazyTensor,
    target: &LazyTensor,
    reduction: Reduction,
) -> Result<LazyTensor> {
    let inp_dims = inp.shape();
    let inp_dims = inp_dims.dims();
    if inp_dims.len() != 2 {
        return Err(crate::Error::Msg(format!(
            "nll: inp must be rank 2 [N, C], got {inp_dims:?}",
        ))
        .bt());
    }
    let target_dims = target.shape();
    let target_dims = target_dims.dims();
    if target_dims.len() != 1 {
        return Err(crate::Error::Msg(format!(
            "nll: target must be rank 1 [N], got {target_dims:?}",
        ))
        .bt());
    }
    if inp_dims[0] != target_dims[0] {
        return Err(crate::Error::Msg(format!(
            "nll: batch size mismatch — inp[0]={} target[0]={}",
            inp_dims[0], target_dims[0],
        ))
        .bt());
    }
    if target.dtype() != DType::U32 {
        return Err(crate::Error::Msg(format!(
            "nll: target must be U32, got {:?}",
            target.dtype(),
        ))
        .bt());
    }
    let n = inp_dims[0];

    let idx = target.unsqueeze(1_usize)?;
    let picked = inp.gather(1_usize, &idx)?;
    let per_sample = picked.squeeze(1_usize)?.neg();
    reduction.apply(per_sample, n)
}

/// Cross-entropy loss with integer class labels.
///
/// `inp` is `[N, C]` raw logits; `target` is `[N]` `I64` class
/// labels (matching PyTorch's `CrossEntropyLoss` convention).
/// Routes to the shipped [`fused_softmax_cross_entropy`] fused
/// op (FusedOpId 17) which collapses log-softmax + NLL into a
/// single graph node.
///
/// [`fused_softmax_cross_entropy`]: LazyTensor::fused_softmax_cross_entropy
pub fn cross_entropy(
    inp: &LazyTensor,
    target: &LazyTensor,
    reduction: Reduction,
) -> Result<LazyTensor> {
    let inp_dims = inp.shape();
    let inp_dims = inp_dims.dims();
    if inp_dims.len() != 2 {
        return Err(crate::Error::Msg(format!(
            "cross_entropy: inp must be rank 2 [N, C], got {inp_dims:?}",
        ))
        .bt());
    }
    let target_dims = target.shape();
    let target_dims = target_dims.dims();
    if target_dims.len() != 1 {
        return Err(crate::Error::Msg(format!(
            "cross_entropy: target must be rank 1 [N], got {target_dims:?}",
        ))
        .bt());
    }
    if inp_dims[0] != target_dims[0] {
        return Err(crate::Error::Msg(format!(
            "cross_entropy: batch size mismatch — inp[0]={} target[0]={}",
            inp_dims[0], target_dims[0],
        ))
        .bt());
    }
    if target.dtype() != DType::I64 {
        return Err(crate::Error::Msg(format!(
            "cross_entropy: target must be I64 (PyTorch convention), got {:?}",
            target.dtype(),
        ))
        .bt());
    }
    let fused_reduction = match reduction {
        Reduction::Mean => fuel_graph::registry::Reduction::Mean,
        Reduction::Sum => fuel_graph::registry::Reduction::Sum,
        Reduction::None => fuel_graph::registry::Reduction::None,
    };
    Ok(inp.fused_softmax_cross_entropy(target, fused_reduction, i64::MIN))
}

/// Binary cross-entropy from logits, numerically stable.
///
/// Formula: `max(x, 0) - x*y + log(1 + exp(-|x|))`. Operates
/// element-wise on tensors of identical shape and produces a
/// scalar (or shape-preserving tensor for [`Reduction::None`]).
pub fn binary_cross_entropy_with_logit(
    inp: &LazyTensor,
    target: &LazyTensor,
    reduction: Reduction,
) -> Result<LazyTensor> {
    if inp.shape().dims() != target.shape().dims() {
        return Err(crate::Error::Msg(format!(
            "bce_with_logit: inp shape {:?} != target shape {:?}",
            inp.shape().dims(),
            target.shape().dims(),
        ))
        .bt());
    }
    if inp.dtype() != target.dtype() {
        return Err(crate::Error::Msg(format!(
            "bce_with_logit: dtype mismatch — inp={:?} target={:?}",
            inp.dtype(),
            target.dtype(),
        ))
        .bt());
    }
    // max(x, 0) - x*y + log(1 + exp(-|x|))
    let relu_inp = inp.relu();
    let neg_abs_inp = inp.abs().neg();
    let xy = inp.mul(target)?;
    let log_term = neg_abs_inp.exp().add_scalar(1.0).log();
    let per_elem = relu_inp.sub(&xy)?.add(&log_term)?;
    let n = inp.elem_count();
    reduction.apply(per_elem, n)
}

/// Mean squared error.
///
/// `mean((inp - target)^2)` for the default reduction; the loss
/// is element-wise `(inp - target)^2` otherwise.
pub fn mse(
    inp: &LazyTensor,
    target: &LazyTensor,
    reduction: Reduction,
) -> Result<LazyTensor> {
    if inp.shape().dims() != target.shape().dims() {
        return Err(crate::Error::Msg(format!(
            "mse: inp shape {:?} != target shape {:?}",
            inp.shape().dims(),
            target.shape().dims(),
        ))
        .bt());
    }
    if inp.dtype() != target.dtype() {
        return Err(crate::Error::Msg(format!(
            "mse: dtype mismatch — inp={:?} target={:?}",
            inp.dtype(),
            target.dtype(),
        ))
        .bt());
    }
    let diff = inp.sub(target)?;
    let per_elem = diff.sqr();
    let n = inp.elem_count();
    reduction.apply(per_elem, n)
}

/// Huber (smooth-L1) loss.
///
/// Quadratic `0.5 * (x - y)^2` for `|x - y| < delta`, linear
/// `delta * (|x - y| - 0.5 * delta)` otherwise. Implemented via
/// element-wise `where_cond` over a `|diff| <= delta` mask.
pub fn huber(
    inp: &LazyTensor,
    target: &LazyTensor,
    delta: f64,
    reduction: Reduction,
) -> Result<LazyTensor> {
    if inp.shape().dims() != target.shape().dims() {
        return Err(crate::Error::Msg(format!(
            "huber: inp shape {:?} != target shape {:?}",
            inp.shape().dims(),
            target.shape().dims(),
        ))
        .bt());
    }
    if inp.dtype() != target.dtype() {
        return Err(crate::Error::Msg(format!(
            "huber: dtype mismatch — inp={:?} target={:?}",
            inp.dtype(),
            target.dtype(),
        ))
        .bt());
    }
    if !(delta > 0.0) {
        return Err(crate::Error::Msg(format!(
            "huber: delta must be > 0, got {delta}",
        ))
        .bt());
    }
    let diff = inp.sub(target)?;
    let abs_diff = diff.abs();
    // mask = (|diff| <= delta) as U8.
    let delta_t = abs_diff_like_scalar(&abs_diff, delta)?;
    let mask = abs_diff.le(&delta_t)?;
    let squared_loss = diff.mul(&diff)?.mul_scalar(0.5);
    let linear_loss = abs_diff.mul_scalar(delta).add_scalar(-0.5 * delta * delta);
    let per_elem = mask.where_cond(&squared_loss, &linear_loss)?;
    let n = inp.elem_count();
    reduction.apply(per_elem, n)
}

/// Build a constant tensor on `host`'s graph filled with `value`,
/// matching `host`'s shape and dtype (limited to the float dtypes
/// the loss functions actually exercise).
fn abs_diff_like_scalar(host: &LazyTensor, value: f64) -> Result<LazyTensor> {
    let shape = host.shape();
    let dims: Vec<usize> = shape.dims().to_vec();
    let n = shape.elem_count();
    let out_shape = Shape::from_dims(&dims);
    match host.dtype() {
        DType::F32 => Ok(host.const_f32_like(vec![value as f32; n], out_shape)),
        DType::F64 => {
            // No `const_f64_like` bridge — go through f32 then cast.
            let t = host.const_f32_like(vec![value as f32; n], out_shape);
            t.to_dtype(DType::F64)
        }
        DType::BF16 => Ok(host.const_bf16_like(
            vec![half::bf16::from_f64(value); n],
            out_shape,
        )),
        DType::F16 => Ok(host.const_f16_like(
            vec![half::f16::from_f64(value); n],
            out_shape,
        )),
        other => Err(crate::Error::Msg(format!(
            "huber: unsupported dtype {other:?}",
        ))
        .bt()),
    }
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
    fn cross_entropy_matches_eager_hand_computed_2x3() {
        // Two samples, three classes. Hand-computed expected:
        //   row 0 logits = [2, 1, 0.1]; target = 0
        //     logsumexp = log(e^2 + e^1 + e^0.1)
        //                = log(7.389056 + 2.718282 + 1.105171)
        //                = log(11.212509) ≈ 2.417067
        //     loss_0 = -(2.0 - 2.417067) = 0.417067
        //   row 1 logits = [0.5, 2.5, 0.3]; target = 1
        //     logsumexp = log(e^0.5 + e^2.5 + e^0.3)
        //                = log(1.648721 + 12.182494 + 1.349859)
        //                = log(15.181074) ≈ 2.720152
        //     loss_1 = -(2.5 - 2.720152) = 0.220152
        //   mean = (0.417067 + 0.220152) / 2 ≈ 0.318609
        let device = Device::cpu();
        let logits = LazyTensor::from_f32(
            vec![2.0_f32, 1.0, 0.1, 0.5, 2.5, 0.3],
            Shape::from_dims(&[2, 3]),
            &device,
        );
        let target = logits.const_i64_like(vec![0_i64, 1], Shape::from_dims(&[2]));
        let loss = cross_entropy(&logits, &target, Reduction::Mean)
            .unwrap()
            .realize_f32();
        assert_eq!(loss.len(), 1);
        assert!(
            approx_eq(loss[0], 0.318609, 1e-4),
            "got {} expected ~0.318609",
            loss[0],
        );
    }

    #[test]
    fn nll_matches_textbook_formula() {
        // log_probs[i, target[i]] gives [-0.1054, -0.1054]; mean
        // of -(log probs) ≈ 0.1054.
        let device = Device::cpu();
        let log_probs = LazyTensor::from_f32(
            vec![
                -0.1054_f32, -2.3026, -6.9078,
                -2.3026,    -0.1054, -6.9078,
            ],
            Shape::from_dims(&[2, 3]),
            &device,
        );
        let targets = log_probs.const_u32_like(vec![0_u32, 1], Shape::from_dims(&[2]));
        let loss = nll(&log_probs, &targets, Reduction::Mean)
            .unwrap()
            .realize_f32();
        assert_eq!(loss.len(), 1);
        assert!(
            approx_eq(loss[0], 0.1054, 1e-5),
            "got {} expected ~0.1054",
            loss[0],
        );
    }

    #[test]
    fn mse_zero_on_equal_inputs() {
        let device = Device::cpu();
        let a = LazyTensor::from_f32(
            vec![0.5_f32, -1.0, 2.0, 3.5],
            Shape::from_dims(&[4]),
            &device,
        );
        let b = a.const_f32_like(
            vec![0.5_f32, -1.0, 2.0, 3.5],
            Shape::from_dims(&[4]),
        );
        let loss = mse(&a, &b, Reduction::Mean).unwrap().realize_f32();
        assert_eq!(loss.len(), 1);
        assert!(loss[0].abs() < 1e-7, "got {} expected ~0", loss[0]);
    }

    #[test]
    fn mse_unit_on_unit_diff() {
        // Every element differs by exactly 1.0 ⇒ mean((1.0)^2) = 1.0.
        let device = Device::cpu();
        let a = LazyTensor::from_f32(
            vec![1.0_f32, 2.0, 3.0, 4.0],
            Shape::from_dims(&[4]),
            &device,
        );
        let b = a.const_f32_like(
            vec![0.0_f32, 1.0, 2.0, 3.0],
            Shape::from_dims(&[4]),
        );
        let loss = mse(&a, &b, Reduction::Mean).unwrap().realize_f32();
        assert_eq!(loss.len(), 1);
        assert!(
            approx_eq(loss[0], 1.0, 1e-6),
            "got {} expected 1.0",
            loss[0],
        );
    }

    #[test]
    fn bce_with_logit_matches_sigmoid_then_nll() {
        // BCE_with_logits(x, y) = max(x, 0) - x*y + log(1 + exp(-|x|))
        // Per-element hand-computed:
        //  (x=1.0, y=1.0): max=1.0, -x*y=-1.0, log(1+e^-1)= log(1.367879) = 0.313262
        //                   sum = 0.313262
        //  (x=-1.0, y=0.0): max=0.0, -x*y=0,   log(1+e^-1)= 0.313262
        //                   sum = 0.313262
        //  (x=0.0, y=1.0):  max=0.0, -x*y=0,   log(1+e^0)  = log(2) ≈ 0.693147
        //                   sum = 0.693147
        // mean = (0.313262 + 0.313262 + 0.693147) / 3 ≈ 0.439890
        let device = Device::cpu();
        let logits = LazyTensor::from_f32(
            vec![1.0_f32, -1.0, 0.0],
            Shape::from_dims(&[3]),
            &device,
        );
        let targets = logits.const_f32_like(
            vec![1.0_f32, 0.0, 1.0],
            Shape::from_dims(&[3]),
        );
        let loss = binary_cross_entropy_with_logit(
            &logits, &targets, Reduction::Mean,
        )
        .unwrap()
        .realize_f32();
        assert_eq!(loss.len(), 1);
        assert!(
            approx_eq(loss[0], 0.439890, 1e-5),
            "got {} expected ~0.439890",
            loss[0],
        );
    }

    #[test]
    fn huber_quadratic_under_delta() {
        // All |diff|=0.5 < delta=1.0 ⇒ 0.5 * 0.25 = 0.125 each;
        // mean = 0.125.
        let device = Device::cpu();
        let inp = LazyTensor::from_f32(
            vec![0.5_f32, 1.5, 2.5, 3.5],
            Shape::from_dims(&[4]),
            &device,
        );
        let tgt = inp.const_f32_like(
            vec![0.0_f32, 1.0, 2.0, 3.0],
            Shape::from_dims(&[4]),
        );
        let loss = huber(&inp, &tgt, 1.0, Reduction::Mean)
            .unwrap()
            .realize_f32();
        assert_eq!(loss.len(), 1);
        assert!(
            approx_eq(loss[0], 0.125, 1e-6),
            "got {} expected 0.125",
            loss[0],
        );
    }

    #[test]
    fn huber_linear_over_delta() {
        // Mix: |diff|=0.5 < 1.0 ⇒ 0.5 * 0.25 = 0.125 (quad branch)
        //      |diff|=2.0 ≥ 1.0 ⇒ 1.0 * (2.0 - 0.5) = 1.5 (linear branch)
        // mean = (0.125 + 1.5) / 2 = 0.8125
        let device = Device::cpu();
        let inp = LazyTensor::from_f32(
            vec![0.5_f32, 3.0],
            Shape::from_dims(&[2]),
            &device,
        );
        let tgt = inp.const_f32_like(
            vec![1.0_f32, 1.0],
            Shape::from_dims(&[2]),
        );
        let loss = huber(&inp, &tgt, 1.0, Reduction::Mean)
            .unwrap()
            .realize_f32();
        assert_eq!(loss.len(), 1);
        assert!(
            approx_eq(loss[0], 0.8125, 1e-4),
            "got {} expected 0.8125",
            loss[0],
        );
    }
}
