// SPDX-License-Identifier: MIT OR Apache-2.0
//! The axis-reduction half of the accuracy-claim seeders (ROADMAP row 39): `SumReduce`,
//! `MaxReduce`, `MinReduce` and `MeanReduce` over f32, each driven through the REGISTERED kernel
//! and compared per element with an `f64` host reduction of the same `f32` inputs. Backend-free
//! like [`super::accuracy_common`], whose claim, case and record types it reuses.
//!
//! The cases cover the layouts a reduce kernel branches on: the last axis (decode and prefill
//! rows), the first axis (a strided walk), a middle axis, several axes at once and a full
//! reduction, each with and without `keepdim`.

use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;

use super::accuracy_common::{Claim, OpAccuracy, Runner, case, f32_tensor, scaled};
use super::bit_stability::VerifyError;
use crate::kernel::{BindingEntry, OpParams};

/// `(shape, reduced dims, keepdim)` of each measured case.
pub(crate) const REDUCE_CASES: &[(&[usize], &[usize], bool)] = &[
    (&[1, 1024], &[1], false),
    (&[16, 3072], &[1], false),
    (&[448, 1024], &[1], true),
    (&[4, 8192], &[1], false),
    (&[1, 65536], &[1], false),
    (&[1024, 64], &[0], false),
    (&[8, 64, 128], &[1], false),
    (&[8, 64, 128], &[0, 2], true),
    (&[8, 64, 128], &[0, 1, 2], false),
];

/// How a reduction folds the elements of one output cell.
#[derive(Clone, Copy)]
pub(crate) enum Fold {
    Sum,
    Max,
    Min,
    Mean,
}

impl Fold {
    pub(crate) fn op(self) -> OpKind {
        match self {
            Fold::Sum => OpKind::SumReduce,
            Fold::Max => OpKind::MaxReduce,
            Fold::Min => OpKind::MinReduce,
            Fold::Mean => OpKind::MeanReduce,
        }
    }

    fn reference_label(self) -> &'static str {
        match self {
            Fold::Sum => "f64 sum of the same f32 inputs",
            Fold::Max => "f64 max of the same f32 inputs",
            Fold::Min => "f64 min of the same f32 inputs",
            Fold::Mean => "f64 sum / count of the same f32 inputs",
        }
    }
}

/// Shape after reducing `dims` (kept as size 1 when `keepdim`).
fn reduced_shape(shape: &[usize], dims: &[usize], keepdim: bool) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| match (dims.contains(&i), keepdim) {
            (false, _) => Some(d),
            (true, true) => Some(1),
            (true, false) => None,
        })
        .collect()
}

/// `f64` host reduction of a row-major `x` of `shape` over `dims`, laid out in the order of the
/// reduced shape (kept dims in their original order).
pub(crate) fn reference_reduce(x: &[f32], shape: &[usize], dims: &[usize], fold: Fold) -> Vec<f64> {
    let kept: Vec<usize> = (0..shape.len()).filter(|i| !dims.contains(i)).collect();
    let out_len: usize = kept.iter().map(|&i| shape[i]).product();
    let count: usize = dims.iter().map(|&i| shape[i]).product();
    let init = match fold {
        Fold::Max => f64::NEG_INFINITY,
        Fold::Min => f64::INFINITY,
        Fold::Sum | Fold::Mean => 0.0,
    };
    let mut out = vec![init; out_len];
    let mut idx = vec![0usize; shape.len()];
    for &v in x {
        let v = v as f64;
        let mut cell = 0usize;
        for &k in &kept {
            cell = cell * shape[k] + idx[k];
        }
        out[cell] = match fold {
            Fold::Sum | Fold::Mean => out[cell] + v,
            Fold::Max => out[cell].max(v),
            Fold::Min => out[cell].min(v),
        };
        for axis in (0..shape.len()).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
    if matches!(fold, Fold::Mean) {
        out.iter_mut().for_each(|o| *o /= count as f64);
    }
    out
}

/// Measure one reduction: the f32 kernel in `entry` over [`REDUCE_CASES`], inputs uniform in
/// `[-1, 1)`.
pub(crate) fn measure_reduce(
    run: &Runner,
    entry: &BindingEntry,
    fold: Fold,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32];
    let mut cases = Vec::new();
    for (idx, &(shape, dims, keepdim)) in REDUCE_CASES.iter().enumerate() {
        let n: usize = shape.iter().product();
        let x = scaled(n, 0x8000 + idx as u64, 1.0);
        let params = OpParams::Reduce {
            dims: dims.to_vec(),
            keepdim,
        };
        let got = run(
            entry,
            params,
            &[f32_tensor(shape.to_vec(), &x)],
            reduced_shape(shape, dims, keepdim),
        )?;
        let want = reference_reduce(&x, shape, dims, fold);
        cases.push(case(
            format!("shape={shape:?} dims={dims:?} keepdim={keepdim}"),
            &got,
            &want,
        ));
    }
    Ok(OpAccuracy::new(
        fold.op(),
        &dt,
        claim,
        entry,
        cases,
        fold.reference_label(),
        "uniform [-1, 1)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_reduces_the_named_axes() {
        // [[1, 2, 3], [4, 5, 6]]
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(reference_reduce(&x, &[2, 3], &[1], Fold::Sum), [6.0, 15.0]);
        assert_eq!(
            reference_reduce(&x, &[2, 3], &[0], Fold::Sum),
            [5.0, 7.0, 9.0]
        );
        assert_eq!(reference_reduce(&x, &[2, 3], &[1], Fold::Max), [3.0, 6.0]);
        assert_eq!(
            reference_reduce(&x, &[2, 3], &[0], Fold::Min),
            [1.0, 2.0, 3.0]
        );
        assert_eq!(reference_reduce(&x, &[2, 3], &[1], Fold::Mean), [2.0, 5.0]);
        assert_eq!(reference_reduce(&x, &[2, 3], &[0, 1], Fold::Sum), [21.0]);
    }

    #[test]
    fn reduced_shape_drops_or_keeps_the_reduced_axes() {
        assert_eq!(reduced_shape(&[8, 64, 128], &[0, 2], true), [1, 64, 1]);
        assert_eq!(reduced_shape(&[8, 64, 128], &[0, 2], false), [64]);
        assert_eq!(
            reduced_shape(&[8, 64, 128], &[0, 1, 2], false),
            Vec::<usize>::new()
        );
    }

    /// The retained sabotage sibling of the measurement: a runner that returns the exact `f32`
    /// rounding of the reference measures (almost) no error, and the same runner with one
    /// element perturbed measures that perturbation. Without the second half a harness that
    /// compared the reference with itself would pass every claim.
    #[test]
    fn the_measurement_sees_a_perturbed_output() {
        use super::super::bit_stability::HostTensor;
        let mut table = crate::kernel::KernelBindingTable::new();
        crate::baracuda_dispatch::register_baracuda_cuda_kernels(&mut table);
        let entry = super::super::accuracy_common::find_entry(
            &table,
            OpKind::MaxReduce,
            &[DType::F32, DType::F32],
            fuel_ir::probe::BackendId::Cuda,
        )
        .expect("max f32 entry");
        let host = |perturb: f32| {
            move |_e: &BindingEntry,
                  p: OpParams,
                  ins: &[HostTensor],
                  _s: Vec<usize>|
                  -> Result<Vec<f32>, VerifyError> {
                let OpParams::Reduce { dims, .. } = p else {
                    panic!("reduce params expected");
                };
                let x = super::super::accuracy_common::f32s(&ins[0]);
                let mut out: Vec<f32> = reference_reduce(&x, &ins[0].shape, &dims, Fold::Max)
                    .into_iter()
                    .map(|v| v as f32)
                    .collect();
                out[0] += perturb;
                Ok(out)
            }
        };
        let claim = Claim {
            max_relative: 1e-6,
            max_absolute: 1e-6,
        };
        let exact = measure_reduce(&host(0.0), entry, Fold::Max, claim).expect("measures");
        assert_eq!(exact.worst_abs, 0.0);
        let off = measure_reduce(&host(0.5), entry, Fold::Max, claim).expect("measures");
        assert!(off.worst_abs >= 0.5, "perturbation seen: {}", off.worst_abs);
        assert!(off.worst_abs > claim.max_absolute);
    }
}
