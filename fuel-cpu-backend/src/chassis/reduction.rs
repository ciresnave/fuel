//! Reduction chassis — one shape/loop pass shared by every
//! per-axis reduce kernel (Sum / Max / Min / Mean, across every
//! supported dtype).
//!
//! ## Design
//!
//! - [`ReduceOp<T>`] is the trait every reduction op implements
//!   for every input dtype `T`. The associated `Acc` type is the
//!   accumulator dtype. For f32/f64 reductions `Acc = T`; for
//!   half-float reductions `Acc = f32` so summing many bf16/f16
//!   values doesn't lose precision (each in-place bf16 add rounds
//!   to ~3 decimal digits — a streaming f32 accumulator gives full
//!   f32 precision up to ~16M elements). The accumulator-promotion
//!   invariant is encoded as the associated type so a future
//!   contributor adding a new low-precision dtype cannot forget it.
//!
//! - [`reduce`] is the single chassis function. It validates the
//!   shapes, walks the input once, folds each element into the
//!   correct output slot's accumulator, then finalizes once per
//!   output slot. All 16 per-(op, dtype) kernel entry-points in
//!   `byte_kernels.rs` are 1-line thunks over this function.
//!
//! - Mean's divide-by-count is encoded in `Mean::finalize`. Mean
//!   also overrides `validate_count` to reject `count == 0` (which
//!   would otherwise produce a NaN result silently). Other ops use
//!   the default `validate_count` which accepts any count.

use bytemuck::Pod;

use crate::byte_storage::CpuStorageBytes;
use fuel_ir::{Error, Result};

// =============================================================================
// Trait
// =============================================================================

/// One reduction operation parameterized by the input element type
/// `T`. Implementations are zero-sized markers (e.g. [`Sum`]).
///
/// The contract: the chassis allocates one accumulator slot per
/// output element, initializes each via [`init`](Self::init), folds
/// every contributing input element into it via [`fold`](Self::fold),
/// then finalizes each slot once via [`finalize`](Self::finalize).
/// The `count` parameter passed to `finalize` is the product of the
/// reduced-dim sizes (i.e. how many inputs contributed to that
/// slot); only Mean reads it.
pub trait ReduceOp<T: Copy> {
    /// Accumulator dtype. For numerically-safe reductions of low-
    /// precision floats, this is wider than `T` (e.g. `Acc = f32`
    /// when `T = bf16`).
    type Acc: Copy;

    /// The accumulator's identity value (e.g. `0` for Sum,
    /// `-INFINITY` for Max).
    fn init() -> Self::Acc;

    /// Fold one input element into the accumulator.
    fn fold(acc: Self::Acc, x: T) -> Self::Acc;

    /// Convert the accumulator into the output dtype after all
    /// contributing inputs have been folded. `count` is the number
    /// of input elements that folded into this slot (= product of
    /// reduced-dim sizes). Unused by Sum / Max / Min; used by Mean.
    fn finalize(acc: Self::Acc, count: usize) -> T;

    /// Reject pathological reduction counts before doing work. The
    /// default accepts any count; Mean overrides to reject
    /// `count == 0` (division by zero).
    ///
    /// `name` is the public kernel name (e.g. `"mean_reduce_f32"`)
    /// so the error message matches the entry-point the caller
    /// invoked.
    fn validate_count(_name: &str, _count: usize) -> Result<()> {
        Ok(())
    }
}

// =============================================================================
// Op markers
// =============================================================================

/// Sum-reduce marker. `Sum::init() = 0`; `Sum::fold = +`; finalize
/// is identity (or narrows half-float accumulator back to its
/// dtype). For bf16/f16, the accumulator is f32.
pub struct Sum;

/// Max-reduce marker. `Max::init() = -INFINITY`; `Max::fold` keeps
/// the larger of the two inputs. Half-float reductions run the
/// extremum in f32 accumulator space for uniform NaN handling, then
/// narrow back.
pub struct Max;

/// Min-reduce marker — mirror of [`Max`] with `+INFINITY` init and
/// the smaller-of-two `fold`.
pub struct Min;

/// Mean-reduce marker. Sums via the same accumulator as [`Sum`]
/// then divides by `count` in `finalize`. Rejects `count == 0` in
/// `validate_count`.
pub struct Mean;

// =============================================================================
// Sum impls
// =============================================================================

impl ReduceOp<f32> for Sum {
    type Acc = f32;
    fn init() -> f32 { 0.0 }
    fn fold(acc: f32, x: f32) -> f32 { acc + x }
    fn finalize(acc: f32, _: usize) -> f32 { acc }
}

impl ReduceOp<f64> for Sum {
    type Acc = f64;
    fn init() -> f64 { 0.0 }
    fn fold(acc: f64, x: f64) -> f64 { acc + x }
    fn finalize(acc: f64, _: usize) -> f64 { acc }
}

impl ReduceOp<half::bf16> for Sum {
    type Acc = f32; // accumulator promotion — load-bearing invariant.
    fn init() -> f32 { 0.0 }
    fn fold(acc: f32, x: half::bf16) -> f32 { acc + x.to_f32() }
    fn finalize(acc: f32, _: usize) -> half::bf16 { half::bf16::from_f32(acc) }
}

impl ReduceOp<half::f16> for Sum {
    type Acc = f32; // accumulator promotion — load-bearing invariant.
    fn init() -> f32 { 0.0 }
    fn fold(acc: f32, x: half::f16) -> f32 { acc + x.to_f32() }
    fn finalize(acc: f32, _: usize) -> half::f16 { half::f16::from_f32(acc) }
}

// =============================================================================
// Max impls
// =============================================================================

impl ReduceOp<f32> for Max {
    type Acc = f32;
    fn init() -> f32 { f32::NEG_INFINITY }
    fn fold(acc: f32, x: f32) -> f32 { acc.max(x) }
    fn finalize(acc: f32, _: usize) -> f32 { acc }
}

impl ReduceOp<f64> for Max {
    type Acc = f64;
    fn init() -> f64 { f64::NEG_INFINITY }
    fn fold(acc: f64, x: f64) -> f64 { acc.max(x) }
    fn finalize(acc: f64, _: usize) -> f64 { acc }
}

impl ReduceOp<half::bf16> for Max {
    type Acc = f32; // extremum in f32 space for uniform NaN handling.
    fn init() -> f32 { f32::NEG_INFINITY }
    fn fold(acc: f32, x: half::bf16) -> f32 { acc.max(x.to_f32()) }
    fn finalize(acc: f32, _: usize) -> half::bf16 { half::bf16::from_f32(acc) }
}

impl ReduceOp<half::f16> for Max {
    type Acc = f32;
    fn init() -> f32 { f32::NEG_INFINITY }
    fn fold(acc: f32, x: half::f16) -> f32 { acc.max(x.to_f32()) }
    fn finalize(acc: f32, _: usize) -> half::f16 { half::f16::from_f32(acc) }
}

// =============================================================================
// Min impls
// =============================================================================

impl ReduceOp<f32> for Min {
    type Acc = f32;
    fn init() -> f32 { f32::INFINITY }
    fn fold(acc: f32, x: f32) -> f32 { acc.min(x) }
    fn finalize(acc: f32, _: usize) -> f32 { acc }
}

impl ReduceOp<f64> for Min {
    type Acc = f64;
    fn init() -> f64 { f64::INFINITY }
    fn fold(acc: f64, x: f64) -> f64 { acc.min(x) }
    fn finalize(acc: f64, _: usize) -> f64 { acc }
}

impl ReduceOp<half::bf16> for Min {
    type Acc = f32;
    fn init() -> f32 { f32::INFINITY }
    fn fold(acc: f32, x: half::bf16) -> f32 { acc.min(x.to_f32()) }
    fn finalize(acc: f32, _: usize) -> half::bf16 { half::bf16::from_f32(acc) }
}

impl ReduceOp<half::f16> for Min {
    type Acc = f32;
    fn init() -> f32 { f32::INFINITY }
    fn fold(acc: f32, x: half::f16) -> f32 { acc.min(x.to_f32()) }
    fn finalize(acc: f32, _: usize) -> half::f16 { half::f16::from_f32(acc) }
}

// =============================================================================
// Mean impls
// =============================================================================
//
// Mean accumulates via the same path as Sum (so the half-float
// accumulator-promotion invariant is preserved) and divides by
// `count` in `finalize`. `validate_count` rejects `count == 0`
// rather than returning silently-NaN outputs.

fn mean_divisor_check(name: &str, count: usize) -> Result<()> {
    if count == 0 {
        return Err(Error::Msg(format!(
            "{name}: divisor zero (reduced dim has size 0)",
        ))
        .bt());
    }
    Ok(())
}

impl ReduceOp<f32> for Mean {
    type Acc = f32;
    fn init() -> f32 { 0.0 }
    fn fold(acc: f32, x: f32) -> f32 { acc + x }
    fn finalize(acc: f32, count: usize) -> f32 { acc / count as f32 }
    fn validate_count(name: &str, count: usize) -> Result<()> {
        mean_divisor_check(name, count)
    }
}

impl ReduceOp<f64> for Mean {
    type Acc = f64;
    fn init() -> f64 { 0.0 }
    fn fold(acc: f64, x: f64) -> f64 { acc + x }
    fn finalize(acc: f64, count: usize) -> f64 { acc / count as f64 }
    fn validate_count(name: &str, count: usize) -> Result<()> {
        mean_divisor_check(name, count)
    }
}

impl ReduceOp<half::bf16> for Mean {
    type Acc = f32;
    fn init() -> f32 { 0.0 }
    fn fold(acc: f32, x: half::bf16) -> f32 { acc + x.to_f32() }
    fn finalize(acc: f32, count: usize) -> half::bf16 {
        half::bf16::from_f32(acc / count as f32)
    }
    fn validate_count(name: &str, count: usize) -> Result<()> {
        mean_divisor_check(name, count)
    }
}

impl ReduceOp<half::f16> for Mean {
    type Acc = f32;
    fn init() -> f32 { 0.0 }
    fn fold(acc: f32, x: half::f16) -> f32 { acc + x.to_f32() }
    fn finalize(acc: f32, count: usize) -> half::f16 {
        half::f16::from_f32(acc / count as f32)
    }
    fn validate_count(name: &str, count: usize) -> Result<()> {
        mean_divisor_check(name, count)
    }
}

// =============================================================================
// Chassis function
// =============================================================================

/// One pass over the input bytes, folding every element into the
/// accumulator for its destination output slot, then finalizing
/// each slot. All per-axis reduction kernels (Sum / Max / Min /
/// Mean across every supported dtype) call this function.
///
/// `name` is the public kernel name (`"sum_reduce_f32"` etc.) — it
/// appears in shape-mismatch and divisor-zero error messages so
/// the diagnostic points at the entry the caller invoked, not at
/// the chassis.
///
/// Caller contract:
/// - `input` holds `total_input * size_of::<T>()` bytes in row-
///   major order matching `input_shape`.
/// - `output` holds `total_output * size_of::<T>()` bytes,
///   pre-allocated. Content is overwritten; it need not be zeroed.
/// - `reduce_dims` is sorted ascending + unique; every entry is in
///   range `0..input_shape.len()`.
pub fn reduce<T, R>(
    name: &str,
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
) -> Result<()>
where
    T: Copy + Pod,
    R: ReduceOp<T>,
{
    let (total_input, kept) =
        check_reduce_shape(name, input, output, input_shape, reduce_dims, std::mem::size_of::<T>())?;
    let count: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();
    R::validate_count(name, count)?;
    let in_view: &[T] = input.as_slice()?;
    let out_view: &mut [T] = output.as_slice_mut()?;
    let total_output = out_view.len();
    let mut acc: Vec<R::Acc> = (0..total_output).map(|_| R::init()).collect();
    let mut mi = vec![0usize; input_shape.len()];
    for flat in 0..total_input {
        decode_multi_index(flat, input_shape, &mut mi);
        let oi = output_index(input_shape, &kept, &mi);
        acc[oi] = R::fold(acc[oi], in_view[flat]);
    }
    for (slot, &a) in out_view.iter_mut().zip(&acc) {
        *slot = R::finalize(a, count);
    }
    Ok(())
}

/// Reduce `input` to `output_shape` — the broadcast-target form
/// used by `Op::ReduceSumTo` and `Op::ReduceMaxTo`. Differs from
/// [`reduce`] only in shape derivation: each output axis (after
/// left-padding to input rank) must equal either the input's size
/// on that axis (axis passes through unchanged) or 1 (axis is
/// reduced away). Any other value is a contract violation.
///
/// Walks the input once, projecting each input multi-index to the
/// corresponding output slot by clamping coords to 0 on collapsed
/// axes, folds via [`ReduceOp::fold`], finalizes once per slot.
pub fn reduce_to<T, R>(
    name: &str,
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<()>
where
    T: Copy + Pod,
    R: ReduceOp<T>,
{
    let padded = align_reduce_to(name, input_shape, output_shape)?;
    let elem = std::mem::size_of::<T>();
    let in_elems: usize = input_shape.iter().product();
    let out_elems: usize = output_shape.iter().product();
    if input.len_bytes() != in_elems.saturating_mul(elem)
        || output.len_bytes() != out_elems.saturating_mul(elem)
    {
        return Err(Error::Msg(format!(
            "{name}: bytes mismatch (in {in_elems} elems, out {out_elems} elems)",
        ))
        .bt());
    }
    // Count of inputs folded into each output slot — the product of
    // input dims on the collapsed axes (uniform across all output
    // slots because each collapsed axis collapses uniformly). Only
    // Mean's `finalize` reads it; for Sum/Max it's ignored. Computed
    // here to keep the trait contract uniform across chassis fns.
    let mut count: usize = 1;
    for (axis, &p) in padded.iter().enumerate() {
        if p == 1 {
            count = count.saturating_mul(input_shape[axis]);
        }
    }
    R::validate_count(name, count)?;
    let in_view: &[T] = input.as_slice()?;
    let out_view: &mut [T] = output.as_slice_mut()?;
    let mut acc: Vec<R::Acc> = (0..out_elems).map(|_| R::init()).collect();
    let rank = input_shape.len();
    let mut in_strides = vec![1_usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
    }
    // Strides for the *padded* output, matching input rank.
    let mut out_strides_padded = vec![1_usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out_strides_padded[i] = out_strides_padded[i + 1] * padded[i + 1];
    }
    for in_flat in 0..in_elems {
        let mut out_flat = 0_usize;
        let mut rem = in_flat;
        for axis in 0..rank {
            let coord = rem / in_strides[axis];
            rem %= in_strides[axis];
            let out_coord = if padded[axis] == 1 { 0 } else { coord };
            out_flat += out_coord * out_strides_padded[axis];
        }
        acc[out_flat] = R::fold(acc[out_flat], in_view[in_flat]);
    }
    for (slot, &a) in out_view.iter_mut().zip(&acc) {
        *slot = R::finalize(a, count);
    }
    Ok(())
}

// =============================================================================
// Shape helpers — private to the chassis
// =============================================================================

/// Left-pad `output_shape` with 1s up to `input_shape`'s rank and
/// validate that every axis matches the input or is 1 (collapsed).
/// Returns the padded shape. Shared by [`reduce_to`].
fn align_reduce_to(
    name: &str,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<usize>> {
    if output_shape.len() > input_shape.len() {
        return Err(Error::Msg(format!(
            "{name}: output rank {} exceeds input rank {}",
            output_shape.len(),
            input_shape.len(),
        ))
        .bt());
    }
    let pad = input_shape.len() - output_shape.len();
    let mut padded = vec![1_usize; pad];
    padded.extend_from_slice(output_shape);
    for (i, (&s, &t)) in input_shape.iter().zip(padded.iter()).enumerate() {
        if t != 1 && t != s {
            return Err(Error::Msg(format!(
                "{name}: axis {i} target {t} must be 1 or input {s}",
            ))
            .bt());
        }
    }
    Ok(padded)
}

/// Validate input/output byte counts against the declared shape and
/// reduce-dim list. Returns `(total_input_elements, kept_dims)`.
fn check_reduce_shape(
    name: &str,
    input: &CpuStorageBytes,
    output: &CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
    elem_size: usize,
) -> Result<(usize, Vec<usize>)> {
    let total_input: usize = input_shape.iter().product();
    if input.len_bytes() != total_input.saturating_mul(elem_size) {
        return Err(Error::Msg(format!(
            "{name}: input bytes={} doesn't match shape {:?}",
            input.len_bytes(),
            input_shape,
        ))
        .bt());
    }
    let rank = input_shape.len();
    for &d in reduce_dims {
        if d >= rank {
            return Err(Error::Msg(format!(
                "{name}: reduce dim {d} out of range for rank {rank}",
            ))
            .bt());
        }
    }
    if reduce_dims.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::Msg(format!(
            "{name}: reduce dims {reduce_dims:?} must be sorted ascending + unique",
        ))
        .bt());
    }
    let kept: Vec<usize> = (0..rank).filter(|d| !reduce_dims.contains(d)).collect();
    let output_count: usize = kept.iter().map(|&d| input_shape[d]).product();
    if output.len_bytes() != output_count.saturating_mul(elem_size) {
        return Err(Error::Msg(format!(
            "{name}: output bytes={} doesn't match reduced shape (kept dims {:?}, count {output_count})",
            output.len_bytes(),
            kept,
        ))
        .bt());
    }
    Ok((total_input, kept))
}

/// Build the row-major flat index in the output tensor for a given
/// input multi-index, by reading only the kept dims.
fn output_index(input_shape: &[usize], kept: &[usize], multi_index: &[usize]) -> usize {
    let mut out_flat = 0;
    for &d in kept {
        out_flat = out_flat * input_shape[d] + multi_index[d];
    }
    out_flat
}

/// Decode a row-major flat index into a multi-index in `multi_index`
/// (must have the right rank).
fn decode_multi_index(flat: usize, input_shape: &[usize], multi_index: &mut [usize]) {
    let mut rem = flat;
    for d in (0..input_shape.len()).rev() {
        let s = input_shape[d];
        multi_index[d] = rem % s;
        rem /= s;
    }
}

// =============================================================================
// Structural tests
// =============================================================================
//
// These prove the trait methods compose correctly for each op.
// They're the structural counterpart to the numerical per-(op,
// dtype) tests in `byte_kernels::tests` — if the trait is right,
// every kernel built on it is right; the per-dtype tests prove the
// trait was right for that dtype.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_op_sum_f32_zero_init_fold_matches_iter_sum() {
        let xs = [1.0_f32, 2.5, -3.0, 4.0];
        let mut acc = <Sum as ReduceOp<f32>>::init();
        for &x in &xs {
            acc = <Sum as ReduceOp<f32>>::fold(acc, x);
        }
        let out = <Sum as ReduceOp<f32>>::finalize(acc, xs.len());
        assert_eq!(out, xs.iter().sum::<f32>());
    }

    #[test]
    fn reduce_op_max_f32_neg_inf_init_finds_largest() {
        let xs = [1.0_f32, -5.0, 3.0, 2.0];
        let mut acc = <Max as ReduceOp<f32>>::init();
        for &x in &xs {
            acc = <Max as ReduceOp<f32>>::fold(acc, x);
        }
        let out = <Max as ReduceOp<f32>>::finalize(acc, xs.len());
        assert_eq!(out, 3.0);
    }

    #[test]
    fn reduce_op_min_f32_pos_inf_init_finds_smallest() {
        let xs = [1.0_f32, -5.0, 3.0, 2.0];
        let mut acc = <Min as ReduceOp<f32>>::init();
        for &x in &xs {
            acc = <Min as ReduceOp<f32>>::fold(acc, x);
        }
        let out = <Min as ReduceOp<f32>>::finalize(acc, xs.len());
        assert_eq!(out, -5.0);
    }

    #[test]
    fn reduce_op_mean_f32_divides_by_count_in_finalize() {
        let xs = [2.0_f32, 4.0, 6.0, 8.0];
        let mut acc = <Mean as ReduceOp<f32>>::init();
        for &x in &xs {
            acc = <Mean as ReduceOp<f32>>::fold(acc, x);
        }
        let out = <Mean as ReduceOp<f32>>::finalize(acc, xs.len());
        assert_eq!(out, 5.0);
    }

    #[test]
    fn reduce_op_mean_rejects_zero_count() {
        let r = <Mean as ReduceOp<f32>>::validate_count("test", 0);
        assert!(r.is_err());
        let r = <Mean as ReduceOp<f32>>::validate_count("test", 5);
        assert!(r.is_ok());
    }

    #[test]
    fn reduce_to_sum_f32_collapses_inner_axis() {
        // input [2,3] → output [2,1] sums each row.
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut output = CpuStorageBytes::from_zero_bytes(2 * 4);
        reduce_to::<f32, Sum>("test_sum_to", &input, &mut output, &[2, 3], &[2, 1])
            .expect("reduce_to sum_f32");
        let r: &[f32] = output.as_slice().unwrap();
        assert_eq!(r, &[6.0, 15.0]);
    }

    #[test]
    fn reduce_to_max_f32_collapses_all_to_scalar() {
        // input [2,3] → output [1,1] picks the global max.
        let input = CpuStorageBytes::from_slice(&[1.0_f32, -5.0, 3.0, 2.0, 0.0, -1.0]);
        let mut output = CpuStorageBytes::from_zero_bytes(4);
        reduce_to::<f32, Max>("test_max_to", &input, &mut output, &[2, 3], &[1, 1])
            .expect("reduce_to max_f32");
        let r: &[f32] = output.as_slice().unwrap();
        assert_eq!(r, &[3.0]);
    }

    #[test]
    fn reduce_to_pad_rank_with_leading_ones() {
        // input [2,3] → output [3] gets padded to [1,3]; collapse outer.
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut output = CpuStorageBytes::from_zero_bytes(3 * 4);
        reduce_to::<f32, Sum>("test_pad", &input, &mut output, &[2, 3], &[3])
            .expect("reduce_to sum_f32 padded");
        let r: &[f32] = output.as_slice().unwrap();
        assert_eq!(r, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn reduce_to_rejects_incompatible_target_axis() {
        // input axis size 3, target axis size 2 → contract violation.
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut output = CpuStorageBytes::from_zero_bytes(2 * 2 * 4);
        let r = reduce_to::<f32, Sum>("test_bad", &input, &mut output, &[2, 3], &[2, 2]);
        assert!(r.is_err());
    }

    #[test]
    fn reduce_op_sum_bf16_accumulator_is_f32() {
        // Architectural invariant: bf16 sum uses f32 accumulator.
        // Verify by summing many bf16 values whose bf16-truncated
        // result would diverge from the f32-accumulated result.
        let xs: Vec<half::bf16> = (0..1000)
            .map(|_| half::bf16::from_f32(0.1))
            .collect();
        let mut acc = <Sum as ReduceOp<half::bf16>>::init();
        for &x in &xs {
            acc = <Sum as ReduceOp<half::bf16>>::fold(acc, x);
        }
        let out = <Sum as ReduceOp<half::bf16>>::finalize(acc, xs.len()).to_f32();
        // 1000 × 0.1 = 100.0 (in f32). bf16 representation of 0.1 is
        // ~0.09961 so 1000 × that ≈ 99.61. Result is within bf16's
        // ~3 decimal digits but the accumulator preserved the path.
        assert!(out > 95.0 && out < 105.0, "got {out}");
    }
}
