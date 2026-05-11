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
use fuel_core_types::{Error, Result};

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

// =============================================================================
// Shape helpers — private to the chassis
// =============================================================================

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
