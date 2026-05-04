//! Typed byte-shaped kernels — Phase 7.5 B5.
//!
//! These kernels operate on [`CpuStorageBytes`] (bytes-based CPU
//! storage). They take typed slices via `bytemuck::cast_slice` /
//! `as_slice<T>` / `as_slice_mut<T>`, do the per-element work, and
//! return.
//!
//! These are the per-T monomorphic units that the dispatch wrapper
//! in `fuel_storage::dispatch::cpu_wrappers` calls after extracting
//! the `CpuStorageBytes` from a `BackendStorage::Cpu(...)` variant.
//!
//! ## Status
//!
//! B5 shipped the proof-of-concept (`add_f32`); Phase C grows the
//! coverage matrix family-by-family. Today's additions:
//!
//! - elementwise binary `f32`: `add`, `sub`, `mul`, `div`
//! - elementwise unary `f32`: `relu`, `neg`, `sqr`, `sqrt`, `recip`,
//!   `abs`, `tanh`

use crate::byte_storage::CpuStorageBytes;
use fuel_core_types::{Error, Layout, Result};

/// Verify three byte buffers have matching lengths.
fn check_lens_3(name: &str, a: usize, b: usize, c: usize) -> Result<()> {
    if a != b || a != c {
        return Err(Error::Msg(format!(
            "{name}: byte length mismatch (lhs={a}, rhs={b}, out={c})",
        ))
        .bt());
    }
    Ok(())
}

/// Verify two byte buffers have matching lengths (unary kernels).
fn check_lens_2(name: &str, a: usize, b: usize) -> Result<()> {
    if a != b {
        return Err(Error::Msg(format!(
            "{name}: byte length mismatch (input={a}, output={b})",
        ))
        .bt());
    }
    Ok(())
}

// =============================================================================
// Elementwise binary kernels (f32)
// =============================================================================

/// Generate a binary f32 kernel of the form
/// `out[i] = op(lhs[i], rhs[i])`.
///
/// Output is pre-allocated by the caller (the dispatch wrapper) and
/// must match the input byte length. The kernel writes into the
/// pre-allocated bytes; it never allocates.
macro_rules! binary_f32_kernel {
    ($name:ident, $op:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(
            lhs: &CpuStorageBytes,
            rhs: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
        ) -> Result<()> {
            check_lens_3(stringify!($name), lhs.len_bytes(), rhs.len_bytes(), out.len_bytes())?;
            let lhs_view: &[f32] = lhs.as_slice()?;
            let rhs_view: &[f32] = rhs.as_slice()?;
            let out_view: &mut [f32] = out.as_slice_mut()?;
            let op: fn(f32, f32) -> f32 = $op;
            for (i, slot) in out_view.iter_mut().enumerate() {
                *slot = op(lhs_view[i], rhs_view[i]);
            }
            Ok(())
        }
    };
}

binary_f32_kernel!(add_f32, |a, b| a + b, "Elementwise `f32` addition: `out[i] = lhs[i] + rhs[i]`.");
binary_f32_kernel!(sub_f32, |a, b| a - b, "Elementwise `f32` subtraction: `out[i] = lhs[i] - rhs[i]`.");
binary_f32_kernel!(mul_f32, |a, b| a * b, "Elementwise `f32` multiplication: `out[i] = lhs[i] * rhs[i]`.");
binary_f32_kernel!(div_f32, |a, b| a / b, "Elementwise `f32` division: `out[i] = lhs[i] / rhs[i]`. Division by zero yields IEEE-754 inf/NaN per platform.");

// =============================================================================
// Elementwise unary kernels (f32)
// =============================================================================

/// Generate a unary f32 kernel of the form `out[i] = op(input[i])`.
///
/// Output is pre-allocated by the caller and must match the input
/// byte length.
macro_rules! unary_f32_kernel {
    ($name:ident, $op:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(input: &CpuStorageBytes, out: &mut CpuStorageBytes) -> Result<()> {
            check_lens_2(stringify!($name), input.len_bytes(), out.len_bytes())?;
            let in_view: &[f32] = input.as_slice()?;
            let out_view: &mut [f32] = out.as_slice_mut()?;
            let op: fn(f32) -> f32 = $op;
            for (i, slot) in out_view.iter_mut().enumerate() {
                *slot = op(in_view[i]);
            }
            Ok(())
        }
    };
}

unary_f32_kernel!(relu_f32, |x| x.max(0.0), "Elementwise `f32` ReLU: `out[i] = max(0, input[i])`.");
unary_f32_kernel!(neg_f32, |x| -x, "Elementwise `f32` negation: `out[i] = -input[i]`.");
unary_f32_kernel!(sqr_f32, |x| x * x, "Elementwise `f32` square: `out[i] = input[i] * input[i]`.");
unary_f32_kernel!(sqrt_f32, |x| x.sqrt(), "Elementwise `f32` square root: `out[i] = sqrt(input[i])`. Negative inputs yield NaN per IEEE-754.");
unary_f32_kernel!(recip_f32, |x| 1.0 / x, "Elementwise `f32` reciprocal: `out[i] = 1 / input[i]`. Zero input yields IEEE-754 inf/NaN.");
unary_f32_kernel!(abs_f32, |x: f32| x.abs(), "Elementwise `f32` absolute value: `out[i] = |input[i]|`.");
unary_f32_kernel!(tanh_f32, |x: f32| x.tanh(), "Elementwise `f32` hyperbolic tangent: `out[i] = tanh(input[i])`.");
unary_f32_kernel!(exp_f32, |x: f32| x.exp(), "Elementwise `f32` exponential: `out[i] = e^input[i]`.");
unary_f32_kernel!(log_f32, |x: f32| x.ln(), "Elementwise `f32` natural log: `out[i] = ln(input[i])`. Negative inputs yield NaN per IEEE-754.");
unary_f32_kernel!(sin_f32, |x: f32| x.sin(), "Elementwise `f32` sine: `out[i] = sin(input[i])`.");
unary_f32_kernel!(cos_f32, |x: f32| x.cos(), "Elementwise `f32` cosine: `out[i] = cos(input[i])`.");
unary_f32_kernel!(sigmoid_f32, |x: f32| 1.0 / (1.0 + (-x).exp()), "Elementwise `f32` logistic sigmoid: `out[i] = 1 / (1 + exp(-input[i]))`.");
unary_f32_kernel!(silu_f32, |x: f32| x / (1.0 + (-x).exp()), "Elementwise `f32` SiLU/Swish: `out[i] = input[i] * sigmoid(input[i])`.");
unary_f32_kernel!(step_f32, |x: f32| if x > 0.0 { 1.0 } else { 0.0 }, "Elementwise `f32` Heaviside step: `out[i] = 1` where `input[i] > 0`, `0` otherwise.");

// =============================================================================
// Contiguize (dtype-agnostic, byte-level)
// =============================================================================

/// Materialize a contiguous-row-major buffer from a (potentially
/// strided / offset / broadcast) input. The output is a freshly
/// allocated [`CpuStorageBytes`] holding `layout.shape().elem_count()
/// * dtype_size` bytes; element `i` of the output corresponds to
/// the i-th element produced by `layout`'s strided iteration over
/// the input.
///
/// Dtype-agnostic: only `dtype_size` matters; the kernel copies
/// that many bytes per element. Broadcast layouts (stride 0)
/// transparently replicate source elements.
///
/// Used by the pipelined executor's auto-Contiguize pass before
/// kernels that require contiguous input (currently every kernel,
/// since today's kernels assume contiguous f32 walks).
pub fn contiguize_cpu(
    input: &CpuStorageBytes,
    layout: &Layout,
    dtype_size: usize,
) -> Result<CpuStorageBytes> {
    let elem_count = layout.shape().elem_count();
    let total_bytes = elem_count
        .checked_mul(dtype_size)
        .ok_or_else(|| Error::Msg("contiguize_cpu: elem_count * dtype_size overflow".to_string()).bt())?;
    let mut out = CpuStorageBytes::from_zero_bytes(total_bytes);
    if elem_count == 0 {
        return Ok(out);
    }
    let in_bytes = input.bytes();
    let out_bytes = out.bytes_mut();
    for (out_i, src_elem_off) in layout.strided_index().enumerate() {
        let src_byte_off = src_elem_off
            .checked_mul(dtype_size)
            .ok_or_else(|| Error::Msg("contiguize_cpu: src byte offset overflow".to_string()).bt())?;
        let dst_byte_off = out_i * dtype_size;
        if src_byte_off + dtype_size > in_bytes.len() {
            return Err(Error::Msg(format!(
                "contiguize_cpu: layout points past input bytes \
                 (src_byte={src_byte_off}, dtype_size={dtype_size}, input_bytes={})",
                in_bytes.len(),
            ))
            .bt());
        }
        out_bytes[dst_byte_off..dst_byte_off + dtype_size]
            .copy_from_slice(&in_bytes[src_byte_off..src_byte_off + dtype_size]);
    }
    Ok(out)
}

// =============================================================================
// Matrix multiplication (f32)
// =============================================================================

/// Rank-2 row-major `f32` matrix multiply:
/// `out[i, j] = Σₖ lhs[i, k] * rhs[k, j]`.
///
/// Inputs are assumed contiguous in row-major order (the pipelined
/// executor's auto-Contiguize pass guarantees this). Output is
/// pre-allocated, zero-initialized, and overwritten by this kernel.
///
/// This is the textbook triple loop in `i, k, j` order — sub-optimal
/// for cache behavior but a correct reference. Phase 7b's vendor-
/// specific BLAS backends (MKL, AOCL) will eclipse this on
/// performance once they're wired into the unified path; this
/// kernel is the always-available fallback.
pub fn matmul_f32(
    lhs: &CpuStorageBytes,
    rhs: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let elem = std::mem::size_of::<f32>();
    if lhs.len_bytes() != m.saturating_mul(k).saturating_mul(elem) {
        return Err(Error::Msg(format!(
            "matmul_f32: lhs bytes={} doesn't match shape [{m}, {k}] (f32)",
            lhs.len_bytes(),
        ))
        .bt());
    }
    if rhs.len_bytes() != k.saturating_mul(n).saturating_mul(elem) {
        return Err(Error::Msg(format!(
            "matmul_f32: rhs bytes={} doesn't match shape [{k}, {n}] (f32)",
            rhs.len_bytes(),
        ))
        .bt());
    }
    if out.len_bytes() != m.saturating_mul(n).saturating_mul(elem) {
        return Err(Error::Msg(format!(
            "matmul_f32: out bytes={} doesn't match shape [{m}, {n}] (f32)",
            out.len_bytes(),
        ))
        .bt());
    }
    let lhs_view: &[f32] = lhs.as_slice()?;
    let rhs_view: &[f32] = rhs.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    // Zero the output even though alloc_cpu_zeroed normally hands us
    // zero bytes — the kernel is robust against stale output buffers
    // (e.g. when a future executor re-uses a buffer).
    for slot in out_view.iter_mut() {
        *slot = 0.0;
    }
    // i, k, j order: each lhs[i, kk] is loaded once and broadcast to
    // every j of the inner loop, giving good lhs reuse.
    for i in 0..m {
        for kk in 0..k {
            let a = lhs_view[i * k + kk];
            let rhs_row_off = kk * n;
            let out_row_off = i * n;
            for j in 0..n {
                out_view[out_row_off + j] += a * rhs_view[rhs_row_off + j];
            }
        }
    }
    Ok(())
}

// =============================================================================
// Reduction kernels (f32)
// =============================================================================

/// Validate a reduction's shape contract: `input_shape`'s product
/// equals `input` element count; reduce_dims are in-range and
/// sorted; output element count equals the product of kept dims.
fn check_reduce_shape(
    name: &str,
    input: &CpuStorageBytes,
    output: &CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
) -> Result<(usize, Vec<usize>)> {
    let total_input: usize = input_shape.iter().product();
    let elem_size = std::mem::size_of::<f32>();
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
    // dims must be sorted ascending and unique — caller's contract.
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
/// (must be the right rank).
fn decode_multi_index(flat: usize, input_shape: &[usize], multi_index: &mut [usize]) {
    let mut rem = flat;
    for d in (0..input_shape.len()).rev() {
        let s = input_shape[d];
        multi_index[d] = rem % s;
        rem /= s;
    }
}

/// Sum-reduce `f32`: walks every input element and accumulates into
/// the output slot determined by the kept dims of the input
/// multi-index. Output is written from scratch (zeroed first); the
/// caller's pre-allocated bytes need not be zeroed.
pub fn sum_reduce_f32(
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
) -> Result<()> {
    let (total_input, kept) =
        check_reduce_shape("sum_reduce_f32", input, output, input_shape, reduce_dims)?;
    let in_view: &[f32] = input.as_slice()?;
    let out_view: &mut [f32] = output.as_slice_mut()?;
    for slot in out_view.iter_mut() {
        *slot = 0.0;
    }
    let mut mi = vec![0usize; input_shape.len()];
    for flat in 0..total_input {
        decode_multi_index(flat, input_shape, &mut mi);
        let oi = output_index(input_shape, &kept, &mi);
        out_view[oi] += in_view[flat];
    }
    Ok(())
}

/// Mean-reduce `f32`: sum-reduce divided by the product of reduced
/// dim sizes.
pub fn mean_reduce_f32(
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
) -> Result<()> {
    sum_reduce_f32(input, output, input_shape, reduce_dims)?;
    let divisor: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();
    if divisor == 0 {
        return Err(Error::Msg(
            "mean_reduce_f32: divisor zero (reduced dim has size 0)".to_string(),
        )
        .bt());
    }
    let inv = 1.0_f32 / divisor as f32;
    let out_view: &mut [f32] = output.as_slice_mut()?;
    for slot in out_view.iter_mut() {
        *slot *= inv;
    }
    Ok(())
}

/// Generic reduction with a custom accumulator init + combine fn.
/// Used by max/min reduce; not exposed (kept private to keep the
/// kernel surface tight).
fn reduce_f32_generic(
    name: &str,
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
    init: f32,
    combine: fn(f32, f32) -> f32,
) -> Result<()> {
    let (total_input, kept) =
        check_reduce_shape(name, input, output, input_shape, reduce_dims)?;
    let in_view: &[f32] = input.as_slice()?;
    let out_view: &mut [f32] = output.as_slice_mut()?;
    for slot in out_view.iter_mut() {
        *slot = init;
    }
    let mut mi = vec![0usize; input_shape.len()];
    for flat in 0..total_input {
        decode_multi_index(flat, input_shape, &mut mi);
        let oi = output_index(input_shape, &kept, &mi);
        out_view[oi] = combine(out_view[oi], in_view[flat]);
    }
    Ok(())
}

/// Max-reduce `f32`. Output slots initialize to `-inf`.
pub fn max_reduce_f32(
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
) -> Result<()> {
    reduce_f32_generic(
        "max_reduce_f32",
        input,
        output,
        input_shape,
        reduce_dims,
        f32::NEG_INFINITY,
        |a, b| a.max(b),
    )
}

/// Min-reduce `f32`. Output slots initialize to `+inf`.
pub fn min_reduce_f32(
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
) -> Result<()> {
    reduce_f32_generic(
        "min_reduce_f32",
        input,
        output,
        input_shape,
        reduce_dims,
        f32::INFINITY,
        |a, b| a.min(b),
    )
}

/// Elementwise GELU using the tanh approximation:
/// `out[i] = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))`.
///
/// Matches `Op::Gelu`'s tanh-approximation semantics in fuel-graph.
pub fn gelu_f32(input: &CpuStorageBytes, out: &mut CpuStorageBytes) -> Result<()> {
    check_lens_2("gelu_f32", input.len_bytes(), out.len_bytes())?;
    let in_view: &[f32] = input.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    // √(2/π) ≈ 0.7978845608
    const COEFF: f32 = 0.797_884_56;
    for (i, slot) in out_view.iter_mut().enumerate() {
        let x = in_view[i];
        let inner = COEFF * (x + 0.044_715 * x * x * x);
        *slot = 0.5 * x * (1.0 + inner.tanh());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: add two f32 storages elementwise.
    #[test]
    fn add_f32_round_trip() {
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let b = CpuStorageBytes::from_slice(&[10.0_f32, 20.0, 30.0, 40.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(16);

        add_f32(&a, &b, &mut out).expect("add");

        let result: &[f32] = out.as_slice().unwrap();
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }

    /// Mismatched byte counts produce an error, not a panic.
    #[test]
    fn add_f32_errors_on_size_mismatch() {
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let b = CpuStorageBytes::from_slice(&[10.0_f32, 20.0, 30.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(8);

        let result = add_f32(&a, &b, &mut out);
        assert!(result.is_err(), "size mismatch must error");
    }

    #[test]
    fn sub_mul_div_f32_round_trip() {
        let a = CpuStorageBytes::from_slice(&[10.0_f32, 20.0, 30.0]);
        let b = CpuStorageBytes::from_slice(&[1.0_f32, 4.0, 5.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(12);

        sub_f32(&a, &b, &mut out).expect("sub");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[9.0, 16.0, 25.0]);

        mul_f32(&a, &b, &mut out).expect("mul");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[10.0, 80.0, 150.0]);

        div_f32(&a, &b, &mut out).expect("div");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[10.0, 5.0, 6.0]);
    }

    #[test]
    fn binary_f32_size_mismatch_errors() {
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let b = CpuStorageBytes::from_slice(&[1.0_f32]);
        let mut out = CpuStorageBytes::from_zero_bytes(8);
        for f in [sub_f32, mul_f32, div_f32] {
            assert!(f(&a, &b, &mut out).is_err(), "size mismatch must error");
        }
    }

    #[test]
    fn relu_f32_clips_negatives() {
        let input = CpuStorageBytes::from_slice(&[-1.0_f32, 0.0, 0.5, -3.5, 7.25]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        relu_f32(&input, &mut out).expect("relu");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[0.0, 0.0, 0.5, 0.0, 7.25]);
    }

    #[test]
    fn unary_f32_round_trip() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, -2.0, 4.0, 0.5]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());

        neg_f32(&input, &mut out).expect("neg");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[-1.0, 2.0, -4.0, -0.5]);

        sqr_f32(&input, &mut out).expect("sqr");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 4.0, 16.0, 0.25]);

        let pos = CpuStorageBytes::from_slice(&[1.0_f32, 4.0, 9.0, 16.0]);
        let mut sqrt_out = CpuStorageBytes::from_zero_bytes(pos.len_bytes());
        sqrt_f32(&pos, &mut sqrt_out).expect("sqrt");
        assert_eq!(sqrt_out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

        recip_f32(&pos, &mut sqrt_out).expect("recip");
        assert_eq!(sqrt_out.as_slice::<f32>().unwrap(), &[1.0, 0.25, 1.0 / 9.0, 1.0 / 16.0]);

        abs_f32(&input, &mut out).expect("abs");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 4.0, 0.5]);
    }

    #[test]
    fn tanh_f32_at_known_points() {
        let input = CpuStorageBytes::from_slice(&[0.0_f32, 1.0, -1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        tanh_f32(&input, &mut out).expect("tanh");
        let result: &[f32] = out.as_slice().unwrap();
        assert!((result[0] - 0.0).abs() < 1e-7);
        assert!((result[1] - 1.0_f32.tanh()).abs() < 1e-7);
        assert!((result[2] - (-1.0_f32).tanh()).abs() < 1e-7);
    }

    #[test]
    fn unary_f32_size_mismatch_errors() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4); // wrong size
        for f in [
            relu_f32, neg_f32, sqr_f32, sqrt_f32, recip_f32, abs_f32, tanh_f32,
            exp_f32, log_f32, sin_f32, cos_f32, sigmoid_f32, silu_f32, step_f32,
            gelu_f32,
        ] {
            assert!(f(&input, &mut out).is_err(), "size mismatch must error");
        }
    }

    #[test]
    fn exp_log_round_trip() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]);
        let mut intermediate = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());

        exp_f32(&input, &mut intermediate).expect("exp");
        log_f32(&intermediate, &mut out).expect("log");

        let result: &[f32] = out.as_slice().unwrap();
        for (got, want) in result.iter().zip(&[1.0_f32, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-5, "exp+log not identity");
        }
    }

    #[test]
    fn sin_cos_at_known_points() {
        let input = CpuStorageBytes::from_slice(&[0.0_f32, std::f32::consts::FRAC_PI_2]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());

        sin_f32(&input, &mut out).expect("sin");
        let r: &[f32] = out.as_slice().unwrap();
        assert!(r[0].abs() < 1e-7);
        assert!((r[1] - 1.0).abs() < 1e-6);

        cos_f32(&input, &mut out).expect("cos");
        let r: &[f32] = out.as_slice().unwrap();
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!(r[1].abs() < 1e-6);
    }

    #[test]
    fn sigmoid_silu_at_known_points() {
        let input = CpuStorageBytes::from_slice(&[0.0_f32, 5.0, -5.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());

        sigmoid_f32(&input, &mut out).expect("sigmoid");
        let r: &[f32] = out.as_slice().unwrap();
        assert!((r[0] - 0.5).abs() < 1e-7);
        assert!(r[1] > 0.99);
        assert!(r[2] < 0.01);

        silu_f32(&input, &mut out).expect("silu");
        let r: &[f32] = out.as_slice().unwrap();
        assert!(r[0].abs() < 1e-7); // 0 * sigmoid(0) = 0
    }

    #[test]
    fn step_clips_correctly() {
        let input = CpuStorageBytes::from_slice(&[-2.0_f32, 0.0, 0.5, 100.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        step_f32(&input, &mut out).expect("step");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn matmul_f32_2x3_times_3x2() {
        // [[1, 2, 3],         [[7,  8],          [[58,  64],
        //  [4, 5, 6]]    @     [9, 10],     =     [139, 154]]
        //                      [11, 12]]
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = CpuStorageBytes::from_slice(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(16); // 2 * 2 * 4

        matmul_f32(&lhs, &rhs, &mut out, 2, 2, 3).expect("matmul");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_f32_identity_returns_input() {
        // [[1, 2],         [[1, 0],         [[1, 2],
        //  [3, 4]]    @     [0, 1]]    =     [3, 4]]
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let rhs = CpuStorageBytes::from_slice(&[1.0_f32, 0.0, 0.0, 1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(16);

        matmul_f32(&lhs, &rhs, &mut out, 2, 2, 2).expect("matmul");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_f32_inner_product_1x3_times_3x1() {
        // Row vector × column vector → 1×1.
        // [1, 2, 3] @ [4; 5; 6] = [32]
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]);
        let rhs = CpuStorageBytes::from_slice(&[4.0_f32, 5.0, 6.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4);

        matmul_f32(&lhs, &rhs, &mut out, 1, 1, 3).expect("matmul");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[32.0]);
    }

    #[test]
    fn matmul_f32_size_mismatch_errors() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]); // 4 elem; bytes = 16
        let rhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]); // 2 elem
        let mut out = CpuStorageBytes::from_zero_bytes(16);
        // Claim shape [2, 2] @ [2, 2] but rhs only has 2 elements
        let r = matmul_f32(&lhs, &rhs, &mut out, 2, 2, 2);
        assert!(r.is_err(), "matmul must error on shape/byte mismatch");
    }

    #[test]
    fn contiguize_cpu_no_op_on_contiguous_input() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let layout = Layout::contiguous(fuel_core_types::Shape::from_dims(&[2, 2]));
        let out = contiguize_cpu(&input, &layout, 4).expect("contiguize");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn contiguize_cpu_realizes_transpose() {
        // shape [2, 3], stored row-major: 1 2 3 / 4 5 6
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Transposed view: shape [3, 2], strides [1, 3]
        let layout = Layout::new(
            fuel_core_types::Shape::from_dims(&[3, 2]),
            fuel_core_types::DimVec::from_slice(&[1usize, 3]),
            0,
        );
        let out = contiguize_cpu(&input, &layout, 4).expect("contiguize");
        // Transposed: column 0 of source (1, 4) becomes row 0; col 1 (2, 5) row 1; col 2 (3, 6) row 2
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn contiguize_cpu_replicates_broadcast() {
        // shape [3] broadcast to [2, 3] — leading dim has stride 0
        let input = CpuStorageBytes::from_slice(&[10.0_f32, 20.0, 30.0]);
        let layout = Layout::new(
            fuel_core_types::Shape::from_dims(&[2, 3]),
            fuel_core_types::DimVec::from_slice(&[0usize, 1]),
            0,
        );
        let out = contiguize_cpu(&input, &layout, 4).expect("contiguize");
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn contiguize_cpu_offset_layout() {
        // input has 5 elements; layout views the last 3 with offset 2
        let input = CpuStorageBytes::from_slice(&[100.0_f32, 200.0, 1.0, 2.0, 3.0]);
        let layout = Layout::new(
            fuel_core_types::Shape::from_dims(&[3]),
            fuel_core_types::DimVec::from_slice(&[1usize]),
            2,
        );
        let out = contiguize_cpu(&input, &layout, 4).expect("contiguize");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn sum_reduce_along_one_dim() {
        // shape [2, 3], reduce dim 1 → output shape [2]
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(8); // 2 f32s

        sum_reduce_f32(&input, &mut out, &[2, 3], &[1]).expect("sum_reduce");
        // row 0: 1+2+3=6; row 1: 4+5+6=15
        assert_eq!(out.as_slice::<f32>().unwrap(), &[6.0, 15.0]);
    }

    #[test]
    fn sum_reduce_all_dims() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4); // 1 f32

        sum_reduce_f32(&input, &mut out, &[2, 2], &[0, 1]).expect("sum_reduce_all");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[10.0]);
    }

    #[test]
    fn sum_reduce_inner_dim_of_rank_3() {
        // shape [2, 2, 3], reduce dim 1 → output shape [2, 3]
        // Layout (row-major):
        //   batch=0: [[1,2,3],[4,5,6]]   → reduce dim 1 → [5,7,9]
        //   batch=1: [[7,8,9],[10,11,12]] → reduce dim 1 → [17,19,21]
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let input = CpuStorageBytes::from_slice(&data);
        let mut out = CpuStorageBytes::from_zero_bytes(24); // 6 f32s

        sum_reduce_f32(&input, &mut out, &[2, 2, 3], &[1]).expect("sum_reduce");
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[5.0, 7.0, 9.0, 17.0, 19.0, 21.0]
        );
    }

    #[test]
    fn mean_reduce_divides_by_reduced_dim_count() {
        let input = CpuStorageBytes::from_slice(&[2.0_f32, 4.0, 6.0, 8.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(8); // [2] output

        mean_reduce_f32(&input, &mut out, &[2, 2], &[1]).expect("mean_reduce");
        // row 0 mean = (2+4)/2 = 3; row 1 mean = (6+8)/2 = 7
        assert_eq!(out.as_slice::<f32>().unwrap(), &[3.0, 7.0]);
    }

    #[test]
    fn max_reduce_f32_basic() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, -5.0, 3.0, 2.0, 0.0, -1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(8); // [2] output

        max_reduce_f32(&input, &mut out, &[2, 3], &[1]).expect("max_reduce");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[3.0, 2.0]);
    }

    #[test]
    fn min_reduce_f32_basic() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, -5.0, 3.0, 2.0, 0.0, -1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(8); // [2] output

        min_reduce_f32(&input, &mut out, &[2, 3], &[1]).expect("min_reduce");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[-5.0, -1.0]);
    }

    #[test]
    fn reduce_errors_on_shape_mismatch() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]); // 3 elems
        let mut out = CpuStorageBytes::from_zero_bytes(8);
        // shape says 4 elems but storage has 3 — must error.
        let r = sum_reduce_f32(&input, &mut out, &[2, 2], &[1]);
        assert!(r.is_err());
    }

    #[test]
    fn reduce_errors_on_unsorted_dims() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4);
        let r = sum_reduce_f32(&input, &mut out, &[2, 2], &[1, 0]); // unsorted
        assert!(r.is_err());
    }

    #[test]
    fn reduce_errors_on_out_of_range_dim() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4);
        let r = sum_reduce_f32(&input, &mut out, &[2], &[5]); // dim 5 doesn't exist
        assert!(r.is_err());
    }

    #[test]
    fn gelu_at_known_points() {
        let input = CpuStorageBytes::from_slice(&[0.0_f32, 1.0, -1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        gelu_f32(&input, &mut out).expect("gelu");
        let r: &[f32] = out.as_slice().unwrap();
        // gelu(0) = 0
        assert!(r[0].abs() < 1e-6);
        // gelu(1) ≈ 0.8412 (tanh approx)
        assert!((r[1] - 0.841_192).abs() < 1e-3);
        // gelu(-1) ≈ -0.1588 (tanh approx)
        assert!((r[2] - (-0.158_808)).abs() < 1e-3);
    }
}
