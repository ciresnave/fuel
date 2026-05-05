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
// Index select (f32 source, U32 indices)
// =============================================================================

/// Pick slices from a source `f32` tensor along one axis using
/// `u32` indices. The source is laid out as
/// `[outer_count, source_dim_size, inner_count]`, indices is a
/// rank-1 `[n_indices]` `u32` array, and the output is
/// `[outer_count, n_indices, inner_count]`.
///
/// Each output element at `(outer, j, inner)` reads from source
/// at `(outer, indices[j], inner)`. Out-of-bounds indices return
/// a typed error rather than reading garbage.
pub fn index_select_f32(
    source: &CpuStorageBytes,
    indices: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    outer_count: usize,
    source_dim_size: usize,
    n_indices: usize,
    inner_count: usize,
) -> Result<()> {
    let elem = std::mem::size_of::<f32>();
    let need_src = outer_count
        .saturating_mul(source_dim_size)
        .saturating_mul(inner_count)
        .saturating_mul(elem);
    let need_idx = n_indices.saturating_mul(std::mem::size_of::<u32>());
    let need_out = outer_count
        .saturating_mul(n_indices)
        .saturating_mul(inner_count)
        .saturating_mul(elem);
    if source.len_bytes() != need_src {
        return Err(Error::Msg(format!(
            "index_select_f32: source bytes={} doesn't match outer={outer_count} × dim={source_dim_size} × inner={inner_count} × {elem}",
            source.len_bytes(),
        ))
        .bt());
    }
    if indices.len_bytes() != need_idx {
        return Err(Error::Msg(format!(
            "index_select_f32: indices bytes={} doesn't match n_indices={n_indices} × 4",
            indices.len_bytes(),
        ))
        .bt());
    }
    if out.len_bytes() != need_out {
        return Err(Error::Msg(format!(
            "index_select_f32: out bytes={} doesn't match outer={outer_count} × n={n_indices} × inner={inner_count} × {elem}",
            out.len_bytes(),
        ))
        .bt());
    }
    let src_view: &[f32] = source.as_slice()?;
    let idx_view: &[u32] = indices.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    for j in 0..n_indices {
        let i = idx_view[j] as usize;
        if i >= source_dim_size {
            return Err(Error::Msg(format!(
                "index_select_f32: index {i} at position {j} out of bounds for source dim {source_dim_size}",
            ))
            .bt());
        }
        for outer in 0..outer_count {
            let src_off = (outer * source_dim_size + i) * inner_count;
            let dst_off = (outer * n_indices + j) * inner_count;
            out_view[dst_off..dst_off + inner_count]
                .copy_from_slice(&src_view[src_off..src_off + inner_count]);
        }
    }
    Ok(())
}

// =============================================================================
// Rotary position embedding (f32, rotate_half convention)
// =============================================================================

/// Fused rotary position embedding. Inputs `(x, cos, sin)`:
///
/// - `x` is laid out as `[outer_count, seq, head_dim]` row-major
///   (the leading "..." dims fold into `outer_count`).
/// - `cos` and `sin` are `[seq, head_dim]`, broadcasting across
///   the outer dims.
/// - `head_dim` must be even; `h = head_dim / 2`.
///
/// rotate_half formula:
/// ```text
///   out[..., s, i]     = x[..., s, i]     * cos[s, i]     - x[..., s, i+h] * sin[s, i]
///   out[..., s, i+h]   = x[..., s, i+h]   * cos[s, i+h]   + x[..., s, i]   * sin[s, i+h]
/// ```
/// for `i ∈ 0..h`. Replaces a 9-op decomposition (slice + neg +
/// concat + broadcast_mul + add + ...) with a single fused kernel.
pub fn rope_f32(
    x: &CpuStorageBytes,
    cos: &CpuStorageBytes,
    sin: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    outer_count: usize,
    seq: usize,
    head_dim: usize,
) -> Result<()> {
    if head_dim % 2 != 0 {
        return Err(Error::Msg(format!(
            "rope_f32: head_dim ({head_dim}) must be even",
        ))
        .bt());
    }
    let elem = std::mem::size_of::<f32>();
    let need_x = outer_count
        .saturating_mul(seq)
        .saturating_mul(head_dim)
        .saturating_mul(elem);
    let need_cs = seq.saturating_mul(head_dim).saturating_mul(elem);
    if x.len_bytes() != need_x {
        return Err(Error::Msg(format!(
            "rope_f32: x bytes={} doesn't match outer={outer_count} × seq={seq} × head_dim={head_dim} × {elem}",
            x.len_bytes(),
        ))
        .bt());
    }
    if cos.len_bytes() != need_cs {
        return Err(Error::Msg(format!(
            "rope_f32: cos bytes={} doesn't match seq={seq} × head_dim={head_dim} × {elem}",
            cos.len_bytes(),
        ))
        .bt());
    }
    if sin.len_bytes() != need_cs {
        return Err(Error::Msg(format!(
            "rope_f32: sin bytes={} doesn't match seq={seq} × head_dim={head_dim} × {elem}",
            sin.len_bytes(),
        ))
        .bt());
    }
    if out.len_bytes() != need_x {
        return Err(Error::Msg(format!(
            "rope_f32: out bytes={} doesn't match x shape",
            out.len_bytes(),
        ))
        .bt());
    }
    if seq == 0 || head_dim == 0 {
        return Ok(());
    }
    let x_view: &[f32] = x.as_slice()?;
    let cos_view: &[f32] = cos.as_slice()?;
    let sin_view: &[f32] = sin.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    let h = head_dim / 2;
    for outer in 0..outer_count {
        for s in 0..seq {
            let x_row_off = (outer * seq + s) * head_dim;
            let cs_row_off = s * head_dim;
            for i in 0..h {
                let x_lo_off = x_row_off + i;
                let x_hi_off = x_row_off + i + h;
                let cs_lo_off = cs_row_off + i;
                let cs_hi_off = cs_row_off + i + h;
                let x_lo = x_view[x_lo_off];
                let x_hi = x_view[x_hi_off];
                let cos_lo = cos_view[cs_lo_off];
                let cos_hi = cos_view[cs_hi_off];
                let sin_lo = sin_view[cs_lo_off];
                let sin_hi = sin_view[cs_hi_off];
                out_view[x_lo_off] = x_lo * cos_lo - x_hi * sin_lo;
                out_view[x_hi_off] = x_hi * cos_hi + x_lo * sin_hi;
            }
        }
    }
    Ok(())
}

// =============================================================================
// Gather (f32 source, U32 indices, same-rank)
// =============================================================================

/// N-dimensional gather along `dim`. Source and output shapes
/// agree on every dim except `dim`. The indices tensor has the
/// output's shape and supplies the source coord for `dim`.
///
/// For each output position (i₀, …, iₙ):
/// `out[i₀, …, iₙ] = source[i₀, …, indices[i₀, …, iₙ], …, iₙ]`.
///
/// Out-of-bounds indices return a typed error.
pub fn gather_f32(
    source: &CpuStorageBytes,
    indices: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    source_shape: &[usize],
    output_shape: &[usize],
    dim: usize,
) -> Result<()> {
    if source_shape.len() != output_shape.len() {
        return Err(Error::Msg(format!(
            "gather_f32: source rank ({}) != output rank ({})",
            source_shape.len(),
            output_shape.len(),
        ))
        .bt());
    }
    let rank = source_shape.len();
    if dim >= rank {
        return Err(Error::Msg(format!(
            "gather_f32: dim {dim} out of range for rank {rank}",
        ))
        .bt());
    }
    for d in 0..rank {
        if d != dim && source_shape[d] != output_shape[d] {
            return Err(Error::Msg(format!(
                "gather_f32: source and output disagree at dim {d} \
                 (source={}, output={}); only `dim`={dim} may differ",
                source_shape[d],
                output_shape[d],
            ))
            .bt());
        }
    }
    let elem = std::mem::size_of::<f32>();
    let source_total: usize = source_shape.iter().product();
    let output_total: usize = output_shape.iter().product();
    if source.len_bytes() != source_total.saturating_mul(elem) {
        return Err(Error::Msg(format!(
            "gather_f32: source bytes={} doesn't match shape {source_shape:?} (f32)",
            source.len_bytes(),
        ))
        .bt());
    }
    if indices.len_bytes() != output_total.saturating_mul(std::mem::size_of::<u32>()) {
        return Err(Error::Msg(format!(
            "gather_f32: indices bytes={} doesn't match output shape {output_shape:?} (u32)",
            indices.len_bytes(),
        ))
        .bt());
    }
    if out.len_bytes() != output_total.saturating_mul(elem) {
        return Err(Error::Msg(format!(
            "gather_f32: out bytes={} doesn't match output shape {output_shape:?} (f32)",
            out.len_bytes(),
        ))
        .bt());
    }
    if output_total == 0 {
        return Ok(());
    }
    let src_view: &[f32] = source.as_slice()?;
    let idx_view: &[u32] = indices.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    // Source strides (row-major).
    let mut src_strides = vec![0usize; rank];
    let mut s = 1;
    for d in (0..rank).rev() {
        src_strides[d] = s;
        s *= source_shape[d];
    }
    let mut multi = vec![0usize; rank];
    for f in 0..output_total {
        // Decode output multi-index from f (row-major).
        let mut rem = f;
        for d in (0..rank).rev() {
            multi[d] = rem % output_shape[d];
            rem /= output_shape[d];
        }
        let src_dim_idx = idx_view[f] as usize;
        if src_dim_idx >= source_shape[dim] {
            return Err(Error::Msg(format!(
                "gather_f32: index {src_dim_idx} at output position {f} out of bounds for source dim {} = {}",
                dim, source_shape[dim],
            ))
            .bt());
        }
        // Compose source flat index.
        let mut src_flat = 0;
        for d in 0..rank {
            let coord = if d == dim { src_dim_idx } else { multi[d] };
            src_flat += coord * src_strides[d];
        }
        out_view[f] = src_view[src_flat];
    }
    Ok(())
}

// =============================================================================
// Softmax (f32)
// =============================================================================

/// Softmax along the last dim, numerically stable. For each of the
/// `outer_count` rows of `last_dim` elements, computes
/// `out[i] = exp(x[i] - max_row) / sum(exp(x - max_row))`. The
/// `max_row` subtraction prevents overflow in `exp`.
///
/// Edge cases:
///   - `last_dim == 0`: no work; returns `Ok(())`.
///   - All-`-inf` row: would divide 0/0; returns NaN per IEEE-754.
///     Caller is expected to ensure inputs are finite.
pub fn softmax_last_dim_f32(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    outer_count: usize,
    last_dim: usize,
) -> Result<()> {
    check_lens_2("softmax_last_dim_f32", input.len_bytes(), out.len_bytes())?;
    let elem = std::mem::size_of::<f32>();
    let need = outer_count
        .saturating_mul(last_dim)
        .saturating_mul(elem);
    if input.len_bytes() != need {
        return Err(Error::Msg(format!(
            "softmax_last_dim_f32: input bytes={} doesn't match outer={outer_count} × last={last_dim} × {elem}",
            input.len_bytes(),
        ))
        .bt());
    }
    if last_dim == 0 {
        return Ok(());
    }
    let in_view: &[f32] = input.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    for row in 0..outer_count {
        let off = row * last_dim;
        let row_in = &in_view[off..off + last_dim];
        // Find row max for numerical stability.
        let mut row_max = row_in[0];
        for &v in &row_in[1..] {
            if v > row_max {
                row_max = v;
            }
        }
        // Compute exp(x - max) and accumulate sum.
        let mut sum = 0.0_f32;
        for j in 0..last_dim {
            let e = (row_in[j] - row_max).exp();
            out_view[off + j] = e;
            sum += e;
        }
        // Normalize.
        let inv = 1.0 / sum;
        for j in 0..last_dim {
            out_view[off + j] *= inv;
        }
    }
    Ok(())
}

// =============================================================================
// RMS / Layer norm along the last dim (f32, no affine)
// =============================================================================

/// Helper: validate that input/output bytes match `outer_count *
/// last_dim * sizeof::<f32>()`. Returns Ok if all three agree.
fn check_norm_lens(
    name: &str,
    input: &CpuStorageBytes,
    out: &CpuStorageBytes,
    outer_count: usize,
    last_dim: usize,
) -> Result<()> {
    let elem = std::mem::size_of::<f32>();
    let need = outer_count
        .saturating_mul(last_dim)
        .saturating_mul(elem);
    if input.len_bytes() != need || out.len_bytes() != need {
        return Err(Error::Msg(format!(
            "{name}: bytes mismatch (input={}, out={}, expected outer={outer_count} × last={last_dim} × {elem} = {need})",
            input.len_bytes(), out.len_bytes(),
        ))
        .bt());
    }
    Ok(())
}

/// RMS normalization along the last dim, no affine params:
/// `out[i] = x[i] / sqrt(mean(x²) + eps)` per row.
/// `eps` is `f64` for graph-API consistency; converted to `f32`
/// before use in the f32 kernel.
pub fn rms_norm_last_dim_f32(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    outer_count: usize,
    last_dim: usize,
    eps: f64,
) -> Result<()> {
    check_norm_lens("rms_norm_last_dim_f32", input, out, outer_count, last_dim)?;
    if last_dim == 0 {
        return Ok(());
    }
    let in_view: &[f32] = input.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    let eps32 = eps as f32;
    let inv_n = 1.0_f32 / last_dim as f32;
    for row in 0..outer_count {
        let off = row * last_dim;
        let row_in = &in_view[off..off + last_dim];
        let mut sum_sq = 0.0_f32;
        for &v in row_in {
            sum_sq += v * v;
        }
        let mean_sq = sum_sq * inv_n;
        let rms_inv = 1.0_f32 / (mean_sq + eps32).sqrt();
        for j in 0..last_dim {
            out_view[off + j] = row_in[j] * rms_inv;
        }
    }
    Ok(())
}

/// Layer normalization along the last dim, no affine params:
/// `out[i] = (x[i] - mean(x)) / sqrt(var(x) + eps)` per row, with
/// `var = mean((x - mean)²)`.
pub fn layer_norm_last_dim_f32(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    outer_count: usize,
    last_dim: usize,
    eps: f64,
) -> Result<()> {
    check_norm_lens("layer_norm_last_dim_f32", input, out, outer_count, last_dim)?;
    if last_dim == 0 {
        return Ok(());
    }
    let in_view: &[f32] = input.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    let eps32 = eps as f32;
    let inv_n = 1.0_f32 / last_dim as f32;
    for row in 0..outer_count {
        let off = row * last_dim;
        let row_in = &in_view[off..off + last_dim];
        let mut sum = 0.0_f32;
        for &v in row_in {
            sum += v;
        }
        let mean = sum * inv_n;
        let mut sum_sq = 0.0_f32;
        for &v in row_in {
            let d = v - mean;
            sum_sq += d * d;
        }
        let var = sum_sq * inv_n;
        let inv_std = 1.0_f32 / (var + eps32).sqrt();
        for j in 0..last_dim {
            out_view[off + j] = (row_in[j] - mean) * inv_std;
        }
    }
    Ok(())
}

// =============================================================================
// Concat (f32)
// =============================================================================

/// Concatenate N `f32` inputs along one dim. The output is laid out
/// as `[outer_count, total_dim, inner_count]` row-major, where
/// `total_dim = sum(input_dim_sizes)`. Each input contributes a
/// `[outer_count, input_dim_sizes[i], inner_count]` slab into the
/// output's `[outer_count, dim_offset_i .. dim_offset_i + input_dim_sizes[i], inner_count]`
/// region.
///
/// Inputs are assumed contiguous in row-major order (the executor's
/// auto-Contiguize pass guarantees this).
pub fn concat_f32(
    inputs: &[&CpuStorageBytes],
    out: &mut CpuStorageBytes,
    outer_count: usize,
    input_dim_sizes: &[usize],
    inner_count: usize,
) -> Result<()> {
    if inputs.len() != input_dim_sizes.len() {
        return Err(Error::Msg(format!(
            "concat_f32: inputs count ({}) != input_dim_sizes len ({})",
            inputs.len(),
            input_dim_sizes.len(),
        ))
        .bt());
    }
    if inputs.is_empty() {
        return Err(Error::Msg("concat_f32: at least one input required".to_string()).bt());
    }
    let elem = std::mem::size_of::<f32>();
    let total_dim: usize = input_dim_sizes.iter().sum();
    let need_out = outer_count
        .saturating_mul(total_dim)
        .saturating_mul(inner_count)
        .saturating_mul(elem);
    if out.len_bytes() != need_out {
        return Err(Error::Msg(format!(
            "concat_f32: out bytes={} doesn't match outer={outer_count} × total_dim={total_dim} × inner={inner_count} × {elem}",
            out.len_bytes(),
        ))
        .bt());
    }
    for (i, input) in inputs.iter().enumerate() {
        let need_in = outer_count
            .saturating_mul(input_dim_sizes[i])
            .saturating_mul(inner_count)
            .saturating_mul(elem);
        if input.len_bytes() != need_in {
            return Err(Error::Msg(format!(
                "concat_f32: input[{i}] bytes={} doesn't match outer={outer_count} × dim={} × inner={inner_count} × {elem}",
                input.len_bytes(),
                input_dim_sizes[i],
            ))
            .bt());
        }
    }
    let out_view: &mut [f32] = out.as_slice_mut()?;
    let mut dim_offset = 0usize;
    for (i, input) in inputs.iter().enumerate() {
        let in_view: &[f32] = input.as_slice()?;
        let d_i = input_dim_sizes[i];
        for outer in 0..outer_count {
            for dim_pos in 0..d_i {
                let src_off = (outer * d_i + dim_pos) * inner_count;
                let dst_off = (outer * total_dim + dim_offset + dim_pos) * inner_count;
                out_view[dst_off..dst_off + inner_count]
                    .copy_from_slice(&in_view[src_off..src_off + inner_count]);
            }
        }
        dim_offset += d_i;
    }
    Ok(())
}

// =============================================================================
// Scalar / clamp / pow / extrema (f32)
// =============================================================================

/// Affine transformation: `out[i] = mul * input[i] + add`. The
/// pipelined executor maps `Op::AddScalar(c)` as `mul=1, add=c`
/// and `Op::MulScalar(c)` as `mul=c, add=0`, so this single kernel
/// covers both.
pub fn affine_f32(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    mul: f32,
    add: f32,
) -> Result<()> {
    check_lens_2("affine_f32", input.len_bytes(), out.len_bytes())?;
    let in_view: &[f32] = input.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    for (i, slot) in out_view.iter_mut().enumerate() {
        *slot = mul * in_view[i] + add;
    }
    Ok(())
}

/// Element-wise clamp: `out[i] = clamp(input[i], min, max)`.
pub fn clamp_f32(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    min: f32,
    max: f32,
) -> Result<()> {
    check_lens_2("clamp_f32", input.len_bytes(), out.len_bytes())?;
    if min > max {
        return Err(Error::Msg(format!(
            "clamp_f32: min ({min}) > max ({max})"
        ))
        .bt());
    }
    let in_view: &[f32] = input.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    for (i, slot) in out_view.iter_mut().enumerate() {
        *slot = in_view[i].clamp(min, max);
    }
    Ok(())
}

/// Element-wise integer power: `out[i] = input[i].powi(exp)`.
pub fn powi_f32(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    exp: i32,
) -> Result<()> {
    check_lens_2("powi_f32", input.len_bytes(), out.len_bytes())?;
    let in_view: &[f32] = input.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    for (i, slot) in out_view.iter_mut().enumerate() {
        *slot = in_view[i].powi(exp);
    }
    Ok(())
}

binary_f32_kernel!(maximum_f32, |a: f32, b: f32| a.max(b), "Element-wise `f32` maximum: `out[i] = max(lhs[i], rhs[i])`. NaN handling follows `f32::max` (NaN-propagating per IEEE-754).");
binary_f32_kernel!(minimum_f32, |a: f32, b: f32| a.min(b), "Element-wise `f32` minimum: `out[i] = min(lhs[i], rhs[i])`. NaN handling follows `f32::min`.");

// =============================================================================
// Dtype conversion (Cast)
// =============================================================================

/// Generate a typed dtype-conversion kernel of the form
/// `out[i] = convert(input[i])`. Validates that the input byte
/// length is a multiple of the source type's size and that the
/// output byte length matches `elem_count * size_of::<TOut>()`.
/// Output is pre-allocated by the caller.
macro_rules! cast_kernel {
    ($name:ident, $TIn:ty, $TOut:ty, $convert:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(input: &CpuStorageBytes, out: &mut CpuStorageBytes) -> Result<()> {
            let in_size = std::mem::size_of::<$TIn>();
            let out_size = std::mem::size_of::<$TOut>();
            if in_size == 0 || input.len_bytes() % in_size != 0 {
                return Err(Error::Msg(format!(
                    "{}: input bytes {} not a multiple of {} ({})",
                    stringify!($name),
                    input.len_bytes(),
                    in_size,
                    stringify!($TIn),
                ))
                .bt());
            }
            let elem_count = input.len_bytes() / in_size;
            let want_out = elem_count.saturating_mul(out_size);
            if out.len_bytes() != want_out {
                return Err(Error::Msg(format!(
                    "{}: output bytes {} doesn't match input elem count {} \
                     × {} ({}) = {}",
                    stringify!($name),
                    out.len_bytes(),
                    elem_count,
                    out_size,
                    stringify!($TOut),
                    want_out,
                ))
                .bt());
            }
            let in_view: &[$TIn] = input.as_slice()?;
            let out_view: &mut [$TOut] = out.as_slice_mut()?;
            let convert: fn($TIn) -> $TOut = $convert;
            for (i, slot) in out_view.iter_mut().enumerate() {
                *slot = convert(in_view[i]);
            }
            Ok(())
        }
    };
}

cast_kernel!(
    cast_f32_to_f64,
    f32, f64,
    |x: f32| x as f64,
    "Convert `f32` → `f64`. Lossless widening."
);
cast_kernel!(
    cast_f64_to_f32,
    f64, f32,
    |x: f64| x as f32,
    "Convert `f64` → `f32`. Lossy narrowing per IEEE-754 rounding."
);
cast_kernel!(
    cast_f32_to_bf16,
    f32, half::bf16,
    half::bf16::from_f32,
    "Convert `f32` → `bf16`. Lossy narrowing — keeps the f32 exponent and the top mantissa bits."
);
cast_kernel!(
    cast_bf16_to_f32,
    half::bf16, f32,
    |x: half::bf16| x.to_f32(),
    "Convert `bf16` → `f32`. Lossless widening (bf16 is a strict subset of f32)."
);
cast_kernel!(
    cast_f32_to_f16,
    f32, half::f16,
    half::f16::from_f32,
    "Convert `f32` → `f16`. Lossy narrowing — clips to f16 range with NaN/inf preserved."
);
cast_kernel!(
    cast_f16_to_f32,
    half::f16, f32,
    |x: half::f16| x.to_f32(),
    "Convert `f16` → `f32`. Lossless widening within f16's representable range."
);

// =============================================================================
// Matrix multiplication (f32)
// =============================================================================

/// Batched row-major `f32` matrix multiply with optional GQA-style
/// batch broadcasting. For each output batch index, the kernel
/// runs the textbook (i, k, j) triple loop on the corresponding
/// `[m, k] @ [k, n]` slice.
///
/// Per-axis the batch dims either match (`lhs_batch_dims[i] ==
/// rhs_batch_dims[i]`) or follow GQA-style divisibility
/// (`lhs_dim > rhs_dim && lhs_dim % rhs_dim == 0`). The kernel
/// maps each lhs batch slot to the corresponding rhs slot via
/// `rhs_axis_idx = lhs_axis_idx / n_rep_axis`. Equal-batch and
/// rank-2 cases fall out as `n_rep == 1` everywhere.
///
/// Inputs are assumed contiguous in row-major order (the pipelined
/// executor's auto-Contiguize pass guarantees this). Output is
/// pre-allocated and overwritten by this kernel.
///
/// This is correctness-first; vendor BLAS backends will eclipse
/// it on performance once they're wired into the unified path.
pub fn matmul_f32(
    lhs: &CpuStorageBytes,
    rhs: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if lhs_batch_dims.len() != rhs_batch_dims.len() {
        return Err(Error::Msg(format!(
            "matmul_f32: batch ranks must match (lhs={}, rhs={}); fuel-graph's \
             auto-broadcast equalizes them at graph construction time",
            lhs_batch_dims.len(),
            rhs_batch_dims.len(),
        ))
        .bt());
    }
    let batch_rank = lhs_batch_dims.len();
    // Per-axis n_rep validation: matched (n_rep=1) or GQA (lhs is a
    // multiple of rhs).
    let mut n_rep: Vec<usize> = Vec::with_capacity(batch_rank);
    for i in 0..batch_rank {
        let la = lhs_batch_dims[i];
        let ra = rhs_batch_dims[i];
        if la == ra {
            n_rep.push(1);
        } else if ra > 0 && la > ra && la % ra == 0 {
            n_rep.push(la / ra);
        } else {
            return Err(Error::Msg(format!(
                "matmul_f32: batch dim {i} disallowed combination (lhs={la}, rhs={ra}); \
                 must be equal or GQA-divisible (lhs > rhs && lhs % rhs == 0)",
            ))
            .bt());
        }
    }
    let elem = std::mem::size_of::<f32>();
    let lhs_per_batch = m.saturating_mul(k);
    let rhs_per_batch = k.saturating_mul(n);
    let out_per_batch = m.saturating_mul(n);
    let lhs_batch_count: usize = lhs_batch_dims.iter().product::<usize>().max(1);
    let rhs_batch_count: usize = rhs_batch_dims.iter().product::<usize>().max(1);
    let need_lhs = lhs_batch_count.saturating_mul(lhs_per_batch).saturating_mul(elem);
    let need_rhs = rhs_batch_count.saturating_mul(rhs_per_batch).saturating_mul(elem);
    let need_out = lhs_batch_count.saturating_mul(out_per_batch).saturating_mul(elem);
    if lhs.len_bytes() != need_lhs {
        return Err(Error::Msg(format!(
            "matmul_f32: lhs bytes={} doesn't match shape {:?} + [{m}, {k}] (f32)",
            lhs.len_bytes(),
            lhs_batch_dims,
        ))
        .bt());
    }
    if rhs.len_bytes() != need_rhs {
        return Err(Error::Msg(format!(
            "matmul_f32: rhs bytes={} doesn't match shape {:?} + [{k}, {n}] (f32)",
            rhs.len_bytes(),
            rhs_batch_dims,
        ))
        .bt());
    }
    if out.len_bytes() != need_out {
        return Err(Error::Msg(format!(
            "matmul_f32: out bytes={} doesn't match shape {:?} + [{m}, {n}] (f32)",
            out.len_bytes(),
            lhs_batch_dims,
        ))
        .bt());
    }
    let lhs_view: &[f32] = lhs.as_slice()?;
    let rhs_view: &[f32] = rhs.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    for slot in out_view.iter_mut() {
        *slot = 0.0;
    }
    // Decode lhs's flat batch index into a multi-index, map per-axis
    // to rhs's multi-index, encode rhs's flat batch index. Reuses
    // two scratch buffers across iterations.
    let mut lhs_multi = vec![0usize; batch_rank];
    let mut rhs_multi = vec![0usize; batch_rank];
    for b in 0..lhs_batch_count {
        // Decode b into lhs_multi (row-major over lhs_batch_dims).
        let mut rem = b;
        for d in (0..batch_rank).rev() {
            let s = lhs_batch_dims[d];
            lhs_multi[d] = rem % s;
            rem /= s;
        }
        // Per-axis GQA mapping.
        for d in 0..batch_rank {
            rhs_multi[d] = lhs_multi[d] / n_rep[d];
        }
        // Encode rhs's flat batch index.
        let mut rhs_b = 0usize;
        for d in 0..batch_rank {
            rhs_b = rhs_b * rhs_batch_dims[d] + rhs_multi[d];
        }
        let lhs_off = b * lhs_per_batch;
        let rhs_off = rhs_b * rhs_per_batch;
        let out_off = b * out_per_batch;
        for i in 0..m {
            for kk in 0..k {
                let a = lhs_view[lhs_off + i * k + kk];
                let rhs_row_off = rhs_off + kk * n;
                let out_row_off = out_off + i * n;
                for j in 0..n {
                    out_view[out_row_off + j] += a * rhs_view[rhs_row_off + j];
                }
            }
        }
    }
    Ok(())
}

// =============================================================================
// 2D Convolution (f32)
// =============================================================================

/// Direct (no-im2col) 2D convolution forward pass on `f32`.
///
/// Shapes:
///   x:      [N, Cin,                Hin, Win]
///   weight: [Cout, Cin/groups,      Kh,  Kw ]
///   bias:   optional [Cout]
///   out:    [N, Cout,               Hout, Wout]
///
/// Out-of-bounds reads (from padding) yield 0. Groups partition Cin
/// and Cout into `groups` even chunks; output channel `co` reads
/// from input channels `(co / Co_per_group) * Ci_per_group ..`.
///
/// Correctness-first; vendor backends (cuDNN, MKL-DNN) will
/// dramatically outperform this once they're wired.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32(
    x: &CpuStorageBytes,
    weight: &CpuStorageBytes,
    bias: Option<&CpuStorageBytes>,
    out: &mut CpuStorageBytes,
    x_shape: [usize; 4],
    w_shape: [usize; 4],
    out_shape: [usize; 4],
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> Result<()> {
    let [n, cin, h_in, w_in] = x_shape;
    let [cout, cin_per_group, kh, kw] = w_shape;
    let [n_out, cout_out, h_out, w_out] = out_shape;
    if n != n_out {
        return Err(Error::Msg(format!(
            "conv2d_f32: input batch {n} != output batch {n_out}"
        ))
        .bt());
    }
    if cout != cout_out {
        return Err(Error::Msg(format!(
            "conv2d_f32: weight Cout {cout} != output Cout {cout_out}"
        ))
        .bt());
    }
    if groups == 0 || cin % groups != 0 || cout % groups != 0 {
        return Err(Error::Msg(format!(
            "conv2d_f32: groups={groups} must divide Cin={cin} and Cout={cout}"
        ))
        .bt());
    }
    if cin / groups != cin_per_group {
        return Err(Error::Msg(format!(
            "conv2d_f32: weight expects Cin/group={cin_per_group}, but \
             Cin/groups = {cin}/{groups} = {}",
            cin / groups,
        ))
        .bt());
    }
    let cout_per_group = cout / groups;
    let elem = std::mem::size_of::<f32>();
    let need_x = n.saturating_mul(cin).saturating_mul(h_in).saturating_mul(w_in)
        .saturating_mul(elem);
    let need_w = cout.saturating_mul(cin_per_group).saturating_mul(kh).saturating_mul(kw)
        .saturating_mul(elem);
    let need_out = n.saturating_mul(cout).saturating_mul(h_out).saturating_mul(w_out)
        .saturating_mul(elem);
    if x.len_bytes() != need_x {
        return Err(Error::Msg(format!(
            "conv2d_f32: x bytes={} doesn't match shape {:?}",
            x.len_bytes(), x_shape,
        ))
        .bt());
    }
    if weight.len_bytes() != need_w {
        return Err(Error::Msg(format!(
            "conv2d_f32: weight bytes={} doesn't match shape {:?}",
            weight.len_bytes(), w_shape,
        ))
        .bt());
    }
    if out.len_bytes() != need_out {
        return Err(Error::Msg(format!(
            "conv2d_f32: out bytes={} doesn't match shape {:?}",
            out.len_bytes(), out_shape,
        ))
        .bt());
    }
    if let Some(b) = bias {
        if b.len_bytes() != cout * elem {
            return Err(Error::Msg(format!(
                "conv2d_f32: bias bytes={} doesn't match Cout={cout} (f32)",
                b.len_bytes(),
            ))
            .bt());
        }
    }
    let x_view: &[f32] = x.as_slice()?;
    let w_view: &[f32] = weight.as_slice()?;
    let bias_view: Option<&[f32]> = match bias {
        Some(b) => Some(b.as_slice()?),
        None => None,
    };
    let out_view: &mut [f32] = out.as_slice_mut()?;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    // Output element [b, co, oh, ow] = bias[co] + sum over (ci_inner, kh_i, kw_i) of
    //   weight[co, ci_inner, kh_i, kw_i] * x[b, group_offset + ci_inner, in_h, in_w]
    // where in_h = oh * sh + kh_i * dh - ph, in_w = ow * sw + kw_i * dw - pw.
    for b in 0..n {
        for co in 0..cout {
            let group = co / cout_per_group;
            let ci_offset = group * cin_per_group;
            let bias_val = bias_view.map(|bv| bv[co]).unwrap_or(0.0);
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc: f32 = bias_val;
                    for ci in 0..cin_per_group {
                        for kh_i in 0..kh {
                            let in_h = (oh * sh + kh_i * dh) as isize - ph as isize;
                            if in_h < 0 || in_h as usize >= h_in {
                                continue;
                            }
                            let in_h = in_h as usize;
                            for kw_i in 0..kw {
                                let in_w = (ow * sw + kw_i * dw) as isize - pw as isize;
                                if in_w < 0 || in_w as usize >= w_in {
                                    continue;
                                }
                                let in_w = in_w as usize;
                                let x_idx = ((b * cin + (ci_offset + ci)) * h_in + in_h) * w_in + in_w;
                                let w_idx = ((co * cin_per_group + ci) * kh + kh_i) * kw + kw_i;
                                acc += x_view[x_idx] * w_view[w_idx];
                            }
                        }
                    }
                    let out_idx = ((b * cout + co) * h_out + oh) * w_out + ow;
                    out_view[out_idx] = acc;
                }
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
    fn rope_f32_identity_when_cos_one_sin_zero() {
        // cos=1, sin=0 everywhere → out == x.
        let x = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let cos = CpuStorageBytes::from_slice(&[1.0_f32, 1.0, 1.0, 1.0]);
        let sin = CpuStorageBytes::from_slice(&[0.0_f32, 0.0, 0.0, 0.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4 * 4);
        // outer=1, seq=1, head_dim=4, h=2
        rope_f32(&x, &cos, &sin, &mut out, 1, 1, 4).expect("rope");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_f32_pi_over_two_swaps_with_sign() {
        // cos=0, sin=1 (i.e. θ = π/2 everywhere). Then:
        //   out[i]   = x[i]*0 - x[i+h]*1 = -x[i+h]
        //   out[i+h] = x[i+h]*0 + x[i]*1 = x[i]
        // So [a, b, c, d] (head_dim=4, h=2) → [-c, -d, a, b].
        let x = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let cos = CpuStorageBytes::from_slice(&[0.0_f32, 0.0, 0.0, 0.0]);
        let sin = CpuStorageBytes::from_slice(&[1.0_f32, 1.0, 1.0, 1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4 * 4);
        rope_f32(&x, &cos, &sin, &mut out, 1, 1, 4).expect("rope");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn rope_f32_broadcasts_cos_sin_over_outer() {
        // outer=2, seq=1, head_dim=2, h=1. Shared cos=[0, 0], sin=[1, 1].
        // x outer 0: [1, 2] → [-2, 1]
        // x outer 1: [10, 20] → [-20, 10]
        let x = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 10.0, 20.0]);
        let cos = CpuStorageBytes::from_slice(&[0.0_f32, 0.0]);
        let sin = CpuStorageBytes::from_slice(&[1.0_f32, 1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4 * 4);
        rope_f32(&x, &cos, &sin, &mut out, 2, 1, 2).expect("rope");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[-2.0, 1.0, -20.0, 10.0]);
    }

    #[test]
    fn rope_f32_rejects_odd_head_dim() {
        let x = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]);
        let cos = CpuStorageBytes::from_slice(&[1.0_f32, 1.0, 1.0]);
        let sin = CpuStorageBytes::from_slice(&[0.0_f32, 0.0, 0.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 4);
        // head_dim=3 is odd
        let r = rope_f32(&x, &cos, &sin, &mut out, 1, 1, 3);
        assert!(r.is_err());
    }

    #[test]
    fn gather_f32_along_inner_dim() {
        // source [2, 4]:
        //   row 0: [10, 20, 30, 40]
        //   row 1: [50, 60, 70, 80]
        // indices [2, 3]:
        //   row 0: [0, 2, 1]
        //   row 1: [3, 0, 0]
        // gather dim=1:
        //   out[0, j] = source[0, indices[0, j]]
        //   out[1, j] = source[1, indices[1, j]]
        let source = CpuStorageBytes::from_slice(&[
            10.0_f32, 20.0, 30.0, 40.0,
            50.0, 60.0, 70.0, 80.0,
        ]);
        let indices = CpuStorageBytes::from_slice(&[0u32, 2, 1, 3, 0, 0]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 3 * 4);
        gather_f32(&source, &indices, &mut out, &[2, 4], &[2, 3], 1).expect("gather");
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[10.0, 30.0, 20.0, 80.0, 50.0, 50.0]
        );
    }

    #[test]
    fn gather_f32_along_outer_dim() {
        // source [3, 2]:
        //   row 0: [1, 2]
        //   row 1: [3, 4]
        //   row 2: [5, 6]
        // indices [4, 2]:
        //   [[0, 1],
        //    [1, 0],
        //    [2, 2],
        //    [0, 0]]
        // gather dim=0:
        //   out[i, j] = source[indices[i, j], j]
        let source = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices = CpuStorageBytes::from_slice(&[0u32, 1, 1, 0, 2, 2, 0, 0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4 * 2 * 4);
        gather_f32(&source, &indices, &mut out, &[3, 2], &[4, 2], 0).expect("gather");
        // row 0: source[0,0]=1, source[1,1]=4 → [1, 4]
        // row 1: source[1,0]=3, source[0,1]=2 → [3, 2]
        // row 2: source[2,0]=5, source[2,1]=6 → [5, 6]
        // row 3: source[0,0]=1, source[0,1]=2 → [1, 2]
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 1.0, 2.0]
        );
    }

    #[test]
    fn gather_f32_rejects_out_of_bounds_index() {
        let source = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]);
        let indices = CpuStorageBytes::from_slice(&[0u32, 5]); // 5 OOB for dim 3
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4);
        let r = gather_f32(&source, &indices, &mut out, &[3], &[2], 0);
        assert!(r.is_err());
    }

    #[test]
    fn gather_f32_rejects_rank_mismatch() {
        let source = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]);
        let indices = CpuStorageBytes::from_slice(&[0u32, 1]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4);
        // source rank=1, output rank=2 → must error
        let r = gather_f32(&source, &indices, &mut out, &[3], &[1, 2], 0);
        assert!(r.is_err());
    }

    #[test]
    fn index_select_f32_embedding_lookup() {
        // Embedding table [4, 3]: rows 0..3 are different.
        // Indices [u32; 3] = [2, 0, 2]; output [3, 3] picks
        // rows 2, 0, 2 in that order.
        let table = CpuStorageBytes::from_slice(&[
            10.0_f32, 11.0, 12.0,    // row 0
            20.0, 21.0, 22.0,        // row 1
            30.0, 31.0, 32.0,        // row 2
            40.0, 41.0, 42.0,        // row 3
        ]);
        let indices = CpuStorageBytes::from_slice(&[2u32, 0, 2]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 3 * 4);
        index_select_f32(&table, &indices, &mut out, 1, 4, 3, 3).expect("index_select");
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[
                30.0, 31.0, 32.0,
                10.0, 11.0, 12.0,
                30.0, 31.0, 32.0,
            ]
        );
    }

    #[test]
    fn index_select_f32_inner_dim() {
        // source [2, 3]: outer=2, dim_size=3, inner=1.
        // Pick along dim=1 with indices [2, 0]: output [2, 2].
        let source = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices = CpuStorageBytes::from_slice(&[2u32, 0]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 2 * 4);
        index_select_f32(&source, &indices, &mut out, 2, 3, 2, 1).expect("index_select");
        // outer 0: row [1, 2, 3], pick (2, 0) → [3, 1]
        // outer 1: row [4, 5, 6], pick (2, 0) → [6, 4]
        assert_eq!(out.as_slice::<f32>().unwrap(), &[3.0, 1.0, 6.0, 4.0]);
    }

    #[test]
    fn index_select_f32_rejects_out_of_bounds_index() {
        let source = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]);
        let indices = CpuStorageBytes::from_slice(&[5u32]); // out of bounds for dim 3
        let mut out = CpuStorageBytes::from_zero_bytes(4);
        let r = index_select_f32(&source, &indices, &mut out, 1, 3, 1, 1);
        assert!(r.is_err());
    }

    #[test]
    fn rms_norm_last_dim_f32_basic() {
        // Input [3, 4]: rms = sqrt(mean(9, 16)) = sqrt(12.5)
        // out = [3 / sqrt(12.5), 4 / sqrt(12.5)]
        let input = CpuStorageBytes::from_slice(&[3.0_f32, 4.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4);
        rms_norm_last_dim_f32(&input, &mut out, 1, 2, 0.0).expect("rms_norm");
        let result: &[f32] = out.as_slice().unwrap();
        let rms = (12.5_f32).sqrt();
        assert!((result[0] - 3.0 / rms).abs() < 1e-6);
        assert!((result[1] - 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_last_dim_f32_eps_prevents_zero_division() {
        // All-zero row → mean_sq = 0; without eps, would divide by 0.
        let input = CpuStorageBytes::from_slice(&[0.0_f32, 0.0, 0.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 4);
        rms_norm_last_dim_f32(&input, &mut out, 1, 3, 1e-5).expect("rms_norm");
        let result: &[f32] = out.as_slice().unwrap();
        for v in result {
            assert!(v.is_finite() && *v == 0.0);
        }
    }

    #[test]
    fn layer_norm_last_dim_f32_zero_mean_unit_var() {
        // Input [1, 2, 3]: mean = 2; var = (1 + 0 + 1) / 3 = 2/3
        // inv_std = 1/sqrt(2/3 + eps)
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 4);
        layer_norm_last_dim_f32(&input, &mut out, 1, 3, 0.0).expect("layer_norm");
        let result: &[f32] = out.as_slice().unwrap();
        // Output mean should be 0; output stddev should be 1.
        let out_sum: f32 = result.iter().sum();
        assert!(out_sum.abs() < 1e-6);
        let out_mean = out_sum / 3.0;
        let out_var: f32 = result.iter().map(|v| (v - out_mean).powi(2)).sum::<f32>() / 3.0;
        assert!((out_var - 1.0).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_last_dim_f32_two_rows_independent() {
        // Two rows of 3; each row's output should have mean ≈ 0.
        let input = CpuStorageBytes::from_slice(&[
            1.0_f32, 2.0, 3.0,
            10.0, 20.0, 30.0,
        ]);
        let mut out = CpuStorageBytes::from_zero_bytes(6 * 4);
        layer_norm_last_dim_f32(&input, &mut out, 2, 3, 0.0).expect("layer_norm");
        let result: &[f32] = out.as_slice().unwrap();
        let r0_mean: f32 = result[..3].iter().sum::<f32>() / 3.0;
        let r1_mean: f32 = result[3..].iter().sum::<f32>() / 3.0;
        assert!(r0_mean.abs() < 1e-6);
        assert!(r1_mean.abs() < 1e-6);
    }

    #[test]
    fn softmax_last_dim_f32_uniform_row() {
        // [1, 1, 1, 1] → uniform softmax = 0.25 each
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 1.0, 1.0, 1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4 * 4);
        softmax_last_dim_f32(&input, &mut out, 1, 4).expect("softmax");
        let result: &[f32] = out.as_slice().unwrap();
        for v in result {
            assert!((v - 0.25).abs() < 1e-7);
        }
    }

    #[test]
    fn softmax_last_dim_f32_sums_to_one_per_row() {
        // Two rows of 3; arbitrary values; each row should sum to 1.
        let input = CpuStorageBytes::from_slice(&[
            1.0_f32, 2.0, 3.0,
            -1.0, 0.0, 1.0,
        ]);
        let mut out = CpuStorageBytes::from_zero_bytes(6 * 4);
        softmax_last_dim_f32(&input, &mut out, 2, 3).expect("softmax");
        let result: &[f32] = out.as_slice().unwrap();
        let row0_sum: f32 = result[..3].iter().sum();
        let row1_sum: f32 = result[3..].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-6);
        assert!((row1_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_last_dim_f32_numerical_stability_at_large_values() {
        // [1000, 1001, 1002] — without max subtraction, exp would
        // overflow. Stable softmax should still produce finite output.
        let input = CpuStorageBytes::from_slice(&[1000.0_f32, 1001.0, 1002.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 4);
        softmax_last_dim_f32(&input, &mut out, 1, 3).expect("softmax");
        let result: &[f32] = out.as_slice().unwrap();
        for v in result {
            assert!(v.is_finite(), "softmax must not overflow at large inputs");
        }
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_last_dim_f32_known_values() {
        // [0, 1] → softmax = [1/(1+e), e/(1+e)]
        let input = CpuStorageBytes::from_slice(&[0.0_f32, 1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4);
        softmax_last_dim_f32(&input, &mut out, 1, 2).expect("softmax");
        let result: &[f32] = out.as_slice().unwrap();
        let e = std::f32::consts::E;
        let expected = [1.0 / (1.0 + e), e / (1.0 + e)];
        for (got, want) in result.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn softmax_last_dim_f32_size_mismatch_errors() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]); // 3 elements
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 4);
        // Claim 2 rows of 4 = 8 elements but only 3 — must error.
        let r = softmax_last_dim_f32(&input, &mut out, 2, 4);
        assert!(r.is_err());
    }

    #[test]
    fn concat_f32_along_inner_dim() {
        // Two [2, 3] tensors concatenated along dim 1 → [2, 6].
        // a = [[1, 2, 3], [4, 5, 6]]
        // b = [[7, 8, 9], [10, 11, 12]]
        // concat dim 1 = [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = CpuStorageBytes::from_slice(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(12 * 4);

        // dim=1: outer=2 (rows), inner=1 (1D inside each cell)
        concat_f32(&[&a, &b], &mut out, 2, &[3, 3], 1).expect("concat");
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn concat_f32_along_outer_dim() {
        // Two [2, 3] tensors concatenated along dim 0 → [4, 3].
        // For dim=0: outer_count=1 (no dims before), inner_count=3.
        // Input dim sizes are 2 and 2.
        // Output: [a-row-0, a-row-1, b-row-0, b-row-1].
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = CpuStorageBytes::from_slice(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(12 * 4);

        concat_f32(&[&a, &b], &mut out, 1, &[2, 2], 3).expect("concat");
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn concat_f32_three_inputs() {
        // Three [2] tensors concatenated → [6].
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let b = CpuStorageBytes::from_slice(&[3.0_f32, 4.0]);
        let c = CpuStorageBytes::from_slice(&[5.0_f32, 6.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(6 * 4);
        concat_f32(&[&a, &b, &c], &mut out, 1, &[2, 2, 2], 1).expect("concat");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn concat_f32_size_mismatch_errors() {
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let b = CpuStorageBytes::from_slice(&[3.0_f32, 4.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(8); // expects 4 elements but only 8 bytes
        let r = concat_f32(&[&a, &b], &mut out, 1, &[3, 2], 1); // claims a has 3 but it has 2
        assert!(r.is_err());
    }

    #[test]
    fn affine_f32_basic() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, -3.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        // y = 2 * x + 1
        affine_f32(&input, &mut out, 2.0, 1.0).expect("affine");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[3.0, 5.0, -5.0]);
    }

    #[test]
    fn affine_f32_handles_addscalar_and_mulscalar() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());

        // AddScalar(10): mul=1, add=10
        affine_f32(&input, &mut out, 1.0, 10.0).expect("addscalar");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[11.0, 12.0, 13.0]);

        // MulScalar(3): mul=3, add=0
        affine_f32(&input, &mut out, 3.0, 0.0).expect("mulscalar");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[3.0, 6.0, 9.0]);
    }

    #[test]
    fn clamp_f32_basic() {
        let input = CpuStorageBytes::from_slice(&[-5.0_f32, -1.0, 0.5, 3.0, 100.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        clamp_f32(&input, &mut out, -2.0, 2.0).expect("clamp");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[-2.0, -1.0, 0.5, 2.0, 2.0]);
    }

    #[test]
    fn clamp_f32_rejects_inverted_bounds() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        let r = clamp_f32(&input, &mut out, 5.0, 1.0);
        assert!(r.is_err());
    }

    #[test]
    fn powi_f32_basic() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        // exp = 3
        powi_f32(&input, &mut out, 3).expect("powi");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 8.0, 27.0, 64.0]);
        // exp = 0 → 1.0 everywhere
        powi_f32(&input, &mut out, 0).expect("powi");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 1.0, 1.0, 1.0]);
        // exp = -1 → reciprocal
        let pos = CpuStorageBytes::from_slice(&[2.0_f32, 4.0, 8.0]);
        let mut out2 = CpuStorageBytes::from_zero_bytes(pos.len_bytes());
        powi_f32(&pos, &mut out2, -1).expect("powi");
        assert_eq!(out2.as_slice::<f32>().unwrap(), &[0.5, 0.25, 0.125]);
    }

    #[test]
    fn maximum_minimum_f32_basic() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 5.0, -3.0]);
        let rhs = CpuStorageBytes::from_slice(&[2.0_f32, 1.0, -1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(lhs.len_bytes());

        maximum_f32(&lhs, &rhs, &mut out).expect("max");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[2.0, 5.0, -1.0]);

        minimum_f32(&lhs, &rhs, &mut out).expect("min");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 1.0, -3.0]);
    }

    #[test]
    fn conv2d_f32_identity_3x3_kernel() {
        // 1×1 input ch, 1×1 output ch, 3×3 kernel with center 1 and rest 0
        // → output equals input.
        let x = CpuStorageBytes::from_slice(&[
            1.0_f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let weight = CpuStorageBytes::from_slice(&[
            0.0_f32, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let mut out = CpuStorageBytes::from_zero_bytes(9 * 4);
        conv2d_f32(
            &x, &weight, None, &mut out,
            [1, 1, 3, 3], [1, 1, 3, 3], [1, 1, 3, 3],
            (1, 1), (1, 1), (1, 1), 1,
        ).expect("conv");
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn conv2d_f32_2x2_sum_kernel_no_padding() {
        // Input 1×1×3×3; kernel 2×2 of all-ones; no padding; stride 1.
        // Output shape 1×1×2×2; each output is sum of a 2×2 window.
        let x = CpuStorageBytes::from_slice(&[
            1.0_f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let weight = CpuStorageBytes::from_slice(&[1.0_f32, 1.0, 1.0, 1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4 * 4);
        conv2d_f32(
            &x, &weight, None, &mut out,
            [1, 1, 3, 3], [1, 1, 2, 2], [1, 1, 2, 2],
            (1, 1), (0, 0), (1, 1), 1,
        ).expect("conv");
        // Window at (0,0): 1+2+4+5 = 12
        // Window at (0,1): 2+3+5+6 = 16
        // Window at (1,0): 4+5+7+8 = 24
        // Window at (1,1): 5+6+8+9 = 28
        assert_eq!(out.as_slice::<f32>().unwrap(), &[12.0, 16.0, 24.0, 28.0]);
    }

    #[test]
    fn conv2d_f32_with_bias() {
        // Same as the 2×2 sum kernel test, plus a bias of 100 — every
        // output gets +100.
        let x = CpuStorageBytes::from_slice(&[
            1.0_f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let weight = CpuStorageBytes::from_slice(&[1.0_f32, 1.0, 1.0, 1.0]);
        let bias = CpuStorageBytes::from_slice(&[100.0_f32]);
        let mut out = CpuStorageBytes::from_zero_bytes(4 * 4);
        conv2d_f32(
            &x, &weight, Some(&bias), &mut out,
            [1, 1, 3, 3], [1, 1, 2, 2], [1, 1, 2, 2],
            (1, 1), (0, 0), (1, 1), 1,
        ).expect("conv");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[112.0, 116.0, 124.0, 128.0]);
    }

    #[test]
    fn conv2d_f32_padding_yields_zero_outside() {
        // 1×1×2×2 input, 2×2 kernel of ones, padding 1 → out shape 1×1×3×3.
        // Output[0,0] reads only x[0,0] (rest is padded zeros) → 1.
        let x = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let weight = CpuStorageBytes::from_slice(&[1.0_f32, 1.0, 1.0, 1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(9 * 4);
        conv2d_f32(
            &x, &weight, None, &mut out,
            [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 3, 3],
            (1, 1), (1, 1), (1, 1), 1,
        ).expect("conv");
        // Output positions:
        //   [0,0]: only x[0,0]=1
        //   [0,1]: x[0,0]+x[0,1]=1+2=3
        //   [0,2]: only x[0,1]=2
        //   [1,0]: x[0,0]+x[1,0]=1+3=4
        //   [1,1]: 1+2+3+4=10
        //   [1,2]: x[0,1]+x[1,1]=2+4=6
        //   [2,0]: only x[1,0]=3
        //   [2,1]: x[1,0]+x[1,1]=3+4=7
        //   [2,2]: only x[1,1]=4
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]
        );
    }

    #[test]
    fn conv2d_f32_depthwise_groups_equal_cin() {
        // 2 channels in, 2 channels out, groups=2 (depthwise). Each
        // output channel reads only its corresponding input channel.
        // x[0, ch=0] = [[1, 2], [3, 4]]; ch=1 = [[10, 20], [30, 40]]
        // weight: ch0 has all-ones 2x2; ch1 has identity
        let x = CpuStorageBytes::from_slice(&[
            1.0_f32, 2.0, 3.0, 4.0,    // ch 0
            10.0, 20.0, 30.0, 40.0,    // ch 1
        ]);
        // weight shape [Cout, Cin/groups=1, 2, 2] = [2, 1, 2, 2]
        let weight = CpuStorageBytes::from_slice(&[
            1.0_f32, 1.0, 1.0, 1.0,    // co 0 sees ci 0
            1.0, 0.0, 0.0, 0.0,        // co 1 sees ci 1, but only top-left
        ]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4); // [1, 2, 1, 1]
        conv2d_f32(
            &x, &weight, None, &mut out,
            [1, 2, 2, 2], [2, 1, 2, 2], [1, 2, 1, 1],
            (1, 1), (0, 0), (1, 1), 2,
        ).expect("conv");
        // co 0: 1+2+3+4 = 10
        // co 1: 10 (only top-left of ch 1)
        assert_eq!(out.as_slice::<f32>().unwrap(), &[10.0, 10.0]);
    }

    #[test]
    fn cast_f32_to_f64_round_trip() {
        let input = CpuStorageBytes::from_slice(&[1.5_f32, -2.25, 100.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 8);
        cast_f32_to_f64(&input, &mut out).expect("cast");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[1.5_f64, -2.25, 100.0]);
    }

    #[test]
    fn cast_f64_to_f32_lossy() {
        let input = CpuStorageBytes::from_slice(&[1.5_f64, -2.25, 100.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 4);
        cast_f64_to_f32(&input, &mut out).expect("cast");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.5_f32, -2.25, 100.0]);
    }

    #[test]
    fn cast_bf16_round_trip_via_f32() {
        let input_f32 = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, -3.0, 0.5]);
        let mut bf16_buf = CpuStorageBytes::from_zero_bytes(4 * 2);
        cast_f32_to_bf16(&input_f32, &mut bf16_buf).expect("to_bf16");

        let mut back_f32 = CpuStorageBytes::from_zero_bytes(4 * 4);
        cast_bf16_to_f32(&bf16_buf, &mut back_f32).expect("to_f32");

        let result: &[f32] = back_f32.as_slice().unwrap();
        // bf16 has ~3 decimal digits of precision; these inputs round-trip exactly.
        assert_eq!(result, &[1.0_f32, 2.0, -3.0, 0.5]);
    }

    #[test]
    fn cast_f16_round_trip_via_f32() {
        let input_f32 = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, -3.0, 0.5]);
        let mut f16_buf = CpuStorageBytes::from_zero_bytes(4 * 2);
        cast_f32_to_f16(&input_f32, &mut f16_buf).expect("to_f16");

        let mut back_f32 = CpuStorageBytes::from_zero_bytes(4 * 4);
        cast_f16_to_f32(&f16_buf, &mut back_f32).expect("to_f32");

        let result: &[f32] = back_f32.as_slice().unwrap();
        // f16 has ~3 decimal digits of precision in this range.
        for (got, want) in result.iter().zip(&[1.0_f32, 2.0, -3.0, 0.5]) {
            assert!((got - want).abs() < 1e-3, "f16 round trip lost too much: {got} vs {want}");
        }
    }

    #[test]
    fn cast_size_mismatch_errors() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let mut out_wrong = CpuStorageBytes::from_zero_bytes(8); // should be 16 (2 f64)
        let r = cast_f32_to_f64(&input, &mut out_wrong);
        assert!(r.is_err());
    }

    #[test]
    fn matmul_f32_2x3_times_3x2() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = CpuStorageBytes::from_slice(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(16);

        matmul_f32(&lhs, &rhs, &mut out, &[], &[], 2, 2, 3).expect("matmul");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_f32_identity_returns_input() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let rhs = CpuStorageBytes::from_slice(&[1.0_f32, 0.0, 0.0, 1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(16);

        matmul_f32(&lhs, &rhs, &mut out, &[], &[], 2, 2, 2).expect("matmul");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_f32_inner_product_1x3_times_3x1() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]);
        let rhs = CpuStorageBytes::from_slice(&[4.0_f32, 5.0, 6.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4);

        matmul_f32(&lhs, &rhs, &mut out, &[], &[], 1, 1, 3).expect("matmul");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[32.0]);
    }

    #[test]
    fn matmul_f32_size_mismatch_errors() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let rhs = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(16);
        let r = matmul_f32(&lhs, &rhs, &mut out, &[], &[], 2, 2, 2);
        assert!(r.is_err(), "matmul must error on shape/byte mismatch");
    }

    #[test]
    fn matmul_f32_batched_2x_2x2_times_2x2() {
        let lhs = CpuStorageBytes::from_slice(&[
            1.0_f32, 2.0, 3.0, 4.0,
            1.0, 0.0, 0.0, 1.0,
        ]);
        let rhs = CpuStorageBytes::from_slice(&[
            5.0_f32, 6.0, 7.0, 8.0,
            10.0, 20.0, 30.0, 40.0,
        ]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4 * 4);

        matmul_f32(&lhs, &rhs, &mut out, &[2], &[2], 2, 2, 2).expect("matmul");
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[19.0, 22.0, 43.0, 50.0, 10.0, 20.0, 30.0, 40.0]
        );
    }

    /// GQA-style: lhs has 4 batch heads, rhs has 2; n_rep = 2.
    /// lhs heads {0, 1} share rhs head 0; lhs heads {2, 3} share rhs head 1.
    #[test]
    fn matmul_f32_gqa_4x_vs_2x() {
        // lhs shape [4, 1, 2]: heads 0..3 are [[1,2]], [[3,4]], [[5,6]], [[7,8]]
        let lhs = CpuStorageBytes::from_slice(&[
            1.0_f32, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ]);
        // rhs shape [2, 2, 1]: heads 0,1 are [[1],[0]], [[0],[1]]
        let rhs = CpuStorageBytes::from_slice(&[
            1.0_f32, 0.0,
            0.0, 1.0,
        ]);
        // output shape [4, 1, 1]:
        //   head 0: [[1, 2]] @ [[1], [0]] = [[1]]
        //   head 1: [[3, 4]] @ [[1], [0]] = [[3]]   (still rhs 0)
        //   head 2: [[5, 6]] @ [[0], [1]] = [[6]]   (rhs 1)
        //   head 3: [[7, 8]] @ [[0], [1]] = [[8]]   (rhs 1)
        let mut out = CpuStorageBytes::from_zero_bytes(4 * 4);

        matmul_f32(&lhs, &rhs, &mut out, &[4], &[2], 1, 1, 2).expect("gqa matmul");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 3.0, 6.0, 8.0]);
    }

    #[test]
    fn matmul_f32_gqa_rejects_non_divisible() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f32; 6]); // [3, 1, 2]
        let rhs = CpuStorageBytes::from_slice(&[1.0_f32; 4]); // [2, 2, 1]
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 4);
        // 3 is NOT a multiple of 2 — must error.
        let r = matmul_f32(&lhs, &rhs, &mut out, &[3], &[2], 1, 1, 2);
        assert!(r.is_err());
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
