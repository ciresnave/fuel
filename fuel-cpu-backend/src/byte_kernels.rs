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

/// Generate a binary kernel parameterized over the element type
/// `$T`: `out[i] = op(lhs[i], rhs[i])`.
///
/// Output is pre-allocated by the caller (the dispatch wrapper) and
/// must match the input byte length. The kernel writes into the
/// pre-allocated bytes; it never allocates.
macro_rules! binary_kernel {
    ($name:ident, $T:ty, $op:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(
            lhs: &CpuStorageBytes,
            rhs: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
        ) -> Result<()> {
            check_lens_3(stringify!($name), lhs.len_bytes(), rhs.len_bytes(), out.len_bytes())?;
            let lhs_view: &[$T] = lhs.as_slice()?;
            let rhs_view: &[$T] = rhs.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let op: fn($T, $T) -> $T = $op;
            for (i, slot) in out_view.iter_mut().enumerate() {
                *slot = op(lhs_view[i], rhs_view[i]);
            }
            Ok(())
        }
    };
}

/// Backward-compat alias for the previous `binary_f32_kernel!`
/// invocations. Existing call sites stay unchanged; new dtypes
/// declare via `binary_kernel!` directly.
macro_rules! binary_f32_kernel {
    ($name:ident, $op:expr, $doc:literal) => {
        binary_kernel!($name, f32, $op, $doc);
    };
}

binary_f32_kernel!(add_f32, |a, b| a + b, "Elementwise `f32` addition: `out[i] = lhs[i] + rhs[i]`.");
binary_f32_kernel!(sub_f32, |a, b| a - b, "Elementwise `f32` subtraction: `out[i] = lhs[i] - rhs[i]`.");
binary_f32_kernel!(mul_f32, |a, b| a * b, "Elementwise `f32` multiplication: `out[i] = lhs[i] * rhs[i]`.");
binary_f32_kernel!(div_f32, |a, b| a / b, "Elementwise `f32` division: `out[i] = lhs[i] / rhs[i]`. Division by zero yields IEEE-754 inf/NaN per platform.");

binary_kernel!(add_f64, f64, |a: f64, b: f64| a + b, "Elementwise `f64` addition.");
binary_kernel!(sub_f64, f64, |a: f64, b: f64| a - b, "Elementwise `f64` subtraction.");
binary_kernel!(mul_f64, f64, |a: f64, b: f64| a * b, "Elementwise `f64` multiplication.");
binary_kernel!(div_f64, f64, |a: f64, b: f64| a / b, "Elementwise `f64` division.");

// =============================================================================
// Elementwise unary kernels (f32)
// =============================================================================

/// Generate a unary kernel parameterized over the element type `$T`:
/// `out[i] = op(input[i])`.
///
/// Output is pre-allocated by the caller and must match the input
/// byte length.
macro_rules! unary_kernel {
    ($name:ident, $T:ty, $op:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(input: &CpuStorageBytes, out: &mut CpuStorageBytes) -> Result<()> {
            check_lens_2(stringify!($name), input.len_bytes(), out.len_bytes())?;
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let op: fn($T) -> $T = $op;
            for (i, slot) in out_view.iter_mut().enumerate() {
                *slot = op(in_view[i]);
            }
            Ok(())
        }
    };
}

/// Backward-compat alias for the previous `unary_f32_kernel!`
/// invocations.
macro_rules! unary_f32_kernel {
    ($name:ident, $op:expr, $doc:literal) => {
        unary_kernel!($name, f32, $op, $doc);
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

// f64 mirrors of the unary set above. Each is a one-liner via the
// parametric `unary_kernel!` macro.
unary_kernel!(relu_f64, f64, |x: f64| x.max(0.0), "Elementwise `f64` ReLU.");
unary_kernel!(neg_f64, f64, |x: f64| -x, "Elementwise `f64` negation.");
unary_kernel!(sqr_f64, f64, |x: f64| x * x, "Elementwise `f64` square.");
unary_kernel!(sqrt_f64, f64, |x: f64| x.sqrt(), "Elementwise `f64` square root.");
unary_kernel!(recip_f64, f64, |x: f64| 1.0 / x, "Elementwise `f64` reciprocal.");
unary_kernel!(abs_f64, f64, |x: f64| x.abs(), "Elementwise `f64` absolute value.");
unary_kernel!(tanh_f64, f64, |x: f64| x.tanh(), "Elementwise `f64` hyperbolic tangent.");
unary_kernel!(exp_f64, f64, |x: f64| x.exp(), "Elementwise `f64` exponential.");
unary_kernel!(log_f64, f64, |x: f64| x.ln(), "Elementwise `f64` natural log.");
unary_kernel!(sin_f64, f64, |x: f64| x.sin(), "Elementwise `f64` sine.");
unary_kernel!(cos_f64, f64, |x: f64| x.cos(), "Elementwise `f64` cosine.");
unary_kernel!(sigmoid_f64, f64, |x: f64| 1.0 / (1.0 + (-x).exp()), "Elementwise `f64` logistic sigmoid.");
unary_kernel!(silu_f64, f64, |x: f64| x / (1.0 + (-x).exp()), "Elementwise `f64` SiLU/Swish.");
unary_kernel!(step_f64, f64, |x: f64| if x > 0.0 { 1.0 } else { 0.0 }, "Elementwise `f64` Heaviside step.");

// f64 versions of the binary extrema (Maximum/Minimum). The f32
// versions sit later in the file (in the scalar/clamp/extrema
// section); placing the f64 mirrors here keeps the elementwise
// arithmetic block contiguous.
binary_kernel!(maximum_f64, f64, |a: f64, b: f64| a.max(b), "Elementwise `f64` maximum.");
binary_kernel!(minimum_f64, f64, |a: f64, b: f64| a.min(b), "Elementwise `f64` minimum.");

// =============================================================================
// bf16 / f16 elementwise — via-f32 round-trip
// =============================================================================
//
// `bf16` and `f16` lack native transcendentals (no `.exp()`,
// `.sin()`, etc. on `half::{bf16, f16}`), and even arithmetic via
// the native `Add`/`Sub`/`Mul`/`Div` impls truncates intermediate
// values. The standard CPU pattern — what NumPy/PyTorch do for
// these dtypes — is to widen each element to `f32`, do the work,
// and narrow back. Performance is sub-optimal (3 conversions per
// op) but correctness-first; vendor backends override this once
// they're wired.

binary_kernel!(add_bf16, half::bf16, |a: half::bf16, b: half::bf16| half::bf16::from_f32(a.to_f32() + b.to_f32()), "Elementwise `bf16` addition (via f32).");
binary_kernel!(sub_bf16, half::bf16, |a: half::bf16, b: half::bf16| half::bf16::from_f32(a.to_f32() - b.to_f32()), "Elementwise `bf16` subtraction (via f32).");
binary_kernel!(mul_bf16, half::bf16, |a: half::bf16, b: half::bf16| half::bf16::from_f32(a.to_f32() * b.to_f32()), "Elementwise `bf16` multiplication (via f32).");
binary_kernel!(div_bf16, half::bf16, |a: half::bf16, b: half::bf16| half::bf16::from_f32(a.to_f32() / b.to_f32()), "Elementwise `bf16` division (via f32).");
binary_kernel!(maximum_bf16, half::bf16, |a: half::bf16, b: half::bf16| half::bf16::from_f32(a.to_f32().max(b.to_f32())), "Elementwise `bf16` maximum (via f32).");
binary_kernel!(minimum_bf16, half::bf16, |a: half::bf16, b: half::bf16| half::bf16::from_f32(a.to_f32().min(b.to_f32())), "Elementwise `bf16` minimum (via f32).");

unary_kernel!(relu_bf16, half::bf16, |x: half::bf16| half::bf16::from_f32(x.to_f32().max(0.0)), "Elementwise `bf16` ReLU (via f32).");
unary_kernel!(neg_bf16, half::bf16, |x: half::bf16| half::bf16::from_f32(-x.to_f32()), "Elementwise `bf16` negation (via f32).");
unary_kernel!(sqr_bf16, half::bf16, |x: half::bf16| { let f = x.to_f32(); half::bf16::from_f32(f * f) }, "Elementwise `bf16` square (via f32).");
unary_kernel!(sqrt_bf16, half::bf16, |x: half::bf16| half::bf16::from_f32(x.to_f32().sqrt()), "Elementwise `bf16` square root (via f32).");
unary_kernel!(recip_bf16, half::bf16, |x: half::bf16| half::bf16::from_f32(1.0 / x.to_f32()), "Elementwise `bf16` reciprocal (via f32).");
unary_kernel!(abs_bf16, half::bf16, |x: half::bf16| half::bf16::from_f32(x.to_f32().abs()), "Elementwise `bf16` absolute value (via f32).");
unary_kernel!(tanh_bf16, half::bf16, |x: half::bf16| half::bf16::from_f32(x.to_f32().tanh()), "Elementwise `bf16` hyperbolic tangent (via f32).");
unary_kernel!(exp_bf16, half::bf16, |x: half::bf16| half::bf16::from_f32(x.to_f32().exp()), "Elementwise `bf16` exponential (via f32).");
unary_kernel!(log_bf16, half::bf16, |x: half::bf16| half::bf16::from_f32(x.to_f32().ln()), "Elementwise `bf16` natural log (via f32).");
unary_kernel!(sin_bf16, half::bf16, |x: half::bf16| half::bf16::from_f32(x.to_f32().sin()), "Elementwise `bf16` sine (via f32).");
unary_kernel!(cos_bf16, half::bf16, |x: half::bf16| half::bf16::from_f32(x.to_f32().cos()), "Elementwise `bf16` cosine (via f32).");
unary_kernel!(sigmoid_bf16, half::bf16, |x: half::bf16| { let f = x.to_f32(); half::bf16::from_f32(1.0 / (1.0 + (-f).exp())) }, "Elementwise `bf16` logistic sigmoid (via f32).");
unary_kernel!(silu_bf16, half::bf16, |x: half::bf16| { let f = x.to_f32(); half::bf16::from_f32(f / (1.0 + (-f).exp())) }, "Elementwise `bf16` SiLU/Swish (via f32).");
unary_kernel!(step_bf16, half::bf16, |x: half::bf16| half::bf16::from_f32(if x.to_f32() > 0.0 { 1.0 } else { 0.0 }), "Elementwise `bf16` Heaviside step (via f32).");

/// `bf16` GELU using the tanh approximation (mirror of [`gelu_f32`]).
pub fn gelu_bf16(input: &CpuStorageBytes, out: &mut CpuStorageBytes) -> Result<()> {
    check_lens_2("gelu_bf16", input.len_bytes(), out.len_bytes())?;
    let in_view: &[half::bf16] = input.as_slice()?;
    let out_view: &mut [half::bf16] = out.as_slice_mut()?;
    const COEFF: f32 = 0.797_884_56;
    for (i, slot) in out_view.iter_mut().enumerate() {
        let x = in_view[i].to_f32();
        let inner = COEFF * (x + 0.044_715 * x * x * x);
        *slot = half::bf16::from_f32(0.5 * x * (1.0 + inner.tanh()));
    }
    Ok(())
}

// f16 mirrors of the bf16 set above. Identical patterns; only the
// concrete type differs.

binary_kernel!(add_f16, half::f16, |a: half::f16, b: half::f16| half::f16::from_f32(a.to_f32() + b.to_f32()), "Elementwise `f16` addition (via f32).");
binary_kernel!(sub_f16, half::f16, |a: half::f16, b: half::f16| half::f16::from_f32(a.to_f32() - b.to_f32()), "Elementwise `f16` subtraction (via f32).");
binary_kernel!(mul_f16, half::f16, |a: half::f16, b: half::f16| half::f16::from_f32(a.to_f32() * b.to_f32()), "Elementwise `f16` multiplication (via f32).");
binary_kernel!(div_f16, half::f16, |a: half::f16, b: half::f16| half::f16::from_f32(a.to_f32() / b.to_f32()), "Elementwise `f16` division (via f32).");
binary_kernel!(maximum_f16, half::f16, |a: half::f16, b: half::f16| half::f16::from_f32(a.to_f32().max(b.to_f32())), "Elementwise `f16` maximum (via f32).");
binary_kernel!(minimum_f16, half::f16, |a: half::f16, b: half::f16| half::f16::from_f32(a.to_f32().min(b.to_f32())), "Elementwise `f16` minimum (via f32).");

unary_kernel!(relu_f16, half::f16, |x: half::f16| half::f16::from_f32(x.to_f32().max(0.0)), "Elementwise `f16` ReLU (via f32).");
unary_kernel!(neg_f16, half::f16, |x: half::f16| half::f16::from_f32(-x.to_f32()), "Elementwise `f16` negation (via f32).");
unary_kernel!(sqr_f16, half::f16, |x: half::f16| { let f = x.to_f32(); half::f16::from_f32(f * f) }, "Elementwise `f16` square (via f32).");
unary_kernel!(sqrt_f16, half::f16, |x: half::f16| half::f16::from_f32(x.to_f32().sqrt()), "Elementwise `f16` square root (via f32).");
unary_kernel!(recip_f16, half::f16, |x: half::f16| half::f16::from_f32(1.0 / x.to_f32()), "Elementwise `f16` reciprocal (via f32).");
unary_kernel!(abs_f16, half::f16, |x: half::f16| half::f16::from_f32(x.to_f32().abs()), "Elementwise `f16` absolute value (via f32).");
unary_kernel!(tanh_f16, half::f16, |x: half::f16| half::f16::from_f32(x.to_f32().tanh()), "Elementwise `f16` hyperbolic tangent (via f32).");
unary_kernel!(exp_f16, half::f16, |x: half::f16| half::f16::from_f32(x.to_f32().exp()), "Elementwise `f16` exponential (via f32).");
unary_kernel!(log_f16, half::f16, |x: half::f16| half::f16::from_f32(x.to_f32().ln()), "Elementwise `f16` natural log (via f32).");
unary_kernel!(sin_f16, half::f16, |x: half::f16| half::f16::from_f32(x.to_f32().sin()), "Elementwise `f16` sine (via f32).");
unary_kernel!(cos_f16, half::f16, |x: half::f16| half::f16::from_f32(x.to_f32().cos()), "Elementwise `f16` cosine (via f32).");
unary_kernel!(sigmoid_f16, half::f16, |x: half::f16| { let f = x.to_f32(); half::f16::from_f32(1.0 / (1.0 + (-f).exp())) }, "Elementwise `f16` logistic sigmoid (via f32).");
unary_kernel!(silu_f16, half::f16, |x: half::f16| { let f = x.to_f32(); half::f16::from_f32(f / (1.0 + (-f).exp())) }, "Elementwise `f16` SiLU/Swish (via f32).");
unary_kernel!(step_f16, half::f16, |x: half::f16| half::f16::from_f32(if x.to_f32() > 0.0 { 1.0 } else { 0.0 }), "Elementwise `f16` Heaviside step (via f32).");

/// `f16` GELU using the tanh approximation (mirror of [`gelu_f32`]).
pub fn gelu_f16(input: &CpuStorageBytes, out: &mut CpuStorageBytes) -> Result<()> {
    check_lens_2("gelu_f16", input.len_bytes(), out.len_bytes())?;
    let in_view: &[half::f16] = input.as_slice()?;
    let out_view: &mut [half::f16] = out.as_slice_mut()?;
    const COEFF: f32 = 0.797_884_56;
    for (i, slot) in out_view.iter_mut().enumerate() {
        let x = in_view[i].to_f32();
        let inner = COEFF * (x + 0.044_715 * x * x * x);
        *slot = half::f16::from_f32(0.5 * x * (1.0 + inner.tanh()));
    }
    Ok(())
}

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
// Index-add / scatter-add (f32 base + src, U32 indices)
// =============================================================================

/// Index-add along one dim with a rank-1 `u32` index tensor.
/// Inputs:
/// - `base` with shape `[outer, base_dim, inner]` (the destination's
///   prior contents; the kernel copies these into `out` first).
/// - `indices` with shape `[n_indices]` (`u32`). Each
///   `indices[i] ∈ [0, base_dim)` is the destination row.
/// - `src` with shape `[outer, n_indices, inner]`.
///
/// Updates: `out[outer, indices[i], inner] += src[outer, i, inner]`
/// for each `i ∈ 0..n_indices`. Out-of-bounds indices return a
/// typed Error.
pub fn index_add_f32(
    base: &CpuStorageBytes,
    indices: &CpuStorageBytes,
    src: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    outer_count: usize,
    base_dim_size: usize,
    n_indices: usize,
    inner_count: usize,
) -> Result<()> {
    let elem = std::mem::size_of::<f32>();
    let need_base = outer_count
        .saturating_mul(base_dim_size)
        .saturating_mul(inner_count)
        .saturating_mul(elem);
    let need_idx = n_indices.saturating_mul(std::mem::size_of::<u32>());
    let need_src = outer_count
        .saturating_mul(n_indices)
        .saturating_mul(inner_count)
        .saturating_mul(elem);
    if base.len_bytes() != need_base || out.len_bytes() != need_base {
        return Err(Error::Msg(format!(
            "index_add_f32: base bytes={} or out bytes={} doesn't match outer={outer_count} × base_dim={base_dim_size} × inner={inner_count} × {elem}",
            base.len_bytes(), out.len_bytes(),
        ))
        .bt());
    }
    if indices.len_bytes() != need_idx {
        return Err(Error::Msg(format!(
            "index_add_f32: indices bytes={} doesn't match n_indices={n_indices} × 4",
            indices.len_bytes(),
        ))
        .bt());
    }
    if src.len_bytes() != need_src {
        return Err(Error::Msg(format!(
            "index_add_f32: src bytes={} doesn't match outer={outer_count} × n={n_indices} × inner={inner_count} × {elem}",
            src.len_bytes(),
        ))
        .bt());
    }
    // Copy base into out as the starting point.
    out.bytes_mut().copy_from_slice(base.bytes());
    if n_indices == 0 {
        return Ok(());
    }
    let idx_view: &[u32] = indices.as_slice()?;
    let src_view: &[f32] = src.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    for i in 0..n_indices {
        let target = idx_view[i] as usize;
        if target >= base_dim_size {
            return Err(Error::Msg(format!(
                "index_add_f32: index {target} at position {i} out of bounds for base dim {base_dim_size}",
            ))
            .bt());
        }
        for outer in 0..outer_count {
            let src_off = (outer * n_indices + i) * inner_count;
            let dst_off = (outer * base_dim_size + target) * inner_count;
            for inner in 0..inner_count {
                out_view[dst_off + inner] += src_view[src_off + inner];
            }
        }
    }
    Ok(())
}

/// Generate an IndexAdd kernel parameterized over native arithmetic
/// type `$T` (f32 / f64). Accumulates in-place using `$T`'s `+=`.
macro_rules! index_add_native_kernel {
    ($name:ident, $T:ty, $T_size:expr) => {
        pub fn $name(
            base: &CpuStorageBytes,
            indices: &CpuStorageBytes,
            src: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            outer_count: usize,
            base_dim_size: usize,
            n_indices: usize,
            inner_count: usize,
        ) -> Result<()> {
            let elem = $T_size;
            let need_base = outer_count
                .saturating_mul(base_dim_size)
                .saturating_mul(inner_count)
                .saturating_mul(elem);
            let need_idx = n_indices.saturating_mul(std::mem::size_of::<u32>());
            let need_src = outer_count
                .saturating_mul(n_indices)
                .saturating_mul(inner_count)
                .saturating_mul(elem);
            if base.len_bytes() != need_base || out.len_bytes() != need_base {
                return Err(Error::Msg(format!(
                    "{}: base/out bytes don't match outer={outer_count} × base_dim={base_dim_size} × inner={inner_count} × {elem}",
                    stringify!($name),
                ))
                .bt());
            }
            if indices.len_bytes() != need_idx {
                return Err(Error::Msg(format!(
                    "{}: indices bytes={} doesn't match n_indices={n_indices} × 4",
                    stringify!($name), indices.len_bytes(),
                ))
                .bt());
            }
            if src.len_bytes() != need_src {
                return Err(Error::Msg(format!(
                    "{}: src bytes={} doesn't match outer={outer_count} × n={n_indices} × inner={inner_count} × {elem}",
                    stringify!($name), src.len_bytes(),
                ))
                .bt());
            }
            out.bytes_mut().copy_from_slice(base.bytes());
            if n_indices == 0 {
                return Ok(());
            }
            let idx_view: &[u32] = indices.as_slice()?;
            let src_view: &[$T] = src.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            for i in 0..n_indices {
                let target = idx_view[i] as usize;
                if target >= base_dim_size {
                    return Err(Error::Msg(format!(
                        "{}: index {target} at position {i} out of bounds for base dim {base_dim_size}",
                        stringify!($name),
                    ))
                    .bt());
                }
                for outer in 0..outer_count {
                    let src_off = (outer * n_indices + i) * inner_count;
                    let dst_off = (outer * base_dim_size + target) * inner_count;
                    for inner in 0..inner_count {
                        out_view[dst_off + inner] += src_view[src_off + inner];
                    }
                }
            }
            Ok(())
        }
    };
}

index_add_native_kernel!(index_add_f64, f64, std::mem::size_of::<f64>());

/// IndexAdd kernel for half-float types (`bf16` / `f16`).
/// Accumulates in f32 (widen → +=  → narrow back).
macro_rules! index_add_half_kernel {
    ($name:ident, $T:ty, $T_size:expr) => {
        pub fn $name(
            base: &CpuStorageBytes,
            indices: &CpuStorageBytes,
            src: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            outer_count: usize,
            base_dim_size: usize,
            n_indices: usize,
            inner_count: usize,
        ) -> Result<()> {
            let elem = $T_size;
            let need_base = outer_count
                .saturating_mul(base_dim_size)
                .saturating_mul(inner_count)
                .saturating_mul(elem);
            let need_idx = n_indices.saturating_mul(std::mem::size_of::<u32>());
            let need_src = outer_count
                .saturating_mul(n_indices)
                .saturating_mul(inner_count)
                .saturating_mul(elem);
            if base.len_bytes() != need_base || out.len_bytes() != need_base {
                return Err(Error::Msg(format!(
                    "{}: base/out bytes don't match",
                    stringify!($name),
                ))
                .bt());
            }
            if indices.len_bytes() != need_idx {
                return Err(Error::Msg(format!(
                    "{}: indices bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if src.len_bytes() != need_src {
                return Err(Error::Msg(format!(
                    "{}: src bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            out.bytes_mut().copy_from_slice(base.bytes());
            if n_indices == 0 {
                return Ok(());
            }
            let idx_view: &[u32] = indices.as_slice()?;
            let src_view: &[$T] = src.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            for i in 0..n_indices {
                let target = idx_view[i] as usize;
                if target >= base_dim_size {
                    return Err(Error::Msg(format!(
                        "{}: index {target} OOB for {base_dim_size}",
                        stringify!($name),
                    ))
                    .bt());
                }
                for outer in 0..outer_count {
                    let src_off = (outer * n_indices + i) * inner_count;
                    let dst_off = (outer * base_dim_size + target) * inner_count;
                    for inner in 0..inner_count {
                        let acc = out_view[dst_off + inner].to_f32()
                            + src_view[src_off + inner].to_f32();
                        out_view[dst_off + inner] = <$T>::from_f32(acc);
                    }
                }
            }
            Ok(())
        }
    };
}

index_add_half_kernel!(index_add_bf16, half::bf16, std::mem::size_of::<half::bf16>());
index_add_half_kernel!(index_add_f16,  half::f16,  std::mem::size_of::<half::f16>());

/// Generate a ScatterAdd kernel parameterized over native arithmetic
/// type `$T` (f32 / f64). Same shape as `scatter_add_f32`.
macro_rules! scatter_add_native_kernel {
    ($name:ident, $T:ty, $T_size:expr) => {
        pub fn $name(
            base: &CpuStorageBytes,
            indices: &CpuStorageBytes,
            src: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            base_shape: &[usize],
            src_shape: &[usize],
            dim: usize,
        ) -> Result<()> {
            if base_shape.len() != src_shape.len() {
                return Err(Error::Msg(format!(
                    "{}: base rank ({}) != src rank ({})",
                    stringify!($name), base_shape.len(), src_shape.len(),
                ))
                .bt());
            }
            let rank = base_shape.len();
            if dim >= rank {
                return Err(Error::Msg(format!(
                    "{}: dim {dim} out of range for rank {rank}",
                    stringify!($name),
                ))
                .bt());
            }
            for d in 0..rank {
                if d != dim && base_shape[d] != src_shape[d] {
                    return Err(Error::Msg(format!(
                        "{}: base/src disagree at dim {d}",
                        stringify!($name),
                    ))
                    .bt());
                }
            }
            let elem = $T_size;
            let base_total: usize = base_shape.iter().product();
            let src_total: usize = src_shape.iter().product();
            if base.len_bytes() != base_total.saturating_mul(elem)
                || out.len_bytes() != base_total.saturating_mul(elem)
            {
                return Err(Error::Msg(format!(
                    "{}: base/out bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if indices.len_bytes() != src_total.saturating_mul(std::mem::size_of::<u32>()) {
                return Err(Error::Msg(format!(
                    "{}: indices bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if src.len_bytes() != src_total.saturating_mul(elem) {
                return Err(Error::Msg(format!(
                    "{}: src bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            out.bytes_mut().copy_from_slice(base.bytes());
            if src_total == 0 {
                return Ok(());
            }
            let idx_view: &[u32] = indices.as_slice()?;
            let src_view: &[$T] = src.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let mut base_strides = vec![0usize; rank];
            let mut s = 1;
            for d in (0..rank).rev() {
                base_strides[d] = s;
                s *= base_shape[d];
            }
            let mut multi = vec![0usize; rank];
            for f in 0..src_total {
                let mut rem = f;
                for d in (0..rank).rev() {
                    multi[d] = rem % src_shape[d];
                    rem /= src_shape[d];
                }
                let dst_dim_idx = idx_view[f] as usize;
                if dst_dim_idx >= base_shape[dim] {
                    return Err(Error::Msg(format!(
                        "{}: index {dst_dim_idx} OOB", stringify!($name),
                    ))
                    .bt());
                }
                let mut dst_flat = 0;
                for d in 0..rank {
                    let coord = if d == dim { dst_dim_idx } else { multi[d] };
                    dst_flat += coord * base_strides[d];
                }
                out_view[dst_flat] += src_view[f];
            }
            Ok(())
        }
    };
}

scatter_add_native_kernel!(scatter_add_f64, f64, std::mem::size_of::<f64>());

/// ScatterAdd for half-float types — accumulates in f32 then narrows.
macro_rules! scatter_add_half_kernel {
    ($name:ident, $T:ty, $T_size:expr) => {
        pub fn $name(
            base: &CpuStorageBytes,
            indices: &CpuStorageBytes,
            src: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            base_shape: &[usize],
            src_shape: &[usize],
            dim: usize,
        ) -> Result<()> {
            if base_shape.len() != src_shape.len() {
                return Err(Error::Msg(format!(
                    "{}: rank mismatch", stringify!($name),
                ))
                .bt());
            }
            let rank = base_shape.len();
            if dim >= rank {
                return Err(Error::Msg(format!(
                    "{}: dim OOB", stringify!($name),
                ))
                .bt());
            }
            for d in 0..rank {
                if d != dim && base_shape[d] != src_shape[d] {
                    return Err(Error::Msg(format!(
                        "{}: shape mismatch at dim {d}", stringify!($name),
                    ))
                    .bt());
                }
            }
            let elem = $T_size;
            let base_total: usize = base_shape.iter().product();
            let src_total: usize = src_shape.iter().product();
            if base.len_bytes() != base_total.saturating_mul(elem)
                || out.len_bytes() != base_total.saturating_mul(elem)
            {
                return Err(Error::Msg(format!(
                    "{}: base/out bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if indices.len_bytes() != src_total.saturating_mul(std::mem::size_of::<u32>()) {
                return Err(Error::Msg(format!(
                    "{}: indices bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if src.len_bytes() != src_total.saturating_mul(elem) {
                return Err(Error::Msg(format!(
                    "{}: src bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            out.bytes_mut().copy_from_slice(base.bytes());
            if src_total == 0 {
                return Ok(());
            }
            let idx_view: &[u32] = indices.as_slice()?;
            let src_view: &[$T] = src.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let mut base_strides = vec![0usize; rank];
            let mut s = 1;
            for d in (0..rank).rev() {
                base_strides[d] = s;
                s *= base_shape[d];
            }
            let mut multi = vec![0usize; rank];
            for f in 0..src_total {
                let mut rem = f;
                for d in (0..rank).rev() {
                    multi[d] = rem % src_shape[d];
                    rem /= src_shape[d];
                }
                let dst_dim_idx = idx_view[f] as usize;
                if dst_dim_idx >= base_shape[dim] {
                    return Err(Error::Msg(format!(
                        "{}: index OOB", stringify!($name),
                    ))
                    .bt());
                }
                let mut dst_flat = 0;
                for d in 0..rank {
                    let coord = if d == dim { dst_dim_idx } else { multi[d] };
                    dst_flat += coord * base_strides[d];
                }
                let acc = out_view[dst_flat].to_f32() + src_view[f].to_f32();
                out_view[dst_flat] = <$T>::from_f32(acc);
            }
            Ok(())
        }
    };
}

scatter_add_half_kernel!(scatter_add_bf16, half::bf16, std::mem::size_of::<half::bf16>());
scatter_add_half_kernel!(scatter_add_f16,  half::f16,  std::mem::size_of::<half::f16>());

/// N-dimensional scatter-add — the functional inverse of
/// [`gather_f32`]. `base_shape` and `src_shape` agree on every
/// dim except `dim`. For each src/indices position `p`, the
/// destination multi-index is the same as `p` except `dim`'s
/// coord is replaced by `indices[p]`, and `src[p]` is added into
/// `out` at that destination.
pub fn scatter_add_f32(
    base: &CpuStorageBytes,
    indices: &CpuStorageBytes,
    src: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    base_shape: &[usize],
    src_shape: &[usize],
    dim: usize,
) -> Result<()> {
    if base_shape.len() != src_shape.len() {
        return Err(Error::Msg(format!(
            "scatter_add_f32: base rank ({}) != src rank ({})",
            base_shape.len(), src_shape.len(),
        ))
        .bt());
    }
    let rank = base_shape.len();
    if dim >= rank {
        return Err(Error::Msg(format!(
            "scatter_add_f32: dim {dim} out of range for rank {rank}",
        ))
        .bt());
    }
    for d in 0..rank {
        if d != dim && base_shape[d] != src_shape[d] {
            return Err(Error::Msg(format!(
                "scatter_add_f32: base and src disagree at dim {d} (base={}, src={}); only `dim`={dim} may differ",
                base_shape[d], src_shape[d],
            ))
            .bt());
        }
    }
    let elem = std::mem::size_of::<f32>();
    let base_total: usize = base_shape.iter().product();
    let src_total: usize = src_shape.iter().product();
    if base.len_bytes() != base_total.saturating_mul(elem) || out.len_bytes() != base_total.saturating_mul(elem) {
        return Err(Error::Msg(format!(
            "scatter_add_f32: base/out bytes don't match shape {base_shape:?} (f32)",
        ))
        .bt());
    }
    if indices.len_bytes() != src_total.saturating_mul(std::mem::size_of::<u32>()) {
        return Err(Error::Msg(format!(
            "scatter_add_f32: indices bytes don't match src shape {src_shape:?} (u32)",
        ))
        .bt());
    }
    if src.len_bytes() != src_total.saturating_mul(elem) {
        return Err(Error::Msg(format!(
            "scatter_add_f32: src bytes don't match shape {src_shape:?} (f32)",
        ))
        .bt());
    }
    // Copy base into out as the starting point.
    out.bytes_mut().copy_from_slice(base.bytes());
    if src_total == 0 {
        return Ok(());
    }
    let idx_view: &[u32] = indices.as_slice()?;
    let src_view: &[f32] = src.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    // Base strides (row-major).
    let mut base_strides = vec![0usize; rank];
    let mut s = 1;
    for d in (0..rank).rev() {
        base_strides[d] = s;
        s *= base_shape[d];
    }
    let mut multi = vec![0usize; rank];
    for f in 0..src_total {
        // Decode src multi-index from f (row-major over src_shape).
        let mut rem = f;
        for d in (0..rank).rev() {
            multi[d] = rem % src_shape[d];
            rem /= src_shape[d];
        }
        let dst_dim_idx = idx_view[f] as usize;
        if dst_dim_idx >= base_shape[dim] {
            return Err(Error::Msg(format!(
                "scatter_add_f32: index {dst_dim_idx} at position {f} out of bounds for base dim {} = {}",
                dim, base_shape[dim],
            ))
            .bt());
        }
        // Compose destination flat index in base.
        let mut dst_flat = 0;
        for d in 0..rank {
            let coord = if d == dim { dst_dim_idx } else { multi[d] };
            dst_flat += coord * base_strides[d];
        }
        out_view[dst_flat] += src_view[f];
    }
    Ok(())
}

// =============================================================================
// Index select (f32 source, U32 indices)
// =============================================================================

/// Pick slices from a source tensor along one axis using `u32`
/// indices. Dtype-agnostic: copies `inner_count * dtype_size`
/// bytes per source slab.
///
/// Each output element at `(outer, j, inner)` reads from source
/// at `(outer, indices[j], inner)`. Out-of-bounds indices return
/// a typed error rather than reading garbage.
pub fn index_select_cpu(
    source: &CpuStorageBytes,
    indices: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    outer_count: usize,
    source_dim_size: usize,
    n_indices: usize,
    inner_count: usize,
    dtype_size: usize,
) -> Result<()> {
    if dtype_size == 0 {
        return Err(Error::Msg("index_select_cpu: dtype_size must be > 0".to_string()).bt());
    }
    let stride_bytes = inner_count.saturating_mul(dtype_size);
    let need_src = outer_count
        .saturating_mul(source_dim_size)
        .saturating_mul(stride_bytes);
    let need_idx = n_indices.saturating_mul(std::mem::size_of::<u32>());
    let need_out = outer_count.saturating_mul(n_indices).saturating_mul(stride_bytes);
    if source.len_bytes() != need_src {
        return Err(Error::Msg(format!(
            "index_select_cpu: source bytes={} doesn't match outer={outer_count} × dim={source_dim_size} × inner={inner_count} × dtype_size={dtype_size}",
            source.len_bytes(),
        ))
        .bt());
    }
    if indices.len_bytes() != need_idx {
        return Err(Error::Msg(format!(
            "index_select_cpu: indices bytes={} doesn't match n_indices={n_indices} × 4",
            indices.len_bytes(),
        ))
        .bt());
    }
    if out.len_bytes() != need_out {
        return Err(Error::Msg(format!(
            "index_select_cpu: out bytes={} doesn't match outer={outer_count} × n={n_indices} × inner={inner_count} × dtype_size={dtype_size}",
            out.len_bytes(),
        ))
        .bt());
    }
    let src_bytes = source.bytes();
    let idx_view: &[u32] = indices.as_slice()?;
    let out_bytes = out.bytes_mut();
    for j in 0..n_indices {
        let i = idx_view[j] as usize;
        if i >= source_dim_size {
            return Err(Error::Msg(format!(
                "index_select_cpu: index {i} at position {j} out of bounds for source dim {source_dim_size}",
            ))
            .bt());
        }
        for outer in 0..outer_count {
            let src_off = (outer * source_dim_size + i) * stride_bytes;
            let dst_off = (outer * n_indices + j) * stride_bytes;
            out_bytes[dst_off..dst_off + stride_bytes]
                .copy_from_slice(&src_bytes[src_off..src_off + stride_bytes]);
        }
    }
    Ok(())
}

/// Backward-compat shim — same shape as the prior f32-only kernel.
pub fn index_select_f32(
    source: &CpuStorageBytes,
    indices: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    outer_count: usize,
    source_dim_size: usize,
    n_indices: usize,
    inner_count: usize,
) -> Result<()> {
    index_select_cpu(
        source, indices, out,
        outer_count, source_dim_size, n_indices, inner_count,
        std::mem::size_of::<f32>(),
    )
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

/// Generate a half-float Rope kernel parameterized over `$T`.
/// Each element is widened to f32 for the rotate_half computation,
/// then narrowed back when written to the output.
macro_rules! rope_half {
    ($name:ident, $T:ty) => {
        pub fn $name(
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
                    "{}: head_dim ({head_dim}) must be even",
                    stringify!($name),
                ))
                .bt());
            }
            let elem = std::mem::size_of::<$T>();
            let need_x = outer_count
                .saturating_mul(seq)
                .saturating_mul(head_dim)
                .saturating_mul(elem);
            let need_cs = seq.saturating_mul(head_dim).saturating_mul(elem);
            if x.len_bytes() != need_x {
                return Err(Error::Msg(format!(
                    "{}: x bytes={} doesn't match outer={outer_count} × seq={seq} × head_dim={head_dim} × {elem}",
                    stringify!($name), x.len_bytes(),
                ))
                .bt());
            }
            if cos.len_bytes() != need_cs {
                return Err(Error::Msg(format!(
                    "{}: cos bytes={} doesn't match seq × head_dim × {elem}",
                    stringify!($name), cos.len_bytes(),
                ))
                .bt());
            }
            if sin.len_bytes() != need_cs {
                return Err(Error::Msg(format!(
                    "{}: sin bytes={} doesn't match seq × head_dim × {elem}",
                    stringify!($name), sin.len_bytes(),
                ))
                .bt());
            }
            if out.len_bytes() != need_x {
                return Err(Error::Msg(format!(
                    "{}: out bytes={} doesn't match x shape",
                    stringify!($name), out.len_bytes(),
                ))
                .bt());
            }
            if seq == 0 || head_dim == 0 {
                return Ok(());
            }
            let x_view: &[$T] = x.as_slice()?;
            let cos_view: &[$T] = cos.as_slice()?;
            let sin_view: &[$T] = sin.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
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
                        let x_lo = x_view[x_lo_off].to_f32();
                        let x_hi = x_view[x_hi_off].to_f32();
                        let cos_lo = cos_view[cs_lo_off].to_f32();
                        let cos_hi = cos_view[cs_hi_off].to_f32();
                        let sin_lo = sin_view[cs_lo_off].to_f32();
                        let sin_hi = sin_view[cs_hi_off].to_f32();
                        out_view[x_lo_off] = <$T>::from_f32(x_lo * cos_lo - x_hi * sin_lo);
                        out_view[x_hi_off] = <$T>::from_f32(x_hi * cos_hi + x_lo * sin_hi);
                    }
                }
            }
            Ok(())
        }
    };
}

rope_half!(rope_bf16, half::bf16);
rope_half!(rope_f16, half::f16);

/// `f64` Rope — native arithmetic, same rotate_half formula as
/// `rope_f32`.
pub fn rope_f64(
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
            "rope_f64: head_dim ({head_dim}) must be even",
        ))
        .bt());
    }
    let elem = std::mem::size_of::<f64>();
    let need_x = outer_count
        .saturating_mul(seq)
        .saturating_mul(head_dim)
        .saturating_mul(elem);
    let need_cs = seq.saturating_mul(head_dim).saturating_mul(elem);
    if x.len_bytes() != need_x {
        return Err(Error::Msg(format!(
            "rope_f64: x bytes={} doesn't match shape (outer={outer_count} × seq={seq} × head_dim={head_dim})",
            x.len_bytes(),
        ))
        .bt());
    }
    if cos.len_bytes() != need_cs || sin.len_bytes() != need_cs {
        return Err(Error::Msg(format!(
            "rope_f64: cos/sin bytes don't match (seq × head_dim × {elem})",
        ))
        .bt());
    }
    if out.len_bytes() != need_x {
        return Err(Error::Msg(format!(
            "rope_f64: out bytes={} doesn't match x shape",
            out.len_bytes(),
        ))
        .bt());
    }
    if seq == 0 || head_dim == 0 {
        return Ok(());
    }
    let x_view: &[f64] = x.as_slice()?;
    let cos_view: &[f64] = cos.as_slice()?;
    let sin_view: &[f64] = sin.as_slice()?;
    let out_view: &mut [f64] = out.as_slice_mut()?;
    let h = head_dim / 2;
    for outer in 0..outer_count {
        for s in 0..seq {
            let x_row_off = (outer * seq + s) * head_dim;
            let cs_row_off = s * head_dim;
            for i in 0..h {
                let x_lo = x_view[x_row_off + i];
                let x_hi = x_view[x_row_off + i + h];
                let cos_lo = cos_view[cs_row_off + i];
                let cos_hi = cos_view[cs_row_off + i + h];
                let sin_lo = sin_view[cs_row_off + i];
                let sin_hi = sin_view[cs_row_off + i + h];
                out_view[x_row_off + i] = x_lo * cos_lo - x_hi * sin_lo;
                out_view[x_row_off + i + h] = x_hi * cos_hi + x_lo * sin_hi;
            }
        }
    }
    Ok(())
}

// =============================================================================
// Gather (f32 source, U32 indices, same-rank)
// =============================================================================

/// N-dimensional gather along `dim`. Dtype-agnostic byte-level
/// version — copies `dtype_size` bytes per output element.
pub fn gather_cpu(
    source: &CpuStorageBytes,
    indices: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    source_shape: &[usize],
    output_shape: &[usize],
    dim: usize,
    dtype_size: usize,
) -> Result<()> {
    if dtype_size == 0 {
        return Err(Error::Msg("gather_cpu: dtype_size must be > 0".to_string()).bt());
    }
    if source_shape.len() != output_shape.len() {
        return Err(Error::Msg(format!(
            "gather_cpu: source rank ({}) != output rank ({})",
            source_shape.len(),
            output_shape.len(),
        ))
        .bt());
    }
    let rank = source_shape.len();
    if dim >= rank {
        return Err(Error::Msg(format!(
            "gather_cpu: dim {dim} out of range for rank {rank}",
        ))
        .bt());
    }
    for d in 0..rank {
        if d != dim && source_shape[d] != output_shape[d] {
            return Err(Error::Msg(format!(
                "gather_cpu: source and output disagree at dim {d} \
                 (source={}, output={}); only `dim`={dim} may differ",
                source_shape[d],
                output_shape[d],
            ))
            .bt());
        }
    }
    let source_total: usize = source_shape.iter().product();
    let output_total: usize = output_shape.iter().product();
    if source.len_bytes() != source_total.saturating_mul(dtype_size) {
        return Err(Error::Msg(format!(
            "gather_cpu: source bytes={} doesn't match shape {source_shape:?} (dtype_size={dtype_size})",
            source.len_bytes(),
        ))
        .bt());
    }
    if indices.len_bytes() != output_total.saturating_mul(std::mem::size_of::<u32>()) {
        return Err(Error::Msg(format!(
            "gather_cpu: indices bytes={} doesn't match output shape {output_shape:?} (u32)",
            indices.len_bytes(),
        ))
        .bt());
    }
    if out.len_bytes() != output_total.saturating_mul(dtype_size) {
        return Err(Error::Msg(format!(
            "gather_cpu: out bytes={} doesn't match output shape {output_shape:?} (dtype_size={dtype_size})",
            out.len_bytes(),
        ))
        .bt());
    }
    if output_total == 0 {
        return Ok(());
    }
    let src_bytes = source.bytes();
    let idx_view: &[u32] = indices.as_slice()?;
    let out_bytes = out.bytes_mut();
    let mut src_strides = vec![0usize; rank];
    let mut s = 1;
    for d in (0..rank).rev() {
        src_strides[d] = s;
        s *= source_shape[d];
    }
    let mut multi = vec![0usize; rank];
    for f in 0..output_total {
        let mut rem = f;
        for d in (0..rank).rev() {
            multi[d] = rem % output_shape[d];
            rem /= output_shape[d];
        }
        let src_dim_idx = idx_view[f] as usize;
        if src_dim_idx >= source_shape[dim] {
            return Err(Error::Msg(format!(
                "gather_cpu: index {src_dim_idx} at output position {f} out of bounds for source dim {} = {}",
                dim, source_shape[dim],
            ))
            .bt());
        }
        let mut src_flat = 0;
        for d in 0..rank {
            let coord = if d == dim { src_dim_idx } else { multi[d] };
            src_flat += coord * src_strides[d];
        }
        let src_off = src_flat * dtype_size;
        let dst_off = f * dtype_size;
        out_bytes[dst_off..dst_off + dtype_size]
            .copy_from_slice(&src_bytes[src_off..src_off + dtype_size]);
    }
    Ok(())
}

/// Backward-compat shim — calls [`gather_cpu`] with `f32`'s dtype size.
pub fn gather_f32(
    source: &CpuStorageBytes,
    indices: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    source_shape: &[usize],
    output_shape: &[usize],
    dim: usize,
) -> Result<()> {
    gather_cpu(source, indices, out, source_shape, output_shape, dim, std::mem::size_of::<f32>())
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

/// Generate a half-float SoftmaxLastDim kernel parameterized
/// over `$T` (`half::bf16` / `half::f16`). All arithmetic happens
/// in f32 (numerically stable max-subtract + exp + sum +
/// normalize); narrowing back to half-float happens only on the
/// final write.
macro_rules! softmax_last_dim_half {
    ($name:ident, $T:ty) => {
        pub fn $name(
            input: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            outer_count: usize,
            last_dim: usize,
        ) -> Result<()> {
            check_lens_2(stringify!($name), input.len_bytes(), out.len_bytes())?;
            let elem = std::mem::size_of::<$T>();
            let need = outer_count.saturating_mul(last_dim).saturating_mul(elem);
            if input.len_bytes() != need {
                return Err(Error::Msg(format!(
                    "{}: input bytes={} doesn't match outer={outer_count} × last={last_dim} × {elem}",
                    stringify!($name), input.len_bytes(),
                ))
                .bt());
            }
            if last_dim == 0 {
                return Ok(());
            }
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            // Per-row scratch buffer for exp(x - max) values.
            let mut exps = vec![0.0_f32; last_dim];
            for row in 0..outer_count {
                let off = row * last_dim;
                let mut row_max = in_view[off].to_f32();
                for j in 1..last_dim {
                    let v = in_view[off + j].to_f32();
                    if v > row_max {
                        row_max = v;
                    }
                }
                let mut sum = 0.0_f32;
                for j in 0..last_dim {
                    let e = (in_view[off + j].to_f32() - row_max).exp();
                    exps[j] = e;
                    sum += e;
                }
                let inv = 1.0_f32 / sum;
                for j in 0..last_dim {
                    out_view[off + j] = <$T>::from_f32(exps[j] * inv);
                }
            }
            Ok(())
        }
    };
}

softmax_last_dim_half!(softmax_last_dim_bf16, half::bf16);
softmax_last_dim_half!(softmax_last_dim_f16, half::f16);

/// Softmax along the last dim for `f64`. Same numerically-stable
/// algorithm as `softmax_last_dim_f32`; native f64 arithmetic.
pub fn softmax_last_dim_f64(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    outer_count: usize,
    last_dim: usize,
) -> Result<()> {
    check_lens_2("softmax_last_dim_f64", input.len_bytes(), out.len_bytes())?;
    let elem = std::mem::size_of::<f64>();
    let need = outer_count.saturating_mul(last_dim).saturating_mul(elem);
    if input.len_bytes() != need {
        return Err(Error::Msg(format!(
            "softmax_last_dim_f64: input bytes={} doesn't match outer={outer_count} × last={last_dim} × {elem}",
            input.len_bytes(),
        ))
        .bt());
    }
    if last_dim == 0 {
        return Ok(());
    }
    let in_view: &[f64] = input.as_slice()?;
    let out_view: &mut [f64] = out.as_slice_mut()?;
    for row in 0..outer_count {
        let off = row * last_dim;
        let row_in = &in_view[off..off + last_dim];
        let mut row_max = row_in[0];
        for &v in &row_in[1..] {
            if v > row_max {
                row_max = v;
            }
        }
        let mut sum = 0.0_f64;
        for j in 0..last_dim {
            let e = (row_in[j] - row_max).exp();
            out_view[off + j] = e;
            sum += e;
        }
        let inv = 1.0_f64 / sum;
        for j in 0..last_dim {
            out_view[off + j] *= inv;
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

/// Generate a half-float RmsNormLastDim kernel parameterized
/// over `$T`. All arithmetic happens in f32 (sum-of-squares,
/// reciprocal sqrt); narrowing back to the half-float type
/// happens only on the final write.
macro_rules! rms_norm_last_dim_half {
    ($name:ident, $T:ty) => {
        pub fn $name(
            input: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            outer_count: usize,
            last_dim: usize,
            eps: f64,
        ) -> Result<()> {
            check_norm_lens_typed::<$T>(stringify!($name), input, out, outer_count, last_dim)?;
            if last_dim == 0 {
                return Ok(());
            }
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let eps32 = eps as f32;
            let inv_n = 1.0_f32 / last_dim as f32;
            for row in 0..outer_count {
                let off = row * last_dim;
                let mut sum_sq = 0.0_f32;
                for j in 0..last_dim {
                    let v = in_view[off + j].to_f32();
                    sum_sq += v * v;
                }
                let mean_sq = sum_sq * inv_n;
                let rms_inv = 1.0_f32 / (mean_sq + eps32).sqrt();
                for j in 0..last_dim {
                    out_view[off + j] =
                        <$T>::from_f32(in_view[off + j].to_f32() * rms_inv);
                }
            }
            Ok(())
        }
    };
}

rms_norm_last_dim_half!(rms_norm_last_dim_bf16, half::bf16);
rms_norm_last_dim_half!(rms_norm_last_dim_f16, half::f16);

/// `f64` RMS norm — native arithmetic, same algorithm as the
/// f32 version.
pub fn rms_norm_last_dim_f64(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    outer_count: usize,
    last_dim: usize,
    eps: f64,
) -> Result<()> {
    check_norm_lens_typed::<f64>("rms_norm_last_dim_f64", input, out, outer_count, last_dim)?;
    if last_dim == 0 {
        return Ok(());
    }
    let in_view: &[f64] = input.as_slice()?;
    let out_view: &mut [f64] = out.as_slice_mut()?;
    let inv_n = 1.0_f64 / last_dim as f64;
    for row in 0..outer_count {
        let off = row * last_dim;
        let row_in = &in_view[off..off + last_dim];
        let mut sum_sq = 0.0_f64;
        for &v in row_in {
            sum_sq += v * v;
        }
        let mean_sq = sum_sq * inv_n;
        let rms_inv = 1.0_f64 / (mean_sq + eps).sqrt();
        for j in 0..last_dim {
            out_view[off + j] = row_in[j] * rms_inv;
        }
    }
    Ok(())
}

// `check_norm_lens` (used by f32 norm helpers) is dtype-specific
// — it currently bakes in `size_of::<f32>()`. The bf16/f16 norm
// kernels above can't call it. Verify shape locally:
fn check_norm_lens_typed<T>(
    name: &str,
    input: &CpuStorageBytes,
    out: &CpuStorageBytes,
    outer_count: usize,
    last_dim: usize,
) -> Result<()> {
    let elem = std::mem::size_of::<T>();
    let need = outer_count.saturating_mul(last_dim).saturating_mul(elem);
    if input.len_bytes() != need || out.len_bytes() != need {
        return Err(Error::Msg(format!(
            "{name}: bytes mismatch (input={}, out={}, expected outer={outer_count} × last={last_dim} × {elem})",
            input.len_bytes(),
            out.len_bytes(),
        ))
        .bt());
    }
    Ok(())
}

/// Generate a half-float LayerNormLastDim kernel (no affine
/// params) parameterized over `$T`. Two-pass per row in f32.
macro_rules! layer_norm_last_dim_half {
    ($name:ident, $T:ty) => {
        pub fn $name(
            input: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            outer_count: usize,
            last_dim: usize,
            eps: f64,
        ) -> Result<()> {
            check_norm_lens_typed::<$T>(stringify!($name), input, out, outer_count, last_dim)?;
            if last_dim == 0 {
                return Ok(());
            }
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let eps32 = eps as f32;
            let inv_n = 1.0_f32 / last_dim as f32;
            for row in 0..outer_count {
                let off = row * last_dim;
                let mut sum = 0.0_f32;
                for j in 0..last_dim {
                    sum += in_view[off + j].to_f32();
                }
                let mean = sum * inv_n;
                let mut sum_sq = 0.0_f32;
                for j in 0..last_dim {
                    let d = in_view[off + j].to_f32() - mean;
                    sum_sq += d * d;
                }
                let var = sum_sq * inv_n;
                let inv_std = 1.0_f32 / (var + eps32).sqrt();
                for j in 0..last_dim {
                    out_view[off + j] =
                        <$T>::from_f32((in_view[off + j].to_f32() - mean) * inv_std);
                }
            }
            Ok(())
        }
    };
}

layer_norm_last_dim_half!(layer_norm_last_dim_bf16, half::bf16);
layer_norm_last_dim_half!(layer_norm_last_dim_f16, half::f16);

/// `f64` layer norm — native arithmetic, same algorithm as the
/// f32 version.
pub fn layer_norm_last_dim_f64(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    outer_count: usize,
    last_dim: usize,
    eps: f64,
) -> Result<()> {
    check_norm_lens_typed::<f64>("layer_norm_last_dim_f64", input, out, outer_count, last_dim)?;
    if last_dim == 0 {
        return Ok(());
    }
    let in_view: &[f64] = input.as_slice()?;
    let out_view: &mut [f64] = out.as_slice_mut()?;
    let inv_n = 1.0_f64 / last_dim as f64;
    for row in 0..outer_count {
        let off = row * last_dim;
        let row_in = &in_view[off..off + last_dim];
        let mut sum = 0.0_f64;
        for &v in row_in {
            sum += v;
        }
        let mean = sum * inv_n;
        let mut sum_sq = 0.0_f64;
        for &v in row_in {
            let d = v - mean;
            sum_sq += d * d;
        }
        let var = sum_sq * inv_n;
        let inv_std = 1.0_f64 / (var + eps).sqrt();
        for j in 0..last_dim {
            out_view[off + j] = (row_in[j] - mean) * inv_std;
        }
    }
    Ok(())
}

// Patch the bf16/f16 RmsNorm kernels to use `check_norm_lens_typed`
// — the version above mistakenly called `check_norm_lens` (which
// hardcodes f32). Replace those calls below.

// =============================================================================
// Concat (dtype-agnostic, byte-level)
// =============================================================================

/// Concatenate N inputs along one dim. Dtype-agnostic: the kernel
/// memcpys `inner_count * dtype_size` bytes per slab, so it works
/// uniformly for f32 / f64 / bf16 / f16 / u32 / etc.
///
/// Output layout: `[outer_count, total_dim, inner_count]` row-major
/// where `total_dim = sum(input_dim_sizes)`. Each input contributes
/// a `[outer_count, input_dim_sizes[i], inner_count]` slab.
pub fn concat_cpu(
    inputs: &[&CpuStorageBytes],
    out: &mut CpuStorageBytes,
    outer_count: usize,
    input_dim_sizes: &[usize],
    inner_count: usize,
    dtype_size: usize,
) -> Result<()> {
    if inputs.len() != input_dim_sizes.len() {
        return Err(Error::Msg(format!(
            "concat_cpu: inputs count ({}) != input_dim_sizes len ({})",
            inputs.len(),
            input_dim_sizes.len(),
        ))
        .bt());
    }
    if inputs.is_empty() {
        return Err(Error::Msg("concat_cpu: at least one input required".to_string()).bt());
    }
    if dtype_size == 0 {
        return Err(Error::Msg("concat_cpu: dtype_size must be > 0".to_string()).bt());
    }
    let total_dim: usize = input_dim_sizes.iter().sum();
    let stride_bytes = inner_count.saturating_mul(dtype_size);
    let need_out = outer_count.saturating_mul(total_dim).saturating_mul(stride_bytes);
    if out.len_bytes() != need_out {
        return Err(Error::Msg(format!(
            "concat_cpu: out bytes={} doesn't match outer={outer_count} × total_dim={total_dim} × inner={inner_count} × dtype_size={dtype_size}",
            out.len_bytes(),
        ))
        .bt());
    }
    for (i, input) in inputs.iter().enumerate() {
        let need_in = outer_count
            .saturating_mul(input_dim_sizes[i])
            .saturating_mul(stride_bytes);
        if input.len_bytes() != need_in {
            return Err(Error::Msg(format!(
                "concat_cpu: input[{i}] bytes={} doesn't match outer={outer_count} × dim={} × inner={inner_count} × dtype_size={dtype_size}",
                input.len_bytes(),
                input_dim_sizes[i],
            ))
            .bt());
        }
    }
    let out_bytes = out.bytes_mut();
    let mut dim_offset = 0usize;
    for (i, input) in inputs.iter().enumerate() {
        let in_bytes = input.bytes();
        let d_i = input_dim_sizes[i];
        for outer in 0..outer_count {
            for dim_pos in 0..d_i {
                let src_off = (outer * d_i + dim_pos) * stride_bytes;
                let dst_off = (outer * total_dim + dim_offset + dim_pos) * stride_bytes;
                out_bytes[dst_off..dst_off + stride_bytes]
                    .copy_from_slice(&in_bytes[src_off..src_off + stride_bytes]);
            }
        }
        dim_offset += d_i;
    }
    Ok(())
}

/// Backward-compat shim: forwards to [`concat_cpu`] with `f32`'s
/// dtype size. Existing callers (the dispatch wrapper and tests)
/// keep working unchanged.
pub fn concat_f32(
    inputs: &[&CpuStorageBytes],
    out: &mut CpuStorageBytes,
    outer_count: usize,
    input_dim_sizes: &[usize],
    inner_count: usize,
) -> Result<()> {
    concat_cpu(inputs, out, outer_count, input_dim_sizes, inner_count, std::mem::size_of::<f32>())
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

// Native-arithmetic Affine / Clamp / PowI for f64.
pub fn affine_f64(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    mul: f64,
    add: f64,
) -> Result<()> {
    check_lens_2("affine_f64", input.len_bytes(), out.len_bytes())?;
    let in_view: &[f64] = input.as_slice()?;
    let out_view: &mut [f64] = out.as_slice_mut()?;
    for (i, slot) in out_view.iter_mut().enumerate() {
        *slot = mul * in_view[i] + add;
    }
    Ok(())
}

pub fn clamp_f64(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    min: f64,
    max: f64,
) -> Result<()> {
    check_lens_2("clamp_f64", input.len_bytes(), out.len_bytes())?;
    if min > max {
        return Err(Error::Msg(format!("clamp_f64: min ({min}) > max ({max})")).bt());
    }
    let in_view: &[f64] = input.as_slice()?;
    let out_view: &mut [f64] = out.as_slice_mut()?;
    for (i, slot) in out_view.iter_mut().enumerate() {
        *slot = in_view[i].clamp(min, max);
    }
    Ok(())
}

pub fn powi_f64(
    input: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    exp: i32,
) -> Result<()> {
    check_lens_2("powi_f64", input.len_bytes(), out.len_bytes())?;
    let in_view: &[f64] = input.as_slice()?;
    let out_view: &mut [f64] = out.as_slice_mut()?;
    for (i, slot) in out_view.iter_mut().enumerate() {
        *slot = in_view[i].powi(exp);
    }
    Ok(())
}

// Half-float Affine / Clamp / PowI via f32 round-trip.
macro_rules! affine_half_kernel {
    ($name:ident, $T:ty) => {
        pub fn $name(
            input: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            mul: f32,
            add: f32,
        ) -> Result<()> {
            check_lens_2(stringify!($name), input.len_bytes(), out.len_bytes())?;
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            for (i, slot) in out_view.iter_mut().enumerate() {
                *slot = <$T>::from_f32(mul * in_view[i].to_f32() + add);
            }
            Ok(())
        }
    };
}

affine_half_kernel!(affine_bf16, half::bf16);
affine_half_kernel!(affine_f16, half::f16);

macro_rules! clamp_half_kernel {
    ($name:ident, $T:ty) => {
        pub fn $name(
            input: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            min: f32,
            max: f32,
        ) -> Result<()> {
            check_lens_2(stringify!($name), input.len_bytes(), out.len_bytes())?;
            if min > max {
                return Err(Error::Msg(format!(
                    "{}: min ({min}) > max ({max})", stringify!($name),
                ))
                .bt());
            }
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            for (i, slot) in out_view.iter_mut().enumerate() {
                *slot = <$T>::from_f32(in_view[i].to_f32().clamp(min, max));
            }
            Ok(())
        }
    };
}

clamp_half_kernel!(clamp_bf16, half::bf16);
clamp_half_kernel!(clamp_f16, half::f16);

macro_rules! powi_half_kernel {
    ($name:ident, $T:ty) => {
        pub fn $name(
            input: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            exp: i32,
        ) -> Result<()> {
            check_lens_2(stringify!($name), input.len_bytes(), out.len_bytes())?;
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            for (i, slot) in out_view.iter_mut().enumerate() {
                *slot = <$T>::from_f32(in_view[i].to_f32().powi(exp));
            }
            Ok(())
        }
    };
}

powi_half_kernel!(powi_bf16, half::bf16);
powi_half_kernel!(powi_f16, half::f16);

// ArgMax/ArgMin extensions to f64/bf16/f16. The existing
// `argextremum_dim_f32` operates on f32 directly; for other
// dtypes we widen to f32 for comparison (uniform NaN handling).
macro_rules! argextremum_dim_via_f32 {
    ($name:ident, $T:ty, $T_size:expr, $is_better:expr, $init:expr) => {
        pub fn $name(
            input: &CpuStorageBytes,
            output: &mut CpuStorageBytes,
            input_shape: &[usize],
            dim: usize,
        ) -> Result<()> {
            if dim >= input_shape.len() {
                return Err(Error::Msg(format!(
                    "{}: dim {dim} out of range for rank {}",
                    stringify!($name), input_shape.len(),
                ))
                .bt());
            }
            let total_input: usize = input_shape.iter().product();
            if input.len_bytes() != total_input.saturating_mul($T_size) {
                return Err(Error::Msg(format!(
                    "{}: input bytes={} doesn't match shape {input_shape:?}",
                    stringify!($name), input.len_bytes(),
                ))
                .bt());
            }
            let outer_count: usize = input_shape[..dim].iter().product();
            let dim_size = input_shape[dim];
            let inner_count: usize = input_shape[dim + 1..].iter().product();
            let output_count = outer_count * inner_count;
            if output.len_bytes() != output_count.saturating_mul(std::mem::size_of::<u32>()) {
                return Err(Error::Msg(format!(
                    "{}: output bytes={} doesn't match",
                    stringify!($name), output.len_bytes(),
                ))
                .bt());
            }
            if dim_size == 0 {
                return Err(Error::Msg(format!(
                    "{}: dim {dim} has size 0",
                    stringify!($name),
                ))
                .bt());
            }
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [u32] = output.as_slice_mut()?;
            let is_better: fn(f32, f32) -> bool = $is_better;
            for outer in 0..outer_count {
                for inner in 0..inner_count {
                    let mut best_val: f32 = $init;
                    let mut best_idx = 0u32;
                    for d in 0..dim_size {
                        let off = (outer * dim_size + d) * inner_count + inner;
                        let v = in_view[off].to_f32();
                        if d == 0 {
                            best_val = v;
                            best_idx = 0;
                        } else if is_better(v, best_val) {
                            best_val = v;
                            best_idx = d as u32;
                        }
                    }
                    out_view[outer * inner_count + inner] = best_idx;
                }
            }
            Ok(())
        }
    };
}

// f64 has a native to_f32 method via the `as` cast — wrap it.
trait ToF32Ext { fn to_f32(self) -> f32; }
impl ToF32Ext for f64 { fn to_f32(self) -> f32 { self as f32 } }

argextremum_dim_via_f32!(argmax_dim_f64, f64, std::mem::size_of::<f64>(),
    |new: f32, best: f32| new > best, f32::NEG_INFINITY);
argextremum_dim_via_f32!(argmin_dim_f64, f64, std::mem::size_of::<f64>(),
    |new: f32, best: f32| new < best, f32::INFINITY);
argextremum_dim_via_f32!(argmax_dim_bf16, half::bf16, std::mem::size_of::<half::bf16>(),
    |new: f32, best: f32| new > best, f32::NEG_INFINITY);
argextremum_dim_via_f32!(argmin_dim_bf16, half::bf16, std::mem::size_of::<half::bf16>(),
    |new: f32, best: f32| new < best, f32::INFINITY);
argextremum_dim_via_f32!(argmax_dim_f16, half::f16, std::mem::size_of::<half::f16>(),
    |new: f32, best: f32| new > best, f32::NEG_INFINITY);
argextremum_dim_via_f32!(argmin_dim_f16, half::f16, std::mem::size_of::<half::f16>(),
    |new: f32, best: f32| new < best, f32::INFINITY);

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

/// Batched row-major matmul for half-float types (bf16/f16) with
/// f32 accumulation. The kernel widens each input element to f32,
/// accumulates the inner-product sum in f32, then narrows the
/// final output back to the half-float type. Matches the standard
/// "f16 ops with f32 accumulator" pattern that cuBLAS / TPU /
/// cuDNN use to keep matmul numerically stable on half floats.
macro_rules! matmul_half_kernel {
    ($name:ident, $T:ty, $type_name:literal) => {
        pub fn $name(
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
                    "{}: batch ranks must match (lhs={}, rhs={})",
                    $type_name,
                    lhs_batch_dims.len(),
                    rhs_batch_dims.len(),
                ))
                .bt());
            }
            let batch_rank = lhs_batch_dims.len();
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
                        "{}: batch dim {i} disallowed combination (lhs={la}, rhs={ra})",
                        $type_name,
                    ))
                    .bt());
                }
            }
            let elem = std::mem::size_of::<$T>();
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
                    "{}: lhs bytes={} doesn't match shape {:?} + [{m}, {k}]",
                    $type_name, lhs.len_bytes(), lhs_batch_dims,
                ))
                .bt());
            }
            if rhs.len_bytes() != need_rhs {
                return Err(Error::Msg(format!(
                    "{}: rhs bytes={} doesn't match shape {:?} + [{k}, {n}]",
                    $type_name, rhs.len_bytes(), rhs_batch_dims,
                ))
                .bt());
            }
            if out.len_bytes() != need_out {
                return Err(Error::Msg(format!(
                    "{}: out bytes={} doesn't match shape {:?} + [{m}, {n}]",
                    $type_name, out.len_bytes(), lhs_batch_dims,
                ))
                .bt());
            }
            let lhs_view: &[$T] = lhs.as_slice()?;
            let rhs_view: &[$T] = rhs.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let mut lhs_multi = vec![0usize; batch_rank];
            let mut rhs_multi = vec![0usize; batch_rank];
            // Per-batch f32 accumulator buffer (reused across batches).
            let mut acc = vec![0.0_f32; out_per_batch];
            for b in 0..lhs_batch_count {
                let mut rem = b;
                for d in (0..batch_rank).rev() {
                    let s = lhs_batch_dims[d];
                    lhs_multi[d] = rem % s;
                    rem /= s;
                }
                for d in 0..batch_rank {
                    rhs_multi[d] = lhs_multi[d] / n_rep[d];
                }
                let mut rhs_b = 0usize;
                for d in 0..batch_rank {
                    rhs_b = rhs_b * rhs_batch_dims[d] + rhs_multi[d];
                }
                let lhs_off = b * lhs_per_batch;
                let rhs_off = rhs_b * rhs_per_batch;
                let out_off = b * out_per_batch;
                for slot in acc.iter_mut() {
                    *slot = 0.0;
                }
                for i in 0..m {
                    for kk in 0..k {
                        let a = lhs_view[lhs_off + i * k + kk].to_f32();
                        let rhs_row_off = rhs_off + kk * n;
                        let acc_row_off = i * n;
                        for j in 0..n {
                            acc[acc_row_off + j] +=
                                a * rhs_view[rhs_row_off + j].to_f32();
                        }
                    }
                }
                for (slot, &v) in out_view[out_off..out_off + out_per_batch]
                    .iter_mut()
                    .zip(&acc)
                {
                    *slot = <$T>::from_f32(v);
                }
            }
            Ok(())
        }
    };
}

matmul_half_kernel!(matmul_bf16, half::bf16, "matmul_bf16");
matmul_half_kernel!(matmul_f16, half::f16, "matmul_f16");

/// Batched row-major `f64` matrix multiply — a direct mirror of
/// [`matmul_f32`] with f64 element type and accumulator. Same
/// per-axis matched/GQA contract; same (i, k, j) inner loop.
pub fn matmul_f64(
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
            "matmul_f64: batch ranks must match (lhs={}, rhs={})",
            lhs_batch_dims.len(),
            rhs_batch_dims.len(),
        ))
        .bt());
    }
    let batch_rank = lhs_batch_dims.len();
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
                "matmul_f64: batch dim {i} disallowed combination (lhs={la}, rhs={ra})",
            ))
            .bt());
        }
    }
    let elem = std::mem::size_of::<f64>();
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
            "matmul_f64: lhs bytes={} doesn't match shape {:?} + [{m}, {k}] (f64)",
            lhs.len_bytes(),
            lhs_batch_dims,
        ))
        .bt());
    }
    if rhs.len_bytes() != need_rhs {
        return Err(Error::Msg(format!(
            "matmul_f64: rhs bytes={} doesn't match shape {:?} + [{k}, {n}] (f64)",
            rhs.len_bytes(),
            rhs_batch_dims,
        ))
        .bt());
    }
    if out.len_bytes() != need_out {
        return Err(Error::Msg(format!(
            "matmul_f64: out bytes={} doesn't match shape {:?} + [{m}, {n}] (f64)",
            out.len_bytes(),
            lhs_batch_dims,
        ))
        .bt());
    }
    let lhs_view: &[f64] = lhs.as_slice()?;
    let rhs_view: &[f64] = rhs.as_slice()?;
    let out_view: &mut [f64] = out.as_slice_mut()?;
    for slot in out_view.iter_mut() {
        *slot = 0.0;
    }
    let mut lhs_multi = vec![0usize; batch_rank];
    let mut rhs_multi = vec![0usize; batch_rank];
    for b in 0..lhs_batch_count {
        let mut rem = b;
        for d in (0..batch_rank).rev() {
            let s = lhs_batch_dims[d];
            lhs_multi[d] = rem % s;
            rem /= s;
        }
        for d in 0..batch_rank {
            rhs_multi[d] = lhs_multi[d] / n_rep[d];
        }
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
// Quantized matmul (Q4_0 / Q8_0 / Q4_K_M weights, F32 activations)
// =============================================================================

/// Generic batched quantized-matmul kernel parameterized over the
/// quantized block type `T: GgmlType`. Activations are F32 with
/// shape `[batch, m, k]`; weights `[n, k / block_size]` blocks
/// (laid out row-major over `n`, then over `k`-blocks). Output is
/// F32 with shape `[batch, m, n]`.
///
/// fuel-quantized's `matmul` takes (m, k, n) + slices and computes
/// the inner products via SIMD-friendly per-column dot products.
/// This kernel iterates batches and calls into it.
/// SAFETY-WRAPPER: reinterpret a `&[u8]` byte stream as `&[T]`
/// where `T` is a GGML block type (`#[repr(C)]` with all-Pod
/// fields, so the cast is sound). Used by the quantized matmul
/// kernels — fuel-quantized exposes `as_t_slice` for `Cow<[u8]>`
/// but we need the `&[u8]` flavor here. Returns `Err` on length
/// or alignment mismatch.
fn block_slice_from_bytes<'a, T>(name: &str, bytes: &'a [u8]) -> Result<&'a [T]> {
    let size = std::mem::size_of::<T>();
    if size == 0 {
        return Err(Error::Msg(format!("{name}: zero-sized block type")).bt());
    }
    if bytes.len() % size != 0 {
        return Err(Error::Msg(format!(
            "{name}: byte length {} not a multiple of block size {size}",
            bytes.len(),
        ))
        .bt());
    }
    let ptr = bytes.as_ptr();
    let align = std::mem::align_of::<T>();
    if (ptr as usize) % align != 0 {
        return Err(Error::Msg(format!(
            "{name}: byte pointer not aligned to block type's alignment {align}",
        ))
        .bt());
    }
    // Safety: `bytes` lifetime extends through this function's
    // borrow; T is a GGML block type which is `#[repr(C)]` with
    // only-Pod fields (f16/u8/u8 arrays), so any byte pattern is
    // a valid T. Length and alignment have just been validated.
    Ok(unsafe { std::slice::from_raw_parts(ptr as *const T, bytes.len() / size) })
}

fn qmatmul_generic_f32<T: fuel_quantized::GgmlType>(
    name: &str,
    activations: &CpuStorageBytes,
    weight_bytes: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let elements_per_block = T::BLCK_SIZE;
    if k % elements_per_block != 0 {
        return Err(Error::Msg(format!(
            "{name}: k={k} must be a multiple of block size {elements_per_block}",
        ))
        .bt());
    }
    let blocks_per_row = k / elements_per_block;
    let total_blocks = n.saturating_mul(blocks_per_row);
    let bytes_per_block = std::mem::size_of::<T>();
    let need_w = total_blocks.saturating_mul(bytes_per_block);
    let need_a = batch_count
        .saturating_mul(m)
        .saturating_mul(k)
        .saturating_mul(std::mem::size_of::<f32>());
    let need_out = batch_count
        .saturating_mul(m)
        .saturating_mul(n)
        .saturating_mul(std::mem::size_of::<f32>());
    if activations.len_bytes() != need_a {
        return Err(Error::Msg(format!(
            "{name}: activations bytes={} doesn't match batch={batch_count} × m={m} × k={k} × 4",
            activations.len_bytes(),
        ))
        .bt());
    }
    if weight_bytes.len_bytes() != need_w {
        return Err(Error::Msg(format!(
            "{name}: weight bytes={} doesn't match n={n} × k/block_size={blocks_per_row} × {bytes_per_block}",
            weight_bytes.len_bytes(),
        ))
        .bt());
    }
    if out.len_bytes() != need_out {
        return Err(Error::Msg(format!(
            "{name}: out bytes={} doesn't match batch={batch_count} × m={m} × n={n} × 4",
            out.len_bytes(),
        ))
        .bt());
    }
    if batch_count == 0 || m == 0 || n == 0 {
        return Ok(());
    }
    let act_view: &[f32] = activations.as_slice()?;
    // Reinterpret the U32-typed weight bytes as `&[T]` blocks via
    // a safety-wrapper around `from_raw_parts`. The GGML block
    // types are `#[repr(C)]` POD layouts so the cast is sound.
    let weight_view: &[T] = block_slice_from_bytes::<T>(name, weight_bytes.bytes())?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    let act_per_batch = m * k;
    let out_per_batch = m * n;
    for b in 0..batch_count {
        let act_off = b * act_per_batch;
        let out_off = b * out_per_batch;
        fuel_quantized::matmul::<T>(
            (m, k, n),
            &act_view[act_off..act_off + act_per_batch],
            weight_view,
            &mut out_view[out_off..out_off + out_per_batch],
        )
        .map_err(|e| Error::Msg(format!("{name}: {e}")).bt())?;
    }
    Ok(())
}

/// Q4_0 quantized matmul. Activations F32, weights `[n,
/// k/32]` `BlockQ4_0`s, output F32.
pub fn qmatmul_q4_0_f32(
    activations: &CpuStorageBytes,
    weight_bytes: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    qmatmul_generic_f32::<fuel_quantized::BlockQ4_0>(
        "qmatmul_q4_0_f32",
        activations, weight_bytes, out,
        batch_count, m, n, k,
    )
}

/// Q8_0 quantized matmul.
pub fn qmatmul_q8_0_f32(
    activations: &CpuStorageBytes,
    weight_bytes: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    qmatmul_generic_f32::<fuel_quantized::BlockQ8_0>(
        "qmatmul_q8_0_f32",
        activations, weight_bytes, out,
        batch_count, m, n, k,
    )
}

/// Q4_K_M (256-element super-block) quantized matmul.
pub fn qmatmul_q4_k_m_f32(
    activations: &CpuStorageBytes,
    weight_bytes: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    qmatmul_generic_f32::<fuel_quantized::BlockQ4K>(
        "qmatmul_q4_k_m_f32",
        activations, weight_bytes, out,
        batch_count, m, n, k,
    )
}

macro_rules! qmatmul_thin_wrapper {
    ($name:ident, $blk:ty, $kname:expr) => {
        pub fn $name(
            activations: &CpuStorageBytes,
            weight_bytes: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<()> {
            qmatmul_generic_f32::<$blk>(
                $kname,
                activations, weight_bytes, out,
                batch_count, m, n, k,
            )
        }
    };
}

qmatmul_thin_wrapper!(qmatmul_q4_1_f32, fuel_quantized::BlockQ4_1, "qmatmul_q4_1_f32");
qmatmul_thin_wrapper!(qmatmul_q5_0_f32, fuel_quantized::BlockQ5_0, "qmatmul_q5_0_f32");
qmatmul_thin_wrapper!(qmatmul_q5_1_f32, fuel_quantized::BlockQ5_1, "qmatmul_q5_1_f32");
qmatmul_thin_wrapper!(qmatmul_q8_1_f32, fuel_quantized::BlockQ8_1, "qmatmul_q8_1_f32");
qmatmul_thin_wrapper!(qmatmul_q2k_f32,  fuel_quantized::BlockQ2K,  "qmatmul_q2k_f32");
qmatmul_thin_wrapper!(qmatmul_q3k_f32,  fuel_quantized::BlockQ3K,  "qmatmul_q3k_f32");
qmatmul_thin_wrapper!(qmatmul_q5k_f32,  fuel_quantized::BlockQ5K,  "qmatmul_q5k_f32");
qmatmul_thin_wrapper!(qmatmul_q6k_f32,  fuel_quantized::BlockQ6K,  "qmatmul_q6k_f32");

// =============================================================================
// 2D Convolution — multi-dtype (f64 native, bf16/f16 via f32 acc)
// =============================================================================

/// Generate a Conv2D kernel where input/weight/output are `$T`
/// (native arithmetic). Used for f64 (and f32 has its own
/// hand-written version).
macro_rules! conv2d_native_kernel {
    ($name:ident, $T:ty, $T_size:expr, $zero:expr) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
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
            if n != n_out || cout != cout_out
                || groups == 0 || cin % groups != 0 || cout % groups != 0
                || cin / groups != cin_per_group
            {
                return Err(Error::Msg(format!(
                    "{}: shape contract violation (x={x_shape:?}, w={w_shape:?}, out={out_shape:?}, groups={groups})",
                    stringify!($name),
                ))
                .bt());
            }
            let cout_per_group = cout / groups;
            let elem = $T_size;
            if x.len_bytes() != n * cin * h_in * w_in * elem
                || weight.len_bytes() != cout * cin_per_group * kh * kw * elem
                || out.len_bytes() != n * cout * h_out * w_out * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if let Some(b) = bias {
                if b.len_bytes() != cout * elem {
                    return Err(Error::Msg(format!(
                        "{}: bias bytes mismatch", stringify!($name),
                    ))
                    .bt());
                }
            }
            let x_view: &[$T] = x.as_slice()?;
            let w_view: &[$T] = weight.as_slice()?;
            let bias_view: Option<&[$T]> = match bias {
                Some(b) => Some(b.as_slice()?),
                None => None,
            };
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let (sh, sw) = stride;
            let (ph, pw) = padding;
            let (dh, dw) = dilation;
            for b_idx in 0..n {
                for co in 0..cout {
                    let group = co / cout_per_group;
                    let ci_offset = group * cin_per_group;
                    let bias_val = bias_view.map(|bv| bv[co]).unwrap_or($zero);
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut acc = bias_val;
                            for ci in 0..cin_per_group {
                                for kh_i in 0..kh {
                                    let in_h = (oh * sh + kh_i * dh) as isize - ph as isize;
                                    if in_h < 0 || in_h as usize >= h_in { continue; }
                                    let in_h = in_h as usize;
                                    for kw_i in 0..kw {
                                        let in_w = (ow * sw + kw_i * dw) as isize - pw as isize;
                                        if in_w < 0 || in_w as usize >= w_in { continue; }
                                        let in_w = in_w as usize;
                                        let x_idx = ((b_idx * cin + (ci_offset + ci)) * h_in + in_h) * w_in + in_w;
                                        let w_idx = ((co * cin_per_group + ci) * kh + kh_i) * kw + kw_i;
                                        acc += x_view[x_idx] * w_view[w_idx];
                                    }
                                }
                            }
                            let out_idx = ((b_idx * cout + co) * h_out + oh) * w_out + ow;
                            out_view[out_idx] = acc;
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

conv2d_native_kernel!(conv2d_f64, f64, std::mem::size_of::<f64>(), 0.0_f64);

/// Conv2D for half-float types — accumulates each output position
/// in f32, narrows at the end. Inputs (x, weight, bias) all in T.
macro_rules! conv2d_half_kernel {
    ($name:ident, $T:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
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
            if n != n_out || cout != cout_out
                || groups == 0 || cin % groups != 0 || cout % groups != 0
                || cin / groups != cin_per_group
            {
                return Err(Error::Msg(format!(
                    "{}: shape contract violation",
                    stringify!($name),
                ))
                .bt());
            }
            let cout_per_group = cout / groups;
            let elem = std::mem::size_of::<$T>();
            if x.len_bytes() != n * cin * h_in * w_in * elem
                || weight.len_bytes() != cout * cin_per_group * kh * kw * elem
                || out.len_bytes() != n * cout * h_out * w_out * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if let Some(b) = bias {
                if b.len_bytes() != cout * elem {
                    return Err(Error::Msg(format!(
                        "{}: bias bytes mismatch", stringify!($name),
                    ))
                    .bt());
                }
            }
            let x_view: &[$T] = x.as_slice()?;
            let w_view: &[$T] = weight.as_slice()?;
            let bias_view: Option<&[$T]> = match bias {
                Some(b) => Some(b.as_slice()?),
                None => None,
            };
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let (sh, sw) = stride;
            let (ph, pw) = padding;
            let (dh, dw) = dilation;
            for b_idx in 0..n {
                for co in 0..cout {
                    let group = co / cout_per_group;
                    let ci_offset = group * cin_per_group;
                    let bias_val = bias_view.map(|bv| bv[co].to_f32()).unwrap_or(0.0_f32);
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut acc: f32 = bias_val;
                            for ci in 0..cin_per_group {
                                for kh_i in 0..kh {
                                    let in_h = (oh * sh + kh_i * dh) as isize - ph as isize;
                                    if in_h < 0 || in_h as usize >= h_in { continue; }
                                    let in_h = in_h as usize;
                                    for kw_i in 0..kw {
                                        let in_w = (ow * sw + kw_i * dw) as isize - pw as isize;
                                        if in_w < 0 || in_w as usize >= w_in { continue; }
                                        let in_w = in_w as usize;
                                        let x_idx = ((b_idx * cin + (ci_offset + ci)) * h_in + in_h) * w_in + in_w;
                                        let w_idx = ((co * cin_per_group + ci) * kh + kh_i) * kw + kw_i;
                                        acc += x_view[x_idx].to_f32() * w_view[w_idx].to_f32();
                                    }
                                }
                            }
                            let out_idx = ((b_idx * cout + co) * h_out + oh) * w_out + ow;
                            out_view[out_idx] = <$T>::from_f32(acc);
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

conv2d_half_kernel!(conv2d_bf16, half::bf16);
conv2d_half_kernel!(conv2d_f16, half::f16);

// =============================================================================
// 2D Transposed Convolution — multi-dtype
// =============================================================================
//
// Shapes:
//   x:      [N, Cin, H_in, W_in]
//   weight: [Cin, Cout/groups, Kh, Kw]   (transposed channel order vs Conv2D)
//   bias:   optional [Cout]
//   out:    [N, Cout, H_out, W_out]
// where:
//   H_out = (H_in - 1) * stride.0 - 2*pad.0 + dil.0*(Kh - 1) + out_pad.0 + 1
//   W_out = (W_in - 1) * stride.1 - 2*pad.1 + dil.1*(Kw - 1) + out_pad.1 + 1
//
// Strategy: zero output, optionally seed with broadcast bias, then for
// every input element scatter-accumulate its kernel-shaped contribution
// into the output. For half-floats we accumulate into a parallel
// `Vec<f32>` buffer and narrow at the end (same f32-accumulator
// pattern Conv2D-half uses).

/// ConvTranspose2D for native arithmetic ($T = f32 or f64).
macro_rules! conv_transpose2d_native_kernel {
    ($name:ident, $T:ty, $T_size:expr, $zero:expr) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
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
            let [cin_w, cout_per_group, kh, kw] = w_shape;
            let [n_out, cout, h_out, w_out] = out_shape;
            if n != n_out
                || groups == 0
                || cin % groups != 0
                || cout % groups != 0
                || cin != cin_w
                || cout / groups != cout_per_group
            {
                return Err(Error::Msg(format!(
                    "{}: shape contract violation (x={x_shape:?}, w={w_shape:?}, out={out_shape:?}, groups={groups})",
                    stringify!($name),
                ))
                .bt());
            }
            let elem = $T_size;
            if x.len_bytes() != n * cin * h_in * w_in * elem
                || weight.len_bytes() != cin * cout_per_group * kh * kw * elem
                || out.len_bytes() != n * cout * h_out * w_out * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if let Some(b) = bias {
                if b.len_bytes() != cout * elem {
                    return Err(Error::Msg(format!(
                        "{}: bias bytes mismatch", stringify!($name),
                    ))
                    .bt());
                }
            }
            let x_view: &[$T] = x.as_slice()?;
            let w_view: &[$T] = weight.as_slice()?;
            let bias_view: Option<&[$T]> = match bias {
                Some(b) => Some(b.as_slice()?),
                None => None,
            };
            let out_view: &mut [$T] = out.as_slice_mut()?;
            // Initialize output: bias broadcast across spatial dims, or zero.
            for n_i in 0..n {
                for co in 0..cout {
                    let bias_val = bias_view.map(|bv| bv[co]).unwrap_or($zero);
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let idx = ((n_i * cout + co) * h_out + oh) * w_out + ow;
                            out_view[idx] = bias_val;
                        }
                    }
                }
            }
            let (sh, sw) = stride;
            let (ph, pw) = padding;
            let (dh, dw) = dilation;
            let cin_per_group = cin / groups;
            for n_i in 0..n {
                for g in 0..groups {
                    for ci_local in 0..cin_per_group {
                        let ci = g * cin_per_group + ci_local;
                        for hi in 0..h_in {
                            for wi in 0..w_in {
                                let val = x_view[((n_i * cin + ci) * h_in + hi) * w_in + wi];
                                if val == $zero { continue; }
                                for co_local in 0..cout_per_group {
                                    let co = g * cout_per_group + co_local;
                                    for kh_i in 0..kh {
                                        let oh_signed = (hi * sh) as isize + (kh_i * dh) as isize - ph as isize;
                                        if oh_signed < 0 || oh_signed as usize >= h_out { continue; }
                                        let oh = oh_signed as usize;
                                        for kw_i in 0..kw {
                                            let ow_signed = (wi * sw) as isize + (kw_i * dw) as isize - pw as isize;
                                            if ow_signed < 0 || ow_signed as usize >= w_out { continue; }
                                            let ow = ow_signed as usize;
                                            let w_idx = ((ci * cout_per_group + co_local) * kh + kh_i) * kw + kw_i;
                                            let out_idx = ((n_i * cout + co) * h_out + oh) * w_out + ow;
                                            out_view[out_idx] += val * w_view[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

conv_transpose2d_native_kernel!(conv_transpose2d_f32, f32, std::mem::size_of::<f32>(), 0.0_f32);
conv_transpose2d_native_kernel!(conv_transpose2d_f64, f64, std::mem::size_of::<f64>(), 0.0_f64);

/// ConvTranspose2D for half-floats — accumulates into a parallel
/// `Vec<f32>` buffer, narrows back at the end. Same f32-accumulator
/// pattern as conv2d_half_kernel!.
macro_rules! conv_transpose2d_half_kernel {
    ($name:ident, $T:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
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
            let [cin_w, cout_per_group, kh, kw] = w_shape;
            let [n_out, cout, h_out, w_out] = out_shape;
            if n != n_out
                || groups == 0
                || cin % groups != 0
                || cout % groups != 0
                || cin != cin_w
                || cout / groups != cout_per_group
            {
                return Err(Error::Msg(format!(
                    "{}: shape contract violation",
                    stringify!($name),
                ))
                .bt());
            }
            let elem = std::mem::size_of::<$T>();
            if x.len_bytes() != n * cin * h_in * w_in * elem
                || weight.len_bytes() != cin * cout_per_group * kh * kw * elem
                || out.len_bytes() != n * cout * h_out * w_out * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if let Some(b) = bias {
                if b.len_bytes() != cout * elem {
                    return Err(Error::Msg(format!(
                        "{}: bias bytes mismatch", stringify!($name),
                    ))
                    .bt());
                }
            }
            let x_view: &[$T] = x.as_slice()?;
            let w_view: &[$T] = weight.as_slice()?;
            let bias_view: Option<&[$T]> = match bias {
                Some(b) => Some(b.as_slice()?),
                None => None,
            };
            let out_view: &mut [$T] = out.as_slice_mut()?;
            // f32 accumulator buffer; seeded with bias broadcast.
            let total = n * cout * h_out * w_out;
            let mut acc = vec![0.0_f32; total];
            for n_i in 0..n {
                for co in 0..cout {
                    let bias_val = bias_view.map(|bv| bv[co].to_f32()).unwrap_or(0.0_f32);
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let idx = ((n_i * cout + co) * h_out + oh) * w_out + ow;
                            acc[idx] = bias_val;
                        }
                    }
                }
            }
            let (sh, sw) = stride;
            let (ph, pw) = padding;
            let (dh, dw) = dilation;
            let cin_per_group = cin / groups;
            for n_i in 0..n {
                for g in 0..groups {
                    for ci_local in 0..cin_per_group {
                        let ci = g * cin_per_group + ci_local;
                        for hi in 0..h_in {
                            for wi in 0..w_in {
                                let val = x_view[((n_i * cin + ci) * h_in + hi) * w_in + wi].to_f32();
                                if val == 0.0_f32 { continue; }
                                for co_local in 0..cout_per_group {
                                    let co = g * cout_per_group + co_local;
                                    for kh_i in 0..kh {
                                        let oh_signed = (hi * sh) as isize + (kh_i * dh) as isize - ph as isize;
                                        if oh_signed < 0 || oh_signed as usize >= h_out { continue; }
                                        let oh = oh_signed as usize;
                                        for kw_i in 0..kw {
                                            let ow_signed = (wi * sw) as isize + (kw_i * dw) as isize - pw as isize;
                                            if ow_signed < 0 || ow_signed as usize >= w_out { continue; }
                                            let ow = ow_signed as usize;
                                            let w_idx = ((ci * cout_per_group + co_local) * kh + kh_i) * kw + kw_i;
                                            let out_idx = ((n_i * cout + co) * h_out + oh) * w_out + ow;
                                            acc[out_idx] += val * w_view[w_idx].to_f32();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (dst, src) in out_view.iter_mut().zip(acc.iter()) {
                *dst = <$T>::from_f32(*src);
            }
            Ok(())
        }
    };
}

conv_transpose2d_half_kernel!(conv_transpose2d_bf16, half::bf16);
conv_transpose2d_half_kernel!(conv_transpose2d_f16, half::f16);

// =============================================================================
// ReduceSumTo — sum-reduce to a broadcast-compatible target shape
// =============================================================================
//
// The output shape is left-padded with 1s if its rank < input rank, so
// every input axis aligns with one output axis. For each axis: if the
// padded output dim == input dim, the axis carries through; if 1, the
// axis is summed away. Any other value is a contract violation.
//
// Strategy: zero output, walk input flat → multi-index, project each
// axis to the output multi-index, accumulate.

fn align_reduce_to(input_shape: &[usize], output_shape: &[usize]) -> Result<Vec<usize>> {
    if output_shape.len() > input_shape.len() {
        return Err(Error::Msg(format!(
            "reduce_sum_to: output rank {} exceeds input rank {}",
            output_shape.len(), input_shape.len(),
        ))
        .bt());
    }
    let pad = input_shape.len() - output_shape.len();
    let mut padded = vec![1_usize; pad];
    padded.extend_from_slice(output_shape);
    for (i, (&s, &t)) in input_shape.iter().zip(padded.iter()).enumerate() {
        if t != 1 && t != s {
            return Err(Error::Msg(format!(
                "reduce_sum_to: axis {i} target {t} must be 1 or input {s}",
            ))
            .bt());
        }
    }
    Ok(padded)
}

fn elem_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Reduce-sum-to for native arithmetic ($T = f32 or f64).
macro_rules! reduce_sum_to_native_kernel {
    ($name:ident, $T:ty, $T_size:expr, $zero:expr) => {
        pub fn $name(
            input: &CpuStorageBytes,
            output: &mut CpuStorageBytes,
            input_shape: &[usize],
            output_shape: &[usize],
        ) -> Result<()> {
            let padded = align_reduce_to(input_shape, output_shape)?;
            let in_elems = elem_count(input_shape);
            let out_elems = elem_count(output_shape);
            let elem = $T_size;
            if input.len_bytes() != in_elems * elem
                || output.len_bytes() != out_elems * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch (in {} elems, out {} elems)",
                    stringify!($name), in_elems, out_elems,
                ))
                .bt());
            }
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = output.as_slice_mut()?;
            for slot in out_view.iter_mut() { *slot = $zero; }
            // Strides for input (row-major).
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
                // Decode input multi-index, project to output.
                let mut out_flat = 0_usize;
                let mut rem = in_flat;
                for axis in 0..rank {
                    let coord = rem / in_strides[axis];
                    rem %= in_strides[axis];
                    let out_coord = if padded[axis] == 1 { 0 } else { coord };
                    out_flat += out_coord * out_strides_padded[axis];
                }
                out_view[out_flat] += in_view[in_flat];
            }
            Ok(())
        }
    };
}

reduce_sum_to_native_kernel!(reduce_sum_to_f32, f32, std::mem::size_of::<f32>(), 0.0_f32);
reduce_sum_to_native_kernel!(reduce_sum_to_f64, f64, std::mem::size_of::<f64>(), 0.0_f64);

/// Reduce-sum-to for half-floats — accumulate into f32 and narrow.
macro_rules! reduce_sum_to_half_kernel {
    ($name:ident, $T:ty) => {
        pub fn $name(
            input: &CpuStorageBytes,
            output: &mut CpuStorageBytes,
            input_shape: &[usize],
            output_shape: &[usize],
        ) -> Result<()> {
            let padded = align_reduce_to(input_shape, output_shape)?;
            let in_elems = elem_count(input_shape);
            let out_elems = elem_count(output_shape);
            let elem = std::mem::size_of::<$T>();
            if input.len_bytes() != in_elems * elem
                || output.len_bytes() != out_elems * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = output.as_slice_mut()?;
            let mut acc = vec![0.0_f32; out_elems];
            let rank = input_shape.len();
            let mut in_strides = vec![1_usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
            }
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
                acc[out_flat] += in_view[in_flat].to_f32();
            }
            for (dst, src) in out_view.iter_mut().zip(acc.iter()) {
                *dst = <$T>::from_f32(*src);
            }
            Ok(())
        }
    };
}

reduce_sum_to_half_kernel!(reduce_sum_to_bf16, half::bf16);
reduce_sum_to_half_kernel!(reduce_sum_to_f16, half::f16);

// =============================================================================
// ReduceMaxTo — max-reduce to a broadcast-compatible target shape
// =============================================================================
//
// Same alignment + projection logic as ReduceSumTo; the only differences
// are the reduction operator (`max` instead of `+`) and the output's
// initial value (negative infinity instead of zero).

/// Reduce-max-to for native arithmetic ($T = f32 or f64).
macro_rules! reduce_max_to_native_kernel {
    ($name:ident, $T:ty, $T_size:expr, $neg_inf:expr) => {
        pub fn $name(
            input: &CpuStorageBytes,
            output: &mut CpuStorageBytes,
            input_shape: &[usize],
            output_shape: &[usize],
        ) -> Result<()> {
            let padded = align_reduce_to(input_shape, output_shape)?;
            let in_elems = elem_count(input_shape);
            let out_elems = elem_count(output_shape);
            let elem = $T_size;
            if input.len_bytes() != in_elems * elem
                || output.len_bytes() != out_elems * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch (in {} elems, out {} elems)",
                    stringify!($name), in_elems, out_elems,
                ))
                .bt());
            }
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = output.as_slice_mut()?;
            for slot in out_view.iter_mut() { *slot = $neg_inf; }
            let rank = input_shape.len();
            let mut in_strides = vec![1_usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
            }
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
                if in_view[in_flat] > out_view[out_flat] {
                    out_view[out_flat] = in_view[in_flat];
                }
            }
            Ok(())
        }
    };
}

reduce_max_to_native_kernel!(reduce_max_to_f32, f32, std::mem::size_of::<f32>(), f32::NEG_INFINITY);
reduce_max_to_native_kernel!(reduce_max_to_f64, f64, std::mem::size_of::<f64>(), f64::NEG_INFINITY);

/// Reduce-max-to for half-floats — accumulate via f32 and narrow.
/// (The f32 widening is overkill for max — straightforward
/// half-arithmetic would also work — but it keeps the macro shape
/// uniform with the sum kernel and avoids edge cases around half-float
/// total-ordering.)
macro_rules! reduce_max_to_half_kernel {
    ($name:ident, $T:ty) => {
        pub fn $name(
            input: &CpuStorageBytes,
            output: &mut CpuStorageBytes,
            input_shape: &[usize],
            output_shape: &[usize],
        ) -> Result<()> {
            let padded = align_reduce_to(input_shape, output_shape)?;
            let in_elems = elem_count(input_shape);
            let out_elems = elem_count(output_shape);
            let elem = std::mem::size_of::<$T>();
            if input.len_bytes() != in_elems * elem
                || output.len_bytes() != out_elems * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = output.as_slice_mut()?;
            let mut acc = vec![f32::NEG_INFINITY; out_elems];
            let rank = input_shape.len();
            let mut in_strides = vec![1_usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
            }
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
                let v = in_view[in_flat].to_f32();
                if v > acc[out_flat] { acc[out_flat] = v; }
            }
            for (dst, src) in out_view.iter_mut().zip(acc.iter()) {
                *dst = <$T>::from_f32(*src);
            }
            Ok(())
        }
    };
}

reduce_max_to_half_kernel!(reduce_max_to_bf16, half::bf16);
reduce_max_to_half_kernel!(reduce_max_to_f16, half::f16);

// =============================================================================
// FusedLinear — matmul + bias-add (3-input)
// =============================================================================
//
// Inputs:
//   a:    [..., M, K]
//   b:    [..., K, N]
//   bias: [N]   (broadcast across all leading dims of the matmul output)
// Output: [..., M, N] where out[..., i, j] = bias[j] + sum_k a[..., i, k] * b[..., k, j]
//
// Shape semantics match `matmul_*` (per-axis GQA broadcasting on the
// batch prefixes). The fused form simply seeds the accumulator with
// bias[j] before the inner product.

fn fused_linear_check<T>(
    name: &str,
    lhs: &CpuStorageBytes,
    rhs: &CpuStorageBytes,
    bias: &CpuStorageBytes,
    out: &CpuStorageBytes,
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<usize>> {
    if lhs_batch_dims.len() != rhs_batch_dims.len() {
        return Err(Error::Msg(format!(
            "{name}: batch ranks must match (lhs={}, rhs={})",
            lhs_batch_dims.len(),
            rhs_batch_dims.len(),
        ))
        .bt());
    }
    let batch_rank = lhs_batch_dims.len();
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
                "{name}: batch dim {i} disallowed (lhs={la}, rhs={ra})",
            ))
            .bt());
        }
    }
    let elem = std::mem::size_of::<T>();
    let lhs_per = m.saturating_mul(k);
    let rhs_per = k.saturating_mul(n);
    let out_per = m.saturating_mul(n);
    let lhs_count: usize = lhs_batch_dims.iter().product::<usize>().max(1);
    let rhs_count: usize = rhs_batch_dims.iter().product::<usize>().max(1);
    if lhs.len_bytes() != lhs_count.saturating_mul(lhs_per).saturating_mul(elem)
        || rhs.len_bytes() != rhs_count.saturating_mul(rhs_per).saturating_mul(elem)
        || out.len_bytes() != lhs_count.saturating_mul(out_per).saturating_mul(elem)
        || bias.len_bytes() != n.saturating_mul(elem)
    {
        return Err(Error::Msg(format!(
            "{name}: bytes mismatch",
        ))
        .bt());
    }
    Ok(n_rep)
}

/// FusedLinear for native arithmetic.
macro_rules! fused_linear_native_kernel {
    ($name:ident, $T:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            lhs: &CpuStorageBytes,
            rhs: &CpuStorageBytes,
            bias: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            lhs_batch_dims: &[usize],
            rhs_batch_dims: &[usize],
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<()> {
            let n_rep = fused_linear_check::<$T>(
                stringify!($name), lhs, rhs, bias, out,
                lhs_batch_dims, rhs_batch_dims, m, n, k,
            )?;
            let lhs_view: &[$T] = lhs.as_slice()?;
            let rhs_view: &[$T] = rhs.as_slice()?;
            let bias_view: &[$T] = bias.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let batch_rank = lhs_batch_dims.len();
            let lhs_per = m * k;
            let rhs_per = k * n;
            let out_per = m * n;
            let lhs_count: usize = lhs_batch_dims.iter().product::<usize>().max(1);
            let mut lhs_multi = vec![0_usize; batch_rank];
            let mut rhs_multi = vec![0_usize; batch_rank];
            for b in 0..lhs_count {
                let mut rem = b;
                for d in (0..batch_rank).rev() {
                    let s = lhs_batch_dims[d];
                    lhs_multi[d] = rem % s;
                    rem /= s;
                }
                for d in 0..batch_rank {
                    rhs_multi[d] = lhs_multi[d] / n_rep[d];
                }
                let mut rhs_b = 0_usize;
                for d in 0..batch_rank {
                    rhs_b = rhs_b * rhs_batch_dims[d] + rhs_multi[d];
                }
                let lhs_off = b * lhs_per;
                let rhs_off = rhs_b * rhs_per;
                let out_off = b * out_per;
                // Seed each output row with the broadcast bias.
                for i in 0..m {
                    let row_off = out_off + i * n;
                    out_view[row_off..row_off + n].copy_from_slice(bias_view);
                }
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
    };
}

fused_linear_native_kernel!(fused_linear_f32, f32);
fused_linear_native_kernel!(fused_linear_f64, f64);

/// FusedLinear for half-floats — accumulate in f32, narrow at end.
macro_rules! fused_linear_half_kernel {
    ($name:ident, $T:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            lhs: &CpuStorageBytes,
            rhs: &CpuStorageBytes,
            bias: &CpuStorageBytes,
            out: &mut CpuStorageBytes,
            lhs_batch_dims: &[usize],
            rhs_batch_dims: &[usize],
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<()> {
            let n_rep = fused_linear_check::<$T>(
                stringify!($name), lhs, rhs, bias, out,
                lhs_batch_dims, rhs_batch_dims, m, n, k,
            )?;
            let lhs_view: &[$T] = lhs.as_slice()?;
            let rhs_view: &[$T] = rhs.as_slice()?;
            let bias_view: &[$T] = bias.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let bias_f32: Vec<f32> = bias_view.iter().map(|v| v.to_f32()).collect();
            let batch_rank = lhs_batch_dims.len();
            let lhs_per = m * k;
            let rhs_per = k * n;
            let out_per = m * n;
            let lhs_count: usize = lhs_batch_dims.iter().product::<usize>().max(1);
            let mut lhs_multi = vec![0_usize; batch_rank];
            let mut rhs_multi = vec![0_usize; batch_rank];
            let mut row_acc = vec![0.0_f32; n];
            for b in 0..lhs_count {
                let mut rem = b;
                for d in (0..batch_rank).rev() {
                    let s = lhs_batch_dims[d];
                    lhs_multi[d] = rem % s;
                    rem /= s;
                }
                for d in 0..batch_rank {
                    rhs_multi[d] = lhs_multi[d] / n_rep[d];
                }
                let mut rhs_b = 0_usize;
                for d in 0..batch_rank {
                    rhs_b = rhs_b * rhs_batch_dims[d] + rhs_multi[d];
                }
                let lhs_off = b * lhs_per;
                let rhs_off = rhs_b * rhs_per;
                let out_off = b * out_per;
                for i in 0..m {
                    row_acc.copy_from_slice(&bias_f32);
                    for kk in 0..k {
                        let a = lhs_view[lhs_off + i * k + kk].to_f32();
                        let rhs_row_off = rhs_off + kk * n;
                        for j in 0..n {
                            row_acc[j] += a * rhs_view[rhs_row_off + j].to_f32();
                        }
                    }
                    let out_row_off = out_off + i * n;
                    for j in 0..n {
                        out_view[out_row_off + j] = <$T>::from_f32(row_acc[j]);
                    }
                }
            }
            Ok(())
        }
    };
}

fused_linear_half_kernel!(fused_linear_bf16, half::bf16);
fused_linear_half_kernel!(fused_linear_f16, half::f16);

// =============================================================================
// FlashAttn — naive multi-head SDPA (math definition)
// =============================================================================
//
// This is the math-definition oracle, not a tiled FlashAttention-2.
// On CPU the win from tiling is marginal compared to GPU, so we keep
// the simpler O(Sq*Sk*D) form per head and let backends ship a tiled
// kernel when one is worth the maintenance cost.
//
// Inputs:
//   q:               [B, Hq,  Sq, D]
//   k, v:            [B, Hkv, Sk, D]   (GQA: Hq must be a multiple of Hkv)
//   alibi_slopes:    [Hq] (optional)
// Output: same shape as q.

#[inline]
fn flash_attn_admissible(
    qi: usize, kj: usize,
    causal: bool,
    window_left: Option<usize>,
    window_right: Option<usize>,
) -> bool {
    if causal && kj > qi { return false; }
    if let Some(w) = window_left { if kj + w < qi { return false; } }
    if let Some(w) = window_right { if kj > qi + w { return false; } }
    true
}

/// FlashAttn kernel for native arithmetic ($T = f32 or f64).
macro_rules! flash_attn_native_kernel {
    ($name:ident, $T:ty, $T_zero:expr) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            q: &CpuStorageBytes,
            k: &CpuStorageBytes,
            v: &CpuStorageBytes,
            alibi_slopes: Option<&CpuStorageBytes>,
            out: &mut CpuStorageBytes,
            b: usize, hq: usize, hkv: usize,
            sq: usize, sk: usize, d: usize,
            softmax_scale: f32,
            causal: bool,
            window_left: Option<usize>,
            window_right: Option<usize>,
            softcap: Option<f32>,
        ) -> Result<()> {
            if hq % hkv != 0 || hkv == 0 {
                return Err(Error::Msg(format!(
                    "{}: Hq={hq} must be a positive multiple of Hkv={hkv}",
                    stringify!($name),
                ))
                .bt());
            }
            let elem = std::mem::size_of::<$T>();
            if q.len_bytes()   != b * hq  * sq * d * elem
                || k.len_bytes()   != b * hkv * sk * d * elem
                || v.len_bytes()   != b * hkv * sk * d * elem
                || out.len_bytes() != b * hq  * sq * d * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if let Some(a) = alibi_slopes {
                if a.len_bytes() != hq * elem {
                    return Err(Error::Msg(format!(
                        "{}: alibi_slopes must be [{hq}] {} bytes, got {}",
                        stringify!($name), hq * elem, a.len_bytes(),
                    ))
                    .bt());
                }
            }
            let q_view: &[$T] = q.as_slice()?;
            let k_view: &[$T] = k.as_slice()?;
            let v_view: &[$T] = v.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let alibi_view: Option<&[$T]> = match alibi_slopes {
                Some(a) => Some(a.as_slice()?),
                None => None,
            };
            let groups = hq / hkv;
            let q_h_stride = sq * d;
            let q_b_stride = hq * q_h_stride;
            let k_h_stride = sk * d;
            let k_b_stride = hkv * k_h_stride;
            let o_h_stride = sq * d;
            let o_b_stride = hq * o_h_stride;
            let scale = softmax_scale as $T;
            // Zero output up front so masked rows stay zero.
            for slot in out_view.iter_mut() { *slot = $T_zero; }
            for bi in 0..b {
                for hi in 0..hq {
                    let kv_h = hi / groups;
                    let q_off = bi * q_b_stride + hi * q_h_stride;
                    let k_off = bi * k_b_stride + kv_h * k_h_stride;
                    let v_off = k_off;
                    let o_off = bi * o_b_stride + hi * o_h_stride;
                    let alibi_h: Option<$T> = alibi_view.map(|a| a[hi]);
                    for qi in 0..sq {
                        // Build admissible scores; track running max.
                        let mut scores = vec![$T_zero; sk];
                        let mut admissible = vec![false; sk];
                        let mut max_score: $T = <$T>::NEG_INFINITY;
                        for kj in 0..sk {
                            if !flash_attn_admissible(qi, kj, causal, window_left, window_right) {
                                continue;
                            }
                            admissible[kj] = true;
                            let mut acc: $T = $T_zero;
                            let q_row = &q_view[q_off + qi * d .. q_off + (qi + 1) * d];
                            let k_row = &k_view[k_off + kj * d .. k_off + (kj + 1) * d];
                            for (qx, kx) in q_row.iter().zip(k_row.iter()) {
                                acc += (*qx) * (*kx);
                            }
                            let mut s = acc * scale;
                            if let Some(c) = softcap {
                                let cc = c as $T;
                                s = (s / cc).tanh() * cc;
                            }
                            if let Some(slope) = alibi_h {
                                let delta = (kj as f32 - qi as f32) as $T;
                                s += slope * delta;
                            }
                            scores[kj] = s;
                            if s > max_score { max_score = s; }
                        }
                        if !max_score.is_finite() { continue; }
                        let mut sum: $T = $T_zero;
                        for (s, ad) in scores.iter_mut().zip(admissible.iter()) {
                            if *ad {
                                *s = (*s - max_score).exp();
                                sum += *s;
                            } else {
                                *s = $T_zero;
                            }
                        }
                        if sum == $T_zero { continue; }
                        let inv_sum = (1.0 as $T) / sum;
                        for kj in 0..sk {
                            if !admissible[kj] { continue; }
                            let p_ij = scores[kj] * inv_sum;
                            if p_ij == $T_zero { continue; }
                            let v_row = &v_view[v_off + kj * d .. v_off + (kj + 1) * d];
                            for (od, vd) in
                                out_view[o_off + qi * d .. o_off + (qi + 1) * d]
                                    .iter_mut()
                                    .zip(v_row.iter())
                            {
                                *od += p_ij * (*vd);
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

flash_attn_native_kernel!(flash_attn_f32, f32, 0.0_f32);
flash_attn_native_kernel!(flash_attn_f64, f64, 0.0_f64);

/// FlashAttn for half-floats — accumulates the dot products and
/// softmax math in f32, narrows back to T for the output.
macro_rules! flash_attn_half_kernel {
    ($name:ident, $T:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            q: &CpuStorageBytes,
            k: &CpuStorageBytes,
            v: &CpuStorageBytes,
            alibi_slopes: Option<&CpuStorageBytes>,
            out: &mut CpuStorageBytes,
            b: usize, hq: usize, hkv: usize,
            sq: usize, sk: usize, d: usize,
            softmax_scale: f32,
            causal: bool,
            window_left: Option<usize>,
            window_right: Option<usize>,
            softcap: Option<f32>,
        ) -> Result<()> {
            if hq % hkv != 0 || hkv == 0 {
                return Err(Error::Msg(format!(
                    "{}: Hq={hq} must be a positive multiple of Hkv={hkv}",
                    stringify!($name),
                ))
                .bt());
            }
            let elem = std::mem::size_of::<$T>();
            if q.len_bytes()   != b * hq  * sq * d * elem
                || k.len_bytes()   != b * hkv * sk * d * elem
                || v.len_bytes()   != b * hkv * sk * d * elem
                || out.len_bytes() != b * hq  * sq * d * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if let Some(a) = alibi_slopes {
                if a.len_bytes() != hq * elem {
                    return Err(Error::Msg(format!(
                        "{}: alibi_slopes bytes mismatch", stringify!($name),
                    ))
                    .bt());
                }
            }
            let q_view: &[$T] = q.as_slice()?;
            let k_view: &[$T] = k.as_slice()?;
            let v_view: &[$T] = v.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let alibi_view: Option<&[$T]> = match alibi_slopes {
                Some(a) => Some(a.as_slice()?),
                None => None,
            };
            let groups = hq / hkv;
            let q_h_stride = sq * d;
            let q_b_stride = hq * q_h_stride;
            let k_h_stride = sk * d;
            let k_b_stride = hkv * k_h_stride;
            let o_h_stride = sq * d;
            let o_b_stride = hq * o_h_stride;
            for slot in out_view.iter_mut() { *slot = <$T>::from_f32(0.0); }
            for bi in 0..b {
                for hi in 0..hq {
                    let kv_h = hi / groups;
                    let q_off = bi * q_b_stride + hi * q_h_stride;
                    let k_off = bi * k_b_stride + kv_h * k_h_stride;
                    let v_off = k_off;
                    let o_off = bi * o_b_stride + hi * o_h_stride;
                    let alibi_h: Option<f32> = alibi_view.map(|a| a[hi].to_f32());
                    for qi in 0..sq {
                        let mut scores = vec![0.0_f32; sk];
                        let mut admissible = vec![false; sk];
                        let mut max_score = f32::NEG_INFINITY;
                        for kj in 0..sk {
                            if !flash_attn_admissible(qi, kj, causal, window_left, window_right) {
                                continue;
                            }
                            admissible[kj] = true;
                            let mut acc = 0.0_f32;
                            let q_row = &q_view[q_off + qi * d .. q_off + (qi + 1) * d];
                            let k_row = &k_view[k_off + kj * d .. k_off + (kj + 1) * d];
                            for (qx, kx) in q_row.iter().zip(k_row.iter()) {
                                acc += qx.to_f32() * kx.to_f32();
                            }
                            let mut s = acc * softmax_scale;
                            if let Some(c) = softcap {
                                s = (s / c).tanh() * c;
                            }
                            if let Some(slope) = alibi_h {
                                let delta = kj as f32 - qi as f32;
                                s += slope * delta;
                            }
                            scores[kj] = s;
                            if s > max_score { max_score = s; }
                        }
                        if !max_score.is_finite() { continue; }
                        let mut sum = 0.0_f32;
                        for (s, ad) in scores.iter_mut().zip(admissible.iter()) {
                            if *ad {
                                *s = (*s - max_score).exp();
                                sum += *s;
                            } else {
                                *s = 0.0;
                            }
                        }
                        if sum == 0.0 { continue; }
                        let inv_sum = 1.0_f32 / sum;
                        let mut row_acc = vec![0.0_f32; d];
                        for kj in 0..sk {
                            if !admissible[kj] { continue; }
                            let p_ij = scores[kj] * inv_sum;
                            if p_ij == 0.0 { continue; }
                            let v_row = &v_view[v_off + kj * d .. v_off + (kj + 1) * d];
                            for (od, vd) in row_acc.iter_mut().zip(v_row.iter()) {
                                *od += p_ij * vd.to_f32();
                            }
                        }
                        for (slot, val) in
                            out_view[o_off + qi * d .. o_off + (qi + 1) * d]
                                .iter_mut()
                                .zip(row_acc.iter())
                        {
                            *slot = <$T>::from_f32(*val);
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

flash_attn_half_kernel!(flash_attn_bf16, half::bf16);
flash_attn_half_kernel!(flash_attn_f16, half::f16);

// =============================================================================
// PagedAttn — paged-KV-cache attention (naive)
// =============================================================================
//
// Inputs:
//   q:             [B, Hq, Sq, D]
//   k_cache:       [num_blocks, block_size, Hkv, D]
//   v_cache:       [num_blocks, block_size, Hkv, D]
//   block_table:   [B, max_num_blocks_per_seq] (U32)  logical → physical block
//   context_lens:  [B] (U32)                          true context length per seq
//   alibi_slopes:  [Hq] (optional)
// Output: [B, Hq, Sq, D] (same shape as q).
//
// Causal masking is implicit: query at position `q_pos = ctx_len - Sq + sq`
// (absolute position in the sequence) admits keys at `kj <= q_pos`.

/// PagedAttn for native arithmetic.
macro_rules! paged_attn_native_kernel {
    ($name:ident, $T:ty, $T_zero:expr) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            q: &CpuStorageBytes,
            k_cache: &CpuStorageBytes,
            v_cache: &CpuStorageBytes,
            block_table: &CpuStorageBytes,
            context_lens: &CpuStorageBytes,
            alibi_slopes: Option<&CpuStorageBytes>,
            out: &mut CpuStorageBytes,
            b: usize, hq: usize, hkv: usize,
            sq: usize, d: usize,
            block_size: usize,
            max_blocks_per_seq: usize,
            num_blocks: usize,
            softmax_scale: f32,
            softcap: Option<f32>,
        ) -> Result<()> {
            if hq % hkv != 0 || hkv == 0 || block_size == 0 {
                return Err(Error::Msg(format!(
                    "{}: contract violation (Hq={hq}, Hkv={hkv}, block_size={block_size})",
                    stringify!($name),
                ))
                .bt());
            }
            let elem = std::mem::size_of::<$T>();
            let u32_elem = std::mem::size_of::<u32>();
            if q.len_bytes() != b * hq * sq * d * elem
                || k_cache.len_bytes() != num_blocks * block_size * hkv * d * elem
                || v_cache.len_bytes() != num_blocks * block_size * hkv * d * elem
                || block_table.len_bytes() != b * max_blocks_per_seq * u32_elem
                || context_lens.len_bytes() != b * u32_elem
                || out.len_bytes() != b * hq * sq * d * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if let Some(a) = alibi_slopes {
                if a.len_bytes() != hq * elem {
                    return Err(Error::Msg(format!(
                        "{}: alibi_slopes bytes mismatch", stringify!($name),
                    ))
                    .bt());
                }
            }
            let q_view: &[$T] = q.as_slice()?;
            let k_view: &[$T] = k_cache.as_slice()?;
            let v_view: &[$T] = v_cache.as_slice()?;
            let bt_view: &[u32] = block_table.as_slice()?;
            let cl_view: &[u32] = context_lens.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let alibi_view: Option<&[$T]> = match alibi_slopes {
                Some(a) => Some(a.as_slice()?),
                None => None,
            };
            let groups = hq / hkv;
            let q_h_stride = sq * d;
            let q_b_stride = hq * q_h_stride;
            let o_h_stride = sq * d;
            let o_b_stride = hq * o_h_stride;
            // KV cache strides for [num_blocks, block_size, Hkv, D]
            let kv_block_stride = block_size * hkv * d;
            let kv_slot_stride = hkv * d;
            let kv_head_stride = d;
            let scale = softmax_scale as $T;
            for slot in out_view.iter_mut() { *slot = $T_zero; }
            for bi in 0..b {
                let ctx_len = cl_view[bi] as usize;
                if ctx_len == 0 { continue; }
                if ctx_len > max_blocks_per_seq * block_size {
                    return Err(Error::Msg(format!(
                        "{}: context_lens[{bi}]={ctx_len} > capacity {} block_size*max_blocks",
                        stringify!($name), max_blocks_per_seq * block_size,
                    ))
                    .bt());
                }
                let bt_off = bi * max_blocks_per_seq;
                for hi in 0..hq {
                    let kv_h = hi / groups;
                    let q_off = bi * q_b_stride + hi * q_h_stride;
                    let o_off = bi * o_b_stride + hi * o_h_stride;
                    let alibi_h: Option<$T> = alibi_view.map(|a| a[hi]);
                    for qi in 0..sq {
                        let q_pos_abs = ctx_len + qi - sq;
                        // Build admissible scores over [0, ctx_len).
                        let mut scores = vec![$T_zero; ctx_len];
                        let mut admissible = vec![false; ctx_len];
                        let mut max_score: $T = <$T>::NEG_INFINITY;
                        for kj in 0..ctx_len {
                            // Implicit causal mask.
                            if kj > q_pos_abs { continue; }
                            let logical_block = kj / block_size;
                            let block_off = kj % block_size;
                            let physical_block = bt_view[bt_off + logical_block] as usize;
                            if physical_block >= num_blocks {
                                return Err(Error::Msg(format!(
                                    "{}: block_table[{bi}, {logical_block}]={physical_block} \
                                     out of range (num_blocks={num_blocks})",
                                    stringify!($name),
                                ))
                                .bt());
                            }
                            admissible[kj] = true;
                            let k_off = physical_block * kv_block_stride
                                + block_off * kv_slot_stride
                                + kv_h * kv_head_stride;
                            let v_off = k_off;
                            let mut acc: $T = $T_zero;
                            let q_row = &q_view[q_off + qi * d .. q_off + (qi + 1) * d];
                            let k_row = &k_view[k_off .. k_off + d];
                            for (qx, kx) in q_row.iter().zip(k_row.iter()) {
                                acc += (*qx) * (*kx);
                            }
                            let mut s = acc * scale;
                            if let Some(c) = softcap {
                                let cc = c as $T;
                                s = (s / cc).tanh() * cc;
                            }
                            if let Some(slope) = alibi_h {
                                let delta = (kj as f32 - q_pos_abs as f32) as $T;
                                s += slope * delta;
                            }
                            scores[kj] = s;
                            if s > max_score { max_score = s; }
                            // The v_off variable is just a name alias for k_off in
                            // this naive impl since v_cache shares the block layout.
                            let _ = v_off;
                        }
                        if !max_score.is_finite() { continue; }
                        let mut sum: $T = $T_zero;
                        for (s, ad) in scores.iter_mut().zip(admissible.iter()) {
                            if *ad {
                                *s = (*s - max_score).exp();
                                sum += *s;
                            } else {
                                *s = $T_zero;
                            }
                        }
                        if sum == $T_zero { continue; }
                        let inv_sum = (1.0 as $T) / sum;
                        for kj in 0..ctx_len {
                            if !admissible[kj] { continue; }
                            let p_ij = scores[kj] * inv_sum;
                            if p_ij == $T_zero { continue; }
                            let logical_block = kj / block_size;
                            let block_off = kj % block_size;
                            let physical_block = bt_view[bt_off + logical_block] as usize;
                            let v_off = physical_block * kv_block_stride
                                + block_off * kv_slot_stride
                                + kv_h * kv_head_stride;
                            let v_row = &v_view[v_off .. v_off + d];
                            for (od, vd) in
                                out_view[o_off + qi * d .. o_off + (qi + 1) * d]
                                    .iter_mut()
                                    .zip(v_row.iter())
                            {
                                *od += p_ij * (*vd);
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

paged_attn_native_kernel!(paged_attn_f32, f32, 0.0_f32);
paged_attn_native_kernel!(paged_attn_f64, f64, 0.0_f64);

/// PagedAttn for half-floats — f32 accumulator.
macro_rules! paged_attn_half_kernel {
    ($name:ident, $T:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            q: &CpuStorageBytes,
            k_cache: &CpuStorageBytes,
            v_cache: &CpuStorageBytes,
            block_table: &CpuStorageBytes,
            context_lens: &CpuStorageBytes,
            alibi_slopes: Option<&CpuStorageBytes>,
            out: &mut CpuStorageBytes,
            b: usize, hq: usize, hkv: usize,
            sq: usize, d: usize,
            block_size: usize,
            max_blocks_per_seq: usize,
            num_blocks: usize,
            softmax_scale: f32,
            softcap: Option<f32>,
        ) -> Result<()> {
            if hq % hkv != 0 || hkv == 0 || block_size == 0 {
                return Err(Error::Msg(format!(
                    "{}: contract violation", stringify!($name),
                ))
                .bt());
            }
            let elem = std::mem::size_of::<$T>();
            let u32_elem = std::mem::size_of::<u32>();
            if q.len_bytes() != b * hq * sq * d * elem
                || k_cache.len_bytes() != num_blocks * block_size * hkv * d * elem
                || v_cache.len_bytes() != num_blocks * block_size * hkv * d * elem
                || block_table.len_bytes() != b * max_blocks_per_seq * u32_elem
                || context_lens.len_bytes() != b * u32_elem
                || out.len_bytes() != b * hq * sq * d * elem
            {
                return Err(Error::Msg(format!(
                    "{}: bytes mismatch", stringify!($name),
                ))
                .bt());
            }
            if let Some(a) = alibi_slopes {
                if a.len_bytes() != hq * elem {
                    return Err(Error::Msg(format!(
                        "{}: alibi_slopes bytes mismatch", stringify!($name),
                    ))
                    .bt());
                }
            }
            let q_view: &[$T] = q.as_slice()?;
            let k_view: &[$T] = k_cache.as_slice()?;
            let v_view: &[$T] = v_cache.as_slice()?;
            let bt_view: &[u32] = block_table.as_slice()?;
            let cl_view: &[u32] = context_lens.as_slice()?;
            let out_view: &mut [$T] = out.as_slice_mut()?;
            let alibi_view: Option<&[$T]> = match alibi_slopes {
                Some(a) => Some(a.as_slice()?),
                None => None,
            };
            let groups = hq / hkv;
            let q_h_stride = sq * d;
            let q_b_stride = hq * q_h_stride;
            let o_h_stride = sq * d;
            let o_b_stride = hq * o_h_stride;
            let kv_block_stride = block_size * hkv * d;
            let kv_slot_stride = hkv * d;
            let kv_head_stride = d;
            for slot in out_view.iter_mut() { *slot = <$T>::from_f32(0.0); }
            for bi in 0..b {
                let ctx_len = cl_view[bi] as usize;
                if ctx_len == 0 { continue; }
                if ctx_len > max_blocks_per_seq * block_size {
                    return Err(Error::Msg(format!(
                        "{}: ctx_len out of capacity", stringify!($name),
                    ))
                    .bt());
                }
                let bt_off = bi * max_blocks_per_seq;
                for hi in 0..hq {
                    let kv_h = hi / groups;
                    let q_off = bi * q_b_stride + hi * q_h_stride;
                    let o_off = bi * o_b_stride + hi * o_h_stride;
                    let alibi_h: Option<f32> = alibi_view.map(|a| a[hi].to_f32());
                    for qi in 0..sq {
                        let q_pos_abs = ctx_len + qi - sq;
                        let mut scores = vec![0.0_f32; ctx_len];
                        let mut admissible = vec![false; ctx_len];
                        let mut max_score = f32::NEG_INFINITY;
                        for kj in 0..ctx_len {
                            if kj > q_pos_abs { continue; }
                            let logical_block = kj / block_size;
                            let block_off = kj % block_size;
                            let physical_block = bt_view[bt_off + logical_block] as usize;
                            if physical_block >= num_blocks {
                                return Err(Error::Msg(format!(
                                    "{}: block_table out of range", stringify!($name),
                                ))
                                .bt());
                            }
                            admissible[kj] = true;
                            let k_off = physical_block * kv_block_stride
                                + block_off * kv_slot_stride
                                + kv_h * kv_head_stride;
                            let mut acc = 0.0_f32;
                            let q_row = &q_view[q_off + qi * d .. q_off + (qi + 1) * d];
                            let k_row = &k_view[k_off .. k_off + d];
                            for (qx, kx) in q_row.iter().zip(k_row.iter()) {
                                acc += qx.to_f32() * kx.to_f32();
                            }
                            let mut s = acc * softmax_scale;
                            if let Some(c) = softcap {
                                s = (s / c).tanh() * c;
                            }
                            if let Some(slope) = alibi_h {
                                let delta = kj as f32 - q_pos_abs as f32;
                                s += slope * delta;
                            }
                            scores[kj] = s;
                            if s > max_score { max_score = s; }
                        }
                        if !max_score.is_finite() { continue; }
                        let mut sum = 0.0_f32;
                        for (s, ad) in scores.iter_mut().zip(admissible.iter()) {
                            if *ad {
                                *s = (*s - max_score).exp();
                                sum += *s;
                            } else {
                                *s = 0.0;
                            }
                        }
                        if sum == 0.0 { continue; }
                        let inv_sum = 1.0_f32 / sum;
                        let mut row_acc = vec![0.0_f32; d];
                        for kj in 0..ctx_len {
                            if !admissible[kj] { continue; }
                            let p_ij = scores[kj] * inv_sum;
                            if p_ij == 0.0 { continue; }
                            let logical_block = kj / block_size;
                            let block_off = kj % block_size;
                            let physical_block = bt_view[bt_off + logical_block] as usize;
                            let v_off = physical_block * kv_block_stride
                                + block_off * kv_slot_stride
                                + kv_h * kv_head_stride;
                            let v_row = &v_view[v_off .. v_off + d];
                            for (od, vd) in row_acc.iter_mut().zip(v_row.iter()) {
                                *od += p_ij * vd.to_f32();
                            }
                        }
                        for (slot, val) in
                            out_view[o_off + qi * d .. o_off + (qi + 1) * d]
                                .iter_mut()
                                .zip(row_acc.iter())
                        {
                            *slot = <$T>::from_f32(*val);
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

paged_attn_half_kernel!(paged_attn_bf16, half::bf16);
paged_attn_half_kernel!(paged_attn_f16, half::f16);

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
        check_reduce_shape("sum_reduce_f32", input, output, input_shape, reduce_dims, std::mem::size_of::<f32>())?;
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

/// Argmax / argmin along a single dim — same shape contract as
/// the value-reducing kernels (output's `dim` is removed) but the
/// output dtype is `u32` carrying the index of the extremum within
/// each row of `dim`.
///
/// Tie-breaking: returns the first index that achieves the
/// extremum (lowest index on ties). NaN handling propagates per
/// IEEE-754 — comparisons against NaN return false, so a NaN
/// row's argmax is whichever non-NaN slot it first encountered
/// (or 0 if every value is NaN).
fn argextremum_dim_f32(
    name: &str,
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    dim: usize,
    is_better: fn(f32, f32) -> bool,
    init: f32,
) -> Result<()> {
    if dim >= input_shape.len() {
        return Err(Error::Msg(format!(
            "{name}: dim {dim} out of range for rank {}",
            input_shape.len(),
        ))
        .bt());
    }
    let f32_size = std::mem::size_of::<f32>();
    let u32_size = std::mem::size_of::<u32>();
    let total_input: usize = input_shape.iter().product();
    if input.len_bytes() != total_input.saturating_mul(f32_size) {
        return Err(Error::Msg(format!(
            "{name}: input bytes={} doesn't match shape {input_shape:?} (f32)",
            input.len_bytes(),
        ))
        .bt());
    }
    let outer_count: usize = input_shape[..dim].iter().product();
    let dim_size = input_shape[dim];
    let inner_count: usize = input_shape[dim + 1..].iter().product();
    let output_count = outer_count * inner_count;
    if output.len_bytes() != output_count.saturating_mul(u32_size) {
        return Err(Error::Msg(format!(
            "{name}: output bytes={} doesn't match (input shape - dim {dim}) × {u32_size}",
            output.len_bytes(),
        ))
        .bt());
    }
    if dim_size == 0 {
        return Err(Error::Msg(format!(
            "{name}: dim {dim} has size 0 — argmax/argmin undefined",
        ))
        .bt());
    }
    let in_view: &[f32] = input.as_slice()?;
    let out_view: &mut [u32] = output.as_slice_mut()?;
    for outer in 0..outer_count {
        for inner in 0..inner_count {
            let mut best_val = init;
            let mut best_idx = 0u32;
            for d in 0..dim_size {
                let off = (outer * dim_size + d) * inner_count + inner;
                let v = in_view[off];
                // Initial slot: take the first valid value seen.
                if d == 0 {
                    best_val = v;
                    best_idx = 0;
                } else if is_better(v, best_val) {
                    best_val = v;
                    best_idx = d as u32;
                }
            }
            out_view[outer * inner_count + inner] = best_idx;
        }
    }
    Ok(())
}

/// Argmax along one dim — `out[i] = argmax over dim of input`.
pub fn argmax_dim_f32(
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    dim: usize,
) -> Result<()> {
    argextremum_dim_f32(
        "argmax_dim_f32",
        input,
        output,
        input_shape,
        dim,
        |new, best| new > best,
        f32::NEG_INFINITY,
    )
}

/// Argmin along one dim — `out[i] = argmin over dim of input`.
pub fn argmin_dim_f32(
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    dim: usize,
) -> Result<()> {
    argextremum_dim_f32(
        "argmin_dim_f32",
        input,
        output,
        input_shape,
        dim,
        |new, best| new < best,
        f32::INFINITY,
    )
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
        check_reduce_shape(name, input, output, input_shape, reduce_dims, std::mem::size_of::<f32>())?;
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

// =============================================================================
// Reduction kernels (f64) — direct mirrors of the f32 versions
// =============================================================================

/// Sum-reduce `f64` — same algorithm as [`sum_reduce_f32`] with
/// f64 element type and accumulator.
pub fn sum_reduce_f64(
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
) -> Result<()> {
    let (total_input, kept) = check_reduce_shape(
        "sum_reduce_f64", input, output, input_shape, reduce_dims, std::mem::size_of::<f64>(),
    )?;
    let in_view: &[f64] = input.as_slice()?;
    let out_view: &mut [f64] = output.as_slice_mut()?;
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

/// Mean-reduce `f64`.
pub fn mean_reduce_f64(
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
) -> Result<()> {
    sum_reduce_f64(input, output, input_shape, reduce_dims)?;
    let divisor: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();
    if divisor == 0 {
        return Err(Error::Msg(
            "mean_reduce_f64: divisor zero (reduced dim has size 0)".to_string(),
        )
        .bt());
    }
    let inv = 1.0_f64 / divisor as f64;
    let out_view: &mut [f64] = output.as_slice_mut()?;
    for slot in out_view.iter_mut() {
        *slot *= inv;
    }
    Ok(())
}

/// Generic reduce helper for max/min on `f64`.
fn reduce_f64_generic(
    name: &str,
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
    init: f64,
    combine: fn(f64, f64) -> f64,
) -> Result<()> {
    let (total_input, kept) = check_reduce_shape(
        name, input, output, input_shape, reduce_dims, std::mem::size_of::<f64>(),
    )?;
    let in_view: &[f64] = input.as_slice()?;
    let out_view: &mut [f64] = output.as_slice_mut()?;
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

/// Max-reduce `f64`.
pub fn max_reduce_f64(
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
) -> Result<()> {
    reduce_f64_generic(
        "max_reduce_f64", input, output, input_shape, reduce_dims,
        f64::NEG_INFINITY, |a, b| a.max(b),
    )
}

/// Min-reduce `f64`.
pub fn min_reduce_f64(
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
) -> Result<()> {
    reduce_f64_generic(
        "min_reduce_f64", input, output, input_shape, reduce_dims,
        f64::INFINITY, |a, b| a.min(b),
    )
}

// =============================================================================
// Reduction kernels (bf16 / f16) — accumulate in f32 for stability
// =============================================================================
//
// Half-float reductions accumulate in f32: summing many bf16 values
// in bf16 loses precision rapidly (each add rounds to ~3 decimal
// digits), but a streaming f32 accumulator gives full f32 precision
// up to ~16M elements. The result is narrowed to bf16/f16 only at
// the end. This matches what cuBLAS / TPU / cuDNN do.

macro_rules! sum_reduce_half {
    ($name:ident, $T:ty, $T_size:expr, $type_name:literal) => {
        pub fn $name(
            input: &CpuStorageBytes,
            output: &mut CpuStorageBytes,
            input_shape: &[usize],
            reduce_dims: &[usize],
        ) -> Result<()> {
            let (total_input, kept) = check_reduce_shape(
                concat!(stringify!($name)), input, output, input_shape, reduce_dims, $T_size,
            )?;
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = output.as_slice_mut()?;
            let total_output = out_view.len();
            let mut f32_acc = vec![0.0_f32; total_output];
            let mut mi = vec![0usize; input_shape.len()];
            for flat in 0..total_input {
                decode_multi_index(flat, input_shape, &mut mi);
                let oi = output_index(input_shape, &kept, &mi);
                f32_acc[oi] += in_view[flat].to_f32();
            }
            for (slot, &v) in out_view.iter_mut().zip(&f32_acc) {
                *slot = <$T>::from_f32(v);
            }
            let _ = $type_name;
            Ok(())
        }
    };
}

sum_reduce_half!(sum_reduce_bf16, half::bf16, std::mem::size_of::<half::bf16>(), "bf16");
sum_reduce_half!(sum_reduce_f16, half::f16, std::mem::size_of::<half::f16>(), "f16");

macro_rules! mean_reduce_half {
    ($name:ident, $sum_kernel:path, $T:ty, $type_name:literal) => {
        pub fn $name(
            input: &CpuStorageBytes,
            output: &mut CpuStorageBytes,
            input_shape: &[usize],
            reduce_dims: &[usize],
        ) -> Result<()> {
            $sum_kernel(input, output, input_shape, reduce_dims)?;
            let divisor: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();
            if divisor == 0 {
                return Err(Error::Msg(format!(
                    "{}: divisor zero (reduced dim has size 0)",
                    concat!(stringify!($name)),
                ))
                .bt());
            }
            let inv = 1.0_f32 / divisor as f32;
            let out_view: &mut [$T] = output.as_slice_mut()?;
            for slot in out_view.iter_mut() {
                *slot = <$T>::from_f32(slot.to_f32() * inv);
            }
            let _ = $type_name;
            Ok(())
        }
    };
}

mean_reduce_half!(mean_reduce_bf16, sum_reduce_bf16, half::bf16, "bf16");
mean_reduce_half!(mean_reduce_f16, sum_reduce_f16, half::f16, "f16");

macro_rules! reduce_half_extremum {
    ($name:ident, $T:ty, $T_size:expr, $init:expr, $combine:expr) => {
        pub fn $name(
            input: &CpuStorageBytes,
            output: &mut CpuStorageBytes,
            input_shape: &[usize],
            reduce_dims: &[usize],
        ) -> Result<()> {
            let (total_input, kept) = check_reduce_shape(
                concat!(stringify!($name)), input, output, input_shape, reduce_dims, $T_size,
            )?;
            let in_view: &[$T] = input.as_slice()?;
            let out_view: &mut [$T] = output.as_slice_mut()?;
            let total_output = out_view.len();
            // Run the reduction in f32 for accuracy + uniform NaN
            // handling, then narrow back to the half-float type.
            let mut f32_acc = vec![$init; total_output];
            let mut mi = vec![0usize; input_shape.len()];
            let combine: fn(f32, f32) -> f32 = $combine;
            for flat in 0..total_input {
                decode_multi_index(flat, input_shape, &mut mi);
                let oi = output_index(input_shape, &kept, &mi);
                f32_acc[oi] = combine(f32_acc[oi], in_view[flat].to_f32());
            }
            for (slot, &v) in out_view.iter_mut().zip(&f32_acc) {
                *slot = <$T>::from_f32(v);
            }
            Ok(())
        }
    };
}

reduce_half_extremum!(max_reduce_bf16, half::bf16, std::mem::size_of::<half::bf16>(), f32::NEG_INFINITY, |a: f32, b: f32| a.max(b));
reduce_half_extremum!(max_reduce_f16, half::f16, std::mem::size_of::<half::f16>(), f32::NEG_INFINITY, |a: f32, b: f32| a.max(b));
reduce_half_extremum!(min_reduce_bf16, half::bf16, std::mem::size_of::<half::bf16>(), f32::INFINITY, |a: f32, b: f32| a.min(b));
reduce_half_extremum!(min_reduce_f16, half::f16, std::mem::size_of::<half::f16>(), f32::INFINITY, |a: f32, b: f32| a.min(b));

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

/// f64 mirror of [`gelu_f32`]. Kept hand-rolled for the same
/// reason — the inner expression is more than a single arithmetic
/// step so the macro form doesn't quite fit.
pub fn gelu_f64(input: &CpuStorageBytes, out: &mut CpuStorageBytes) -> Result<()> {
    check_lens_2("gelu_f64", input.len_bytes(), out.len_bytes())?;
    let in_view: &[f64] = input.as_slice()?;
    let out_view: &mut [f64] = out.as_slice_mut()?;
    const COEFF: f64 = 0.797_884_560_802_865_4;
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
    fn qmatmul_q4_0_f32_zero_activations_yields_zero() {
        // Smallest valid shape: n=2, k=32 (one Q4_0 block per row).
        // With zero activations, the weight contents are irrelevant —
        // every dot product is 0. Sanity-checks the dispatch + size
        // validation without needing valid quantized data.
        let act = CpuStorageBytes::from_slice(&[0.0_f32; 32]);
        let block_size = std::mem::size_of::<fuel_quantized::BlockQ4_0>();
        // Must be a multiple of 4 for U32-aligned storage.
        assert!(block_size % 2 == 0);
        let w = CpuStorageBytes::from_bytes(&vec![0u8; 2 * block_size]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4);
        qmatmul_q4_0_f32(&act, &w, &mut out, 1, 1, 2, 32).expect("qmatmul");
        let r: &[f32] = out.as_slice().unwrap();
        assert_eq!(r, &[0.0, 0.0]);
    }

    #[test]
    fn qmatmul_q4_0_f32_unit_weight_sums_activations() {
        // Construct a Q4_0 weight where every weight = 1.0 by
        // setting d (scale) = 1.0 and every nibble = 9 (so the
        // effective weight is 1 * (9 - 8) = 1).
        // Then A @ W^T computes per-row sum of activations.
        use half::f16;
        let block_size = std::mem::size_of::<fuel_quantized::BlockQ4_0>();
        let mut w_bytes = vec![0u8; 2 * block_size];
        for block_idx in 0..2 {
            let off = block_idx * block_size;
            // d = f16(1.0) — little-endian bytes
            let d_bytes = f16::from_f32(1.0).to_le_bytes();
            w_bytes[off..off + 2].copy_from_slice(&d_bytes);
            // Every nibble = 9 → packed byte = 0x99 (low nibble first)
            for i in 0..16 {
                w_bytes[off + 2 + i] = 0x99;
            }
        }
        let w = CpuStorageBytes::from_bytes(&w_bytes);

        // Activations: [1, 2, 3, ..., 32]; sum = 528.
        let act_vec: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let act = CpuStorageBytes::from_slice(&act_vec);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4);
        qmatmul_q4_0_f32(&act, &w, &mut out, 1, 1, 2, 32).expect("qmatmul");
        let r: &[f32] = out.as_slice().unwrap();
        // Both output rows = sum over k of activations = 528.
        // Tolerance ~0.5 — fuel_quantized's matmul re-quantizes the
        // f32 activations to Q8_1 internally, introducing small
        // round-trip error.
        assert!((r[0] - 528.0).abs() < 0.5, "got {}, want 528", r[0]);
        assert!((r[1] - 528.0).abs() < 0.5, "got {}, want 528", r[1]);
    }

    #[test]
    fn qmatmul_q4_0_f32_rejects_bad_k() {
        // k=33 isn't a multiple of 32 — must error.
        let act = CpuStorageBytes::from_slice(&[0.0_f32; 33]);
        let w = CpuStorageBytes::from_bytes(&[0u8; 18]); // one block
        let mut out = CpuStorageBytes::from_zero_bytes(4);
        let r = qmatmul_q4_0_f32(&act, &w, &mut out, 1, 1, 1, 33);
        assert!(r.is_err(), "k must be a multiple of 32 for Q4_0");
    }

    #[test]
    fn softmax_last_dim_bf16_uniform_row() {
        let v: Vec<half::bf16> = [1.0_f32, 1.0, 1.0, 1.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let input = CpuStorageBytes::from_slice(&v);
        let mut out = CpuStorageBytes::from_zero_bytes(4 * 2);
        softmax_last_dim_bf16(&input, &mut out, 1, 4).expect("softmax bf16");
        let r: &[half::bf16] = out.as_slice().unwrap();
        for x in r {
            assert!((x.to_f32() - 0.25).abs() < 0.01);
        }
    }

    #[test]
    fn softmax_last_dim_f16_sums_to_one() {
        let v: Vec<half::f16> = [1.0_f32, 2.0, 3.0]
            .iter().map(|&x| half::f16::from_f32(x)).collect();
        let input = CpuStorageBytes::from_slice(&v);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 2);
        softmax_last_dim_f16(&input, &mut out, 1, 3).expect("softmax f16");
        let r: &[half::f16] = out.as_slice().unwrap();
        let sum: f32 = r.iter().map(|x| x.to_f32()).sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn rms_norm_last_dim_bf16_basic() {
        let v: Vec<half::bf16> = [3.0_f32, 4.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let input = CpuStorageBytes::from_slice(&v);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 2);
        rms_norm_last_dim_bf16(&input, &mut out, 1, 2, 0.0).expect("rms_norm bf16");
        let r: &[half::bf16] = out.as_slice().unwrap();
        let rms = (12.5_f32).sqrt();
        // bf16 has ~3 decimal digits — accept small absolute error.
        assert!((r[0].to_f32() - 3.0 / rms).abs() < 0.05);
        assert!((r[1].to_f32() - 4.0 / rms).abs() < 0.05);
    }

    #[test]
    fn layer_norm_last_dim_f16_zero_mean() {
        let v: Vec<half::f16> = [1.0_f32, 2.0, 3.0]
            .iter().map(|&x| half::f16::from_f32(x)).collect();
        let input = CpuStorageBytes::from_slice(&v);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 2);
        layer_norm_last_dim_f16(&input, &mut out, 1, 3, 0.0).expect("layer_norm f16");
        let r: &[half::f16] = out.as_slice().unwrap();
        // Output mean should be ≈ 0; output var should be ≈ 1.
        let result_f32: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        let mean: f32 = result_f32.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 0.01);
        let var: f32 = result_f32.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 3.0;
        assert!((var - 1.0).abs() < 0.05);
    }

    #[test]
    fn rope_bf16_pi_over_two_swaps_with_sign() {
        // x [1, 1, 4] = [1, 2, 3, 4]; cos=0, sin=1.
        // Expected: [-3, -4, 1, 2].
        let x_v: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let zero_v: Vec<half::bf16> = [0.0_f32; 4]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let one_v: Vec<half::bf16> = [1.0_f32; 4]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let x = CpuStorageBytes::from_slice(&x_v);
        let cos = CpuStorageBytes::from_slice(&zero_v);
        let sin = CpuStorageBytes::from_slice(&one_v);
        let mut out = CpuStorageBytes::from_zero_bytes(4 * 2);
        rope_bf16(&x, &cos, &sin, &mut out, 1, 1, 4).expect("rope bf16");
        let r: &[half::bf16] = out.as_slice().unwrap();
        let result_f32: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        assert_eq!(result_f32, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn sum_reduce_bf16_along_one_dim() {
        let v: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let input = CpuStorageBytes::from_slice(&v);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 2);
        sum_reduce_bf16(&input, &mut out, &[2, 3], &[1]).expect("sum bf16");
        let r: &[half::bf16] = out.as_slice().unwrap();
        let f32_out: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        assert_eq!(f32_out, vec![6.0, 15.0]);
    }

    #[test]
    fn max_min_reduce_bf16() {
        let v: Vec<half::bf16> = [1.0_f32, -5.0, 3.0, 2.0, 0.0, -1.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let input = CpuStorageBytes::from_slice(&v);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 2);
        max_reduce_bf16(&input, &mut out, &[2, 3], &[1]).expect("max bf16");
        let r: &[half::bf16] = out.as_slice().unwrap();
        let f32_out: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        assert_eq!(f32_out, vec![3.0, 2.0]);

        min_reduce_bf16(&input, &mut out, &[2, 3], &[1]).expect("min bf16");
        let r: &[half::bf16] = out.as_slice().unwrap();
        let f32_out: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        assert_eq!(f32_out, vec![-5.0, -1.0]);
    }

    #[test]
    fn mean_reduce_f16_divides_by_count() {
        let v: Vec<half::f16> = [2.0_f32, 4.0, 6.0, 8.0]
            .iter().map(|&x| half::f16::from_f32(x)).collect();
        let input = CpuStorageBytes::from_slice(&v);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 2);
        mean_reduce_f16(&input, &mut out, &[2, 2], &[1]).expect("mean f16");
        let r: &[half::f16] = out.as_slice().unwrap();
        let f32_out: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        assert_eq!(f32_out, vec![3.0, 7.0]);
    }

    #[test]
    fn matmul_bf16_2x3_times_3x2() {
        let lhs_v: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let rhs_v: Vec<half::bf16> = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]
            .iter().map(|&x| half::bf16::from_f32(x)).collect();
        let lhs = CpuStorageBytes::from_slice(&lhs_v);
        let rhs = CpuStorageBytes::from_slice(&rhs_v);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 2 * 2);
        matmul_bf16(&lhs, &rhs, &mut out, &[], &[], 2, 2, 3).expect("matmul bf16");
        let r: &[half::bf16] = out.as_slice().unwrap();
        let f32_out: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        // bf16 has ~3 decimal digits; the integer values 58, 64, 139,
        // 154 don't round-trip exactly through bf16 — accept ~1%
        // tolerance.
        let expected = [58.0_f32, 64.0, 139.0, 154.0];
        for (got, want) in f32_out.iter().zip(&expected) {
            assert!((got - want).abs() / want < 0.01,
                "got {got}, want {want}");
        }
    }

    #[test]
    fn matmul_f16_identity() {
        // Identity matmul on f16 — bytes round-trip exactly.
        let lhs_v: Vec<half::f16> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter().map(|&x| half::f16::from_f32(x)).collect();
        let rhs_v: Vec<half::f16> = [1.0_f32, 0.0, 0.0, 1.0]
            .iter().map(|&x| half::f16::from_f32(x)).collect();
        let lhs = CpuStorageBytes::from_slice(&lhs_v);
        let rhs = CpuStorageBytes::from_slice(&rhs_v);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 2 * 2);
        matmul_f16(&lhs, &rhs, &mut out, &[], &[], 2, 2, 2).expect("matmul f16");
        let r: &[half::f16] = out.as_slice().unwrap();
        let f32_out: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        assert_eq!(f32_out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_bf16_batched_2x_2x2_times_2x2() {
        let lhs_v: Vec<half::bf16> = [
            1.0_f32, 2.0, 3.0, 4.0,
            1.0, 0.0, 0.0, 1.0,
        ].iter().map(|&x| half::bf16::from_f32(x)).collect();
        let rhs_v: Vec<half::bf16> = [
            5.0_f32, 6.0, 7.0, 8.0,
            10.0, 20.0, 30.0, 40.0,
        ].iter().map(|&x| half::bf16::from_f32(x)).collect();
        let lhs = CpuStorageBytes::from_slice(&lhs_v);
        let rhs = CpuStorageBytes::from_slice(&rhs_v);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4 * 2);
        matmul_bf16(&lhs, &rhs, &mut out, &[2], &[2], 2, 2, 2).expect("matmul bf16");
        let r: &[half::bf16] = out.as_slice().unwrap();
        let f32_out: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        // Expected (in f32 reference): [19, 22, 43, 50, 10, 20, 30, 40]
        let expected = [19.0_f32, 22.0, 43.0, 50.0, 10.0, 20.0, 30.0, 40.0];
        for (got, want) in f32_out.iter().zip(&expected) {
            // Small powers-of-two and integers up to 50 round-trip
            // through bf16 exactly.
            assert!((got - want).abs() < 0.5, "got {got}, want {want}");
        }
    }

    #[test]
    fn add_bf16_round_trips_through_f32() {
        let a_vec = vec![half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)];
        let b_vec = vec![half::bf16::from_f32(10.0), half::bf16::from_f32(20.0)];
        let a = CpuStorageBytes::from_slice(&a_vec);
        let b = CpuStorageBytes::from_slice(&b_vec);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 2);
        add_bf16(&a, &b, &mut out).expect("add bf16");
        let result: &[half::bf16] = out.as_slice().unwrap();
        // bf16 has ~3 decimal digits; small integers round-trip exactly.
        assert_eq!(result[0].to_f32(), 11.0);
        assert_eq!(result[1].to_f32(), 22.0);
    }

    #[test]
    fn relu_bf16_clips_negatives() {
        let v: Vec<half::bf16> = [-1.0_f32, 0.0, 0.5, -3.5, 7.25]
            .iter()
            .map(|&x| half::bf16::from_f32(x))
            .collect();
        let input = CpuStorageBytes::from_slice(&v);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        relu_bf16(&input, &mut out).expect("relu bf16");
        let r: &[half::bf16] = out.as_slice().unwrap();
        let result_f32: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        assert_eq!(result_f32, vec![0.0, 0.0, 0.5, 0.0, 7.25]);
    }

    #[test]
    fn exp_log_bf16_round_trip_within_precision() {
        // bf16 has ~3 decimal digits of mantissa precision. Test
        // values chosen so exp + log doesn't drift much.
        let v: Vec<half::bf16> = [1.0_f32, 2.0, 3.0]
            .iter()
            .map(|&x| half::bf16::from_f32(x))
            .collect();
        let input = CpuStorageBytes::from_slice(&v);
        let mut intermediate = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        exp_bf16(&input, &mut intermediate).expect("exp");
        log_bf16(&intermediate, &mut out).expect("log");
        let r: &[half::bf16] = out.as_slice().unwrap();
        for (got, want) in r.iter().zip(&[1.0_f32, 2.0, 3.0]) {
            assert!(
                (got.to_f32() - want).abs() < 0.05,
                "bf16 exp+log lost too much: {} vs {}", got.to_f32(), want,
            );
        }
    }

    #[test]
    fn add_f16_round_trips_through_f32() {
        let a_vec = vec![half::f16::from_f32(1.0), half::f16::from_f32(2.5)];
        let b_vec = vec![half::f16::from_f32(0.5), half::f16::from_f32(-1.0)];
        let a = CpuStorageBytes::from_slice(&a_vec);
        let b = CpuStorageBytes::from_slice(&b_vec);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 2);
        add_f16(&a, &b, &mut out).expect("add f16");
        let result: &[half::f16] = out.as_slice().unwrap();
        assert_eq!(result[0].to_f32(), 1.5);
        assert_eq!(result[1].to_f32(), 1.5);
    }

    #[test]
    fn relu_f16_clips_negatives() {
        let v: Vec<half::f16> = [-2.0_f32, 0.0, 4.0, -0.5]
            .iter()
            .map(|&x| half::f16::from_f32(x))
            .collect();
        let input = CpuStorageBytes::from_slice(&v);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        relu_f16(&input, &mut out).expect("relu f16");
        let r: &[half::f16] = out.as_slice().unwrap();
        let result_f32: Vec<f32> = r.iter().map(|x| x.to_f32()).collect();
        assert_eq!(result_f32, vec![0.0, 0.0, 4.0, 0.0]);
    }

    #[test]
    fn sigmoid_bf16_at_zero_is_half() {
        let v = vec![half::bf16::from_f32(0.0_f32)];
        let input = CpuStorageBytes::from_slice(&v);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        sigmoid_bf16(&input, &mut out).expect("sigmoid bf16");
        let r: &[half::bf16] = out.as_slice().unwrap();
        assert!((r[0].to_f32() - 0.5).abs() < 0.01);
    }

    #[test]
    fn gelu_bf16_at_zero_is_zero() {
        let v = vec![half::bf16::from_f32(0.0_f32)];
        let input = CpuStorageBytes::from_slice(&v);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        gelu_bf16(&input, &mut out).expect("gelu bf16");
        let r: &[half::bf16] = out.as_slice().unwrap();
        assert!(r[0].to_f32().abs() < 0.001);
    }

    #[test]
    fn sum_reduce_f64_along_one_dim() {
        let input = CpuStorageBytes::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 8);
        sum_reduce_f64(&input, &mut out, &[2, 3], &[1]).expect("sum");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[6.0_f64, 15.0]);
    }

    #[test]
    fn max_min_reduce_f64_basic() {
        let input = CpuStorageBytes::from_slice(&[1.0_f64, -5.0, 3.0, 2.0, 0.0, -1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 8);
        max_reduce_f64(&input, &mut out, &[2, 3], &[1]).expect("max");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[3.0_f64, 2.0]);
        min_reduce_f64(&input, &mut out, &[2, 3], &[1]).expect("min");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[-5.0_f64, -1.0]);
    }

    #[test]
    fn mean_reduce_f64_divides_by_count() {
        let input = CpuStorageBytes::from_slice(&[2.0_f64, 4.0, 6.0, 8.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 8);
        mean_reduce_f64(&input, &mut out, &[2, 2], &[1]).expect("mean");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[3.0_f64, 7.0]);
    }

    #[test]
    fn matmul_f64_2x3_times_3x2() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = CpuStorageBytes::from_slice(&[7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 2 * 8);
        matmul_f64(&lhs, &rhs, &mut out, &[], &[], 2, 2, 3).expect("matmul");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[58.0_f64, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_f64_batched_2x_2x2_times_2x2() {
        let lhs = CpuStorageBytes::from_slice(&[
            1.0_f64, 2.0, 3.0, 4.0,
            1.0, 0.0, 0.0, 1.0,
        ]);
        let rhs = CpuStorageBytes::from_slice(&[
            5.0_f64, 6.0, 7.0, 8.0,
            10.0, 20.0, 30.0, 40.0,
        ]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4 * 8);
        matmul_f64(&lhs, &rhs, &mut out, &[2], &[2], 2, 2, 2).expect("matmul");
        assert_eq!(
            out.as_slice::<f64>().unwrap(),
            &[19.0_f64, 22.0, 43.0, 50.0, 10.0, 20.0, 30.0, 40.0]
        );
    }

    #[test]
    fn add_f64_basic() {
        let a = CpuStorageBytes::from_slice(&[1.0_f64, 2.0, 3.0]);
        let b = CpuStorageBytes::from_slice(&[10.0_f64, 20.0, 30.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 8);
        add_f64(&a, &b, &mut out).expect("add f64");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn relu_f64_clips_negatives() {
        let input = CpuStorageBytes::from_slice(&[-1.0_f64, 0.0, 0.5, -3.5, 7.25]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        relu_f64(&input, &mut out).expect("relu f64");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[0.0, 0.0, 0.5, 0.0, 7.25]);
    }

    #[test]
    fn unary_f64_round_trip_sample() {
        // Smoke-test several f64 unaries on one input.
        let input = CpuStorageBytes::from_slice(&[1.0_f64, -2.0, 4.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());

        neg_f64(&input, &mut out).expect("neg f64");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[-1.0, 2.0, -4.0]);

        sqr_f64(&input, &mut out).expect("sqr f64");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 4.0, 16.0]);

        let pos = CpuStorageBytes::from_slice(&[1.0_f64, 4.0, 9.0]);
        let mut out2 = CpuStorageBytes::from_zero_bytes(pos.len_bytes());
        sqrt_f64(&pos, &mut out2).expect("sqrt f64");
        assert_eq!(out2.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn maximum_minimum_f64_basic() {
        let lhs = CpuStorageBytes::from_slice(&[1.0_f64, 5.0, -3.0]);
        let rhs = CpuStorageBytes::from_slice(&[2.0_f64, 1.0, -1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(lhs.len_bytes());
        maximum_f64(&lhs, &rhs, &mut out).expect("max f64");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 5.0, -1.0]);
        minimum_f64(&lhs, &rhs, &mut out).expect("min f64");
        assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 1.0, -3.0]);
    }

    #[test]
    fn gelu_f64_at_known_points() {
        let input = CpuStorageBytes::from_slice(&[0.0_f64, 1.0, -1.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        gelu_f64(&input, &mut out).expect("gelu f64");
        let r: &[f64] = out.as_slice().unwrap();
        // gelu(0) = 0
        assert!(r[0].abs() < 1e-12);
        // gelu(1) ≈ 0.8412 (tanh approx); f64 should be ~12 digits accurate
        assert!((r[1] - 0.841_192).abs() < 1e-3);
        assert!((r[2] - (-0.158_808)).abs() < 1e-3);
    }

    #[test]
    fn argmax_dim_f32_basic() {
        // input [2, 3]: row 0 = [1, 5, 2], row 1 = [9, 0, 4]
        // argmax along dim=1 → output [2] with [1, 0]
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 5.0, 2.0, 9.0, 0.0, 4.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4);
        argmax_dim_f32(&input, &mut out, &[2, 3], 1).expect("argmax");
        assert_eq!(out.as_slice::<u32>().unwrap(), &[1u32, 0]);
    }

    #[test]
    fn argmin_dim_f32_basic() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 5.0, 2.0, 9.0, 0.0, 4.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4);
        argmin_dim_f32(&input, &mut out, &[2, 3], 1).expect("argmin");
        // Row 0 min at index 0 (= 1.0); row 1 min at index 1 (= 0.0)
        assert_eq!(out.as_slice::<u32>().unwrap(), &[0u32, 1]);
    }

    #[test]
    fn argmax_dim_f32_outer_dim() {
        // input [3, 2]:
        //   [[1, 4],
        //    [3, 2],
        //    [0, 9]]
        // argmax along dim=0 → output [2]
        // col 0: max at index 1 (= 3); col 1: max at index 2 (= 9)
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 4.0, 3.0, 2.0, 0.0, 9.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4);
        argmax_dim_f32(&input, &mut out, &[3, 2], 0).expect("argmax");
        assert_eq!(out.as_slice::<u32>().unwrap(), &[1u32, 2]);
    }

    #[test]
    fn argmax_dim_f32_first_index_on_tie() {
        // All equal — argmax must return index 0 (first occurrence).
        let input = CpuStorageBytes::from_slice(&[5.0_f32, 5.0, 5.0, 5.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4);
        argmax_dim_f32(&input, &mut out, &[4], 0).expect("argmax");
        assert_eq!(out.as_slice::<u32>().unwrap(), &[0u32]);
    }

    #[test]
    fn argmax_dim_f32_rejects_zero_dim() {
        let input = CpuStorageBytes::from_zero_bytes(0);
        let mut out = CpuStorageBytes::from_zero_bytes(0);
        let r = argmax_dim_f32(&input, &mut out, &[0], 0);
        assert!(r.is_err());
    }

    #[test]
    fn index_add_f32_simple_accumulate() {
        // base [3] = [10, 20, 30]; indices [2] = [0, 0]; src [2] = [1, 2]
        // → out = [10 + 1 + 2, 20, 30] = [13, 20, 30]
        let base = CpuStorageBytes::from_slice(&[10.0_f32, 20.0, 30.0]);
        let indices = CpuStorageBytes::from_slice(&[0u32, 0]);
        let src = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 4);
        index_add_f32(&base, &indices, &src, &mut out, 1, 3, 2, 1).expect("index_add");
        assert_eq!(out.as_slice::<f32>().unwrap(), &[13.0, 20.0, 30.0]);
    }

    #[test]
    fn index_add_f32_along_inner_dim() {
        // base [2, 4]; indices [3] = [1, 3, 1]; src [2, 3].
        // For each row, accumulate src cols at base cols indices.
        let base = CpuStorageBytes::from_slice(&[
            10.0_f32, 20.0, 30.0, 40.0,
            50.0, 60.0, 70.0, 80.0,
        ]);
        let indices = CpuStorageBytes::from_slice(&[1u32, 3, 1]);
        let src = CpuStorageBytes::from_slice(&[
            1.0_f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        let mut out = CpuStorageBytes::from_zero_bytes(2 * 4 * 4);
        // outer=2, base_dim=4, n_indices=3, inner=1
        index_add_f32(&base, &indices, &src, &mut out, 2, 4, 3, 1).expect("index_add");
        // row 0: base [10, 20, 30, 40]. src [1, 2, 3] at indices [1, 3, 1]:
        //   col 0: 10
        //   col 1: 20 + 1 + 3 = 24
        //   col 2: 30
        //   col 3: 40 + 2 = 42
        // row 1: base [50, 60, 70, 80]. src [4, 5, 6] at indices [1, 3, 1]:
        //   col 0: 50
        //   col 1: 60 + 4 + 6 = 70
        //   col 2: 70
        //   col 3: 80 + 5 = 85
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[10.0, 24.0, 30.0, 42.0, 50.0, 70.0, 70.0, 85.0]
        );
    }

    #[test]
    fn index_add_f32_rejects_out_of_bounds_index() {
        let base = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]);
        let indices = CpuStorageBytes::from_slice(&[5u32]);
        let src = CpuStorageBytes::from_slice(&[10.0_f32]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 4);
        let r = index_add_f32(&base, &indices, &src, &mut out, 1, 3, 1, 1);
        assert!(r.is_err());
    }

    #[test]
    fn scatter_add_f32_along_outer_dim() {
        // base [3, 2] = zeros; indices [2, 2]:
        //   [[0, 1],
        //    [2, 0]]
        // src [2, 2] = [[1, 2], [3, 4]]
        // dim=0:
        //   src[0, 0]=1 → out[indices[0,0]=0, 0] += 1 → out[0, 0] += 1
        //   src[0, 1]=2 → out[indices[0,1]=1, 1] += 2 → out[1, 1] += 2
        //   src[1, 0]=3 → out[indices[1,0]=2, 0] += 3 → out[2, 0] += 3
        //   src[1, 1]=4 → out[indices[1,1]=0, 1] += 4 → out[0, 1] += 4
        // → [[1, 4], [0, 2], [3, 0]]
        let base = CpuStorageBytes::from_slice(&[0.0_f32; 6]);
        let indices = CpuStorageBytes::from_slice(&[0u32, 1, 2, 0]);
        let src = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 2 * 4);
        scatter_add_f32(&base, &indices, &src, &mut out, &[3, 2], &[2, 2], 0)
            .expect("scatter_add");
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[1.0, 4.0, 0.0, 2.0, 3.0, 0.0]
        );
    }

    #[test]
    fn scatter_add_f32_starts_from_base() {
        // Same as above but base is already nonzero — verifies that
        // out copies base before accumulating.
        let base = CpuStorageBytes::from_slice(&[100.0_f32, 200.0, 300.0]);
        let indices = CpuStorageBytes::from_slice(&[0u32, 0, 2]);
        let src = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(3 * 4);
        scatter_add_f32(&base, &indices, &src, &mut out, &[3], &[3], 0).expect("scatter_add");
        // out[0] = 100 + 1 + 2 = 103
        // out[1] = 200 (untouched)
        // out[2] = 300 + 3 = 303
        assert_eq!(out.as_slice::<f32>().unwrap(), &[103.0, 200.0, 303.0]);
    }

    #[test]
    fn scatter_add_f32_rejects_shape_mismatch() {
        let base = CpuStorageBytes::from_slice(&[0.0_f32; 4]);
        let indices = CpuStorageBytes::from_slice(&[0u32, 1]);
        let src = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(4 * 4);
        // base shape claims [2, 2] but src claims [3] — rank mismatch.
        let r = scatter_add_f32(&base, &indices, &src, &mut out, &[2, 2], &[3], 0);
        assert!(r.is_err());
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
