//! Byte-level CUDA kernels — Phase 7.5 unified-storage migration.
//!
//! These kernels operate on `CudaStorageBytes` (raw `DeviceBuffer<u8>`)
//! rather than the dtype-tagged legacy `CudaStorage` enum. Dispatch
//! to the right CUDA function happens via wrappers in
//! `fuel-storage::dispatch::register_cuda_kernels`; the typed kernel
//! functions in `fuel-cuda-kernels` are launched by passing
//! `&DeviceBuffer<u8>` as the kernel arg — at the CUDA driver level
//! the typed pointer (`f32*`, `f64*`, etc.) and the byte pointer have
//! the same value, and the kernel's compiled code interprets the
//! bytes per its declared type.
//!
//! The kernels in `fuel-cuda-kernels` (e.g. `badd_f32`) accept the
//! signature `(elem_count, ndims, dims_strides_or_null, lhs, rhs,
//! out)`. A null `dims_strides_or_null` selects the kernel's
//! contiguous fast path; the unified executor's auto-Contiguize pass
//! guarantees inputs are contiguous before kernel call, so the
//! wrappers always pass null.

use std::sync::Arc;

use fuel_core_types::{DType, Layout, Result};
use fuel_cuda_kernels as kernels;

use crate::builder_arg as barg;
use crate::byte_storage::CudaStorageBytes;
use crate::device::LaunchConfig;
use crate::error::WrapErr;
use crate::storage::SlicePtrOrNull;

/// Phase 7.5 first CUDA kernel through the unified path.
/// Element-wise add of two F32 `CudaStorageBytes`. Layouts describe how
/// the kernel walks the input bytes: when both layouts are contiguous +
/// zero-offset, the contiguous fast path is used; otherwise the kernel
/// walks input strides explicitly and the result is shaped by
/// `lhs_layout` (which equals `rhs_layout.shape()` when both come from
/// the executor — Op::BroadcastTo normalizes shapes upstream). Output
/// is freshly allocated on the same device as `lhs`; caller is
/// responsible for storing it where the unified executor expects it.
pub fn add_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, lhs_layout, rhs_layout, "badd_f32")
}

/// Element-wise subtraction (lhs - rhs) of two F32 `CudaStorageBytes`.
/// Same shape as [`add_elementwise_f32`]; only the launched kernel
/// name differs.
pub fn sub_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, lhs_layout, rhs_layout, "bsub_f32")
}

/// Element-wise multiplication (lhs * rhs) of two F32 `CudaStorageBytes`.
pub fn mul_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, lhs_layout, rhs_layout, "bmul_f32")
}

/// Element-wise division (lhs / rhs) of two F32 `CudaStorageBytes`.
pub fn div_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, lhs_layout, rhs_layout, "bdiv_f32")
}

/// Element-wise maximum (max(lhs, rhs)) of two F32 `CudaStorageBytes`.
pub fn maximum_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, lhs_layout, rhs_layout, "bmaximum_f32")
}

/// Element-wise minimum (min(lhs, rhs)) of two F32 `CudaStorageBytes`.
pub fn minimum_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, lhs_layout, rhs_layout, "bminimum_f32")
}

/// Element-wise ReLU (max(x, 0)) of one F32 `CudaStorageBytes`.
/// First unary op through the unified binding table; extracts the
/// shared [`unary_elementwise_f32`] helper for the rest of the F32
/// unary fanout to delegate to.
pub fn relu_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "urelu_f32")
}

/// Element-wise negation (-x) of one F32 `CudaStorageBytes`.
pub fn neg_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "uneg_f32")
}

/// Element-wise square (x * x) of one F32 `CudaStorageBytes`.
pub fn sqr_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "usqr_f32")
}

/// Element-wise square root (sqrt(x)) of one F32 `CudaStorageBytes`.
pub fn sqrt_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "usqrt_f32")
}

/// Element-wise reciprocal (1/x) of one F32 `CudaStorageBytes`.
pub fn recip_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "urecip_f32")
}

/// Element-wise absolute value (|x|) of one F32 `CudaStorageBytes`.
pub fn abs_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "uabs_f32")
}

/// Element-wise hyperbolic tangent (tanh(x)) of one F32 `CudaStorageBytes`.
pub fn tanh_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "utanh_f32")
}

/// Element-wise exp(x) of one F32 `CudaStorageBytes`.
pub fn exp_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "uexp_f32")
}

/// Element-wise natural log (ln(x)) of one F32 `CudaStorageBytes`.
pub fn log_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "ulog_f32")
}

/// Element-wise sin(x) of one F32 `CudaStorageBytes`.
pub fn sin_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "usin_f32")
}

/// Element-wise cos(x) of one F32 `CudaStorageBytes`.
pub fn cos_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "ucos_f32")
}

/// Element-wise sigmoid (1 / (1 + exp(-x))) of one F32 `CudaStorageBytes`.
pub fn sigmoid_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "usigmoid_f32")
}

/// Element-wise SiLU (x * sigmoid(x)) of one F32 `CudaStorageBytes`.
pub fn silu_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "usilu_f32")
}

/// Element-wise GELU (tanh approximation) of one F32 `CudaStorageBytes`.
/// Maps to `ugelu_f32` (the kernel's `gelu_fwd`); the erf variant is
/// `ugelu_erf_f32` and is exposed by `OpKind::GeluErfElementwise` if/when
/// it's added to the binding table.
pub fn gelu_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "ugelu_f32")
}

/// Element-wise Heaviside step (1.0 if x > 0 else 0.0) of one F32
/// `CudaStorageBytes`. Maps to `ustep_f32`, which was added to
/// `fuel-cuda-kernels::UNARY` (via `unary.cu`) in the same commit
/// that introduced this wrapper — the rest of the legacy unary
/// kernels predated it.
pub fn step_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "ustep_f32")
}

/// Sum-reduce one F32 `CudaStorageBytes` along the dims listed in
/// `reduce_dims`. First reduction op through the unified binding
/// table; extracts the shared [`reduce_f32`] helper for Max/Min/Mean
/// to delegate to. Output is freshly allocated, sized
/// `prod(non-reduced dims) * sizeof(f32)`.
pub fn sum_reduce_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    reduce_dims: &[usize],
) -> Result<CudaStorageBytes> {
    reduce_f32(src, input_layout, reduce_dims, "fast_sum_f32")
}

/// Max-reduce one F32 `CudaStorageBytes` along the dims listed in
/// `reduce_dims`. Same shape contract as [`sum_reduce_f32`]; only
/// the launched kernel name differs.
pub fn max_reduce_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    reduce_dims: &[usize],
) -> Result<CudaStorageBytes> {
    reduce_f32(src, input_layout, reduce_dims, "fast_max_f32")
}

/// Min-reduce one F32 `CudaStorageBytes` along the dims listed in
/// `reduce_dims`. Same shape contract as [`sum_reduce_f32`].
pub fn min_reduce_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    reduce_dims: &[usize],
) -> Result<CudaStorageBytes> {
    reduce_f32(src, input_layout, reduce_dims, "fast_min_f32")
}

/// Mean-reduce one F32 `CudaStorageBytes` along the dims listed in
/// `reduce_dims`. Composed: launch `fast_sum_f32` (via the shared
/// reduce helper), then launch `affine_f32` with `mul = 1/divisor`
/// and `add = 0` to scale the sum into the mean. Mirrors the CPU
/// `mean_reduce_f32` (sum then in-place scale). The two-launch
/// shape avoids needing a dedicated `fast_mean_f32` PTX kernel; if
/// profiling later shows the second launch matters, a fused kernel
/// is the natural follow-on.
pub fn mean_reduce_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    reduce_dims: &[usize],
) -> Result<CudaStorageBytes> {
    let sum = reduce_f32(src, input_layout, reduce_dims, "fast_sum_f32")?;
    let src_dims = input_layout.shape().dims();
    let divisor: usize = reduce_dims.iter().map(|&d| src_dims[d]).product();
    if divisor == 0 {
        return Err(fuel_core_types::Error::Msg(
            "mean_reduce_f32: divisor zero (reduced dim has size 0)".to_string(),
        )
        .bt());
    }
    let inv = 1.0_f32 / divisor as f32;
    affine_f32(&sum, inv, 0.0)
}

/// Compute the list of input axes that need reducing to align with a
/// broadcast-compatible target shape. Mirrors the CPU
/// `align_reduce_to` validation: target left-pads with 1s to match
/// input rank; an axis is reduced when the padded target dim is 1
/// and the input dim is greater than 1.
fn reduce_dims_from_shapes(
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<usize>> {
    if output_shape.len() > input_shape.len() {
        return Err(fuel_core_types::Error::Msg(format!(
            "reduce_to: output rank {} exceeds input rank {}",
            output_shape.len(), input_shape.len(),
        )).bt());
    }
    let pad = input_shape.len() - output_shape.len();
    let mut padded = vec![1_usize; pad];
    padded.extend_from_slice(output_shape);
    let mut reduce_dims: Vec<usize> = Vec::new();
    for (axis, (&s, &t)) in input_shape.iter().zip(padded.iter()).enumerate() {
        if t == s {
            // Pass-through axis.
        } else if t == 1 {
            // Axis being reduced. Only push when input dim > 1; if
            // input dim is also 1 the reduction is a no-op and the
            // existing reduce_f32 kernel handles it correctly via the
            // empty-stride single-element path either way.
            if s > 1 {
                reduce_dims.push(axis);
            }
        } else {
            return Err(fuel_core_types::Error::Msg(format!(
                "reduce_to: axis {axis} target {t} must be 1 or input {s}",
            )).bt());
        }
    }
    Ok(reduce_dims)
}

/// Sum-reduce a CUDA F32 tensor to a smaller broadcast-compatible
/// shape. Maps the broadcast-aligned target shape to a list of
/// reduce dims and dispatches through the existing `fast_sum_f32`
/// kernel. The output's byte count matches what the executor
/// pre-allocates for `output_shape` (since the reduced byte count is
/// determined entirely by which dims are reduced, regardless of
/// whether they're dropped or kept as size-1).
///
/// Mirrors the CPU `reduce_sum_to_f32` byte kernel; on CUDA the
/// keepdim form is free because the result bytes are the same as
/// dropping the reduced dim — only the metadata shape differs and
/// is set by the wrapper's pre-allocated output.
pub fn reduce_sum_to_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<CudaStorageBytes> {
    let reduce_dims = reduce_dims_from_shapes(input_shape, output_shape)?;
    // Empty reduce_dims is an "identity reduce_to" — input_shape ==
    // padded(output_shape) on every axis. The reduce kernel handles
    // it correctly (each output element sums over one input element)
    // with the cost of one extra kernel launch; not a hot path so
    // not worth a special-case.
    reduce_f32(src, input_layout, &reduce_dims, "fast_sum_f32")
}

/// Max-reduce a CUDA F32 tensor to a smaller broadcast-compatible
/// shape — the max-symmetric counterpart of [`reduce_sum_to_f32`].
pub fn reduce_max_to_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<CudaStorageBytes> {
    let reduce_dims = reduce_dims_from_shapes(input_shape, output_shape)?;
    reduce_f32(src, input_layout, &reduce_dims, "fast_max_f32")
}

/// Index-select an F32 `CudaStorageBytes` along one dim using a U32
/// index tensor. Mirrors the legacy `IndexSelect` map (storage.rs:441)
/// but on byte buffers, and parameterized by the four counts the
/// executor pre-computes into `OpParams::IndexSelect`.
///
/// Shape contract: source has shape `[..outer_count, source_dim_size,
/// ..inner_count]` and the rank-1 `ids` tensor has `n_indices`
/// elements; the output has shape `[..outer_count, n_indices,
/// ..inner_count]` (rank preserved). `outer_count` and `inner_count`
/// are the products of dims before/after the selected axis.
///
/// The launched kernel is `is_u32_f32` (from
/// `fuel-cuda-kernels::INDEXING`); both `src` and `ids` must be
/// contiguous (auto-Contiguize guarantees this on the unified path).
/// The kernel's `info` parameter passes `[n_indices, 1]` so its
/// internal `is_contiguous` check selects the contiguous fast path —
/// the strided fallback would only fire if a non-contiguous src
/// reached us, which the executor's auto-Contiguize prevents.
pub fn index_select_f32(
    src: &CudaStorageBytes,
    ids: &CudaStorageBytes,
    outer_count: usize,
    source_dim_size: usize,
    n_indices: usize,
    inner_count: usize,
) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    let expected_src_bytes = outer_count
        .saturating_mul(source_dim_size)
        .saturating_mul(inner_count)
        .saturating_mul(elem);
    if src.len_bytes() != expected_src_bytes {
        return Err(fuel_core_types::Error::Msg(format!(
            "index_select_f32: src bytes {} disagrees with \
             outer_count*source_dim_size*inner_count*4 = {}",
            src.len_bytes(),
            expected_src_bytes,
        ))
        .bt());
    }
    let expected_ids_bytes = n_indices.saturating_mul(std::mem::size_of::<u32>());
    if ids.len_bytes() != expected_ids_bytes {
        return Err(fuel_core_types::Error::Msg(format!(
            "index_select_f32: ids bytes {} disagrees with n_indices*4 = {}",
            ids.len_bytes(),
            expected_ids_bytes,
        ))
        .bt());
    }

    let device = src.device().clone();
    let dst_el = outer_count
        .saturating_mul(n_indices)
        .saturating_mul(inner_count);
    let dst_bytes = dst_el.saturating_mul(elem);
    if dst_el == 0 {
        return CudaStorageBytes::alloc(&device, dst_bytes);
    }

    let mut out = device.alloc_zeros::<u8>(dst_bytes)?;
    let cfg = LaunchConfig::for_num_elems(dst_el as u32);
    let func = device.get_or_load_func("is_u32_f32", &kernels::INDEXING)?;

    // info = [dims | strides] for a 1D contiguous "view" of the source —
    // is_contiguous(num_dims=1, dims=[n_indices], strides=[1]) returns
    // true so the kernel takes its fast path. The kernel's strided arm
    // is dead code on the unified path (auto-Contiguize); these values
    // exist purely to satisfy the parameter shape.
    let info = device.clone_htod(&[n_indices, 1_usize])?;
    let mut builder = func.builder();
    barg!(builder, dst_el);
    barg!(builder, 1_usize); // num_dims (matches the 1D info above)
    builder.arg(&info);
    builder.arg(ids.buffer());
    builder.arg(src.buffer());
    builder.arg(&mut out);
    barg!(builder, outer_count);     // left_size
    barg!(builder, source_dim_size); // src_dim_size
    barg!(builder, n_indices);       // ids_dim_size
    barg!(builder, inner_count);     // right_size
    // SAFETY: kernel signature matches the args above — same shape as
    // the legacy `IndexSelect::f`, just on byte buffers.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, dst_bytes))
}

/// Shared launch path for F32 argmax/argmin along one dim.
/// Mirrors [`reduce_f32`] but the output is `dst_el * sizeof(u32)`
/// bytes (not f32) and the kernel writes `uint32_t *dst`. The reduce
/// dim is reordered to last and the existing parallel argmax/argmin
/// kernels (which compute `idx % dims[last]` for the per-block
/// index) work as-is — they were written for the "reduce over last
/// dim" shape, and the dim-reorder normalizes any dim to that shape.
fn arg_extremum_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    dim: usize,
    kernel_name: &'static str,
) -> Result<CudaStorageBytes> {
    let f32_size = std::mem::size_of::<f32>();
    let u32_size = std::mem::size_of::<u32>();
    if src.len_bytes() % f32_size != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: src.len_bytes={} not a multiple of f32 size",
            src.len_bytes(),
        ))
        .bt());
    }
    let src_dims = input_layout.shape().dims();
    let src_stride = input_layout.stride_unsigned();
    let src_el: usize = src_dims.iter().product();
    if src_el * f32_size != src.len_bytes() {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: src element count {} (from layout shape {:?}) \
             disagrees with byte length {} / sizeof(f32)",
            src_el,
            src_dims,
            src.len_bytes(),
        ))
        .bt());
    }
    if dim >= src_dims.len() {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: dim {dim} out of range for rank {}",
            src_dims.len(),
        ))
        .bt());
    }
    if src_dims[dim] == 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: dim {dim} has size 0 — argmax/argmin undefined",
        ))
        .bt());
    }

    // Reorder so the reduce dim is last (mirrors reduce_f32). The
    // fast_argmax/fast_argmin kernels expect the contiguous reduce-
    // axis to be last and compute the per-block index as
    // `idx % dims[num_dims - 1]`.
    let mut dims: Vec<usize> = Vec::with_capacity(src_dims.len());
    let mut stride: Vec<usize> = Vec::with_capacity(src_dims.len());
    let mut dst_el: usize = 1;
    for (dim_idx, &d) in src_dims.iter().enumerate() {
        if dim_idx != dim {
            dst_el *= d;
            dims.push(d);
            stride.push(src_stride[dim_idx]);
        }
    }
    dims.push(src_dims[dim]);
    stride.push(src_stride[dim]);

    let dst_bytes = dst_el * u32_size;
    let device = src.device().clone();
    if src_el == 0 || dst_el == 0 {
        return CudaStorageBytes::alloc(&device, dst_bytes);
    }
    let el_to_sum_per_block = src_el / dst_el;
    let block_dim = usize::min(1024, el_to_sum_per_block).next_power_of_two();
    let cfg = LaunchConfig {
        grid_dim: (dst_el as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut out = device.alloc_zeros::<u8>(dst_bytes)?;
    let ds = device.clone_htod(&[dims.as_slice(), stride.as_slice()].concat())?;
    let func = device.get_or_load_func(kernel_name, &kernels::REDUCE)?;
    let mut builder = func.builder();
    barg!(builder, src_el);
    barg!(builder, el_to_sum_per_block);
    barg!(builder, src_dims.len());
    builder.arg(&ds);
    builder.arg(src.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above — same shape as
    // reduce_f32 but with `uint32_t *dst` output (FAST_OP macro,
    // ARGMIN/ARGMAX entries).
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, dst_bytes))
}

/// Argmax over one dim of an F32 `CudaStorageBytes`. Output is U32
/// indices into the reduce dim, with the reduce dim removed from the
/// output shape (same shape contract as the CPU `argmax_dim_f32`).
/// Launches the existing `fast_argmax_f32` parallel-reduction kernel
/// after reordering the reduce axis to last; tie-breaking is "first
/// index encountered" within the per-block stride pattern.
pub fn argmax_dim_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    dim: usize,
) -> Result<CudaStorageBytes> {
    arg_extremum_f32(src, input_layout, dim, "fast_argmax_f32")
}

/// Argmin over one dim — sister of [`argmax_dim_f32`]. Same shape
/// contract; only the launched kernel name differs.
pub fn argmin_dim_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    dim: usize,
) -> Result<CudaStorageBytes> {
    arg_extremum_f32(src, input_layout, dim, "fast_argmin_f32")
}

/// Concatenate N F32 inputs along one dim. Each input has shape
/// `[..outer_count, input_dim_sizes[i], ..inner_count]`; output has
/// shape `[..outer_count, sum(input_dim_sizes), ..inner_count]`.
/// Mirrors the CPU `concat_cpu` byte kernel.
///
/// Implementation: launches `concat_f32` (from
/// `fuel-cuda-kernels::INDEXING`) once per input. Each launch writes
/// the input's contribution to the right slice of the pre-allocated
/// output, advancing `input_idx_offset` by `input_dim_sizes[i]`
/// between launches. The kernel walks the input's element count and
/// scatters into the right output slot per element.
pub fn concat_f32(
    inputs: &[&CudaStorageBytes],
    outer_count: usize,
    input_dim_sizes: &[usize],
    inner_count: usize,
) -> Result<CudaStorageBytes> {
    if inputs.is_empty() {
        return Err(fuel_core_types::Error::Msg(
            "concat_f32: at least one input required".to_string(),
        )
        .bt());
    }
    if inputs.len() != input_dim_sizes.len() {
        return Err(fuel_core_types::Error::Msg(format!(
            "concat_f32: inputs count ({}) != input_dim_sizes len ({})",
            inputs.len(),
            input_dim_sizes.len(),
        ))
        .bt());
    }
    let elem = std::mem::size_of::<f32>();
    let total_dim: usize = input_dim_sizes.iter().sum();
    let out_elem = outer_count
        .saturating_mul(total_dim)
        .saturating_mul(inner_count);
    let dst_bytes = out_elem.saturating_mul(elem);
    let device = inputs[0].device().clone();

    // Validate per-input byte counts match their declared dim_size.
    for (i, input) in inputs.iter().enumerate() {
        let need = outer_count
            .saturating_mul(input_dim_sizes[i])
            .saturating_mul(inner_count)
            .saturating_mul(elem);
        if input.len_bytes() != need {
            return Err(fuel_core_types::Error::Msg(format!(
                "concat_f32: input[{i}] bytes={} doesn't match outer={outer_count} × dim={} × inner={inner_count} × 4",
                input.len_bytes(),
                input_dim_sizes[i],
            ))
            .bt());
        }
    }

    if out_elem == 0 {
        return CudaStorageBytes::alloc(&device, dst_bytes);
    }

    let mut out = device.alloc_zeros::<u8>(dst_bytes)?;
    let func = device.get_or_load_func("concat_f32", &kernels::INDEXING)?;

    let mut dim_offset: usize = 0;
    for (i, input) in inputs.iter().enumerate() {
        let input_dim_size = input_dim_sizes[i];
        let in_elem = outer_count
            .saturating_mul(input_dim_size)
            .saturating_mul(inner_count);
        if in_elem == 0 {
            dim_offset += input_dim_size;
            continue;
        }
        let cfg = LaunchConfig::for_num_elems(in_elem as u32);
        let mut builder = func.builder();
        barg!(builder, in_elem);          // numel for this input
        barg!(builder, outer_count);
        barg!(builder, inner_count);
        barg!(builder, total_dim);
        barg!(builder, dim_offset);       // input_idx_offset
        barg!(builder, input_dim_size);
        builder.arg(input.buffer());
        builder.arg(&mut out);
        // SAFETY: kernel signature matches the args above (CONCAT_OP
        // macro in indexing.cu).
        unsafe { builder.launch(cfg) }.w()?;
        dim_offset += input_dim_size;
    }
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, dst_bytes))
}

/// N-dimensional gather along `dim` for F32 source + U32 indices, on
/// byte-shaped CUDA storage. Mirrors the legacy `Gather` map
/// (storage.rs:494) and the CPU `gather_cpu` byte kernel: source and
/// indices share rank, output shape equals indices shape, and only
/// `source_shape[dim]` may differ from `output_shape[dim]`.
///
/// The launched kernel is `gather_u32_f32` (from
/// `fuel-cuda-kernels::INDEXING`); it has no `info`/`num_dims`
/// parameters (unlike `is_u32_f32`) — it walks the linear output
/// index directly through `(left_size, src_dim_size, ids_dim_size,
/// right_size)`. Both `src` and `ids` must be contiguous; auto-
/// Contiguize on the unified path guarantees this.
pub fn gather_f32(
    src: &CudaStorageBytes,
    ids: &CudaStorageBytes,
    source_shape: &[usize],
    output_shape: &[usize],
    dim: usize,
) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    if source_shape.len() != output_shape.len() {
        return Err(fuel_core_types::Error::Msg(format!(
            "gather_f32: source rank ({}) != output rank ({})",
            source_shape.len(),
            output_shape.len(),
        ))
        .bt());
    }
    let rank = source_shape.len();
    if dim >= rank {
        return Err(fuel_core_types::Error::Msg(format!(
            "gather_f32: dim {dim} out of range for rank {rank}",
        ))
        .bt());
    }
    for d in 0..rank {
        if d != dim && source_shape[d] != output_shape[d] {
            return Err(fuel_core_types::Error::Msg(format!(
                "gather_f32: source and output disagree at dim {d} \
                 (source={}, output={}); only `dim`={dim} may differ",
                source_shape[d],
                output_shape[d],
            ))
            .bt());
        }
    }
    let source_total: usize = source_shape.iter().product();
    let output_total: usize = output_shape.iter().product();
    if src.len_bytes() != source_total.saturating_mul(elem) {
        return Err(fuel_core_types::Error::Msg(format!(
            "gather_f32: source bytes={} doesn't match shape {source_shape:?} (f32)",
            src.len_bytes(),
        ))
        .bt());
    }
    if ids.len_bytes() != output_total.saturating_mul(std::mem::size_of::<u32>()) {
        return Err(fuel_core_types::Error::Msg(format!(
            "gather_f32: ids bytes={} doesn't match output shape {output_shape:?} (u32)",
            ids.len_bytes(),
        ))
        .bt());
    }

    let left_size: usize = source_shape[..dim].iter().product();
    let src_dim_size = source_shape[dim];
    let ids_dim_size = output_shape[dim];
    let right_size: usize = source_shape[dim + 1..].iter().product();

    let device = src.device().clone();
    let dst_bytes = output_total.saturating_mul(elem);
    if output_total == 0 {
        return CudaStorageBytes::alloc(&device, dst_bytes);
    }

    let mut out = device.alloc_zeros::<u8>(dst_bytes)?;
    let cfg = LaunchConfig::for_num_elems(output_total as u32);
    let func = device.get_or_load_func("gather_u32_f32", &kernels::INDEXING)?;

    let mut builder = func.builder();
    barg!(builder, output_total); // numel
    builder.arg(ids.buffer());    // ids
    builder.arg(src.buffer());    // inp
    builder.arg(&mut out);        // out
    barg!(builder, left_size);
    barg!(builder, src_dim_size);
    barg!(builder, ids_dim_size);
    barg!(builder, right_size);
    // SAFETY: kernel signature matches the args above — same shape as
    // the legacy `Gather::f`, just on byte buffers.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, dst_bytes))
}

/// Batched row-major F32 matmul through cuBLAS, on byte-shaped inputs.
/// Shape contract per `OpParams::Matmul`: `lhs [..lhs_batch.., m, k] @
/// rhs [..rhs_batch.., k, n] → out [..lhs_batch.., m, n]`. Inputs are
/// guaranteed contiguous by the executor's auto-Contiguize pass, so
/// per-batch element strides are `m*k`, `k*n`, `m*n` respectively.
///
/// The cuBLAS row-major-via-col-major trick: pass our `rhs` as cuBLAS
/// `A` and our `lhs` as cuBLAS `B`, swap `m` and `n` in the call, and
/// use no transposes. cuBLAS computes `C^T = B^T × A^T` in col-major
/// terms, which equals `A_row × B_row` viewed back as row-major. See
/// the legacy `matmul_via_cublas` (`storage.rs::CudaStorage::matmul`)
/// — same mechanic.
///
/// Two paths:
/// - **Equal-batch fast path** (all per-axis dims match): single
///   `gemm_strided_batched_ex` call with `batch_count = lhs_batch_count`.
/// - **GQA per-batch loop** (per-axis `lhs_dim = n_rep_axis * rhs_dim`):
///   one `gemm_ex` call per lhs batch slot, with the rhs slot index
///   computed via the per-axis `n_rep` mapping (mirrors CPU's
///   `matmul_f32`). Slow but correct for any GQA pattern; if profiling
///   shows it matters, the natural follow-on is per-rhs-slot grouping
///   for innermost-axis-only n_rep (the GQA-attention common case).
pub fn matmul_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaStorageBytes> {
    if lhs_batch_dims.len() != rhs_batch_dims.len() {
        return Err(fuel_core_types::Error::Msg(format!(
            "matmul_f32: batch ranks must match (lhs={}, rhs={}); fuel-graph's \
             auto-broadcast equalizes them at graph construction time",
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
            return Err(fuel_core_types::Error::Msg(format!(
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
        return Err(fuel_core_types::Error::Msg(format!(
            "matmul_f32: lhs bytes={} doesn't match shape {:?} + [{m}, {k}] (f32)",
            lhs.len_bytes(),
            lhs_batch_dims,
        ))
        .bt());
    }
    if rhs.len_bytes() != need_rhs {
        return Err(fuel_core_types::Error::Msg(format!(
            "matmul_f32: rhs bytes={} doesn't match shape {:?} + [{k}, {n}] (f32)",
            rhs.len_bytes(),
            rhs_batch_dims,
        ))
        .bt());
    }
    let device = lhs.device().clone();
    if rhs.device().id() != device.id() {
        return Err(fuel_core_types::Error::Msg(
            "matmul_f32: lhs and rhs are on different CUDA devices; cross-device \
             matmul is the caller's responsibility (insert Op::Move first)"
                .to_string(),
        )
        .bt());
    }
    if need_out == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out = device.alloc_zeros::<u8>(need_out)?;

    use baracuda_cublas::{cublasComputeType_t, cudaDataType_t, Op};
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let alpha_ptr = (&alpha) as *const f32 as *const std::ffi::c_void;
    let beta_ptr = (&beta) as *const f32 as *const std::ffi::c_void;
    // cuBLAS A = our rhs (logical [k, n] row-major, viewed col-major
    // as [n, k]). lda = n. cuBLAS B = our lhs (logical [m, k] row-
    // major, viewed col-major as [k, m]). ldb = k. cuBLAS C = our out
    // (logical [m, n] row-major, viewed col-major as [n, m]). ldc = n.
    let lda = n.max(1) as i32;
    let ldb = k.max(1) as i32;
    let ldc = n.max(1) as i32;
    let cublas = device.cublas_handle();
    let compute_type = cublasComputeType_t::Compute32F;
    let lhs_base = lhs.buffer().as_raw().0;
    let rhs_base = rhs.buffer().as_raw().0;
    let out_base = out.as_raw().0;

    let all_equal = n_rep.iter().all(|&r| r == 1);
    if all_equal {
        let a_ptr = rhs_base as *const std::ffi::c_void;
        let b_ptr = lhs_base as *const std::ffi::c_void;
        let c_ptr = out_base as *mut std::ffi::c_void;
        // SAFETY: pointers are valid for the call (lhs, rhs, out
        // outlive the launch); shape parameters match byte-length
        // validation above. Sync follows so result is observable on
        // return (sync KernelRef per locked design decision).
        unsafe {
            baracuda_cublas::gemm_strided_batched_ex(
                &cublas.0,
                Op::N,
                Op::N,
                n as i32,                       // cuBLAS m
                m as i32,                       // cuBLAS n
                k as i32,                       // cuBLAS k
                alpha_ptr,
                a_ptr,                          // cuBLAS A = our rhs
                cudaDataType_t::R_32F,
                lda,
                rhs_per_batch as i64,           // stride_a
                b_ptr,                          // cuBLAS B = our lhs
                cudaDataType_t::R_32F,
                ldb,
                lhs_per_batch as i64,           // stride_b
                beta_ptr,
                c_ptr,                          // cuBLAS C = our out
                cudaDataType_t::R_32F,
                ldc,
                out_per_batch as i64,           // stride_c
                lhs_batch_count as i32,
                compute_type,
                99_i32,                         // CUBLAS_GEMM_DEFAULT
            )
        }
        .map_err(|e| fuel_core_types::Error::Msg(format!("cublas gemm: {e:?}")).bt())?;
    } else {
        // GQA path: walk lhs flat batch index in row-major, decode to
        // multi-index, encode rhs flat batch index via per-axis n_rep
        // mapping, single gemm per batch. Mirrors CPU's per-batch
        // loop in `fuel-cpu-backend::byte_kernels::matmul_f32`.
        let mut lhs_multi = vec![0usize; batch_rank];
        for b in 0..lhs_batch_count {
            let mut rem = b;
            for d in (0..batch_rank).rev() {
                let s = lhs_batch_dims[d];
                lhs_multi[d] = rem % s;
                rem /= s;
            }
            let mut rhs_b = 0usize;
            for d in 0..batch_rank {
                rhs_b = rhs_b * rhs_batch_dims[d] + (lhs_multi[d] / n_rep[d]);
            }
            let lhs_off_bytes = (b * lhs_per_batch * elem) as u64;
            let rhs_off_bytes = (rhs_b * rhs_per_batch * elem) as u64;
            let out_off_bytes = (b * out_per_batch * elem) as u64;
            let a_ptr = (rhs_base + rhs_off_bytes) as *const std::ffi::c_void;
            let b_ptr = (lhs_base + lhs_off_bytes) as *const std::ffi::c_void;
            let c_ptr = (out_base + out_off_bytes) as *mut std::ffi::c_void;
            // SAFETY: pointer offsets are within validated byte ranges
            // (b < lhs_batch_count and rhs_b < rhs_batch_count by
            // construction; per-batch byte counts verified above).
            unsafe {
                baracuda_cublas::gemm_ex(
                    &cublas.0,
                    Op::N,
                    Op::N,
                    n as i32,
                    m as i32,
                    k as i32,
                    alpha_ptr,
                    a_ptr,
                    cudaDataType_t::R_32F,
                    lda,
                    b_ptr,
                    cudaDataType_t::R_32F,
                    ldb,
                    beta_ptr,
                    c_ptr,
                    cudaDataType_t::R_32F,
                    ldc,
                    compute_type,
                    99_i32,
                )
            }
            .map_err(|e| fuel_core_types::Error::Msg(format!("cublas gemm: {e:?}")).bt())?;
        }
    }
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, need_out))
}

/// Element-wise affine `y = mul * x + add` for one F32 `CudaStorageBytes`.
/// Backs `OpKind::Affine` (and is the building block `mean_reduce_f32`
/// uses for its post-sum scaling step). The legacy `Affine` struct
/// in `storage.rs` provides the same math; this is the byte-level
/// path through the unified binding table.
///
/// Allocates a fresh output buffer (the affine kernel's signature
/// has separate `inp` and `out` pointers, and the wrapper takes
/// `&out` mutably so it can't alias `inp`). Output size matches
/// input size.
pub fn affine_f32(src: &CudaStorageBytes, mul: f32, add: f32) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    if src.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "affine_f32: src.len_bytes={} not a multiple of f32 size",
            src.len_bytes(),
        ))
        .bt());
    }
    let elem_count = src.len_bytes() / elem;
    let device = src.device().clone();
    if elem_count == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let mut out = device.alloc_zeros::<u8>(src.len_bytes())?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let func = device.get_or_load_func("affine_f32", &kernels::AFFINE)?;
    // Affine kernel signature: (numel, num_dims, info, inp, out, mul, add).
    let dims_strides: SlicePtrOrNull<usize> = SlicePtrOrNull::Null;
    let mut builder = func.builder();
    barg!(builder, elem_count);
    barg!(builder, 1_usize); // ndims (ignored on the contiguous path)
    dims_strides.builder_arg(&mut builder);
    builder.arg(src.buffer());
    builder.arg(&mut out);
    barg!(builder, mul);
    barg!(builder, add);
    // SAFETY: kernel signature matches the args above — same shape
    // as the legacy `Map1::f` for `Affine`, just on byte buffers.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out),
        device,
        src.len_bytes(),
    ))
}

/// Element-wise clamp `y = min(max(x, lo), hi)` for one F32
/// `CudaStorageBytes`. Mirrors the CPU `clamp_f32` byte kernel: the
/// input shape is preserved, output is freshly allocated with the
/// same byte count, and the `uclamp_f32` PTX kernel does
/// `ming(maxg(x, lo), hi)` per element through the `UNARY_OP2`
/// macro (a unary kernel with two scalar parameters). The kernel
/// signature is `(numel, num_dims, info, lo, hi, inp, out)` —
/// passing `info=null` selects the contiguous fast path.
pub fn clamp_f32(src: &CudaStorageBytes, lo: f32, hi: f32) -> Result<CudaStorageBytes> {
    if !(lo <= hi) {
        return Err(fuel_core_types::Error::Msg(format!(
            "clamp_f32: lo ({lo}) > hi ({hi}) (or NaN); refusing to launch"
        ))
        .bt());
    }
    let elem = std::mem::size_of::<f32>();
    if src.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "clamp_f32: src.len_bytes={} not a multiple of f32 size",
            src.len_bytes(),
        ))
        .bt());
    }
    let elem_count = src.len_bytes() / elem;
    let device = src.device().clone();
    if elem_count == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let mut out = device.alloc_zeros::<u8>(src.len_bytes())?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let func = device.get_or_load_func("uclamp_f32", &kernels::UNARY)?;
    // uclamp signature: (numel, num_dims, info, lo, hi, inp, out).
    let dims_strides: SlicePtrOrNull<usize> = SlicePtrOrNull::Null;
    let mut builder = func.builder();
    barg!(builder, elem_count);
    barg!(builder, 1_usize); // ndims (ignored on the contiguous path)
    dims_strides.builder_arg(&mut builder);
    barg!(builder, lo);
    barg!(builder, hi);
    builder.arg(src.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above — UNARY_OP2.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out),
        device,
        src.len_bytes(),
    ))
}

/// Element-wise integer power `y = x^exp` for one F32
/// `CudaStorageBytes`. Mirrors `f32::powi(i32)` semantics via
/// square-and-multiply on the absolute exponent (with reciprocal for
/// negative exponents). The launched kernel `upowi_f32` is a unary
/// kernel with a single `int` scalar param (UPOWI_OP macro in
/// unary.cu); its signature is `(numel, num_dims, info, exp, inp,
/// out)` with `info=null` selecting the contiguous fast path.
///
/// This is distinct from the existing `upowf_f32` (UNARY_OP1 with
/// float exponent → `powf(x, exp)`); using `upowf_f32` for integer
/// exponents would route through CUDA's `powf` approximation and lose
/// bit-exact parity with the CPU `f32::powi` for some shapes.
pub fn powi_f32(src: &CudaStorageBytes, exp: i32) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    if src.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "powi_f32: src.len_bytes={} not a multiple of f32 size",
            src.len_bytes(),
        ))
        .bt());
    }
    let elem_count = src.len_bytes() / elem;
    let device = src.device().clone();
    if elem_count == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let mut out = device.alloc_zeros::<u8>(src.len_bytes())?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let func = device.get_or_load_func("upowi_f32", &kernels::UNARY)?;
    // upowi signature: (numel, num_dims, info, exp, inp, out).
    let dims_strides: SlicePtrOrNull<usize> = SlicePtrOrNull::Null;
    let mut builder = func.builder();
    barg!(builder, elem_count);
    barg!(builder, 1_usize); // ndims (ignored on the contiguous path)
    dims_strides.builder_arg(&mut builder);
    barg!(builder, exp);
    builder.arg(src.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above — UPOWI_OP.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out),
        device,
        src.len_bytes(),
    ))
}

/// Element-wise dtype cast. Element count is preserved; the byte
/// length of the output differs from the input when source and
/// destination have different `size_in_bytes`. Picks the
/// `cast_<src>_<dst>` kernel from `fuel_cuda_kernels::CAST` based on
/// the dtype pair; missing-kernel cases (e.g. an FP8 cast on a GPU
/// where FP8 wasn't compiled in) surface at kernel-load time with the
/// kernel name in the error.
///
/// Sub-byte source/destination types (`F4`/`F6E2M3`/`F6E3M2`) are not
/// supported — they would need a packed-bytes representation that the
/// unified storage doesn't currently expose. Sub-byte arrives as a
/// follow-up if/when those dtypes become load-bearing.
pub fn cast(
    src: &CudaStorageBytes,
    src_dtype: DType,
    dst_dtype: DType,
) -> Result<CudaStorageBytes> {
    let src_elem_size = src_dtype.size_in_bytes();
    let dst_elem_size = dst_dtype.size_in_bytes();
    if src_elem_size == 0 || dst_elem_size == 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "cast({src_dtype:?} -> {dst_dtype:?}): sub-byte dtypes \
             are not supported through the unified path"
        ))
        .bt());
    }
    if src.len_bytes() % src_elem_size != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "cast({src_dtype:?} -> {dst_dtype:?}): src.len_bytes={} \
             not a multiple of src elem size {}",
            src.len_bytes(),
            src_elem_size,
        ))
        .bt());
    }
    let elem_count = src.len_bytes() / src_elem_size;
    let device = src.device().clone();
    if elem_count == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_bytes = elem_count * dst_elem_size;
    let mut out = device.alloc_zeros::<u8>(out_bytes)?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let kernel_name = format!("cast_{}_{}", src_dtype.as_str(), dst_dtype.as_str());
    let func = device.get_or_load_func(&kernel_name, &kernels::CAST)?;
    // Cast kernel signature: (numel, num_dims, info, inp, out).
    // info=null selects the contiguous fast path.
    let dims_strides: SlicePtrOrNull<usize> = SlicePtrOrNull::Null;
    let mut builder = func.builder();
    barg!(builder, elem_count);
    barg!(builder, 1_usize); // ndims (ignored on the contiguous path)
    dims_strides.builder_arg(&mut builder);
    builder.arg(src.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above — same shape as
    // the legacy `to_dtype` impl in `storage.rs::CudaStorage`.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out),
        device,
        out_bytes,
    ))
}

/// Shared launch path for F32 elementwise binary ops. Layouts pick the
/// path: both contiguous + zero-offset → fast path (matches the legacy
/// pre-Layout-on-Node behavior); otherwise → strided path that hands
/// the kernel a `[dims | lhs_strides | rhs_strides]` blob and lets it
/// walk strides itself. The strided path is what enables broadcast (a
/// dim with stride 0 walks one element repeatedly) and transpose
/// without prior materialization.
///
/// Output element count comes from `lhs_layout.shape()`, which equals
/// `rhs_layout.shape()` for executor-driven calls (Op::BroadcastTo
/// normalizes both partners to the broadcast shape). Direct callers
/// (tests) must uphold the same invariant.
///
/// Synchronizes the default stream so the result is observable on
/// return (sync KernelRef per locked design decision).
fn binary_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    kernel_name: &'static str,
) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    if lhs.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: lhs.len_bytes={} not a multiple of f32 size",
            lhs.len_bytes(),
        ))
        .bt());
    }
    if rhs.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: rhs.len_bytes={} not a multiple of f32 size",
            rhs.len_bytes(),
        ))
        .bt());
    }
    let lhs_dims = lhs_layout.shape().dims();
    let rhs_dims = rhs_layout.shape().dims();
    if lhs_dims != rhs_dims {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: layouts disagree on shape (lhs {:?} vs rhs {:?}); \
             broadcast partners must be normalized to a common shape upstream",
            lhs_dims, rhs_dims,
        ))
        .bt());
    }
    let elem_count = lhs_layout.shape().elem_count();
    let out_bytes = elem_count * elem;
    let device = lhs.device().clone();
    if elem_count == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }

    let lhs_fast = lhs_layout.is_contiguous() && lhs_layout.start_offset() == 0;
    let rhs_fast = rhs_layout.is_contiguous() && rhs_layout.start_offset() == 0;
    let take_fast_path = lhs_fast && rhs_fast;

    let mut out = device.alloc_zeros::<u8>(out_bytes)?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let func = device.get_or_load_func(kernel_name, &kernels::BINARY)?;

    let dims_strides: SlicePtrOrNull<usize> = if take_fast_path {
        // Fast path: lhs.len_bytes() must equal out_bytes (the contiguous
        // fast loop reads `lhs[i]` and `rhs[i]` for i in 0..elem_count).
        if lhs.len_bytes() != out_bytes || rhs.len_bytes() != out_bytes {
            return Err(fuel_core_types::Error::Msg(format!(
                "{kernel_name}: contiguous fast path requires byte sizes \
                 equal to elem_count*4 — got lhs={}, rhs={}, expected={}",
                lhs.len_bytes(),
                rhs.len_bytes(),
                out_bytes,
            ))
            .bt());
        }
        SlicePtrOrNull::Null
    } else {
        // Strided path: kernel reads the [dims | lhs_strides | rhs_strides]
        // blob and walks indices itself. Stride 0 on any axis collapses
        // that axis to repeated reads, which is exactly broadcast.
        SlicePtrOrNull::Ptr(device.clone_htod(
            &crate::storage::dims_strides_strides_usize(lhs_dims, lhs_layout, rhs_layout),
        )?)
    };

    let mut builder = func.builder();
    barg!(builder, elem_count);
    barg!(builder, lhs_dims.len());
    dims_strides.builder_arg(&mut builder);
    builder.arg(lhs.buffer());
    builder.arg(rhs.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above. Both fast and
    // strided variants are exposed by the same PTX function via
    // BINARY_OP_OUT in fuel-cuda-kernels — the kernel branches on
    // `dims_and_strides == nullptr` internally.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, out_bytes))
}

/// Shared launch path for F32 elementwise unary ops. Mirrors
/// [`binary_elementwise_f32`] but with a single input. The
/// fuel-cuda-kernels UNARY function signature is
/// `(elem_count, ndims, dims_strides_or_null, src, out)` — same as
/// the legacy `Map1::f` for `UnaryOpT`. A null `dims_strides_or_null`
/// selects the contiguous fast path; auto-Contiguize guarantees
/// that on the unified path.
fn unary_elementwise_f32(
    src: &CudaStorageBytes,
    kernel_name: &'static str,
) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    if src.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: src.len_bytes={} not a multiple of f32 size",
            src.len_bytes(),
        ))
        .bt());
    }
    let elem_count = src.len_bytes() / elem;
    let device = src.device().clone();
    if elem_count == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let mut out = device.alloc_zeros::<u8>(src.len_bytes())?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let func = device.get_or_load_func(kernel_name, &kernels::UNARY)?;
    let dims_strides: SlicePtrOrNull<usize> = SlicePtrOrNull::Null;
    let mut builder = func.builder();
    barg!(builder, elem_count);
    barg!(builder, 1_usize); // ndims (ignored on the contiguous path)
    dims_strides.builder_arg(&mut builder);
    builder.arg(src.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above — same shape as
    // the legacy `Map1::f` for `UnaryOpT`, just on byte buffers.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out),
        device,
        src.len_bytes(),
    ))
}

/// Shared launch path for F32 reductions (Sum/Max/Min). Mirrors the
/// legacy `Map1Any` for `FastReduce` (storage.rs:317): reorders dims
/// so reduced axes come last, builds a `[dims | strides]` device
/// buffer, and launches with `grid_dim = dst_el` and `block_dim =
/// next_power_of_two(min(1024, el_to_sum_per_block))`. The kernel
/// signature is `(src_numel, el_to_sum_per_block, num_dims, info,
/// src, dst)`.
///
/// Auto-Contiguize guarantees the input is contiguous before this
/// runs, so `input_layout.stride()` is the row-major stride. The
/// strides side-band is still passed because the kernel uses
/// `get_strided_index` unconditionally.
fn reduce_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    reduce_dims: &[usize],
    kernel_name: &'static str,
) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    if src.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: src.len_bytes={} not a multiple of f32 size",
            src.len_bytes(),
        ))
        .bt());
    }
    let src_dims = input_layout.shape().dims();
    let src_stride = input_layout.stride_unsigned();
    let src_el: usize = src_dims.iter().product();
    if src_el * elem != src.len_bytes() {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: src element count {} (from layout shape {:?}) \
             disagrees with byte length {} / sizeof(f32)",
            src_el,
            src_dims,
            src.len_bytes(),
        ))
        .bt());
    }

    // Reorder dims/strides so the reduced axes are at the end —
    // matches the legacy `FastReduce::f` precondition that the
    // kernel iterates over the last `el_to_sum_per_block` elements
    // per block.
    let mut dims: Vec<usize> = Vec::with_capacity(src_dims.len());
    let mut stride: Vec<usize> = Vec::with_capacity(src_dims.len());
    let mut dst_el: usize = 1;
    for (dim_idx, &d) in src_dims.iter().enumerate() {
        if !reduce_dims.contains(&dim_idx) {
            dst_el *= d;
            dims.push(d);
            stride.push(src_stride[dim_idx]);
        }
    }
    for &dim_idx in reduce_dims.iter() {
        dims.push(src_dims[dim_idx]);
        stride.push(src_stride[dim_idx]);
    }

    let dst_bytes = dst_el * elem;
    let device = src.device().clone();
    if src_el == 0 || dst_el == 0 {
        return CudaStorageBytes::alloc(&device, dst_bytes);
    }
    let el_to_sum_per_block = src_el / dst_el;
    // Pow-of-two block size so the in-block parallel reduction's
    // halving loop is well-defined (matches legacy).
    let block_dim = usize::min(1024, el_to_sum_per_block).next_power_of_two();
    let cfg = LaunchConfig {
        grid_dim: (dst_el as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut out = device.alloc_zeros::<u8>(dst_bytes)?;
    let ds = device.clone_htod(&[dims.as_slice(), stride.as_slice()].concat())?;
    let func = device.get_or_load_func(kernel_name, &kernels::REDUCE)?;
    let mut builder = func.builder();
    barg!(builder, src_el);
    barg!(builder, el_to_sum_per_block);
    barg!(builder, src_dims.len());
    builder.arg(&ds);
    builder.arg(src.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above — same shape as
    // the legacy `FastReduce::f`, just on byte buffers.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, dst_bytes))
}
