//! Phase 7.6 step 8 — Layer-1 cost model for primitive ops.
//!
//! Each primitive `OpKind` registered in [`crate::kernel::KernelBindingTable`]
//! gets a [`crate::kernel::CostFn`] that returns a [`crate::fused::CostEstimate`]
//! (FLOPs + bytes moved + launch overhead). The architecture's
//! Layer-1 model is conservative + static: it counts FLOPs from
//! shapes and `OpParams`, estimates bandwidth as operand count ×
//! element count × dtype byte width, and assigns a per-family
//! launch-overhead constant (CPU launches measure in tens of ns).
//! Layer-2 — empirical refinement from per-deployment telemetry —
//! composes on top via the framework that lands with step 11
//! (community-aggregated cache).
//!
//! ## Structure
//!
//! - Per-family cost functions ([`cost_elementwise_unary_cpu`],
//!   [`cost_elementwise_binary_cpu`], [`cost_reduction_cpu`], etc.).
//!   Each is the [`crate::kernel::CostFn`] signature and operates
//!   on whatever shapes + `OpParams` payload its OpKind family
//!   carries.
//! - [`default_cost_for_op_kind`] — the dispatcher consumed by
//!   [`crate::kernel::KernelBindingTable::fill_unset_cpu_cost`].
//!   Maps every `OpKind` variant to the family function appropriate
//!   for its shape contract.
//!
//! ## Conventions
//!
//! - FLOP counts use the "FMA = 2 FLOPs" convention matching the
//!   step-6 fused-op cost functions (matmul = 2·M·N·K, etc.).
//! - Bandwidth assumes 4 B/element as a midpoint (F32 reference);
//!   half-precision is conservatively over-counted, F64 under. The
//!   step-8 Layer-1 model is intentionally coarse — empirical
//!   refinement tightens per-dtype later.
//! - Launch overhead: 50 ns for elementwise + reduction families
//!   (matches the fused-op precedent); 100-200 ns for matmul /
//!   attention (more setup work).

use crate::fused::CostEstimate;
use crate::kernel::{unknown_cost, CostFn, OpParams};
use fuel_core_types::backend::BackendCapabilities;
use fuel_core_types::dispatch::OpKind;
use fuel_core_types::{DType, Shape};

// =============================================================================
// Helpers
// =============================================================================

/// Bytes per element for the supplied dtype. Used to convert
/// element counts into bandwidth. Returns 4 (F32 byte width) for
/// non-numeric dtypes that the cost model doesn't differentiate.
fn dtype_bytes(dt: DType) -> u64 {
    match dt {
        DType::F32 | DType::U32 | DType::I32 => 4,
        DType::F64 | DType::I64 => 8,
        DType::BF16 | DType::F16 | DType::I16 => 2,
        DType::U8 | DType::I8 => 1,
        DType::F8E4M3 | DType::F8E8M0 => 1,
        // 6-bit and 4-bit float micro-types are byte-packed; treat
        // them as 1 byte/elem for the bandwidth estimate.
        DType::F6E2M3 | DType::F6E3M2 | DType::F4 => 1,
    }
}

/// Output element count from the **last** shape in the operand
/// list (the output shape, per binding-table convention).
fn out_elem_count(shapes: &[Shape]) -> u64 {
    shapes
        .last()
        .map(|s| s.dims().iter().map(|&d| d as u64).product::<u64>())
        .unwrap_or(0)
}

/// Total element count summed across the input operands (every
/// shape except the last). Used for bandwidth claims when inputs
/// have differing shapes (e.g. Concat).
fn input_elem_count(shapes: &[Shape]) -> u64 {
    if shapes.len() <= 1 {
        return 0;
    }
    shapes[..shapes.len() - 1]
        .iter()
        .map(|s| s.dims().iter().map(|&d| d as u64).product::<u64>())
        .sum()
}

// =============================================================================
// Cost-family functions
// =============================================================================

/// Cost for elementwise unary ops with cheap math (Neg/Sqr/Recip/
/// Abs/Relu/Step/Floor/Ceil/Round/Sign/Sqrt). ~1 FLOP/element;
/// 1 read + 1 write per element.
pub fn cost_elementwise_unary_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n = out_elem_count(shapes);
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: n,
        bytes_moved: 2 * n * dsize,
        kernel_overhead_ns: 50,
    }
}

/// Cost for elementwise unary transcendentals (Exp/Log/Sin/Cos/
/// Tanh/Sigmoid/Silu/Gelu/GeluErf/Erf/Rsqrt). ~10 FLOPs/element
/// (transcendentals lower to 5-15 hardware ops on average).
pub fn cost_elementwise_unary_transcendental_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n = out_elem_count(shapes);
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: 10 * n,
        bytes_moved: 2 * n * dsize,
        kernel_overhead_ns: 50,
    }
}

/// Cost for elementwise binary ops (Add/Sub/Mul/Div/Maximum/Minimum/
/// Pow/Rem). ~1 FLOP/element (Pow/Rem run higher but the bandwidth
/// dominates for typical sizes); 2 reads + 1 write per element.
pub fn cost_elementwise_binary_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n = out_elem_count(shapes);
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: n,
        bytes_moved: 3 * n * dsize,
        kernel_overhead_ns: 50,
    }
}

/// Cost for comparison ops (Equal/Ne/Lt/Le/Gt/Ge). Same shape
/// contract as elementwise binary but output is U8 (1 B/element).
pub fn cost_comparison_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n = out_elem_count(shapes);
    let dsize_in = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: n,
        // 2 reads × input dtype size + 1 write × 1 byte (U8).
        bytes_moved: 2 * n * dsize_in + n,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `Where` (ternary select). Inputs `[cond:U8, a:T, b:T, out:T]`.
pub fn cost_where_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n = out_elem_count(shapes);
    let dsize = dtypes.get(1).map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: n,
        bytes_moved: n + 3 * n * dsize, // 1 byte cond + 3 × dsize values
        kernel_overhead_ns: 50,
    }
}

/// Cost for full-tensor reductions (SumReduce/MaxReduce/MinReduce/
/// MeanReduce). FLOPs ≈ input element count; one write per output
/// element.
pub fn cost_reduction_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n_in = input_elem_count(shapes);
    let n_out = out_elem_count(shapes);
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: n_in,
        bytes_moved: (n_in + n_out) * dsize,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `ReduceSumTo` / `ReduceMaxTo`. Same shape as reductions
/// but the broadcast-to-target structure may iterate the input
/// multiple times depending on which axes reduce; we approximate
/// as `in_count` FLOPs.
pub fn cost_reduce_to_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n_in = input_elem_count(shapes);
    let n_out = out_elem_count(shapes);
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: n_in,
        bytes_moved: (n_in + n_out) * dsize,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `MatMul`. Reads `Matmul { m, n, k, lhs_batch_dims,
/// rhs_batch_dims }` from `OpParams` to compute the exact FLOP
/// count (`2·batch·M·N·K`).
pub fn cost_matmul_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (m, n, k, batch) = match params {
        OpParams::Matmul { m, n, k, lhs_batch_dims, .. } => {
            let batch: u64 = lhs_batch_dims.iter().map(|&d| d as u64).product::<u64>().max(1);
            (*m as u64, *n as u64, *k as u64, batch)
        }
        _ => return CostEstimate::default(),
    };
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    let flops = 2 * batch * m * n * k;
    let bytes_moved = batch * (m * k + k * n + m * n) * dsize;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 100,
    }
}

/// Cost for `FusedLinear` (matmul + bias-add). Reads `Matmul`
/// params (same as `MatMul`) — bias-add contributes M·N FLOPs per
/// batch on top of the matmul.
pub fn cost_fused_linear_primitive_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
    caps: &BackendCapabilities,
) -> CostEstimate {
    let mm = cost_matmul_cpu(shapes, dtypes, params, caps);
    let (m, n, batch) = match params {
        OpParams::Matmul { m, n, lhs_batch_dims, .. } => {
            let batch: u64 = lhs_batch_dims.iter().map(|&d| d as u64).product::<u64>().max(1);
            (*m as u64, *n as u64, batch)
        }
        _ => return mm,
    };
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    let bias_flops = batch * m * n;
    let bias_bytes = n * dsize;
    CostEstimate {
        flops: mm.flops + bias_flops,
        bytes_moved: mm.bytes_moved + bias_bytes,
        kernel_overhead_ns: 100,
    }
}

/// Cost for `Cast`. Pure dtype conversion — no FLOPs (or ~1
/// conversion/element, dwarfed by bandwidth). Bytes are
/// `in_size + out_size` per element with their respective dtype
/// widths.
pub fn cost_cast_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n = out_elem_count(shapes);
    let dsize_in  = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    let dsize_out = dtypes.last().map(|d| dtype_bytes(*d)).unwrap_or(dsize_in);
    CostEstimate {
        flops: 0,
        bytes_moved: n * (dsize_in + dsize_out),
        kernel_overhead_ns: 50,
    }
}

/// Cost for scalar-by-tensor ops (Affine = mul·x + add, Clamp,
/// PowI). 1-2 FLOPs/element; 1 read + 1 write per element.
pub fn cost_scalar_op_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n = out_elem_count(shapes);
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: 2 * n,
        bytes_moved: 2 * n * dsize,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `MaskedFill` (x, mask:U8 → fill where mask != 0).
pub fn cost_masked_fill_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n = out_elem_count(shapes);
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: 0,
        bytes_moved: n * dsize + n + n * dsize, // value-in + mask + value-out
        kernel_overhead_ns: 50,
    }
}

/// Cost for materializing shape-rearrangement ops (Flip, Roll,
/// Triu, Tril, CumSum, Pad, PadBackward). FLOPs are zero (CumSum
/// is the exception — it does N adds — but bandwidth dominates).
pub fn cost_shape_op_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n_in = input_elem_count(shapes);
    let n_out = out_elem_count(shapes);
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: 0,
        bytes_moved: (n_in + n_out) * dsize,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `Concat`. Sum of N input element counts; one output
/// of equal total size. No FLOPs (memcpy).
pub fn cost_concat_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n_in = input_elem_count(shapes);
    let n_out = out_elem_count(shapes);
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: 0,
        bytes_moved: (n_in + n_out) * dsize,
        kernel_overhead_ns: 50,
    }
}

/// Cost for indexing-style ops (IndexSelect, Gather, IndexAdd,
/// ScatterAdd). FLOPs = 0 for select/gather, = src_count for add/
/// scatter (accumulation). Bytes ≈ (data + indices + out) ×
/// respective dsizes.
pub fn cost_indexing_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n_out = out_elem_count(shapes);
    let dsize_data = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    let dsize_idx  = dtypes.get(1).map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: n_out, // conservative — gather is 0, scatter is N; midpoint.
        bytes_moved: 2 * n_out * dsize_data + n_out * dsize_idx,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `ArgMaxDim` / `ArgMinDim`. Inputs `[in:T, out:U32]`.
/// FLOPs ≈ input element count (per-row compare); bytes are
/// in×dsize + out×4 bytes.
pub fn cost_argindex_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n_in = input_elem_count(shapes);
    let n_out = out_elem_count(shapes);
    let dsize_in = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: n_in,
        bytes_moved: n_in * dsize_in + n_out * 4,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `Conv2D` (binding-table form). Reads
/// `OpParams::Conv2D` for shape geometry.
pub fn cost_conv2d_primitive_cpu(
    _shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (x_shape, w_shape, out_shape, groups) = match params {
        OpParams::Conv2D { x_shape, w_shape, out_shape, groups, .. } => {
            (x_shape, w_shape, out_shape, *groups as u64)
        }
        _ => return CostEstimate::default(),
    };
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    let n = out_shape[0] as u64;
    let cout = out_shape[1] as u64;
    let h_out = out_shape[2] as u64;
    let w_out = out_shape[3] as u64;
    let cin_per_g = w_shape[1] as u64;
    let kh = w_shape[2] as u64;
    let kw = w_shape[3] as u64;
    let conv_flops = 2 * n * cout * h_out * w_out * cin_per_g * kh * kw;
    let elems_in   = (x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]) as u64;
    let elems_w    = (w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]) as u64;
    let elems_out  = n * cout * h_out * w_out;
    let _ = groups; // accounted for via cin_per_g already.
    CostEstimate {
        flops: conv_flops,
        bytes_moved: (elems_in + elems_w + elems_out) * dsize,
        kernel_overhead_ns: 100,
    }
}

/// Cost for `ConvTranspose2D` (binding-table form). Same shape
/// algorithm as Conv2D — the transposed pass moves comparable
/// bytes and does comparable FLOPs.
pub fn cost_conv_transpose2d_primitive_cpu(
    _shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (x_shape, w_shape, out_shape, groups) = match params {
        OpParams::ConvTranspose2D { x_shape, w_shape, out_shape, groups, .. } => {
            (x_shape, w_shape, out_shape, *groups as u64)
        }
        _ => return CostEstimate::default(),
    };
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    let n = out_shape[0] as u64;
    let cout = out_shape[1] as u64;
    let h_out = out_shape[2] as u64;
    let w_out = out_shape[3] as u64;
    let cin_per_g = (x_shape[1] as u64) / groups;
    let kh = w_shape[2] as u64;
    let kw = w_shape[3] as u64;
    let flops = 2 * n * cout * h_out * w_out * cin_per_g * kh * kw;
    let elems_in   = (x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]) as u64;
    let elems_w    = (w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]) as u64;
    let elems_out  = n * cout * h_out * w_out;
    CostEstimate {
        flops,
        bytes_moved: (elems_in + elems_w + elems_out) * dsize,
        kernel_overhead_ns: 100,
    }
}

/// Cost for `FlashAttn` (binding-table form). Reads geometry from
/// `OpParams::FlashAttn`.
pub fn cost_flash_attn_primitive_cpu(
    _shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (b, hq, _hkv, sq, sk, d) = match params {
        OpParams::FlashAttn { b, hq, hkv, sq, sk, d, .. } => {
            (*b as u64, *hq as u64, *hkv as u64, *sq as u64, *sk as u64, *d as u64)
        }
        _ => return CostEstimate::default(),
    };
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    let mm_flops = 4 * b * hq * sq * sk * d;
    let sm_flops = 5 * b * hq * sq * sk;
    let elems_qkv = b * hq * sq * d + 2 * b * hq * sk * d;
    let elems_out = b * hq * sq * d;
    CostEstimate {
        flops: mm_flops + sm_flops,
        bytes_moved: (elems_qkv + elems_out) * dsize,
        kernel_overhead_ns: 200,
    }
}

/// Cost for `PagedAttn` (binding-table form). Reads geometry from
/// `OpParams::PagedAttn`; treats `num_blocks · block_size` as the
/// effective `Sk` upper bound.
pub fn cost_paged_attn_primitive_cpu(
    _shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (b, hq, sq, d, block_size, num_blocks) = match params {
        OpParams::PagedAttn { b, hq, sq, d, block_size, num_blocks, .. } => {
            (*b as u64, *hq as u64, *sq as u64, *d as u64,
             *block_size as u64, *num_blocks as u64)
        }
        _ => return CostEstimate::default(),
    };
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    let sk_upper = block_size * num_blocks;
    let mm_flops = 4 * b * hq * sq * sk_upper * d;
    let elems_q  = b * hq * sq * d;
    let elems_kv = 2 * num_blocks * block_size * d;
    let elems_out = elems_q;
    CostEstimate {
        flops: mm_flops,
        bytes_moved: (elems_q + elems_kv + elems_out) * dsize,
        kernel_overhead_ns: 200,
    }
}

/// Cost for `SoftmaxLastDim` / `SoftmaxLastDimBackward` —
/// outer_count rows × last_dim with constant FLOPs/element.
pub fn cost_softmax_last_dim_primitive_cpu(
    _shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (outer, last_dim) = match params {
        OpParams::SoftmaxLastDim { outer_count, last_dim } => {
            (*outer_count as u64, *last_dim as u64)
        }
        _ => return CostEstimate::default(),
    };
    let n = outer * last_dim;
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: 5 * n, // max-sub + exp + sum + div ≈ 5 FLOPs/elem (Exp counted as 1 for the elementwise body; transcendental cost is in cost_elementwise_unary_transcendental_cpu)
        bytes_moved: 3 * n * dsize,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `SelectiveScan` (binding-table form). FLOPs scale with
/// `batch × seqlen × dim × dstate`. Conservative ~16 FLOPs per inner
/// iteration (exp ≈ 10, plus the FMAs for h update and y accumulate).
pub fn cost_selective_scan_primitive_cpu(
    _shapes: &[Shape],
    _dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (batch, seqlen, dim, dstate) = match params {
        OpParams::SelectiveScan { batch, seqlen, dim, dstate, .. } => {
            (*batch as u64, *seqlen as u64, *dim as u64, *dstate as u64)
        }
        _ => return CostEstimate::default(),
    };
    let flops = batch * seqlen * dim * dstate * 16;
    let bytes_moved = 2 * batch * seqlen * dim * 4
        + 2 * batch * seqlen * dstate * 4
        + dim * dstate * 4
        + batch * seqlen * dim * 4;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `CausalConv1d` (binding-table form). Reads
/// `batch × channels × seq_out × kernel` FLOPs (2 per FMA) plus an
/// optional ~10-FLOP SiLU per output. Bandwidth approximation: x +
/// weight + bias + out, all F32.
pub fn cost_causal_conv1d_primitive_cpu(
    _shapes: &[Shape],
    _dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (batch, channels, seq_in, seq_out, kernel, use_silu) = match params {
        OpParams::CausalConv1d {
            batch, channels, seq_in, seq_out, kernel, use_silu,
        } => (
            *batch as u64, *channels as u64,
            *seq_in as u64, *seq_out as u64,
            *kernel as u64, *use_silu,
        ),
        _ => return CostEstimate::default(),
    };
    let per_out_flops = 2 * kernel + if use_silu { 10 } else { 0 };
    let flops = batch * channels * seq_out * per_out_flops;
    let bytes_moved = batch * channels * seq_in * 4
        + channels * kernel * 4
        + channels * 4
        + batch * channels * seq_out * 4;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `FusedSoftmaxCrossEntropy` (binding-table form). Reads
/// `n_rows × vocab` from `OpParams::FusedSoftmaxCrossEntropy`. Two
/// passes per row (max + sum_exp), plus one transcendental log per
/// row; bandwidth is logits + targets + scalar output. The static
/// estimate intentionally underestimates the kernel-launch /
/// per-row overhead — the empirical layer compensates.
pub fn cost_fused_softmax_cross_entropy_primitive_cpu(
    _shapes: &[Shape],
    _dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (n_rows, vocab) = match params {
        OpParams::FusedSoftmaxCrossEntropy { n_rows, vocab, .. } => {
            (*n_rows as u64, *vocab as u64)
        }
        _ => return CostEstimate::default(),
    };
    // 1 max-compare + 2-FLOP exp+sum per row element + ~10 FLOP log
    // (the transcendental). Comparable shape to the fused-side
    // estimate in fuel-storage::fused::cost_fused_softmax_cross_entropy_cpu;
    // duplicated here because the dispatcher needs a CostFn over
    // OpParams, not FusedOpParams.
    let per_row_flops = 3 * vocab + 10;
    let flops = n_rows * per_row_flops;
    // logits (f32) + targets (i64) + output (4 bytes scalar; conservative).
    let bytes_moved = n_rows * vocab * 4 + n_rows * 8 + 4;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `NormLastDim` family — `LayerNormLastDim`,
/// `RmsNormLastDim`, and their backwards. Same outer × last_dim
/// shape as softmax.
pub fn cost_norm_last_dim_primitive_cpu(
    _shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (outer, last_dim) = match params {
        OpParams::NormLastDim { outer_count, last_dim, .. } => {
            (*outer_count as u64, *last_dim as u64)
        }
        _ => return CostEstimate::default(),
    };
    let n = outer * last_dim;
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: 7 * n, // mean + sub + sqr + sum + sqrt + div ≈ 7 FLOPs/elem
        bytes_moved: 3 * n * dsize,
        kernel_overhead_ns: 50,
    }
}

/// Cost for `Rope` (binding-table form).
pub fn cost_rope_primitive_cpu(
    _shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (outer, seq, head_dim) = match params {
        OpParams::Rope { outer_count, seq, head_dim } => {
            (*outer_count as u64, *seq as u64, *head_dim as u64)
        }
        _ => return CostEstimate::default(),
    };
    let n = outer * seq * head_dim;
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: 4 * n, // 2 FMA pairs across the two rotation planes
        bytes_moved: 2 * n * dsize + 2 * seq * head_dim * dsize, // x + out + cos/sin tables
        kernel_overhead_ns: 50,
    }
}

/// Cost for `QMatMul` (binding-table form). Reads dimensions from
/// `OpParams::QMatMul`.
pub fn cost_qmatmul_primitive_cpu(
    _shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let (batch_count, m, n, k) = match params {
        OpParams::QMatMul { batch_count, m, n, k, .. } => {
            (*batch_count as u64, *m as u64, *n as u64, *k as u64)
        }
        _ => return CostEstimate::default(),
    };
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    let flops = 2 * batch_count * m * n * k;
    let bytes_moved = batch_count * (m * k + m * n) * dsize + n * k / 2; // packed Q weights
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 100,
    }
}

/// Cost for `ReduceMaxToBackward` — 5-pass kernel (max recompute +
/// tie-mask + count + scale + gate).
pub fn cost_reduce_max_to_backward_primitive_cpu(
    shapes: &[Shape],
    dtypes: &[DType],
    _params: &OpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    let n_in = shapes.first()
        .map(|s| s.dims().iter().map(|&d| d as u64).product::<u64>())
        .unwrap_or(0);
    let n_out = shapes.get(1)
        .map(|s| s.dims().iter().map(|&d| d as u64).product::<u64>())
        .unwrap_or(0);
    let dsize = dtypes.first().map(|d| dtype_bytes(*d)).unwrap_or(4);
    CostEstimate {
        flops: 5 * n_in + 2 * n_out,
        bytes_moved: (5 * n_in + 3 * n_out) * dsize,
        kernel_overhead_ns: 80,
    }
}

// =============================================================================
// OpKind → cost-family dispatcher
// =============================================================================

/// Phase 7.6 step 8: maps every `OpKind` variant to its default
/// cost-family function. Consumed by
/// [`crate::kernel::KernelBindingTable::fill_unset_cpu_cost`] at
/// the end of `register_cpu_kernels`.
///
/// **Contract**: every `OpKind` variant must return a real
/// (non-[`unknown_cost`]) function pointer. The step-8 coverage
/// lint asserts this — if you add a new `OpKind` without an arm
/// here, the test fails.
pub fn default_cost_for_op_kind(op: OpKind) -> CostFn {
    use OpKind::*;
    match op {
        // Elementwise unary — cheap math.
        ReluElementwise | NegElementwise | SqrElementwise | SqrtElementwise
        | RecipElementwise | AbsElementwise | StepElementwise
        | FloorElementwise | CeilElementwise | RoundElementwise
        | SignElementwise => cost_elementwise_unary_cpu,

        // Elementwise unary — transcendental.
        TanhElementwise | ExpElementwise | LogElementwise
        | SinElementwise | CosElementwise | SigmoidElementwise
        | SiluElementwise | GeluElementwise | GeluErfElementwise
        | ErfElementwise | RsqrtElementwise => cost_elementwise_unary_transcendental_cpu,

        // Elementwise binary.
        AddElementwise | SubElementwise | MulElementwise | DivElementwise
        | MaximumElementwise | MinimumElementwise | PowElementwise
        | RemElementwise => cost_elementwise_binary_cpu,

        // Comparisons (output U8).
        EqualElementwise | NotEqualElementwise | LessElementwise
        | LessEqualElementwise | GreaterElementwise | GreaterEqualElementwise
            => cost_comparison_cpu,

        Where => cost_where_cpu,

        // Reductions.
        SumReduce | MaxReduce | MinReduce | MeanReduce => cost_reduction_cpu,
        ReduceSumTo | ReduceMaxTo => cost_reduce_to_cpu,

        // Dense linear algebra.
        MatMul => cost_matmul_cpu,
        FusedLinear => cost_fused_linear_primitive_cpu,
        QMatMul => cost_qmatmul_primitive_cpu,

        // Convolutions.
        Conv2D => cost_conv2d_primitive_cpu,
        ConvTranspose2D => cost_conv_transpose2d_primitive_cpu,

        // Attention.
        FlashAttn => cost_flash_attn_primitive_cpu,
        PagedAttn => cost_paged_attn_primitive_cpu,

        // Cast.
        Cast => cost_cast_cpu,

        // Scalar/affine.
        Affine | ClampElementwise | PowIElementwise => cost_scalar_op_cpu,

        // Shape-rearrangement / pad / mask / cumsum.
        Flip | Roll | Triu | Tril | CumSum | Pad | PadBackward => cost_shape_op_cpu,

        // MaskedFill.
        MaskedFill => cost_masked_fill_cpu,

        // Concat.
        Concat => cost_concat_cpu,

        // Softmax / norm family (forward + backward).
        SoftmaxLastDim | SoftmaxLastDimBackward
        | LogSoftmaxLastDim | LogSoftmaxLastDimBackward
            => cost_softmax_last_dim_primitive_cpu,
        FusedSoftmaxCrossEntropy => cost_fused_softmax_cross_entropy_primitive_cpu,
        CausalConv1d => cost_causal_conv1d_primitive_cpu,
        SelectiveScan => cost_selective_scan_primitive_cpu,
        RmsNormLastDim | RmsNormLastDimBackward
        | LayerNormLastDim | LayerNormLastDimBackward
            => cost_norm_last_dim_primitive_cpu,
        ReduceMaxToBackward => cost_reduce_max_to_backward_primitive_cpu,

        // Rope.
        Rope => cost_rope_primitive_cpu,

        // Indexing / scatter.
        IndexSelect | Gather | IndexAdd | ScatterAdd => cost_indexing_cpu,

        // ArgMax/ArgMin.
        ArgMaxDim | ArgMinDim => cost_argindex_cpu,

        // Op::Copy — cross-device byte transfer. No FLOPs; bandwidth-
        // bound (read source + write destination). Same shape as
        // Concat / shape-op costs. The per-source-backend kernel
        // overhead (PCIe latency for D2H) is the dominant term for
        // small tensors; the family default underestimates that but
        // is correct for the bytes-moved axis. A Layer-2 empirical
        // refinement can attach later.
        Copy => cost_shape_op_cpu,

        // Op::WriteSlice — in-place rectangular scatter write. No
        // FLOPs; bandwidth-bound (read source slab + write
        // destination slab). Same shape-op cost; the per-backend
        // overhead (kernel launch on GPU vs nested loop on CPU) is
        // a Layer-2 calibration concern.
        WriteSlice => cost_shape_op_cpu,

        // OpKind is `#[non_exhaustive]` — new variants get
        // [`unknown_cost`] until an explicit arm is added here.
        // The step-8 lint catches this immediately by asserting
        // every registered CPU OpKind has a non-`unknown_cost`
        // function.
        _ => unknown_cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: every cost-family function returns non-zero values
    /// for non-empty shapes. The lint elsewhere asserts the
    /// dispatcher covers every OpKind; this just verifies the
    /// families themselves compute something.
    #[test]
    fn elementwise_unary_cost_scales_with_elem_count() {
        use fuel_core_types::backend::{BackendCapabilities, TransferPath};
        use fuel_core_types::probe::BackendId;
        use fuel_core_types::DeviceLocation;
        use std::collections::HashSet;

        let in_shape = Shape::from_dims(&[8, 16]);
        let out_shape = Shape::from_dims(&[8, 16]);
        let caps = BackendCapabilities {
            backend_id: BackendId::Cpu,
            device_location: DeviceLocation::Cpu,
            op_dtype_support: HashSet::new(),
            required_alignment: 1,
            access_granularity_bits: 8,
            transfer_paths: vec![(DeviceLocation::Cpu, TransferPath::SameDevice)],
        };
        let c = cost_elementwise_unary_cpu(
            &[in_shape, out_shape],
            &[DType::F32, DType::F32],
            &OpParams::None,
            &caps,
        );
        // 8×16 = 128 elements; 1 FLOP/elem; 2 reads/writes × 4 bytes.
        assert_eq!(c.flops, 128);
        assert_eq!(c.bytes_moved, 128 * 2 * 4);
    }

    /// Smoke: unknown_cost returns all-zero estimate.
    #[test]
    fn unknown_cost_returns_default() {
        use fuel_core_types::backend::BackendCapabilities;
        use fuel_core_types::probe::BackendId;
        use fuel_core_types::DeviceLocation;
        use std::collections::HashSet;

        let caps = BackendCapabilities {
            backend_id: BackendId::Cpu,
            device_location: DeviceLocation::Cpu,
            op_dtype_support: HashSet::new(),
            required_alignment: 1,
            access_granularity_bits: 8,
            transfer_paths: vec![],
        };
        let c = unknown_cost(&[], &[], &OpParams::None, &caps);
        assert_eq!(c, CostEstimate::default());
    }
}
