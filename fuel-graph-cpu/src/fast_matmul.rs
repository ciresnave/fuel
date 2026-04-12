//! Gemm-backed matmul dispatch for f32 and f64. This is the sole
//! source of speedup versus the reference executor — every other op
//! in the graph is orders of magnitude cheaper than matmul in a
//! transformer forward pass, so speeding up matmul alone gives you
//! ~95% of the possible win with ~5% of the code.
//!
//! Handles any rank ≥ 2 input by flattening the batch prefix and
//! looping over batch slices, calling `gemm::gemm` once per slice.
//! Each slice is treated as a contiguous row-major matrix with row
//! stride `= cols` and column stride `= 1` — the same convention
//! `fuel-cpu-backend` uses and the natural layout for our row-major
//! `RefTensor`.

use fuel_core_types::Shape;
use fuel_reference_backend::RefTensor;
use gemm::{gemm, Parallelism};

/// Parallelism knob passed to every `gemm` call.
///
/// `Parallelism::Rayon(0)` means "use all threads in the current Rayon
/// pool," which for a default-initialized pool is the number of logical
/// CPUs. For the weight matmuls in a transformer forward pass this is
/// what we want — the dominant cost is a handful of large GEMMs that
/// scale almost linearly across cores. Small matmuls (attention
/// `Q @ K^T` on short sequences) fall below gemm's internal threading
/// threshold and transparently run single-threaded anyway, so there's
/// no overhead for short-sequence decode steps.
const PARALLELISM: Parallelism = Parallelism::Rayon(0);

/// Compute `a @ b` as an `f32` tensor of shape `[..., m, n]`. Both
/// operands must have the same rank ≥ 2 and matching batch prefix.
/// Uses gemm under the hood — typically 50-200× faster than the
/// reference triple-loop matmul on large matrices.
pub fn matmul_f32(a: &RefTensor<f32>, b: &RefTensor<f32>) -> RefTensor<f32> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let (m, k, n, batch_dims) = validate_and_dims(a_dims, b_dims);
    let batch_count: usize = batch_dims.iter().product::<usize>().max(1);

    let mut out_dims: Vec<usize> = batch_dims.to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let mut out = vec![0.0_f32; batch_count * m * n];

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let out_batch_stride = m * n;

    for batch in 0..batch_count {
        // Safety: every pointer math stays within the bounds of its
        // respective slice — we just computed those slices from the
        // input shapes. The gemm call below writes exactly m*n f32s
        // into out[batch * out_batch_stride..], again in-bounds.
        unsafe {
            let a_ptr = a_data.as_ptr().add(batch * a_batch_stride);
            let b_ptr = b_data.as_ptr().add(batch * b_batch_stride);
            let c_ptr = out.as_mut_ptr().add(batch * out_batch_stride);
            gemm(
                m,                          // rows of output / rows of lhs
                n,                          // cols of output / cols of rhs
                k,                          // inner / cols of lhs / rows of rhs
                c_ptr,                      // dst pointer
                1_isize,                    // dst col stride (contiguous row-major)
                n as isize,                 // dst row stride = N
                false,                      // read_dst = false → overwrite
                a_ptr,                      // lhs pointer
                1_isize,                    // lhs col stride
                k as isize,                 // lhs row stride = K
                b_ptr,                      // rhs pointer
                1_isize,                    // rhs col stride
                n as isize,                 // rhs row stride = N
                0.0_f32,                    // alpha (dst scalar, ignored w/ read_dst=false)
                1.0_f32,                    // beta (product scalar — dst = 1 * lhs @ rhs)
                false,                      // conj_dst
                false,                      // conj_lhs
                false,                      // conj_rhs
                PARALLELISM,
            );
        }
    }

    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

/// Same as [`matmul_f32`] but for `f64`.
pub fn matmul_f64(a: &RefTensor<f64>, b: &RefTensor<f64>) -> RefTensor<f64> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let (m, k, n, batch_dims) = validate_and_dims(a_dims, b_dims);
    let batch_count: usize = batch_dims.iter().product::<usize>().max(1);

    let mut out_dims: Vec<usize> = batch_dims.to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let mut out = vec![0.0_f64; batch_count * m * n];

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let out_batch_stride = m * n;

    for batch in 0..batch_count {
        unsafe {
            let a_ptr = a_data.as_ptr().add(batch * a_batch_stride);
            let b_ptr = b_data.as_ptr().add(batch * b_batch_stride);
            let c_ptr = out.as_mut_ptr().add(batch * out_batch_stride);
            gemm(
                m,
                n,
                k,
                c_ptr,
                1_isize,
                n as isize,
                false,
                a_ptr,
                1_isize,
                k as isize,
                b_ptr,
                1_isize,
                n as isize,
                0.0_f64,
                1.0_f64,
                false,
                false,
                false,
                PARALLELISM,
            );
        }
    }

    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

/// Validate shape compatibility for matmul and return `(m, k, n,
/// batch_dims)`. Used by both the f32 and f64 paths.
fn validate_and_dims<'a>(
    a_dims: &'a [usize],
    b_dims: &'a [usize],
) -> (usize, usize, usize, &'a [usize]) {
    assert!(
        a_dims.len() >= 2 && b_dims.len() >= 2,
        "matmul: both operands must be rank ≥ 2, got {a_dims:?} and {b_dims:?}",
    );
    assert_eq!(
        a_dims.len(),
        b_dims.len(),
        "matmul: operands must have the same rank, got {} and {}",
        a_dims.len(),
        b_dims.len(),
    );
    let rank = a_dims.len();
    let batch_rank = rank - 2;
    for i in 0..batch_rank {
        assert_eq!(
            a_dims[i], b_dims[i],
            "matmul: batch dim mismatch at axis {i}: {} vs {}",
            a_dims[i], b_dims[i],
        );
    }
    let m = a_dims[rank - 2];
    let k = a_dims[rank - 1];
    let k2 = b_dims[rank - 2];
    let n = b_dims[rank - 1];
    assert_eq!(
        k, k2,
        "matmul: inner dim mismatch (lhs k={k}, rhs k={k2})",
    );
    (m, k, n, &a_dims[..batch_rank])
}
