//! Per-operation reference implementations.
//!
//! Every function here is written for maximum obviousness, not speed. When
//! in doubt about the correct output of an op, this is the file that tells
//! you what the answer *should* be. If a function in here is clever enough
//! that you have to think about whether it is correct, it is too clever —
//! rewrite it into a form that is trivially correct by inspection.
//!
//! ## Dtype coverage
//!
//! Functions are generic over any `T: num_traits::Float`. In practice this
//! means `f32`, `f64`, `half::bf16`, and `half::f16` all flow through the
//! same implementation. Integer dtypes are not covered here and will be
//! added in a separate pass when there is a concrete validation need.

use crate::RefTensor;
use fuel_core_types::Shape;
use num_traits::Float;

/// Internal helper: build a tensor of type `T` from an `f64` constant. Used
/// to express literal constants like `0.044715` in dtype-generic functions.
#[inline]
fn cst<T: Float>(v: f64) -> T {
    T::from(v).expect("constant cannot be represented in target dtype")
}

// ---------- unary ops -------------------------------------------------------

/// Element-wise negation: `y[i] = -x[i]`.
pub fn neg<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| -v).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Rectified linear unit: `y[i] = max(0, x[i])`.
pub fn relu<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let zero = T::zero();
    let data: Vec<T> = x
        .as_slice()
        .iter()
        .map(|&v| if v > zero { v } else { zero })
        .collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Element-wise square: `y[i] = x[i] * x[i]`.
pub fn sqr<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| v * v).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Element-wise square root: `y[i] = sqrt(x[i])`.
pub fn sqrt<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| v.sqrt()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Element-wise exponential: `y[i] = e^x[i]`.
pub fn exp<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| v.exp()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Sign function: `y[i] = -1 if x[i] < 0, 0 if x[i] == 0, 1 if x[i] > 0`.
pub fn sign<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let zero = T::zero();
    let one = T::one();
    let data: Vec<T> = x
        .as_slice()
        .iter()
        .map(|&v| {
            if v > zero {
                one
            } else if v < zero {
                -one
            } else {
                zero
            }
        })
        .collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Natural logarithm: `y[i] = ln(x[i])`. Defined for positive inputs only;
/// non-positive inputs pass straight through to the IEEE 754 `ln` which
/// produces `NaN` or `-inf`.
pub fn log<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| v.ln()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Sine: `y[i] = sin(x[i])`.
pub fn sin<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| v.sin()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cosine: `y[i] = cos(x[i])`.
pub fn cos<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| v.cos()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Absolute value: `y[i] = |x[i]|`.
pub fn abs<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| v.abs()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Reciprocal: `y[i] = 1 / x[i]`. Callers are responsible for ensuring
/// inputs are non-zero; `1 / 0` produces `inf` by IEEE 754.
pub fn recip<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let one = T::one();
    let data: Vec<T> = x.as_slice().iter().map(|&v| one / v).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Hyperbolic tangent: `y[i] = tanh(x[i])`.
pub fn tanh<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| v.tanh()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Floor: `y[i] = floor(x[i])`.
pub fn floor<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| v.floor()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Ceiling: `y[i] = ceil(x[i])`.
pub fn ceil<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| v.ceil()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Logistic sigmoid: `y[i] = 1 / (1 + exp(-x[i]))`. Implemented in the
/// numerically stable split form to avoid overflow in `exp(-x)` for large
/// negative `x`.
pub fn sigmoid<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let zero = T::zero();
    let one = T::one();
    let data: Vec<T> = x
        .as_slice()
        .iter()
        .map(|&v| {
            if v >= zero {
                let e = (-v).exp();
                one / (one + e)
            } else {
                let e = v.exp();
                e / (one + e)
            }
        })
        .collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// SiLU activation (also called Swish): `y[i] = x[i] * sigmoid(x[i])`.
pub fn silu<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let sig = sigmoid(x);
    let a = x.as_slice();
    let b = sig.as_slice();
    let data: Vec<T> = a.iter().zip(b).map(|(&v, &s)| v * s).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// GELU activation using the tanh approximation:
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gelu<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let c: T = cst::<T>(2.0 / std::f64::consts::PI).sqrt();
    let half: T = cst(0.5);
    let one: T = T::one();
    let k: T = cst(0.044715);
    let data: Vec<T> = x
        .as_slice()
        .iter()
        .map(|&v| {
            let inner = c * (v + k * v * v * v);
            half * v * (one + inner.tanh())
        })
        .collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Heaviside step function: `y[i] = 1 if x[i] > 0 else 0`. Serves as the
/// subgradient of `relu` at nonzero inputs, and as a general indicator
/// function. At exactly `x = 0` we return `0` (the standard convention
/// for subgradient-based autodiff).
pub fn step<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let zero = T::zero();
    let one = T::one();
    let data: Vec<T> = x
        .as_slice()
        .iter()
        .map(|&v| if v > zero { one } else { zero })
        .collect();
    RefTensor::from_vec(data, x.shape().clone())
}

// ---------- binary ops ------------------------------------------------------

fn assert_same_shape(a: &Shape, b: &Shape, op: &'static str) {
    assert_eq!(
        a.dims(),
        b.dims(),
        "reference {op}: shape mismatch: lhs={:?}, rhs={:?}",
        a.dims(),
        b.dims(),
    );
}

/// Element-wise addition: `y[i] = a[i] + b[i]`.
///
/// Requires matching shapes. No broadcasting — broadcast the inputs into
/// contiguous tensors of matching shape before calling.
pub fn add<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    assert_same_shape(a.shape(), b.shape(), "add");
    let data: Vec<T> = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&x, &y)| x + y)
        .collect();
    RefTensor::from_vec(data, a.shape().clone())
}

/// Element-wise subtraction: `y[i] = a[i] - b[i]`.
pub fn sub<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    assert_same_shape(a.shape(), b.shape(), "sub");
    let data: Vec<T> = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&x, &y)| x - y)
        .collect();
    RefTensor::from_vec(data, a.shape().clone())
}

/// Element-wise multiplication: `y[i] = a[i] * b[i]`.
pub fn mul<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    assert_same_shape(a.shape(), b.shape(), "mul");
    let data: Vec<T> = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&x, &y)| x * y)
        .collect();
    RefTensor::from_vec(data, a.shape().clone())
}

/// Element-wise division: `y[i] = a[i] / b[i]`.
pub fn div<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    assert_same_shape(a.shape(), b.shape(), "div");
    let data: Vec<T> = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&x, &y)| x / y)
        .collect();
    RefTensor::from_vec(data, a.shape().clone())
}

// ---------- reductions ------------------------------------------------------

/// Sum-reduce all elements to a scalar (a rank-0 tensor).
pub fn sum_all<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let total = x
        .as_slice()
        .iter()
        .copied()
        .fold(T::zero(), |acc, v| acc + v);
    RefTensor::from_vec(vec![total], Shape::from_dims(&[]))
}

/// Max-reduce all elements to a scalar. Returns `-inf` for an empty tensor
/// (the mathematical identity for max).
pub fn max_all<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let mut best = T::neg_infinity();
    for &v in x.as_slice() {
        if v > best {
            best = v;
        }
    }
    RefTensor::from_vec(vec![best], Shape::from_dims(&[]))
}

/// Min-reduce all elements to a scalar. Returns `+inf` for an empty tensor
/// (the mathematical identity for min).
pub fn min_all<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let mut best = T::infinity();
    for &v in x.as_slice() {
        if v < best {
            best = v;
        }
    }
    RefTensor::from_vec(vec![best], Shape::from_dims(&[]))
}

/// Mean of all elements to a scalar. Returns `NaN` for an empty tensor (the
/// arithmetic mean of zero samples is undefined).
pub fn mean_all<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let n = x.elem_count();
    if n == 0 {
        return RefTensor::from_vec(vec![T::nan()], Shape::from_dims(&[]));
    }
    let total = x
        .as_slice()
        .iter()
        .copied()
        .fold(T::zero(), |acc, v| acc + v);
    RefTensor::from_vec(vec![total / cst::<T>(n as f64)], Shape::from_dims(&[]))
}

/// Row-major strides for a given shape. Internal helper for axis reductions
/// and reshaping. `strides[i]` is the number of flat elements you advance
/// when the multi-index increments along dimension `i` by one.
fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let n = dims.len();
    let mut strides = vec![1_usize; n];
    if n == 0 {
        return strides;
    }
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Shared internal implementation for axis reductions.
///
/// `init` is the identity element for the reduction (`T::zero()` for sum,
/// `T::neg_infinity()` for max, `T::infinity()` for min). `combine` folds
/// each contributing element into the running accumulator.
fn reduce_dim<T, F>(x: &RefTensor<T>, dim: usize, init: T, combine: F) -> RefTensor<T>
where
    T: Float,
    F: Fn(T, T) -> T,
{
    let in_dims = x.shape().dims();
    assert!(
        dim < in_dims.len(),
        "reduce_dim: dim {dim} out of bounds for shape {in_dims:?}",
    );

    let out_dims: Vec<usize> = in_dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, &d)| d)
        .collect();
    let out_count: usize = if out_dims.is_empty() {
        1
    } else {
        out_dims.iter().product()
    };
    let mut out = vec![init; out_count];

    let in_strides = row_major_strides(in_dims);
    let out_strides = row_major_strides(&out_dims);

    let data = x.as_slice();
    for in_flat in 0..data.len() {
        // Unflatten in_flat using in_strides.
        let mut remainder = in_flat;
        let mut in_multi = vec![0_usize; in_dims.len()];
        for i in 0..in_dims.len() {
            in_multi[i] = remainder / in_strides[i];
            remainder %= in_strides[i];
        }
        // Drop the reduced coordinate.
        let out_multi: Vec<usize> = in_multi
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, &v)| v)
            .collect();
        // Flatten the output multi-index.
        let out_flat: usize = out_multi
            .iter()
            .zip(&out_strides)
            .map(|(&c, &s)| c * s)
            .sum();
        out[out_flat] = combine(out[out_flat], data[in_flat]);
    }

    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

/// Sum along a single dimension. The reduced dimension is removed from the
/// output shape (no keepdim). For an input of shape `[a, b, c]` and
/// `dim = 1`, the output has shape `[a, c]`.
pub fn sum_dim<T: Float>(x: &RefTensor<T>, dim: usize) -> RefTensor<T> {
    reduce_dim(x, dim, T::zero(), |acc, v| acc + v)
}

/// Maximum along a single dimension. The reduced dimension is removed from
/// the output shape. Identity for empty reductions is `-inf`.
pub fn max_dim<T: Float>(x: &RefTensor<T>, dim: usize) -> RefTensor<T> {
    reduce_dim(x, dim, T::neg_infinity(), |acc, v| if v > acc { v } else { acc })
}

/// Minimum along a single dimension. The reduced dimension is removed from
/// the output shape. Identity for empty reductions is `+inf`.
pub fn min_dim<T: Float>(x: &RefTensor<T>, dim: usize) -> RefTensor<T> {
    reduce_dim(x, dim, T::infinity(), |acc, v| if v < acc { v } else { acc })
}

/// Mean along a single dimension. The reduced dimension is removed from the
/// output shape. Computes `sum_dim(x, dim) / dims[dim]`.
pub fn mean_dim<T: Float>(x: &RefTensor<T>, dim: usize) -> RefTensor<T> {
    let n: T = cst(x.shape().dims()[dim] as f64);
    let s = sum_dim(x, dim);
    let data: Vec<T> = s.as_slice().iter().map(|&v| v / n).collect();
    RefTensor::from_vec(data, s.shape().clone())
}

/// Index of the maximum along `dim`. Returns a `u32` tensor with the
/// reduced dim removed. On ties, returns the smallest index (PyTorch
/// convention).
pub fn argmax_dim<T: Float>(x: &RefTensor<T>, dim: usize) -> RefTensor<u32> {
    argindex_dim(x, dim, /*is_max=*/ true)
}

/// Index of the minimum along `dim`. Returns a `u32` tensor with the
/// reduced dim removed. On ties, returns the smallest index.
pub fn argmin_dim<T: Float>(x: &RefTensor<T>, dim: usize) -> RefTensor<u32> {
    argindex_dim(x, dim, /*is_max=*/ false)
}

/// Shared implementation of argmax_dim / argmin_dim.
fn argindex_dim<T: Float>(x: &RefTensor<T>, dim: usize, is_max: bool) -> RefTensor<u32> {
    let in_dims = x.shape().dims();
    assert!(
        dim < in_dims.len(),
        "argindex_dim: dim {dim} out of bounds for {in_dims:?}",
    );
    let out_dims: Vec<usize> = in_dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, &d)| d)
        .collect();
    let out_count: usize = if out_dims.is_empty() {
        1
    } else {
        out_dims.iter().product()
    };
    let reduced_size = in_dims[dim];
    let in_strides = row_major_strides(in_dims);
    let out_strides = row_major_strides(&out_dims);
    let data = x.as_slice();
    let mut out = vec![0_u32; out_count];

    for out_flat in 0..out_count {
        // Unflatten out_flat in out_dims.
        let mut remainder = out_flat;
        let mut out_multi = vec![0_usize; out_dims.len()];
        for i in 0..out_dims.len() {
            out_multi[i] = remainder / out_strides[i];
            remainder %= out_strides[i];
        }
        // Build the "seed" input multi-index with the reduced dim at 0.
        let mut in_multi: Vec<usize> = Vec::with_capacity(in_dims.len());
        for i in 0..in_dims.len() {
            if i < dim {
                in_multi.push(out_multi[i]);
            } else if i == dim {
                in_multi.push(0);
            } else {
                in_multi.push(out_multi[i - 1]);
            }
        }
        let base_flat: usize = in_multi.iter().zip(&in_strides).map(|(&c, &s)| c * s).sum();
        let mut best_val = data[base_flat];
        let mut best_idx: u32 = 0;
        for k in 1..reduced_size {
            let flat = base_flat + k * in_strides[dim];
            let v = data[flat];
            let better = if is_max { v > best_val } else { v < best_val };
            if better {
                best_val = v;
                best_idx = k as u32;
            }
        }
        out[out_flat] = best_idx;
    }
    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

// ---------- matmul ----------------------------------------------------------

/// N-D batched matrix multiply. The last two dimensions of each operand
/// are the matrix dims; any leading dims are batch dims and must match
/// exactly (no batch broadcasting). Rank ≥ 2 required for both operands.
///
/// - `a.shape()` = `[...batch, m, k]`
/// - `b.shape()` = `[...batch, k, n]`
/// - output shape = `[...batch, m, n]`
///
/// Textbook: loop over every batch index, then over (i, j, k) of the
/// per-batch rank-2 matmul. This is the op graph-level `MatMul` realizes
/// to for any rank; the rank-2 fast path goes through [`matmul_2d`].
pub fn matmul<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
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
    // Check batch prefix matches exactly.
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

    // Shortcut for rank-2: defer to the existing simple implementation.
    if rank == 2 {
        return matmul_2d(a, b);
    }

    // Build output shape and flat buffer.
    let mut out_dims: Vec<usize> = a_dims[..batch_rank].to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let batch_count: usize = a_dims[..batch_rank].iter().product::<usize>().max(1);
    let out_count: usize = out_dims.iter().product();
    let mut out = vec![T::zero(); out_count];

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let out_batch_stride = m * n;

    for batch in 0..batch_count {
        let a_off = batch * a_batch_stride;
        let b_off = batch * b_batch_stride;
        let out_off = batch * out_batch_stride;
        for i in 0..m {
            for j in 0..n {
                let mut acc = T::zero();
                for kk in 0..k {
                    acc = acc + a_data[a_off + i * k + kk] * b_data[b_off + kk * n + j];
                }
                out[out_off + i * n + j] = acc;
            }
        }
    }
    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

/// Textbook 2-D matrix multiply: `C[i,j] = sum_k A[i,k] * B[k,j]`.
///
/// Rank-2 only. For a batched or rank-3 matmul, loop over the batch dimension
/// in the caller and invoke this per slice.
pub fn matmul_2d<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    assert_eq!(
        a_dims.len(),
        2,
        "reference matmul_2d: lhs must be rank 2, got shape {a_dims:?}",
    );
    assert_eq!(
        b_dims.len(),
        2,
        "reference matmul_2d: rhs must be rank 2, got shape {b_dims:?}",
    );
    let (m, k) = (a_dims[0], a_dims[1]);
    let (k2, n) = (b_dims[0], b_dims[1]);
    assert_eq!(
        k, k2,
        "reference matmul_2d: inner dim mismatch: lhs k={k}, rhs k={k2}",
    );

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let mut out = vec![T::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = T::zero();
            for kk in 0..k {
                acc = acc + a_data[i * k + kk] * b_data[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    RefTensor::from_vec(out, Shape::from_dims(&[m, n]))
}

// ---------- 2-D convolution ------------------------------------------------

/// Textbook 2-D convolution. Bit-match reference for the `Op::Conv2D`
/// op. Inputs:
///   - `x`:      `[N, Cin, H, W]`
///   - `weight`: `[Cout, Cin/groups, Kh, Kw]`
///   - `bias`:   `Option<[Cout]>`
///   - `stride`: `(stride_h, stride_w)`
///   - `padding`: `(pad_h, pad_w)` — symmetric zero-pad on each side
///   - `groups`: channel-group count (1 = regular, Cin=Cout=groups = depthwise)
///
/// Output `[N, Cout, Hout, Wout]` with
///   Hout = (H + 2·pad_h − Kh) / stride_h + 1
///   Wout = (W + 2·pad_w − Kw) / stride_w + 1
///
/// Written as five nested loops for obviousness; fast backends wrap
/// im2col + gemm instead. This function is the oracle their output
/// must match.
pub fn conv2d<T: Float>(
    x: &RefTensor<T>,
    weight: &RefTensor<T>,
    bias: Option<&RefTensor<T>>,
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
) -> RefTensor<T> {
    let xd = x.shape().dims();
    let wd = weight.shape().dims();
    assert_eq!(xd.len(), 4, "conv2d: x must be rank 4, got {xd:?}");
    assert_eq!(wd.len(), 4, "conv2d: weight must be rank 4, got {wd:?}");
    let s = fuel_conv::ConvShape {
        batch: xd[0], c_in: xd[1], h: xd[2], w: xd[3],
        c_out: wd[0], k_h: wd[2], k_w: wd[3],
        stride, padding, groups,
    };
    s.validate().expect("conv2d shape validation");
    let mut out = vec![T::zero(); s.output_len()];
    fuel_conv::conv2d_direct(
        x.as_slice(),
        weight.as_slice(),
        bias.map(|b| b.as_slice()),
        &s,
        &mut out,
    );
    RefTensor::from_vec(
        out,
        Shape::from_dims(&[s.batch, s.c_out, s.h_out(), s.w_out()]),
    )
}

/// Textbook 2-D transposed convolution (a.k.a. "deconv"). Bit-match
/// reference for `Op::ConvTranspose2D`.
///
/// Inputs:
///   - `x`:      `[N, Cin, H, W]`
///   - `weight`: `[Cin, Cout/groups, Kh, Kw]` (note transposed channel
///     order vs `conv2d`)
///
/// Output `[N, Cout, Hout, Wout]` with
///   Hout = (H − 1)·stride.0 − 2·pad.0 + dil.0·(Kh − 1) + out_pad.0 + 1
///   Wout = (W − 1)·stride.1 − 2·pad.1 + dil.1·(Kw − 1) + out_pad.1 + 1
///
/// Written as the obvious nested-loop form: scatter each input element
/// into the output through every kernel position. Slow but correct.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d<T: Float>(
    x: &RefTensor<T>,
    weight: &RefTensor<T>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> RefTensor<T> {
    let xd = x.shape().dims();
    let wd = weight.shape().dims();
    assert_eq!(xd.len(), 4, "conv_transpose2d: x must be rank 4, got {xd:?}");
    assert_eq!(wd.len(), 4, "conv_transpose2d: weight must be rank 4, got {wd:?}");
    let (n, cin, h_in, w_in) = (xd[0], xd[1], xd[2], xd[3]);
    let (cin_w, cout_per_g, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
    assert_eq!(cin, cin_w, "conv_transpose2d: x has {cin} in-channels but weight has {cin_w}");
    assert_eq!(cin % groups, 0, "conv_transpose2d: Cin={cin} must be divisible by groups={groups}");
    let cin_per_g = cin / groups;
    let cout = cout_per_g * groups;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (oph, opw) = output_padding;
    let (dh, dw) = dilation;
    let h_out_unpadded = (h_in.saturating_sub(1)) * sh + dh * (kh - 1) + oph + 1;
    let w_out_unpadded = (w_in.saturating_sub(1)) * sw + dw * (kw - 1) + opw + 1;
    assert!(
        h_out_unpadded > 2 * ph && w_out_unpadded > 2 * pw,
        "conv_transpose2d: padding larger than produced output dims",
    );
    let h_out = h_out_unpadded - 2 * ph;
    let w_out = w_out_unpadded - 2 * pw;
    let xs = x.as_slice();
    let ws = weight.as_slice();
    let mut out = vec![T::zero(); n * cout * h_out * w_out];
    for ni in 0..n {
        for g in 0..groups {
            for ic_in_g in 0..cin_per_g {
                let ic = g * cin_per_g + ic_in_g;
                for oc_in_g in 0..cout_per_g {
                    let oc = g * cout_per_g + oc_in_g;
                    for h in 0..h_in {
                        for w in 0..w_in {
                            let xv = xs[((ni * cin + ic) * h_in + h) * w_in + w];
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    // Transposed conv scatters: each input pixel
                                    // contributes to a kh×kw region in output.
                                    let oh_unpadded = h * sh + ki * dh;
                                    let ow_unpadded = w * sw + kj * dw;
                                    if oh_unpadded < ph || ow_unpadded < pw { continue; }
                                    let oh = oh_unpadded - ph;
                                    let ow = ow_unpadded - pw;
                                    if oh >= h_out || ow >= w_out { continue; }
                                    let wv = ws[(((ic * cout_per_g) + oc_in_g) * kh + ki) * kw + kj];
                                    let off = ((ni * cout + oc) * h_out + oh) * w_out + ow;
                                    out[off] = out[off] + xv * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    RefTensor::from_vec(out, Shape::from_dims(&[n, cout, h_out, w_out]))
}

// ---------- reshape --------------------------------------------------------

/// Reshape a tensor to a new shape. The element count of the new shape
/// must match the input. Data is unchanged; only the shape metadata
/// changes. Since `RefTensor` is always contiguous and row-major, reshape
/// is just a move of the underlying data with a new `Shape` stamp.
///
/// The bound is `T: Clone` rather than `T: Float` so this function also
/// works on integer index tensors (`RefTensor<u32>` and similar). Reshape
/// is fundamentally a shape-only operation and does not care about the
/// numeric properties of `T`.
pub fn reshape<T: Clone>(x: &RefTensor<T>, target: &Shape) -> RefTensor<T> {
    assert_eq!(
        x.elem_count(),
        target.elem_count(),
        "reshape: element count mismatch: source {} vs target {}",
        x.elem_count(),
        target.elem_count(),
    );
    // Reshape is a pure metadata operation on a contiguous row-major
    // tensor — the element order doesn't change, just how we interpret
    // the shape tuple. Share the underlying `Arc<[T]>` and give it a
    // new shape. No allocation, no memcpy.
    RefTensor::from_arc(x.as_arc().clone(), target.clone())
}

// ---------- reduce-sum-to-shape (inverse of broadcast) --------------------

/// Sum-reduce a tensor to a target shape. The target shape must be
/// broadcast-compatible into the source shape — i.e., `broadcast_to(t,
/// source_shape)` would work. This is the backward of `broadcast_to`.
///
/// Implementation: iterate every element of the input and accumulate it
/// into the corresponding output position, where output coordinates are
/// derived by mapping each input multi-index through the same rules
/// `broadcast_to` uses (but summing instead of copying).
pub fn reduce_sum_to<T: Float>(x: &RefTensor<T>, target: &Shape) -> RefTensor<T> {
    let src_dims = x.shape().dims();
    let dst_dims = target.dims();
    assert!(
        dst_dims.len() <= src_dims.len(),
        "reduce_sum_to: target rank {} exceeds source rank {}",
        dst_dims.len(),
        src_dims.len(),
    );
    let pad = src_dims.len() - dst_dims.len();
    for (i, &d) in dst_dims.iter().enumerate() {
        let s = src_dims[pad + i];
        assert!(
            d == s || d == 1,
            "reduce_sum_to: source dim {} ({s}) is not broadcast-compatible with target dim {i} ({d})",
            pad + i,
        );
    }

    let out_count: usize = if dst_dims.is_empty() {
        1
    } else {
        dst_dims.iter().product()
    };
    let mut out = vec![T::zero(); out_count];
    let src_strides = row_major_strides(src_dims);
    let dst_strides = row_major_strides(dst_dims);
    let src = x.as_slice();

    for src_flat in 0..src.len() {
        // Unflatten src_flat into multi-index.
        let mut remainder = src_flat;
        let mut src_multi = vec![0_usize; src_dims.len()];
        for i in 0..src_dims.len() {
            src_multi[i] = remainder / src_strides[i];
            remainder %= src_strides[i];
        }
        // Compute destination flat index: drop leading `pad` dims and
        // collapse any size-1 destination dims to coord 0.
        let mut dst_flat = 0_usize;
        for i in 0..dst_dims.len() {
            let coord = if dst_dims[i] == 1 {
                0
            } else {
                src_multi[pad + i]
            };
            dst_flat += coord * dst_strides[i];
        }
        out[dst_flat] = out[dst_flat] + src[src_flat];
    }
    RefTensor::from_vec(out, target.clone())
}

/// Max-reduce a tensor to a target shape — the max-symmetric counterpart
/// of [`reduce_sum_to`]. Same alignment rules, different reduction:
/// every input element contributes to its projected output position via
/// `max` instead of `+`. Output is initialized to `-infinity`.
pub fn reduce_max_to<T: Float>(x: &RefTensor<T>, target: &Shape) -> RefTensor<T> {
    let src_dims = x.shape().dims();
    let dst_dims = target.dims();
    assert!(
        dst_dims.len() <= src_dims.len(),
        "reduce_max_to: target rank {} exceeds source rank {}",
        dst_dims.len(),
        src_dims.len(),
    );
    let pad = src_dims.len() - dst_dims.len();
    for (i, &d) in dst_dims.iter().enumerate() {
        let s = src_dims[pad + i];
        assert!(
            d == s || d == 1,
            "reduce_max_to: source dim {} ({s}) is not broadcast-compatible with target dim {i} ({d})",
            pad + i,
        );
    }

    let out_count: usize = if dst_dims.is_empty() {
        1
    } else {
        dst_dims.iter().product()
    };
    let mut out = vec![T::neg_infinity(); out_count];
    let src_strides = row_major_strides(src_dims);
    let dst_strides = row_major_strides(dst_dims);
    let src = x.as_slice();

    for src_flat in 0..src.len() {
        let mut remainder = src_flat;
        let mut src_multi = vec![0_usize; src_dims.len()];
        for i in 0..src_dims.len() {
            src_multi[i] = remainder / src_strides[i];
            remainder %= src_strides[i];
        }
        let mut dst_flat = 0_usize;
        for i in 0..dst_dims.len() {
            let coord = if dst_dims[i] == 1 {
                0
            } else {
                src_multi[pad + i]
            };
            dst_flat += coord * dst_strides[i];
        }
        if src[src_flat] > out[dst_flat] {
            out[dst_flat] = src[src_flat];
        }
    }
    RefTensor::from_vec(out, target.clone())
}

/// Backward for [`reduce_max_to`]. Given the original input `x`
/// (shape S_in), the upstream gradient (shape S_target == the forward
/// `target` shape, broadcast-compatible into S_in), and the same
/// `target` shape, returns dL/dx of shape S_in. Routes the upstream
/// to the position(s) where `x` equals its per-window max; when
/// multiple positions tie, the gradient is split equally (fair-share
/// subgradient).
///
/// Implementation:
/// 1. Recompute `y = reduce_max_to(x, target)`.
/// 2. Build `mask`, shape S_in: 1 where x[i] == y[broadcast(i)], else 0.
/// 3. Compute `count = reduce_sum_to(mask, target)` — number of tied
///    positions per output cell.
/// 4. `count_safe = max(count, 1.0)` (degenerate empty-window guard).
/// 5. `scaled_upstream = upstream / count_safe` — shape S_target.
/// 6. `grad_x = broadcast(scaled_upstream) * mask` — shape S_in.
pub fn reduce_max_to_backward<T: Float>(
    x: &RefTensor<T>,
    upstream: &RefTensor<T>,
    target: &Shape,
) -> RefTensor<T> {
    let in_shape = x.shape().clone();
    let in_dims = in_shape.dims();

    // Step 1: recompute the forward max.
    let max_y = reduce_max_to(x, target);

    // Step 2: broadcast max_y to S_in and build the mask.
    let max_b = broadcast_to(&max_y, &in_shape);
    let n: usize = in_dims.iter().product();
    let x_data = x.as_slice();
    let max_data = max_b.as_slice();
    let mut mask_data = vec![T::zero(); n];
    for i in 0..n {
        if x_data[i] == max_data[i] {
            mask_data[i] = T::one();
        }
    }
    let mask = RefTensor::from_vec(mask_data.clone(), in_shape.clone());

    // Step 3: count ties per output cell.
    let count = reduce_sum_to(&mask, target);

    // Step 4: clamp count to >= 1 to avoid 0/0. A zero count would
    // mean no input position equaled the recomputed max, which can
    // only happen for empty windows or NaN inputs — defensive guard.
    let count_safe_data: Vec<T> = count
        .as_slice()
        .iter()
        .map(|&c| if c < T::one() { T::one() } else { c })
        .collect();
    let count_safe = RefTensor::from_vec(count_safe_data, count.shape().clone());

    // Step 5: scale upstream.
    let up_data = upstream.as_slice();
    let cs_data = count_safe.as_slice();
    let mut scaled_data = Vec::with_capacity(up_data.len());
    for (u, c) in up_data.iter().zip(cs_data.iter()) {
        scaled_data.push(*u / *c);
    }
    let scaled = RefTensor::from_vec(scaled_data, count_safe.shape().clone());

    // Step 6: broadcast and gate by mask.
    let scaled_b = broadcast_to(&scaled, &in_shape);
    let scaled_b_data = scaled_b.as_slice();
    let mut grad_data = vec![T::zero(); n];
    for i in 0..n {
        grad_data[i] = scaled_b_data[i] * mask_data[i];
    }
    RefTensor::from_vec(grad_data, in_shape)
}

// ---------- broadcasting to a target shape --------------------------------

/// Broadcast `x` to `target_shape` using NumPy rules: right-align, pad the
/// shorter shape with leading 1s, expand any size-1 dim of `x` to the
/// target's size in that dim. Panics if the shapes are not broadcast-
/// compatible.
///
/// This is the forward op used by the executor when realizing a
/// `BroadcastTo` graph node. It is also the backward rule for `SumAll`
/// and `MeanAll`.
pub fn broadcast_to<T: Float>(x: &RefTensor<T>, target_shape: &Shape) -> RefTensor<T> {
    let src_dims = x.shape().dims();
    let dst_dims = target_shape.dims();
    assert!(
        src_dims.len() <= dst_dims.len(),
        "broadcast_to: source rank {} exceeds target rank {}",
        src_dims.len(),
        dst_dims.len(),
    );
    let pad = dst_dims.len() - src_dims.len();
    for (i, &s) in src_dims.iter().enumerate() {
        let d = dst_dims[pad + i];
        assert!(
            s == d || s == 1,
            "broadcast_to: source dim {i} ({s}) cannot broadcast to target dim {} ({d})",
            pad + i,
        );
    }

    // Pure-padding fast path: the source either matches its aligned
    // target dim exactly (no expansion) or expands from 1 on a padding
    // prefix that is itself size 1. In that case the element count
    // hasn't changed and the row-major layout is identical — we can
    // reuse the source buffer (now an `Arc<[T]>`) with just a new
    // shape. No allocation, no memcpy.
    //
    // This is the hot path for matmul's rank-2 × rank-3 case, where a
    // weight `[dim, out_dim]` is "broadcast" to `[1, dim, out_dim]`
    // but the payload never changes. Without this shortcut we'd
    // allocate and fill a fresh buffer the size of every weight in
    // the model on every forward pass.
    let mut pure_pad = true;
    for i in 0..pad {
        if dst_dims[i] != 1 {
            pure_pad = false;
            break;
        }
    }
    if pure_pad {
        for i in 0..src_dims.len() {
            if src_dims[i] != dst_dims[pad + i] {
                pure_pad = false;
                break;
            }
        }
    }
    if pure_pad {
        return RefTensor::from_arc(x.as_arc().clone(), target_shape.clone());
    }

    let out_count: usize = if dst_dims.is_empty() {
        1
    } else {
        dst_dims.iter().product()
    };
    let mut out = vec![T::zero(); out_count];
    let src_strides = row_major_strides(src_dims);
    let dst_strides = row_major_strides(dst_dims);
    let src = x.as_slice();

    // Preallocate the output multi-index scratch space once; the old
    // version reallocated it on every single element, which for any
    // multi-gigabyte tensor turned into tens of millions of heap
    // allocations per forward pass.
    let mut out_multi = vec![0_usize; dst_dims.len()];
    for out_flat in 0..out_count {
        // Unflatten out_flat into multi-index.
        let mut remainder = out_flat;
        for i in 0..dst_dims.len() {
            out_multi[i] = remainder / dst_strides[i];
            remainder %= dst_strides[i];
        }
        // Compute the source flat index: pad-align and collapse size-1 dims.
        let mut src_flat = 0_usize;
        for i in 0..src_dims.len() {
            let coord = if src_dims[i] == 1 {
                0
            } else {
                out_multi[pad + i]
            };
            src_flat += coord * src_strides[i];
        }
        out[out_flat] = src[src_flat];
    }
    RefTensor::from_vec(out, target_shape.clone())
}

// ---------- dtype casts ----------------------------------------------------

/// Cast `f32 → f64` element-wise. Lossless except for underlying IEEE semantics.
pub fn cast_f32_to_f64(x: &RefTensor<f32>) -> RefTensor<f64> {
    let data: Vec<f64> = x.as_slice().iter().map(|&v| v as f64).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `f32 → bf16`. Rounds to nearest bf16 representable value.
pub fn cast_f32_to_bf16(x: &RefTensor<f32>) -> RefTensor<half::bf16> {
    let data: Vec<half::bf16> = x.as_slice().iter().map(|&v| half::bf16::from_f32(v)).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `f32 → f16`. Rounds to nearest f16 representable value.
pub fn cast_f32_to_f16(x: &RefTensor<f32>) -> RefTensor<half::f16> {
    let data: Vec<half::f16> = x.as_slice().iter().map(|&v| half::f16::from_f32(v)).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `f64 → f32`. Lossy for values outside f32's representable range.
pub fn cast_f64_to_f32(x: &RefTensor<f64>) -> RefTensor<f32> {
    let data: Vec<f32> = x.as_slice().iter().map(|&v| v as f32).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `f64 → bf16`.
pub fn cast_f64_to_bf16(x: &RefTensor<f64>) -> RefTensor<half::bf16> {
    let data: Vec<half::bf16> = x.as_slice().iter().map(|&v| half::bf16::from_f64(v)).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `f64 → f16`.
pub fn cast_f64_to_f16(x: &RefTensor<f64>) -> RefTensor<half::f16> {
    let data: Vec<half::f16> = x.as_slice().iter().map(|&v| half::f16::from_f64(v)).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `bf16 → f32`. Lossless (bf16 is a subset of f32).
pub fn cast_bf16_to_f32(x: &RefTensor<half::bf16>) -> RefTensor<f32> {
    let data: Vec<f32> = x.as_slice().iter().map(|&v| v.to_f32()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `bf16 → f64`.
pub fn cast_bf16_to_f64(x: &RefTensor<half::bf16>) -> RefTensor<f64> {
    let data: Vec<f64> = x.as_slice().iter().map(|&v| v.to_f64()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `bf16 → f16`. Lossy both ways (different mantissa/exponent layouts);
/// routes through `f32` which is lossless for both.
pub fn cast_bf16_to_f16(x: &RefTensor<half::bf16>) -> RefTensor<half::f16> {
    let data: Vec<half::f16> = x
        .as_slice()
        .iter()
        .map(|&v| half::f16::from_f32(v.to_f32()))
        .collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `f16 → f32`. Lossless (f16 is a subset of f32).
pub fn cast_f16_to_f32(x: &RefTensor<half::f16>) -> RefTensor<f32> {
    let data: Vec<f32> = x.as_slice().iter().map(|&v| v.to_f32()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `f16 → f64`.
pub fn cast_f16_to_f64(x: &RefTensor<half::f16>) -> RefTensor<f64> {
    let data: Vec<f64> = x.as_slice().iter().map(|&v| v.to_f64()).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `f16 → bf16`. Lossy both ways; routes through `f32`.
pub fn cast_f16_to_bf16(x: &RefTensor<half::f16>) -> RefTensor<half::bf16> {
    let data: Vec<half::bf16> = x
        .as_slice()
        .iter()
        .map(|&v| half::bf16::from_f32(v.to_f32()))
        .collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `u32 → f32`. Safe for the full `u32` range (f32 has 24-bit
/// mantissa so integers above 2^24 lose precision, but small label
/// indices and counts always round-trip losslessly).
pub fn cast_u32_to_f32(x: &RefTensor<u32>) -> RefTensor<f32> {
    let data: Vec<f32> = x.as_slice().iter().map(|&v| v as f32).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `u32 → f64`. Lossless — f64's 53-bit mantissa covers u32 exactly.
pub fn cast_u32_to_f64(x: &RefTensor<u32>) -> RefTensor<f64> {
    let data: Vec<f64> = x.as_slice().iter().map(|&v| v as f64).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `f32 → u32` via truncation toward zero. Values outside
/// `[0, u32::MAX]` produce implementation-defined results and should
/// not occur in well-formed graphs.
pub fn cast_f32_to_u32(x: &RefTensor<f32>) -> RefTensor<u32> {
    let data: Vec<u32> = x.as_slice().iter().map(|&v| v as u32).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Cast `f64 → u32` via truncation toward zero.
pub fn cast_f64_to_u32(x: &RefTensor<f64>) -> RefTensor<u32> {
    let data: Vec<u32> = x.as_slice().iter().map(|&v| v as u32).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

// ---------- gather / index_select by tensor --------------------------------

/// Index-select along `dim` using a 1-D `u32` index tensor. The output
/// shape is the same as `x` except dimension `dim` is replaced by
/// `indices.elem_count()`.
///
/// This is the graph-driven counterpart to [`index_select`], which takes
/// a `&[usize]` index slice directly. Both share the same semantics; the
/// tensor-based version exists because the graph executor receives a
/// full [`RefTensor<u32>`] as the index operand, matching what every real
/// backend (Candle, CUDA, Metal) stores for index tensors.
pub fn index_select_tensor<T, I>(
    x: &RefTensor<T>,
    dim: usize,
    indices: &RefTensor<I>,
) -> RefTensor<T>
where
    T: Clone + Default,
    I: num_traits::PrimInt + num_traits::ToPrimitive,
{
    let idx_dims = indices.shape().dims();
    assert_eq!(
        idx_dims.len(),
        1,
        "index_select_tensor: index tensor must be rank 1, got {idx_dims:?}",
    );
    // Convert the u32 (or other PrimInt) slice to a Vec<usize> once, then
    // reuse the existing `&[usize]`-based index_select. Keeping one
    // canonical implementation avoids reimplementing the stride math.
    let usize_indices: Vec<usize> = indices
        .as_slice()
        .iter()
        .map(|v| v.to_usize().expect("index_select_tensor: index -> usize conversion"))
        .collect();
    // index_select requires T: Float; we're generic over T: Clone+Default.
    // Inline the logic here to avoid dragging in the Float bound.
    let in_dims = x.shape().dims();
    assert!(
        dim < in_dims.len(),
        "index_select_tensor: dim {dim} out of bounds for shape {in_dims:?}",
    );
    let in_d = in_dims[dim];
    for (k, &idx) in usize_indices.iter().enumerate() {
        assert!(
            idx < in_d,
            "index_select_tensor: indices[{k}] = {idx} out of bounds for dim size {in_d}",
        );
    }
    let mut out_dims: Vec<usize> = in_dims.to_vec();
    out_dims[dim] = usize_indices.len();
    let out_count: usize = if out_dims.is_empty() {
        0
    } else {
        out_dims.iter().product()
    };
    let mut out: Vec<T> = vec![T::default(); out_count];

    let in_strides = row_major_strides(in_dims);
    let out_strides = row_major_strides(&out_dims);
    let src = x.as_slice();
    for out_flat in 0..out_count {
        let mut remainder = out_flat;
        let mut out_multi = vec![0_usize; out_dims.len()];
        for i in 0..out_dims.len() {
            out_multi[i] = remainder / out_strides[i];
            remainder %= out_strides[i];
        }
        let mut in_multi = out_multi.clone();
        in_multi[dim] = usize_indices[out_multi[dim]];
        let in_flat: usize = in_multi
            .iter()
            .zip(&in_strides)
            .map(|(&c, &s)| c * s)
            .sum();
        out[out_flat] = src[in_flat].clone();
    }
    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

/// N-dimensional gather along `dim`, PyTorch semantics. The `indices`
/// tensor must have the same rank as `x`. The output has the same shape
/// as `indices`. For each position `p` in the output:
///
/// `out[p] = x[p with p[dim] replaced by indices[p]]`
///
/// Example (dim = 1):
///
/// ```text
/// x:       [[1, 2, 3],            indices: [[0, 2],
///           [4, 5, 6]]                      [1, 0]]
///
/// out[0,0] = x[0, indices[0,0]] = x[0, 0] = 1
/// out[0,1] = x[0, indices[0,1]] = x[0, 2] = 3
/// out[1,0] = x[1, indices[1,0]] = x[1, 1] = 5
/// out[1,1] = x[1, indices[1,1]] = x[1, 0] = 4
/// ```
pub fn gather<T, I>(x: &RefTensor<T>, dim: usize, indices: &RefTensor<I>) -> RefTensor<T>
where
    T: Clone + Default,
    I: num_traits::PrimInt + num_traits::ToPrimitive,
{
    let x_dims = x.shape().dims();
    let idx_dims = indices.shape().dims();
    assert_eq!(
        x_dims.len(),
        idx_dims.len(),
        "gather: data and index must have the same rank, got {} vs {}",
        x_dims.len(),
        idx_dims.len(),
    );
    assert!(
        dim < x_dims.len(),
        "gather: dim {dim} out of bounds for data rank {}",
        x_dims.len(),
    );
    let out_count: usize = if idx_dims.is_empty() {
        1
    } else {
        idx_dims.iter().product()
    };
    let mut out: Vec<T> = vec![T::default(); out_count];

    let x_strides = row_major_strides(x_dims);
    let idx_strides = row_major_strides(idx_dims);
    let x_data = x.as_slice();
    let idx_data = indices.as_slice();

    for idx_flat in 0..out_count {
        // Unflatten idx_flat into multi-index in the index/output shape.
        let mut remainder = idx_flat;
        let mut multi = vec![0_usize; idx_dims.len()];
        for i in 0..idx_dims.len() {
            multi[i] = remainder / idx_strides[i];
            remainder %= idx_strides[i];
        }
        // Replace the `dim` coordinate with the index value.
        let idx_val = idx_data[idx_flat]
            .to_usize()
            .expect("gather: index -> usize conversion failed");
        assert!(
            idx_val < x_dims[dim],
            "gather: index value {idx_val} out of bounds for data dim size {}",
            x_dims[dim],
        );
        multi[dim] = idx_val;
        let x_flat: usize = multi.iter().zip(&x_strides).map(|(&c, &s)| c * s).sum();
        out[idx_flat] = x_data[x_flat].clone();
    }
    RefTensor::from_vec(out, indices.shape().clone())
}

// ---------- concat and slice ----------------------------------------------

/// Concatenate two tensors along `dim`. Both must have the same rank,
/// same dtype, and equal sizes in every dim except `dim`.
pub fn concat<T: Clone + Default>(
    a: &RefTensor<T>,
    b: &RefTensor<T>,
    dim: usize,
) -> RefTensor<T> {
    let ad = a.shape().dims();
    let bd = b.shape().dims();
    assert_eq!(ad.len(), bd.len(), "concat: rank mismatch");
    assert!(dim < ad.len(), "concat: dim out of bounds");
    for i in 0..ad.len() {
        if i != dim {
            assert_eq!(
                ad[i], bd[i],
                "concat: non-dim shape mismatch at dim {i}: {} vs {}",
                ad[i], bd[i],
            );
        }
    }
    let mut out_dims: Vec<usize> = ad.to_vec();
    out_dims[dim] = ad[dim] + bd[dim];
    let out_count: usize = out_dims.iter().product();
    let mut out: Vec<T> = vec![T::default(); out_count];
    let out_strides = row_major_strides(&out_dims);
    let a_strides = row_major_strides(ad);
    let b_strides = row_major_strides(bd);
    let a_data = a.as_slice();
    let b_data = b.as_slice();

    // Walk every output position and pick from a or b based on the dim coord.
    for out_flat in 0..out_count {
        let mut remainder = out_flat;
        let mut out_multi = vec![0_usize; out_dims.len()];
        for i in 0..out_dims.len() {
            out_multi[i] = remainder / out_strides[i];
            remainder %= out_strides[i];
        }
        let dim_coord = out_multi[dim];
        let (src_data, src_strides, src_dim_coord) = if dim_coord < ad[dim] {
            (a_data, &a_strides, dim_coord)
        } else {
            (b_data, &b_strides, dim_coord - ad[dim])
        };
        let mut src_multi = out_multi.clone();
        src_multi[dim] = src_dim_coord;
        let src_flat: usize = src_multi
            .iter()
            .zip(src_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();
        out[out_flat] = src_data[src_flat].clone();
    }
    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

/// Slice (narrow) a tensor along `dim`: take elements `[start, start+len)`.
pub fn slice<T: Clone + Default>(
    x: &RefTensor<T>,
    dim: usize,
    start: usize,
    len: usize,
) -> RefTensor<T> {
    let in_dims = x.shape().dims();
    assert!(dim < in_dims.len(), "slice: dim out of bounds");
    assert!(
        start + len <= in_dims[dim],
        "slice: [start={start}, len={len}) exceeds dim size {}",
        in_dims[dim],
    );
    let mut out_dims: Vec<usize> = in_dims.to_vec();
    out_dims[dim] = len;
    let out_count: usize = out_dims.iter().product();
    let mut out: Vec<T> = vec![T::default(); out_count];
    let in_strides = row_major_strides(in_dims);
    let out_strides = row_major_strides(&out_dims);
    let src = x.as_slice();
    for out_flat in 0..out_count {
        let mut remainder = out_flat;
        let mut out_multi = vec![0_usize; out_dims.len()];
        for i in 0..out_dims.len() {
            out_multi[i] = remainder / out_strides[i];
            remainder %= out_strides[i];
        }
        let mut in_multi = out_multi.clone();
        in_multi[dim] += start;
        let in_flat: usize = in_multi
            .iter()
            .zip(&in_strides)
            .map(|(&c, &s)| c * s)
            .sum();
        out[out_flat] = src[in_flat].clone();
    }
    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

// ---------- scalar-by-tensor ops -------------------------------------------

/// Add a scalar to every element: `y[i] = x[i] + c`.
pub fn add_scalar<T: Float>(x: &RefTensor<T>, c: f64) -> RefTensor<T> {
    let ct: T = cst(c);
    let data: Vec<T> = x.as_slice().iter().map(|&v| v + ct).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Multiply every element by a scalar: `y[i] = x[i] * c`.
pub fn mul_scalar<T: Float>(x: &RefTensor<T>, c: f64) -> RefTensor<T> {
    let ct: T = cst(c);
    let data: Vec<T> = x.as_slice().iter().map(|&v| v * ct).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Raise every element to an integer power: `y[i] = x[i]^n`. Negative
/// exponents give reciprocals. Uses repeated multiplication rather than
/// `exp(n * ln(x))`, so it's defined for negative and zero inputs.
pub fn powi<T: Float>(x: &RefTensor<T>, n: i32) -> RefTensor<T> {
    let data: Vec<T> = x.as_slice().iter().map(|&v| v.powi(n)).collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Clamp every element to `[min, max]`.
pub fn clamp<T: Float>(x: &RefTensor<T>, min: f64, max: f64) -> RefTensor<T> {
    let mn: T = cst(min);
    let mx: T = cst(max);
    let data: Vec<T> = x
        .as_slice()
        .iter()
        .map(|&v| {
            if v < mn {
                mn
            } else if v > mx {
                mx
            } else {
                v
            }
        })
        .collect();
    RefTensor::from_vec(data, x.shape().clone())
}

/// Element-wise maximum: `y[i] = max(a[i], b[i])`. Shapes must match.
pub fn maximum<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    assert_same_shape(a.shape(), b.shape(), "maximum");
    let data: Vec<T> = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&x, &y)| if x > y { x } else { y })
        .collect();
    RefTensor::from_vec(data, a.shape().clone())
}

/// Element-wise minimum: `y[i] = min(a[i], b[i])`.
pub fn minimum<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    assert_same_shape(a.shape(), b.shape(), "minimum");
    let data: Vec<T> = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&x, &y)| if x < y { x } else { y })
        .collect();
    RefTensor::from_vec(data, a.shape().clone())
}

// ---------- scatter_add and index_add --------------------------------------

/// Index-add (functional): returns a copy of `base` with `src` added to
/// positions along `dim` given by the 1-D index vector `indices`.
/// `out[..., indices[i], ...] = base[..., indices[i], ...] + src[..., i, ...]`.
///
/// This is the backward rule for `index_select`: the upstream gradient
/// (shape matches `src` here) gets scattered back to a zeros tensor of
/// the original data's shape, then added to `base` (which is zero in
/// the backward case).
pub fn index_add<T, I>(
    base: &RefTensor<T>,
    dim: usize,
    indices: &RefTensor<I>,
    src: &RefTensor<T>,
) -> RefTensor<T>
where
    T: Clone + std::ops::Add<Output = T>,
    I: num_traits::PrimInt + num_traits::ToPrimitive,
{
    let base_dims = base.shape().dims();
    let src_dims = src.shape().dims();
    assert_eq!(
        base_dims.len(),
        src_dims.len(),
        "index_add: base and src must have the same rank",
    );
    assert!(dim < base_dims.len(), "index_add: dim out of bounds");
    assert_eq!(
        indices.shape().dims().len(),
        1,
        "index_add: index must be rank 1",
    );
    let k = indices.shape().dims()[0];
    assert_eq!(src_dims[dim], k, "index_add: src dim {dim} must match index length");
    // Check all non-dim dims match.
    for i in 0..base_dims.len() {
        if i == dim {
            continue;
        }
        assert_eq!(
            base_dims[i], src_dims[i],
            "index_add: non-dim shapes must match (dim {i}: base {} vs src {})",
            base_dims[i], src_dims[i],
        );
    }

    // Start with a copy of base.
    let mut out: Vec<T> = base.as_slice().to_vec();
    let base_strides = row_major_strides(base_dims);
    let src_strides = row_major_strides(src_dims);
    let src_data = src.as_slice();
    let idx_data = indices.as_slice();

    // Iterate over every element of src and add it to the corresponding
    // position in out.
    for src_flat in 0..src_data.len() {
        // Unflatten src_flat in src.shape.
        let mut remainder = src_flat;
        let mut src_multi = vec![0_usize; src_dims.len()];
        for i in 0..src_dims.len() {
            src_multi[i] = remainder / src_strides[i];
            remainder %= src_strides[i];
        }
        // Translate the `dim` coordinate through indices.
        let idx_val = idx_data[src_multi[dim]]
            .to_usize()
            .expect("index_add: index -> usize conversion");
        assert!(
            idx_val < base_dims[dim],
            "index_add: indices[{}] = {} out of bounds for base dim {}",
            src_multi[dim], idx_val, base_dims[dim],
        );
        let mut base_multi = src_multi.clone();
        base_multi[dim] = idx_val;
        let base_flat: usize = base_multi
            .iter()
            .zip(&base_strides)
            .map(|(&c, &s)| c * s)
            .sum();
        out[base_flat] = out[base_flat].clone() + src_data[src_flat].clone();
    }
    RefTensor::from_vec(out, base.shape().clone())
}

/// Scatter-add (functional): returns a copy of `base` with values from
/// `src` accumulated at positions given by the N-D `indices` tensor
/// (which has the same shape as `src`). For each position `p` in `src`:
/// `out[p with dim ← indices[p]] += src[p]`.
pub fn scatter_add<T, I>(
    base: &RefTensor<T>,
    dim: usize,
    indices: &RefTensor<I>,
    src: &RefTensor<T>,
) -> RefTensor<T>
where
    T: Clone + std::ops::Add<Output = T>,
    I: num_traits::PrimInt + num_traits::ToPrimitive,
{
    let base_dims = base.shape().dims();
    let src_dims = src.shape().dims();
    assert_eq!(
        base_dims.len(),
        src_dims.len(),
        "scatter_add: base and src must have the same rank",
    );
    assert!(dim < base_dims.len(), "scatter_add: dim out of bounds");
    assert_eq!(
        indices.shape().dims(),
        src_dims,
        "scatter_add: indices and src must have the same shape",
    );

    let mut out: Vec<T> = base.as_slice().to_vec();
    let base_strides = row_major_strides(base_dims);
    let src_strides = row_major_strides(src_dims);
    let src_data = src.as_slice();
    let idx_data = indices.as_slice();

    for src_flat in 0..src_data.len() {
        // Unflatten src_flat in src.shape.
        let mut remainder = src_flat;
        let mut src_multi = vec![0_usize; src_dims.len()];
        for i in 0..src_dims.len() {
            src_multi[i] = remainder / src_strides[i];
            remainder %= src_strides[i];
        }
        // Replace the dim coordinate with the indexed value.
        let idx_val = idx_data[src_flat]
            .to_usize()
            .expect("scatter_add: index -> usize conversion");
        assert!(
            idx_val < base_dims[dim],
            "scatter_add: index value {idx_val} out of bounds for base dim {}",
            base_dims[dim],
        );
        let mut base_multi = src_multi.clone();
        base_multi[dim] = idx_val;
        let base_flat: usize = base_multi
            .iter()
            .zip(&base_strides)
            .map(|(&c, &s)| c * s)
            .sum();
        out[base_flat] = out[base_flat].clone() + src_data[src_flat].clone();
    }
    RefTensor::from_vec(out, base.shape().clone())
}

// ---------- indexing -------------------------------------------------------

/// Select slices from `x` along `dim` using `indices`. For a rank-N input,
/// the output has the same rank; only the size of dimension `dim` changes
/// from `x.shape()[dim]` to `indices.len()`. The selected values are copied
/// from `x[..., indices[k], ...]` to `out[..., k, ...]`.
///
/// Indices are plain `usize`, kept as a `&[usize]` rather than a tensor
/// because the reference backend does not yet carry a separate index-tensor
/// type. General `gather`/`scatter` with multi-dimensional index tensors
/// will land in a follow-up pass alongside `RefIndexTensor`.
pub fn index_select<T: Float>(
    x: &RefTensor<T>,
    dim: usize,
    indices: &[usize],
) -> RefTensor<T> {
    let in_dims = x.shape().dims();
    assert!(
        dim < in_dims.len(),
        "index_select: dim {dim} out of bounds for shape {in_dims:?}",
    );
    let in_d = in_dims[dim];
    for (k, &idx) in indices.iter().enumerate() {
        assert!(
            idx < in_d,
            "index_select: indices[{k}] = {idx} out of bounds for dim size {in_d}",
        );
    }
    let mut out_dims: Vec<usize> = in_dims.to_vec();
    out_dims[dim] = indices.len();
    let out_count: usize = if out_dims.is_empty() {
        0
    } else {
        out_dims.iter().product()
    };
    let mut out = vec![T::zero(); out_count];

    let in_strides = row_major_strides(in_dims);
    let out_strides = row_major_strides(&out_dims);

    let src = x.as_slice();
    for out_flat in 0..out_count {
        // Unflatten out_flat using out_strides to get the output multi-index.
        let mut remainder = out_flat;
        let mut out_multi = vec![0_usize; out_dims.len()];
        for i in 0..out_dims.len() {
            out_multi[i] = remainder / out_strides[i];
            remainder %= out_strides[i];
        }
        // Build the input multi-index: same as output except at `dim`,
        // where we translate through `indices`.
        let mut in_multi = out_multi.clone();
        in_multi[dim] = indices[out_multi[dim]];
        // Flatten the input multi-index.
        let in_flat: usize = in_multi
            .iter()
            .zip(&in_strides)
            .map(|(&c, &s)| c * s)
            .sum();
        out[out_flat] = src[in_flat];
    }

    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

/// Embedding lookup: for a rank-2 `table` of shape `[V, D]` (vocab size × hidden
/// dim) and a flat slice of token `ids`, produce a rank-2 tensor of shape
/// `[ids.len(), D]` where row `i` is `table[ids[i]]`.
///
/// Higher-rank id tensors (e.g. `[batch, seq]`) are handled by the caller
/// reshaping the output afterward. Keeping this function rank-2-in /
/// rank-2-out makes the semantics trivially obvious.
pub fn embedding<T: Float>(table: &RefTensor<T>, ids: &[usize]) -> RefTensor<T> {
    let table_dims = table.shape().dims();
    assert_eq!(
        table_dims.len(),
        2,
        "embedding: table must be rank 2, got shape {table_dims:?}",
    );
    let v = table_dims[0];
    let d = table_dims[1];
    let n = ids.len();
    let mut out = vec![T::zero(); n * d];
    let src = table.as_slice();
    for (i, &id) in ids.iter().enumerate() {
        assert!(
            id < v,
            "embedding: ids[{i}] = {id} out of bounds for vocab size {v}",
        );
        let src_start = id * d;
        let dst_start = i * d;
        out[dst_start..dst_start + d].copy_from_slice(&src[src_start..src_start + d]);
    }
    RefTensor::from_vec(out, Shape::from_dims(&[n, d]))
}

// ---------- broadcasting ---------------------------------------------------

/// Compute the broadcast shape of two shapes using NumPy rules: align from
/// the right, pad the shorter with 1s, and for each aligned dimension
/// either sizes match or exactly one is 1 (which expands to the other).
fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len().max(b.len());
    let a_pad = n - a.len();
    let b_pad = n - b.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ai = if i < a_pad { 1 } else { a[i - a_pad] };
        let bi = if i < b_pad { 1 } else { b[i - b_pad] };
        let oi = if ai == bi {
            ai
        } else if ai == 1 {
            bi
        } else if bi == 1 {
            ai
        } else {
            panic!(
                "broadcast_shape: incompatible shapes lhs={a:?} rhs={b:?} at axis {i}: {ai} vs {bi}",
            );
        };
        out.push(oi);
    }
    out
}

/// Given an output multi-index into a broadcast shape, compute the flat
/// index into a source tensor whose shape is right-aligned and may have
/// size-1 dims that collapse to coordinate 0.
fn broadcast_src_flat(
    out_multi: &[usize],
    src_dims: &[usize],
    src_strides: &[usize],
) -> usize {
    let out_rank = out_multi.len();
    let src_rank = src_dims.len();
    let offset = out_rank - src_rank;
    let mut flat = 0_usize;
    for i in 0..src_rank {
        let coord = if src_dims[i] == 1 {
            0
        } else {
            out_multi[offset + i]
        };
        flat += coord * src_strides[i];
    }
    flat
}

/// Shared implementation for broadcast-aware elementwise binary ops.
fn broadcast_binary<T, F>(a: &RefTensor<T>, b: &RefTensor<T>, f: F) -> RefTensor<T>
where
    T: Float,
    F: Fn(T, T) -> T,
{
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let out_dims = broadcast_shape(a_dims, b_dims);
    let out_count: usize = if out_dims.is_empty() {
        1
    } else {
        out_dims.iter().product()
    };
    let out_strides = row_major_strides(&out_dims);
    let a_strides = row_major_strides(a_dims);
    let b_strides = row_major_strides(b_dims);

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let mut out = vec![T::zero(); out_count];

    for out_flat in 0..out_count {
        // Unflatten out_flat into an out_multi index.
        let mut remainder = out_flat;
        let mut out_multi = vec![0_usize; out_dims.len()];
        for i in 0..out_dims.len() {
            out_multi[i] = remainder / out_strides[i];
            remainder %= out_strides[i];
        }
        let a_flat = broadcast_src_flat(&out_multi, a_dims, &a_strides);
        let b_flat = broadcast_src_flat(&out_multi, b_dims, &b_strides);
        out[out_flat] = f(a_data[a_flat], b_data[b_flat]);
    }
    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

/// Broadcast-aware element-wise addition.
pub fn broadcast_add<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    broadcast_binary(a, b, |x, y| x + y)
}

/// Broadcast-aware element-wise subtraction.
pub fn broadcast_sub<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    broadcast_binary(a, b, |x, y| x - y)
}

/// Broadcast-aware element-wise multiplication.
pub fn broadcast_mul<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    broadcast_binary(a, b, |x, y| x * y)
}

/// Broadcast-aware element-wise division.
pub fn broadcast_div<T: Float>(a: &RefTensor<T>, b: &RefTensor<T>) -> RefTensor<T> {
    broadcast_binary(a, b, |x, y| x / y)
}

// ---------- convolution and pooling ----------------------------------------

/// 2-D convolution on a rank-4 input, the textbook 7-nested-loop form.
/// Superseded by [`conv2d`] — this is the simpler variant with no bias
/// and no groups support, kept for the tests that exercised it before
/// the full op landed.
///
/// - Input shape: `[N, C_in, H, W]`
/// - Kernel shape: `[C_out, C_in, kH, kW]`
/// - Output shape: `[N, C_out, H_out, W_out]` where
///   `H_out = (H + 2*padding - kH) / stride + 1` and likewise for W.
pub fn conv2d_simple<T: Float>(
    x: &RefTensor<T>,
    kernel: &RefTensor<T>,
    stride: usize,
    padding: usize,
) -> RefTensor<T> {
    let xd = x.shape().dims();
    let kd = kernel.shape().dims();
    assert_eq!(xd.len(), 4, "conv2d: input must be rank 4, got {xd:?}");
    assert_eq!(kd.len(), 4, "conv2d: kernel must be rank 4, got {kd:?}");
    let (n, c_in, h, w) = (xd[0], xd[1], xd[2], xd[3]);
    let (c_out, c_in_k, kh, kw) = (kd[0], kd[1], kd[2], kd[3]);
    assert_eq!(
        c_in, c_in_k,
        "conv2d: input C_in ({c_in}) != kernel C_in ({c_in_k})",
    );
    assert!(stride > 0, "conv2d: stride must be positive");

    let h_out = (h + 2 * padding).saturating_sub(kh) / stride + 1;
    let w_out = (w + 2 * padding).saturating_sub(kw) / stride + 1;

    let x_data = x.as_slice();
    let k_data = kernel.as_slice();
    let mut out = vec![T::zero(); n * c_out * h_out * w_out];

    // Strides for the four dimensions (contiguous row-major).
    let x_stride_n = c_in * h * w;
    let x_stride_c = h * w;
    let x_stride_h = w;
    let k_stride_out = c_in * kh * kw;
    let k_stride_in = kh * kw;
    let k_stride_h = kw;
    let o_stride_n = c_out * h_out * w_out;
    let o_stride_c = h_out * w_out;
    let o_stride_h = w_out;

    for b in 0..n {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = T::zero();
                    for ic in 0..c_in {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                // Input coordinates (may lie outside padding).
                                let ih_signed = (oh * stride) as isize
                                    + ky as isize
                                    - padding as isize;
                                let iw_signed = (ow * stride) as isize
                                    + kx as isize
                                    - padding as isize;
                                if ih_signed < 0
                                    || iw_signed < 0
                                    || (ih_signed as usize) >= h
                                    || (iw_signed as usize) >= w
                                {
                                    // Zero-padded region contributes 0; skip.
                                    continue;
                                }
                                let ih = ih_signed as usize;
                                let iw = iw_signed as usize;
                                let x_flat = b * x_stride_n
                                    + ic * x_stride_c
                                    + ih * x_stride_h
                                    + iw;
                                let k_flat = oc * k_stride_out
                                    + ic * k_stride_in
                                    + ky * k_stride_h
                                    + kx;
                                acc = acc + x_data[x_flat] * k_data[k_flat];
                            }
                        }
                    }
                    let out_flat = b * o_stride_n
                        + oc * o_stride_c
                        + oh * o_stride_h
                        + ow;
                    out[out_flat] = acc;
                }
            }
        }
    }
    RefTensor::from_vec(out, Shape::from_dims(&[n, c_out, h_out, w_out]))
}

/// 2-D max pooling on a rank-4 input, with no padding. For each
/// `kernel_size × kernel_size` window stepped by `stride`, emit the
/// maximum value in the window.
///
/// - Input shape: `[N, C, H, W]`
/// - Output shape: `[N, C, H_out, W_out]` where
///   `H_out = (H - kernel_size) / stride + 1` and likewise for W.
pub fn max_pool2d<T: Float>(
    x: &RefTensor<T>,
    kernel_size: usize,
    stride: usize,
) -> RefTensor<T> {
    let xd = x.shape().dims();
    assert_eq!(xd.len(), 4, "max_pool2d: input must be rank 4, got {xd:?}");
    let (n, c, h, w) = (xd[0], xd[1], xd[2], xd[3]);
    assert!(kernel_size > 0 && stride > 0, "max_pool2d: zero kernel_size or stride");
    assert!(kernel_size <= h && kernel_size <= w, "max_pool2d: kernel larger than input");

    let h_out = (h - kernel_size) / stride + 1;
    let w_out = (w - kernel_size) / stride + 1;

    let x_data = x.as_slice();
    let mut out = vec![T::zero(); n * c * h_out * w_out];

    let x_stride_n = c * h * w;
    let x_stride_c = h * w;
    let x_stride_h = w;
    let o_stride_n = c * h_out * w_out;
    let o_stride_c = h_out * w_out;
    let o_stride_h = w_out;

    for b in 0..n {
        for cc in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut best = T::neg_infinity();
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let ih = oh * stride + ky;
                            let iw = ow * stride + kx;
                            let v = x_data[b * x_stride_n
                                + cc * x_stride_c
                                + ih * x_stride_h
                                + iw];
                            if v > best {
                                best = v;
                            }
                        }
                    }
                    let out_flat = b * o_stride_n
                        + cc * o_stride_c
                        + oh * o_stride_h
                        + ow;
                    out[out_flat] = best;
                }
            }
        }
    }
    RefTensor::from_vec(out, Shape::from_dims(&[n, c, h_out, w_out]))
}

// ---------- reshaping ------------------------------------------------------

/// Permute the axes of a tensor. `out.shape[i] = x.shape[axes[i]]`. The
/// `axes` slice must be a permutation of `0..rank`.
///
/// This is the general N-D rearrangement primitive that [`transpose_last_two`]
/// is a rank-2-final specialization of. It physically reorders the
/// elements so the output is row-major contiguous in its new shape.
pub fn permute<T: Clone + Default>(x: &RefTensor<T>, axes: &[usize]) -> RefTensor<T> {
    let in_dims = x.shape().dims();
    let rank = in_dims.len();
    assert_eq!(
        axes.len(),
        rank,
        "permute: axes length {} must match tensor rank {}",
        axes.len(),
        rank,
    );
    let mut seen = vec![false; rank];
    for &ax in axes {
        assert!(ax < rank, "permute: axis {ax} out of bounds");
        assert!(!seen[ax], "permute: duplicate axis {ax}");
        seen[ax] = true;
    }
    let out_dims: Vec<usize> = axes.iter().map(|&ax| in_dims[ax]).collect();
    let out_count: usize = if out_dims.is_empty() {
        1
    } else {
        out_dims.iter().product()
    };
    let mut out: Vec<T> = vec![T::default(); out_count];
    let in_strides = row_major_strides(in_dims);
    let out_strides = row_major_strides(&out_dims);
    let src = x.as_slice();

    for out_flat in 0..out_count {
        // Unflatten out_flat into multi-index in the output shape.
        let mut remainder = out_flat;
        let mut out_multi = vec![0_usize; rank];
        for i in 0..rank {
            out_multi[i] = remainder / out_strides[i];
            remainder %= out_strides[i];
        }
        // Build input multi-index: `in_multi[axes[i]] = out_multi[i]`.
        let mut in_multi = vec![0_usize; rank];
        for i in 0..rank {
            in_multi[axes[i]] = out_multi[i];
        }
        let in_flat: usize = in_multi
            .iter()
            .zip(&in_strides)
            .map(|(&c, &s)| c * s)
            .sum();
        out[out_flat] = src[in_flat].clone();
    }
    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

/// Transpose the last two dims of a tensor of any rank ≥ 2. Leaves all
/// leading dims unchanged. For shape `[..., m, n]` → `[..., n, m]`, with
/// every batch slice transposed independently.
///
/// This is what MatMul's backward rule needs for batched operands
/// (`dA = dY @ B^T`, `dB = A^T @ dY` — the transposes here are always
/// over the last two dims regardless of batch rank).
pub fn transpose_last_two<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let dims = x.shape().dims();
    assert!(
        dims.len() >= 2,
        "transpose_last_two: input must be rank ≥ 2, got shape {dims:?}",
    );
    let rank = dims.len();
    // Shortcut for the rank-2 case.
    if rank == 2 {
        return transpose_2d(x);
    }
    let m = dims[rank - 2];
    let n = dims[rank - 1];
    let batch_count: usize = dims[..rank - 2].iter().product::<usize>().max(1);

    let mut out_dims: Vec<usize> = dims[..rank - 2].to_vec();
    out_dims.push(n);
    out_dims.push(m);
    let src = x.as_slice();
    let mut out = vec![T::zero(); batch_count * m * n];
    let batch_stride = m * n;
    for batch in 0..batch_count {
        let off = batch * batch_stride;
        for i in 0..m {
            for j in 0..n {
                out[off + j * m + i] = src[off + i * n + j];
            }
        }
    }
    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

/// Transpose of a rank-2 tensor: `y[j, i] = x[i, j]`.
///
/// Rank-2 only. General N-D transpose is deferred until there is a concrete
/// validation need — the specialization avoids introducing stride/permute
/// handling into the reference before it is required.
pub fn transpose_2d<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let dims = x.shape().dims();
    assert_eq!(
        dims.len(),
        2,
        "transpose_2d: input must be rank 2, got shape {dims:?}",
    );
    let (m, n) = (dims[0], dims[1]);
    let src = x.as_slice();
    let mut out = vec![T::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = src[i * n + j];
        }
    }
    RefTensor::from_vec(out, Shape::from_dims(&[n, m]))
}

// ---------- compositions ---------------------------------------------------

/// Softmax along the last dimension, in the numerically stable form
/// `softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))`.
///
/// Specialized to the last dimension so no broadcasting infrastructure is
/// needed. For an input of shape `[..., n]` the output has the same shape
/// and each length-`n` slice along the last axis is a probability
/// distribution.
pub fn softmax_last_dim<T: Float>(x: &RefTensor<T>) -> RefTensor<T> {
    let dims = x.shape().dims();
    assert!(
        !dims.is_empty(),
        "softmax_last_dim: input must be rank >= 1",
    );
    let last = dims[dims.len() - 1];
    let row_count: usize = if dims.len() == 1 {
        1
    } else {
        dims[..dims.len() - 1].iter().product()
    };

    let src = x.as_slice();
    let mut out = vec![T::zero(); src.len()];

    for r in 0..row_count {
        let start = r * last;
        // Row max for numerical stability.
        let mut row_max = T::neg_infinity();
        for i in 0..last {
            let v = src[start + i];
            if v > row_max {
                row_max = v;
            }
        }
        // exp(x - max) and accumulate denominator.
        let mut row_sum = T::zero();
        for i in 0..last {
            let e = (src[start + i] - row_max).exp();
            out[start + i] = e;
            row_sum = row_sum + e;
        }
        // Normalize.
        for i in 0..last {
            out[start + i] = out[start + i] / row_sum;
        }
    }

    RefTensor::from_vec(out, x.shape().clone())
}

/// Softmax-last-dim backward: given the forward softmax output `y` and
/// upstream gradient `g`, compute `dL/dx = y * (g - sum(y * g, last_dim,
/// keepdim=true))`. Implemented as a single textbook loop per row.
pub fn softmax_last_dim_backward<T: Float>(
    y: &RefTensor<T>,
    g: &RefTensor<T>,
) -> RefTensor<T> {
    let dims = y.shape().dims();
    assert_eq!(
        dims,
        g.shape().dims(),
        "softmax_last_dim_backward: shape mismatch",
    );
    assert!(
        !dims.is_empty(),
        "softmax_last_dim_backward: input must be rank >= 1",
    );
    let last = dims[dims.len() - 1];
    let row_count: usize = if dims.len() == 1 {
        1
    } else {
        dims[..dims.len() - 1].iter().product()
    };
    let y_data = y.as_slice();
    let g_data = g.as_slice();
    let mut out = vec![T::zero(); y_data.len()];

    for r in 0..row_count {
        let start = r * last;
        // dot = sum_i(y[i] * g[i])
        let mut dot = T::zero();
        for i in 0..last {
            dot = dot + y_data[start + i] * g_data[start + i];
        }
        // out[i] = y[i] * (g[i] - dot)
        for i in 0..last {
            out[start + i] = y_data[start + i] * (g_data[start + i] - dot);
        }
    }
    RefTensor::from_vec(out, y.shape().clone())
}

/// Layer-norm-last-dim backward: given the original input `x` and
/// upstream gradient `g`, compute `dL/dx`. Uses the canonical formula:
///
/// `dL/dx_i = rstd * (g_i - mean(g) - y_i * mean(g * y))`
///
/// where `y_i = (x_i - μ) / σ` is the normalized value at position `i`,
/// `μ = mean(x)`, `σ = sqrt(var(x) + eps)`, and `rstd = 1/σ`. Statistics
/// are computed per-row along the last dim. Affine (gamma/beta) is not
/// part of this op — callers apply it as a separate mul+add.
pub fn layer_norm_last_dim_backward<T: Float>(
    x: &RefTensor<T>,
    g: &RefTensor<T>,
    eps: f64,
) -> RefTensor<T> {
    let dims = x.shape().dims();
    assert_eq!(
        dims,
        g.shape().dims(),
        "layer_norm_last_dim_backward: shape mismatch",
    );
    assert!(!dims.is_empty(), "layer_norm_last_dim_backward: rank >= 1");
    let last = dims[dims.len() - 1];
    let row_count: usize = if dims.len() == 1 {
        1
    } else {
        dims[..dims.len() - 1].iter().product()
    };
    let x_data = x.as_slice();
    let g_data = g.as_slice();
    let mut out = vec![T::zero(); x_data.len()];
    let n = cst::<T>(last as f64);
    let eps_t: T = cst(eps);

    for r in 0..row_count {
        let start = r * last;
        // Recompute mean and variance.
        let mut mean = T::zero();
        for i in 0..last {
            mean = mean + x_data[start + i];
        }
        mean = mean / n;
        let mut var = T::zero();
        for i in 0..last {
            let d = x_data[start + i] - mean;
            var = var + d * d;
        }
        var = var / n;
        let rstd = T::one() / (var + eps_t).sqrt();
        // Compute mean(g) and mean(g * y) where y = (x - μ) * rstd.
        let mut mean_g = T::zero();
        let mut mean_g_y = T::zero();
        for i in 0..last {
            let yi = (x_data[start + i] - mean) * rstd;
            mean_g = mean_g + g_data[start + i];
            mean_g_y = mean_g_y + g_data[start + i] * yi;
        }
        mean_g = mean_g / n;
        mean_g_y = mean_g_y / n;
        // grad_x_i = rstd * (g_i - mean_g - y_i * mean_g_y)
        for i in 0..last {
            let yi = (x_data[start + i] - mean) * rstd;
            out[start + i] = rstd * (g_data[start + i] - mean_g - yi * mean_g_y);
        }
    }
    RefTensor::from_vec(out, x.shape().clone())
}

/// Layer normalization along the last dimension, without affine parameters.
///
/// Computes `(x - mean) / sqrt(var + eps)` where the mean and variance are
/// taken along the last dimension of each row. Variance uses the biased
/// estimator (divides by `n`, not `n - 1`), matching PyTorch's `LayerNorm`.
///
/// Affine parameters (gamma/beta) are applied as a separate `mul + add` step
/// by the caller. Keeping them out of this function makes each primitive
/// easier to validate in isolation. `eps` is taken as `f64` and converted to
/// `T` internally so callers do not need to express the epsilon in the
/// target dtype.
pub fn layer_norm_last_dim<T: Float>(x: &RefTensor<T>, eps: f64) -> RefTensor<T> {
    let dims = x.shape().dims();
    assert!(
        !dims.is_empty(),
        "layer_norm_last_dim: input must be rank >= 1",
    );
    let last = dims[dims.len() - 1];
    assert!(
        last > 0,
        "layer_norm_last_dim: last dim must be non-zero",
    );
    let row_count: usize = if dims.len() == 1 {
        1
    } else {
        dims[..dims.len() - 1].iter().product()
    };
    let n: T = cst(last as f64);
    let eps_t: T = cst(eps);
    let one = T::one();

    let src = x.as_slice();
    let mut out = vec![T::zero(); src.len()];

    for r in 0..row_count {
        let start = r * last;
        // Row mean.
        let mut mean = T::zero();
        for i in 0..last {
            mean = mean + src[start + i];
        }
        mean = mean / n;
        // Row (biased) variance.
        let mut var = T::zero();
        for i in 0..last {
            let d = src[start + i] - mean;
            var = var + d * d;
        }
        var = var / n;
        // Normalize.
        let rstd = one / (var + eps_t).sqrt();
        for i in 0..last {
            out[start + i] = (src[start + i] - mean) * rstd;
        }
    }

    RefTensor::from_vec(out, x.shape().clone())
}

/// Root-mean-square normalization along the last dimension, no affine
/// parameters. Formula:
///   y = x / sqrt(mean(x², last) + eps)
/// See [`Op::RmsNormLastDim`](../../fuel_graph/enum.Op.html) for
/// rationale. Textbook reference — a loop per row, no SIMD.
pub fn rms_norm_last_dim<T: Float>(x: &RefTensor<T>, eps: f64) -> RefTensor<T> {
    let dims = x.shape().dims();
    assert!(
        !dims.is_empty(),
        "rms_norm_last_dim: input must be rank >= 1",
    );
    let last = dims[dims.len() - 1];
    assert!(
        last > 0,
        "rms_norm_last_dim: last dim must be non-zero",
    );
    let row_count: usize = if dims.len() == 1 {
        1
    } else {
        dims[..dims.len() - 1].iter().product()
    };
    let n: T = cst(last as f64);
    let eps_t: T = cst(eps);
    let one = T::one();

    let src = x.as_slice();
    let mut out = vec![T::zero(); src.len()];

    for r in 0..row_count {
        let start = r * last;
        // mean(x²)
        let mut mean_sq = T::zero();
        for i in 0..last {
            let v = src[start + i];
            mean_sq = mean_sq + v * v;
        }
        mean_sq = mean_sq / n;
        // 1/sqrt(mean_sq + eps)
        let rrms = one / (mean_sq + eps_t).sqrt();
        for i in 0..last {
            out[start + i] = src[start + i] * rrms;
        }
    }

    RefTensor::from_vec(out, x.shape().clone())
}

/// Fused backward for [`rms_norm_last_dim`]. Closed-form gradient:
///
/// ```text
///   let s       = sum_i(g_y_i * x_i)
///   let mean_sq = mean(x²)
///   grad_x_j    = r_rms * (g_y_j - x_j * s / (n * (mean_sq + eps)))
///                         where r_rms = 1 / sqrt(mean_sq + eps)
/// ```
pub fn rms_norm_last_dim_backward<T: Float>(
    x: &RefTensor<T>,
    g_y: &RefTensor<T>,
    eps: f64,
) -> RefTensor<T> {
    let dims = x.shape().dims();
    assert!(
        !dims.is_empty(),
        "rms_norm_last_dim_backward: input must be rank >= 1",
    );
    assert_eq!(dims, g_y.shape().dims(), "rms_norm_last_dim_backward: shape mismatch");
    let last = dims[dims.len() - 1];
    let row_count: usize = if dims.len() == 1 {
        1
    } else {
        dims[..dims.len() - 1].iter().product()
    };
    let n: T = cst(last as f64);
    let eps_t: T = cst(eps);
    let one = T::one();

    let xs = x.as_slice();
    let gs = g_y.as_slice();
    let mut out = vec![T::zero(); xs.len()];

    for r in 0..row_count {
        let off = r * last;
        let mut sum_sq = T::zero();
        let mut sum_gx = T::zero();
        for i in 0..last {
            let xi = xs[off + i];
            let gi = gs[off + i];
            sum_sq = sum_sq + xi * xi;
            sum_gx = sum_gx + gi * xi;
        }
        let mean_sq = sum_sq / n;
        let denom_sq = mean_sq + eps_t;
        let r_rms = one / denom_sq.sqrt();
        let coeff = sum_gx / (n * denom_sq);
        for i in 0..last {
            let xi = xs[off + i];
            let gi = gs[off + i];
            out[off + i] = r_rms * (gi - xi * coeff);
        }
    }

    RefTensor::from_vec(out, x.shape().clone())
}

/// Fused rotary position embedding. `x` has shape `[..., seq, head_dim]`;
/// `cos`/`sin` have shape `[seq, head_dim]` and broadcast across leading
/// dims. head_dim must be even. See the `Op::Rope` docs for the formula.
pub fn rope<T: Float>(
    x: &RefTensor<T>,
    cos: &RefTensor<T>,
    sin: &RefTensor<T>,
) -> RefTensor<T> {
    let x_dims = x.shape().dims();
    let rank = x_dims.len();
    assert!(rank >= 2, "rope: input rank must be >= 2, got {x_dims:?}");
    let seq = x_dims[rank - 2];
    let head_dim = x_dims[rank - 1];
    assert!(head_dim % 2 == 0, "rope: head_dim must be even, got {head_dim}");
    let cos_dims = cos.shape().dims();
    let sin_dims = sin.shape().dims();
    assert_eq!(cos_dims, &[seq, head_dim], "rope: cos shape mismatch");
    assert_eq!(sin_dims, &[seq, head_dim], "rope: sin shape mismatch");

    let half = head_dim / 2;
    let outer: usize = x_dims[..rank - 2].iter().product();

    let xs = x.as_slice();
    let cs = cos.as_slice();
    let ss = sin.as_slice();
    let mut out = vec![T::zero(); xs.len()];

    for o in 0..outer {
        for s in 0..seq {
            let row_off = (o * seq + s) * head_dim;
            let table_off = s * head_dim;
            for i in 0..half {
                let x0 = xs[row_off + i];
                let x1 = xs[row_off + i + half];
                let c0 = cs[table_off + i];
                let s0 = ss[table_off + i];
                let c1 = cs[table_off + i + half];
                let s1 = ss[table_off + i + half];
                out[row_off + i] = x0 * c0 - x1 * s0;
                out[row_off + i + half] = x1 * c1 + x0 * s1;
            }
        }
    }

    RefTensor::from_vec(out, x.shape().clone())
}

// ---------- tests -----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f32>, dims: &[usize]) -> RefTensor<f32> {
        RefTensor::from_vec(data, Shape::from_dims(dims))
    }

    // ----- unary -----

    #[test]
    fn neg_flips_sign() {
        let x = t(vec![-1.0, 2.0, -3.0, 4.0], &[4]);
        let y = neg(&x);
        assert_eq!(y.as_slice(), &[1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn relu_clamps_negatives_to_zero() {
        let x = t(vec![-1.0, 0.0, 2.0, -3.5, 4.0], &[5]);
        let y = relu(&x);
        assert_eq!(y.as_slice(), &[0.0, 0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn sqr_squares_each_element() {
        let x = t(vec![-2.0, 0.0, 3.0, 4.0], &[4]);
        let y = sqr(&x);
        assert_eq!(y.as_slice(), &[4.0, 0.0, 9.0, 16.0]);
    }

    #[test]
    fn sqrt_of_squares() {
        let x = t(vec![4.0, 9.0, 16.0, 25.0], &[4]);
        let y = sqrt(&x);
        assert_eq!(y.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn exp_of_zero_is_one() {
        let x = t(vec![0.0, 1.0], &[2]);
        let y = exp(&x);
        assert!((y.as_slice()[0] - 1.0).abs() < 1e-6);
        assert!((y.as_slice()[1] - std::f32::consts::E).abs() < 1e-6);
    }

    #[test]
    fn sign_returns_minus_one_zero_one() {
        let x = t(vec![-5.0, 0.0, 3.0, -0.0, 2.5], &[5]);
        let y = sign(&x);
        assert_eq!(y.as_slice(), &[-1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    // ----- binary -----

    #[test]
    fn add_is_pointwise() {
        let a = t(vec![1.0, 2.0, 3.0], &[3]);
        let b = t(vec![4.0, 5.0, 6.0], &[3]);
        let c = add(&a, &b);
        assert_eq!(c.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn sub_is_pointwise() {
        let a = t(vec![10.0, 20.0, 30.0], &[3]);
        let b = t(vec![1.0, 2.0, 3.0], &[3]);
        let c = sub(&a, &b);
        assert_eq!(c.as_slice(), &[9.0, 18.0, 27.0]);
    }

    #[test]
    fn mul_is_pointwise() {
        let a = t(vec![2.0, 3.0, 4.0], &[3]);
        let b = t(vec![5.0, 6.0, 7.0], &[3]);
        let c = mul(&a, &b);
        assert_eq!(c.as_slice(), &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn div_is_pointwise() {
        let a = t(vec![10.0, 20.0, 30.0], &[3]);
        let b = t(vec![2.0, 4.0, 5.0], &[3]);
        let c = div(&a, &b);
        assert_eq!(c.as_slice(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn add_panics_on_shape_mismatch() {
        let a = t(vec![1.0, 2.0, 3.0], &[3]);
        let b = t(vec![1.0, 2.0], &[2]);
        let _ = add(&a, &b);
    }

    // ----- reductions -----

    #[test]
    fn sum_all_sums_everything() {
        let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let y = sum_all(&x);
        assert_eq!(y.as_slice(), &[15.0]);
        assert_eq!(y.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn max_all_finds_the_maximum() {
        let x = t(vec![3.0, -1.0, 7.0, 2.0, 5.0], &[5]);
        let y = max_all(&x);
        assert_eq!(y.as_slice(), &[7.0]);
    }

    #[test]
    fn min_all_finds_the_minimum() {
        let x = t(vec![3.0, -1.0, 7.0, -4.0, 5.0], &[5]);
        let y = min_all(&x);
        assert_eq!(y.as_slice(), &[-4.0]);
    }

    // ----- matmul -----

    #[test]
    fn matmul_2x3_by_3x2_hand_computed() {
        // A = [[1, 2, 3],
        //      [4, 5, 6]]    shape (2, 3)
        let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // B = [[7,  8],
        //      [9, 10],
        //      [11,12]]      shape (3, 2)
        let b = t(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        // Expected:
        //   [[1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12],
        //    [4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12]]
        // = [[58, 64], [139, 154]]
        let c = matmul_2d(&a, &b);
        assert_eq!(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
        assert_eq!(c.shape().dims(), &[2, 2]);
    }

    #[test]
    fn matmul_identity_is_noop() {
        // 3x3 identity times an arbitrary 3x3 matrix gives the matrix back.
        let eye = t(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
        );
        let m = t(
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[3, 3],
        );
        let c = matmul_2d(&eye, &m);
        assert_eq!(c.as_slice(), m.as_slice());
    }

    #[test]
    #[should_panic(expected = "inner dim mismatch")]
    fn matmul_panics_on_inner_dim_mismatch() {
        let a = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(vec![1.0, 2.0, 3.0], &[3, 1]);
        let _ = matmul_2d(&a, &b);
    }

    // ----- additional unary -----

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn log_of_exp_is_identity() {
        // log(exp(x)) should recover x within a small tolerance.
        let x = t(vec![-2.0, -0.5, 0.0, 0.5, 2.0], &[5]);
        let y = log(&exp(&x));
        for (&a, &b) in x.as_slice().iter().zip(y.as_slice()) {
            assert!(approx_eq(a, b, 1e-6), "expected {a}, got {b}");
        }
    }

    #[test]
    fn sin_cos_pythagorean_identity() {
        // sin^2 + cos^2 == 1 for every element.
        let x = t(vec![-1.5, -0.5, 0.0, 0.5, 1.5, 3.0], &[6]);
        let s = sin(&x);
        let c = cos(&x);
        for i in 0..x.elem_count() {
            let v = s.as_slice()[i].powi(2) + c.as_slice()[i].powi(2);
            assert!(approx_eq(v, 1.0, 1e-6), "sin^2+cos^2 = {v} at index {i}");
        }
    }

    #[test]
    fn abs_drops_sign() {
        let x = t(vec![-3.0, -0.0, 0.0, 4.5, -7.5], &[5]);
        let y = abs(&x);
        assert_eq!(y.as_slice(), &[3.0, 0.0, 0.0, 4.5, 7.5]);
    }

    #[test]
    fn recip_of_recip_is_identity() {
        let x = t(vec![1.0, 2.0, 4.0, -5.0, 10.0], &[5]);
        let y = recip(&recip(&x));
        for (&a, &b) in x.as_slice().iter().zip(y.as_slice()) {
            assert!(approx_eq(a, b, 1e-6));
        }
    }

    #[test]
    fn tanh_of_zero_is_zero_and_bounded() {
        let x = t(vec![-100.0, -1.0, 0.0, 1.0, 100.0], &[5]);
        let y = tanh(&x);
        assert!(approx_eq(y.as_slice()[2], 0.0, 1e-6));
        // Large positive → 1, large negative → -1
        assert!(approx_eq(y.as_slice()[0], -1.0, 1e-6));
        assert!(approx_eq(y.as_slice()[4], 1.0, 1e-6));
        // Every output in [-1, 1]
        for &v in y.as_slice() {
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn floor_and_ceil_bracket_input() {
        let x = t(vec![-1.5, -0.5, 0.0, 0.5, 1.5, 2.9], &[6]);
        let lo = floor(&x);
        let hi = ceil(&x);
        for i in 0..x.elem_count() {
            assert!(lo.as_slice()[i] <= x.as_slice()[i]);
            assert!(hi.as_slice()[i] >= x.as_slice()[i]);
        }
        assert_eq!(lo.as_slice(), &[-2.0, -1.0, 0.0, 0.0, 1.0, 2.0]);
        assert_eq!(hi.as_slice(), &[-1.0, -0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn sigmoid_of_zero_is_half_and_bounded() {
        let x = t(vec![-100.0, -1.0, 0.0, 1.0, 100.0], &[5]);
        let y = sigmoid(&x);
        assert!(approx_eq(y.as_slice()[2], 0.5, 1e-6));
        assert!(approx_eq(y.as_slice()[0], 0.0, 1e-6));
        assert!(approx_eq(y.as_slice()[4], 1.0, 1e-6));
        for &v in y.as_slice() {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn silu_of_zero_is_zero() {
        let x = t(vec![-2.0, 0.0, 2.0], &[3]);
        let y = silu(&x);
        assert!(approx_eq(y.as_slice()[1], 0.0, 1e-6));
        // silu(2) = 2 * sigmoid(2) ≈ 2 * 0.8808 ≈ 1.7616
        assert!(approx_eq(y.as_slice()[2], 2.0 * 0.8807971, 1e-5));
    }

    #[test]
    fn gelu_of_zero_is_zero_and_gelu_of_large_matches_input() {
        let x = t(vec![-10.0, 0.0, 10.0], &[3]);
        let y = gelu(&x);
        // gelu(0) = 0
        assert!(approx_eq(y.as_slice()[1], 0.0, 1e-6));
        // gelu(10) ≈ 10 (saturates), gelu(-10) ≈ 0
        assert!(approx_eq(y.as_slice()[2], 10.0, 1e-4));
        assert!(approx_eq(y.as_slice()[0], 0.0, 1e-4));
    }

    // ----- mean -----

    #[test]
    fn mean_all_averages_everything() {
        let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let y = mean_all(&x);
        assert_eq!(y.as_slice(), &[3.0]);
        assert_eq!(y.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn mean_all_of_empty_is_nan() {
        let x = t(vec![], &[0]);
        let y = mean_all(&x);
        assert!(y.as_slice()[0].is_nan());
    }

    // ----- axis reductions -----

    #[test]
    fn row_major_strides_matches_shape() {
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(row_major_strides(&[5]), vec![1]);
        assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn sum_dim_reduces_rows_of_2x3() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // sum along dim 0 → column sums [5, 7, 9], shape [3]
        let s0 = sum_dim(&x, 0);
        assert_eq!(s0.shape().dims(), &[3]);
        assert_eq!(s0.as_slice(), &[5.0, 7.0, 9.0]);
        // sum along dim 1 → row sums [6, 15], shape [2]
        let s1 = sum_dim(&x, 1);
        assert_eq!(s1.shape().dims(), &[2]);
        assert_eq!(s1.as_slice(), &[6.0, 15.0]);
    }

    #[test]
    fn sum_dim_on_rank_3() {
        // Shape [2, 2, 3]:
        //   [[[1, 2, 3], [4, 5, 6]],
        //    [[7, 8, 9], [10, 11, 12]]]
        let x = t(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[2, 2, 3],
        );
        // sum along dim 2 (innermost) → shape [2, 2]:
        //   [[6, 15], [24, 33]]
        let s2 = sum_dim(&x, 2);
        assert_eq!(s2.shape().dims(), &[2, 2]);
        assert_eq!(s2.as_slice(), &[6.0, 15.0, 24.0, 33.0]);
        // sum along dim 0 (outermost) → shape [2, 3]:
        //   [[1+7, 2+8, 3+9], [4+10, 5+11, 6+12]] = [[8, 10, 12], [14, 16, 18]]
        let s0 = sum_dim(&x, 0);
        assert_eq!(s0.shape().dims(), &[2, 3]);
        assert_eq!(s0.as_slice(), &[8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
    }

    #[test]
    fn max_dim_and_min_dim_on_2x3() {
        let x = t(vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0], &[2, 3]);
        // max along dim 0 (columns): max(1,4)=4, max(5,3)=5, max(2,6)=6
        let mx0 = max_dim(&x, 0);
        assert_eq!(mx0.as_slice(), &[4.0, 5.0, 6.0]);
        // min along dim 1 (rows): min of row 0 = 1, row 1 = 3
        let mn1 = min_dim(&x, 1);
        assert_eq!(mn1.as_slice(), &[1.0, 3.0]);
    }

    #[test]
    fn mean_dim_on_2x3() {
        let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // mean along dim 1 → row means [2, 5]
        let m = mean_dim(&x, 1);
        assert_eq!(m.as_slice(), &[2.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn sum_dim_panics_on_bad_dim() {
        let x = t(vec![1.0, 2.0, 3.0], &[3]);
        let _ = sum_dim(&x, 5);
    }

    // ----- reshaping -----

    #[test]
    fn transpose_2d_swaps_rows_and_columns() {
        // [[1, 2, 3],
        //  [4, 5, 6]]  shape (2, 3)
        let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = transpose_2d(&x);
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]      shape (3, 2)
        assert_eq!(y.shape().dims(), &[3, 2]);
        assert_eq!(y.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_2d_is_self_inverse() {
        let x = t(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
        );
        let y = transpose_2d(&transpose_2d(&x));
        assert_eq!(y.shape().dims(), &[2, 4]);
        assert_eq!(y.as_slice(), x.as_slice());
    }

    // ----- compositions -----

    #[test]
    fn softmax_last_dim_sums_to_one_per_row() {
        // Shape [2, 3]: two rows, each a 3-element distribution.
        let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = softmax_last_dim(&x);
        assert_eq!(y.shape().dims(), &[2, 3]);
        for row in 0..2 {
            let start = row * 3;
            let s: f32 = y.as_slice()[start..start + 3].iter().sum();
            assert!(approx_eq(s, 1.0, 1e-6), "row {row} sum = {s}");
            // All probabilities in [0, 1]
            for &v in &y.as_slice()[start..start + 3] {
                assert!((0.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn softmax_last_dim_stability_with_large_inputs() {
        // Large inputs must not overflow thanks to max-subtraction.
        let x = t(vec![1000.0, 1001.0, 999.0], &[3]);
        let y = softmax_last_dim(&x);
        let s: f32 = y.as_slice().iter().sum();
        assert!(approx_eq(s, 1.0, 1e-6));
        for &v in y.as_slice() {
            assert!(v.is_finite());
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn softmax_last_dim_uniform_input_is_uniform_output() {
        let x = t(vec![3.0, 3.0, 3.0, 3.0], &[4]);
        let y = softmax_last_dim(&x);
        for &v in y.as_slice() {
            assert!(approx_eq(v, 0.25, 1e-6));
        }
    }

    #[test]
    fn layer_norm_last_dim_produces_zero_mean_unit_variance() {
        // Shape [2, 4] with two rows to normalize independently.
        let x = t(
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            &[2, 4],
        );
        let y = layer_norm_last_dim(&x, 1e-12);
        for row in 0..2 {
            let start = row * 4;
            let slice = &y.as_slice()[start..start + 4];
            let mean: f32 = slice.iter().sum::<f32>() / 4.0;
            let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
            assert!(approx_eq(mean, 0.0, 1e-5), "row {row} mean = {mean}");
            assert!(approx_eq(var, 1.0, 1e-5), "row {row} var = {var}");
        }
    }

    #[test]
    fn layer_norm_last_dim_constant_row_is_zeros() {
        // A constant row has variance 0, so the normalized output is 0
        // (with `eps` preventing division by zero).
        let x = t(vec![5.0, 5.0, 5.0, 5.0], &[4]);
        let y = layer_norm_last_dim(&x, 1e-5);
        for &v in y.as_slice() {
            assert!(approx_eq(v, 0.0, 1e-3));
        }
    }

    // ----- multi-dtype coverage ------------------------------------------
    //
    // These tests demonstrate that the generic implementations work across
    // f32, f64, bf16, and f16. Tolerances are per-dtype because bf16/f16
    // have far less precision than f32/f64 — bf16 in particular has only
    // ~3 decimal digits of precision.

    use half::{bf16, f16};

    /// Generic tensor constructor from raw `f64` values, converting to the
    /// target dtype via `num_traits::NumCast`. Keeps the test bodies free of
    /// per-dtype literal gymnastics.
    fn tg<T: Float>(data: Vec<f64>, dims: &[usize]) -> RefTensor<T> {
        let converted: Vec<T> = data.into_iter().map(|v| cst(v)).collect();
        RefTensor::from_vec(converted, Shape::from_dims(dims))
    }

    fn approx_eq_t<T: Float>(a: T, b: T, tol: T) -> bool {
        (a - b).abs() <= tol
    }

    // ----- f64 -----

    #[test]
    fn f64_add_is_pointwise() {
        let a = tg::<f64>(vec![1.0, 2.0, 3.0], &[3]);
        let b = tg::<f64>(vec![4.0, 5.0, 6.0], &[3]);
        let c = add(&a, &b);
        assert_eq!(c.as_slice(), &[5.0_f64, 7.0, 9.0]);
    }

    #[test]
    fn f64_sqrt_of_squares() {
        let x = tg::<f64>(vec![4.0, 9.0, 16.0, 25.0], &[4]);
        let y = sqrt(&x);
        assert_eq!(y.as_slice(), &[2.0_f64, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn f64_matmul_2x3_by_3x2_hand_computed() {
        let a = tg::<f64>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = tg::<f64>(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = matmul_2d(&a, &b);
        // f64 should be bit-exact for these small integers.
        assert_eq!(c.as_slice(), &[58.0_f64, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn f64_softmax_sums_to_one() {
        let x = tg::<f64>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = softmax_last_dim(&x);
        for row in 0..2 {
            let s: f64 = y.as_slice()[row * 3..row * 3 + 3].iter().sum();
            // f64 is tighter than f32 — use a tighter tolerance.
            assert!(approx_eq_t(s, 1.0_f64, 1e-12));
        }
    }

    #[test]
    fn f64_layer_norm_zero_mean_unit_variance() {
        let x = tg::<f64>(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let y = layer_norm_last_dim(&x, 1e-12);
        let slice = y.as_slice();
        let mean: f64 = slice.iter().sum::<f64>() / 4.0;
        let var: f64 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
        assert!(approx_eq_t(mean, 0.0_f64, 1e-12));
        assert!(approx_eq_t(var, 1.0_f64, 1e-12));
    }

    // ----- bf16 -----
    //
    // bf16 has ~3 decimal digits of precision (8-bit significand) so
    // tolerances are deliberately loose.

    #[test]
    fn bf16_add_is_pointwise() {
        let a = tg::<bf16>(vec![1.0, 2.0, 3.0], &[3]);
        let b = tg::<bf16>(vec![4.0, 5.0, 6.0], &[3]);
        let c = add(&a, &b);
        let tol = bf16::from_f32(0.1);
        assert!(approx_eq_t(c.as_slice()[0], bf16::from_f32(5.0), tol));
        assert!(approx_eq_t(c.as_slice()[1], bf16::from_f32(7.0), tol));
        assert!(approx_eq_t(c.as_slice()[2], bf16::from_f32(9.0), tol));
    }

    #[test]
    fn bf16_relu_clamps_negatives() {
        let x = tg::<bf16>(vec![-1.0, 0.0, 2.5, -3.5, 4.0], &[5]);
        let y = relu(&x);
        let zero = bf16::from_f32(0.0);
        assert_eq!(y.as_slice()[0], zero);
        assert_eq!(y.as_slice()[1], zero);
        assert!(y.as_slice()[2] > zero);
        assert_eq!(y.as_slice()[3], zero);
        assert!(y.as_slice()[4] > zero);
    }

    #[test]
    fn bf16_matmul_hand_computed() {
        // Use small integers that bf16 can represent exactly.
        let a = tg::<bf16>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = tg::<bf16>(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = matmul_2d(&a, &b);
        // Expected: [58, 64, 139, 154] — bf16 can represent 58 and 64
        // exactly but not 139 or 154 (those round). Use a tolerance.
        let tol = bf16::from_f32(2.0);
        assert!(approx_eq_t(c.as_slice()[0], bf16::from_f32(58.0), tol));
        assert!(approx_eq_t(c.as_slice()[1], bf16::from_f32(64.0), tol));
        assert!(approx_eq_t(c.as_slice()[2], bf16::from_f32(139.0), tol));
        assert!(approx_eq_t(c.as_slice()[3], bf16::from_f32(154.0), tol));
    }

    #[test]
    fn bf16_softmax_sums_to_approximately_one() {
        let x = tg::<bf16>(vec![1.0, 2.0, 3.0], &[3]);
        let y = softmax_last_dim(&x);
        let s: bf16 = y
            .as_slice()
            .iter()
            .copied()
            .fold(bf16::from_f32(0.0), |a, b| a + b);
        let tol = bf16::from_f32(0.02);
        assert!(
            approx_eq_t(s, bf16::from_f32(1.0), tol),
            "bf16 softmax sum was {s}, expected ~1.0",
        );
    }

    // ----- f16 -----
    //
    // f16 has ~3-4 decimal digits of precision (10-bit significand) —
    // tighter than bf16 but still much looser than f32.

    #[test]
    fn f16_add_is_pointwise() {
        let a = tg::<f16>(vec![1.0, 2.0, 3.0], &[3]);
        let b = tg::<f16>(vec![4.0, 5.0, 6.0], &[3]);
        let c = add(&a, &b);
        assert_eq!(c.as_slice()[0], f16::from_f32(5.0));
        assert_eq!(c.as_slice()[1], f16::from_f32(7.0));
        assert_eq!(c.as_slice()[2], f16::from_f32(9.0));
    }

    #[test]
    fn f16_sqrt_of_perfect_squares() {
        let x = tg::<f16>(vec![4.0, 9.0, 16.0, 25.0], &[4]);
        let y = sqrt(&x);
        let tol = f16::from_f32(1e-2);
        assert!(approx_eq_t(y.as_slice()[0], f16::from_f32(2.0), tol));
        assert!(approx_eq_t(y.as_slice()[1], f16::from_f32(3.0), tol));
        assert!(approx_eq_t(y.as_slice()[2], f16::from_f32(4.0), tol));
        assert!(approx_eq_t(y.as_slice()[3], f16::from_f32(5.0), tol));
    }

    #[test]
    fn f16_sum_all_on_small_tensor() {
        let x = tg::<f16>(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let y = sum_all(&x);
        assert_eq!(y.as_slice()[0], f16::from_f32(15.0));
    }

    #[test]
    fn f16_layer_norm_zero_mean_unit_variance() {
        let x = tg::<f16>(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let y = layer_norm_last_dim(&x, 1e-5);
        // f16 precision is limited — use a loose tolerance on mean/var.
        let sum = y
            .as_slice()
            .iter()
            .copied()
            .fold(f16::from_f32(0.0), |a, b| a + b);
        let mean = sum / f16::from_f32(4.0);
        let tol = f16::from_f32(1e-2);
        assert!(
            approx_eq_t(mean, f16::from_f32(0.0), tol),
            "f16 layer_norm mean = {mean}",
        );
    }

    // ----- indexing -----

    #[test]
    fn index_select_2d_by_rows() {
        // [[1, 2, 3],
        //  [4, 5, 6],
        //  [7, 8, 9]]
        let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        // Pick rows 2, 0, 2 → [[7,8,9], [1,2,3], [7,8,9]]
        let y = index_select(&x, 0, &[2, 0, 2]);
        assert_eq!(y.shape().dims(), &[3, 3]);
        assert_eq!(
            y.as_slice(),
            &[7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
        );
    }

    #[test]
    fn index_select_2d_by_columns() {
        // Pick columns 2, 0 from the same matrix →
        // [[3, 1], [6, 4], [9, 7]]
        let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let y = index_select(&x, 1, &[2, 0]);
        assert_eq!(y.shape().dims(), &[3, 2]);
        assert_eq!(y.as_slice(), &[3.0, 1.0, 6.0, 4.0, 9.0, 7.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_select_panics_on_bad_index() {
        let x = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let _ = index_select(&x, 0, &[5]);
    }

    #[test]
    fn embedding_looks_up_rows() {
        // 4-vocab × 3-dim embedding table.
        let table = t(
            vec![
                1.0, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
                7.0, 8.0, 9.0, // row 2
                10.0, 11.0, 12.0, // row 3
            ],
            &[4, 3],
        );
        let out = embedding(&table, &[2, 0, 3]);
        assert_eq!(out.shape().dims(), &[3, 3]);
        assert_eq!(
            out.as_slice(),
            &[7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        );
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn embedding_panics_on_oov_id() {
        let table = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let _ = embedding(&table, &[3]);
    }

    // ----- broadcasting -----

    #[test]
    fn broadcast_shape_rules() {
        // Scalar broadcasts against any shape.
        assert_eq!(broadcast_shape(&[], &[3, 4]), vec![3, 4]);
        // Right-aligned, padding with 1s on the left.
        assert_eq!(broadcast_shape(&[4], &[3, 4]), vec![3, 4]);
        // Size-1 dims expand.
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]), vec![3, 4]);
        assert_eq!(broadcast_shape(&[1, 1, 5], &[2, 3, 5]), vec![2, 3, 5]);
    }

    #[test]
    #[should_panic(expected = "incompatible shapes")]
    fn broadcast_shape_rejects_incompatible() {
        let _ = broadcast_shape(&[3, 4], &[2, 4]);
    }

    #[test]
    fn broadcast_add_column_vector_plus_row_vector() {
        // Column: [3, 1] with values [10, 20, 30]
        let col = t(vec![10.0, 20.0, 30.0], &[3, 1]);
        // Row: [1, 4] with values [1, 2, 3, 4]
        let row = t(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        // Result: [3, 4] where out[i, j] = col[i] + row[j]
        let out = broadcast_add(&col, &row);
        assert_eq!(out.shape().dims(), &[3, 4]);
        let expected = vec![
            11.0, 12.0, 13.0, 14.0, // row 0: 10 + [1..4]
            21.0, 22.0, 23.0, 24.0, // row 1: 20 + [1..4]
            31.0, 32.0, 33.0, 34.0, // row 2: 30 + [1..4]
        ];
        assert_eq!(out.as_slice(), expected.as_slice());
    }

    #[test]
    fn broadcast_mul_scalar_against_vector() {
        let scalar = t(vec![2.5], &[]);
        let v = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let out = broadcast_mul(&scalar, &v);
        assert_eq!(out.shape().dims(), &[4]);
        assert_eq!(out.as_slice(), &[2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn broadcast_sub_matrix_minus_row_mean() {
        // A common pattern: subtract row-wise mean from each element.
        // matrix: [2, 3]
        let m = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // row means: [2, 5]
        let means = t(vec![2.0, 5.0], &[2, 1]);
        let out = broadcast_sub(&m, &means);
        assert_eq!(out.shape().dims(), &[2, 3]);
        assert_eq!(out.as_slice(), &[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn broadcast_div_normalizes_rows() {
        // Divide each row by its own row-sum.
        let m = t(vec![1.0, 2.0, 3.0, 4.0, 4.0, 4.0], &[2, 3]);
        let sums = t(vec![6.0, 12.0], &[2, 1]); // row sums
        let out = broadcast_div(&m, &sums);
        // Row 0: [1/6, 2/6, 3/6], row 1: [4/12, 4/12, 4/12]
        let expected = [
            1.0 / 6.0,
            2.0 / 6.0,
            3.0 / 6.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ];
        for (&a, &b) in out.as_slice().iter().zip(expected.iter()) {
            assert!(approx_eq(a, b, 1e-6));
        }
    }

    // ----- convolution and pooling -----

    #[test]
    fn conv2d_identity_kernel_returns_input() {
        // 1×1×3×3 input, 1×1×1×1 kernel with value 1.0, stride 1, pad 0.
        // Output should equal input.
        let x = t(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        );
        let k = t(vec![1.0], &[1, 1, 1, 1]);
        let y = conv2d_simple(&x, &k, 1, 0);
        assert_eq!(y.shape().dims(), &[1, 1, 3, 3]);
        assert_eq!(y.as_slice(), x.as_slice());
    }

    #[test]
    fn conv2d_sum_kernel_computes_sliding_sum() {
        // 1×1×3×3 input with a 1×1×2×2 all-ones kernel, stride 1, pad 0.
        // Output is 1×1×2×2 and each element is the sum of its 2×2 window.
        let x = t(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        );
        let k = t(vec![1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);
        let y = conv2d_simple(&x, &k, 1, 0);
        assert_eq!(y.shape().dims(), &[1, 1, 2, 2]);
        // Windows: (1+2+4+5)=12, (2+3+5+6)=16, (4+5+7+8)=24, (5+6+8+9)=28
        assert_eq!(y.as_slice(), &[12.0, 16.0, 24.0, 28.0]);
    }

    #[test]
    fn conv2d_with_padding_preserves_spatial_dims() {
        // Pad 1, kernel 3, stride 1 → output spatial dims == input.
        let x = t(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        );
        // Kernel that picks out the center only.
        let k = t(
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            &[1, 1, 3, 3],
        );
        let y = conv2d_simple(&x, &k, 1, 1);
        assert_eq!(y.shape().dims(), &[1, 1, 3, 3]);
        // Each output pixel is the center of the 3×3 window = the input pixel itself.
        assert_eq!(y.as_slice(), x.as_slice());
    }

    #[test]
    fn conv2d_multi_channel_and_multi_output() {
        // Input: [1, 2, 2, 2], kernel: [3, 2, 1, 1]
        // Each output channel is a weighted sum of input channels.
        let x = t(
            vec![
                1.0, 2.0, 3.0, 4.0, // channel 0
                5.0, 6.0, 7.0, 8.0, // channel 1
            ],
            &[1, 2, 2, 2],
        );
        let k = t(
            vec![
                1.0, 0.0, // out_ch 0: 1*in0 + 0*in1
                0.0, 1.0, // out_ch 1: 0*in0 + 1*in1
                1.0, 1.0, // out_ch 2: 1*in0 + 1*in1
            ],
            &[3, 2, 1, 1],
        );
        let y = conv2d_simple(&x, &k, 1, 0);
        assert_eq!(y.shape().dims(), &[1, 3, 2, 2]);
        // out_ch 0 = in 0, out_ch 1 = in 1, out_ch 2 = in 0 + in 1
        assert_eq!(
            y.as_slice(),
            &[
                1.0, 2.0, 3.0, 4.0, // out 0 = in 0
                5.0, 6.0, 7.0, 8.0, // out 1 = in 1
                6.0, 8.0, 10.0, 12.0, // out 2 = sum
            ],
        );
    }

    #[test]
    fn max_pool2d_2x2_stride_2() {
        // 1×1×4×4 input, pool 2×2 stride 2 → 1×1×2×2 output.
        let x = t(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
            ],
            &[1, 1, 4, 4],
        );
        let y = max_pool2d(&x, 2, 2);
        assert_eq!(y.shape().dims(), &[1, 1, 2, 2]);
        // Windows: max(1,2,5,6)=6, max(3,4,7,8)=8, max(9,10,13,14)=14, max(11,12,15,16)=16
        assert_eq!(y.as_slice(), &[6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn max_pool2d_preserves_batch_and_channel() {
        // 2 batches × 3 channels, each 2×2. Kernel=2, stride=1 → output 1×1 per channel.
        let mut data = Vec::new();
        for b in 0..2 {
            for c in 0..3 {
                // Fill each 2×2 with distinct values.
                let base = (b * 3 + c) as f32 * 10.0;
                data.extend([base + 1.0, base + 2.0, base + 3.0, base + 4.0]);
            }
        }
        let x = t(data, &[2, 3, 2, 2]);
        let y = max_pool2d(&x, 2, 1);
        assert_eq!(y.shape().dims(), &[2, 3, 1, 1]);
        // Max of each 2×2 window is base+4.0.
        let expected: Vec<f32> = (0..6).map(|i| i as f32 * 10.0 + 4.0).collect();
        assert_eq!(y.as_slice(), expected.as_slice());
    }
}
