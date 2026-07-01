use crate::tensor::Tensor;
use crate::Result;

#[macro_export]
macro_rules! test_device {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $test_cpu: ident, $test_cuda: ident, $test_metal: ident) => {
        #[test]
        fn $test_cpu() -> Result<()> {
            $fn_name(&Device::cpu())
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn $test_cuda() -> Result<()> {
            $fn_name(&$crate::cuda_backend::new_device(0)?)
        }

        #[cfg(feature = "metal")]
        #[test]
        fn $test_metal() -> Result<()> {
            $fn_name(&$crate::metal_backend::new_device(0)?)
        }
    };
}

/// Asserts that two tensors have the same shape and identical element values.
///
/// # Example
///
/// ```rust
/// use fuel_core::{Tensor, Device, DType};
/// use fuel_core::test_utils::assert_tensor_eq;
/// let a = Tensor::new(&[1f32, 2., 3.], &Device::cpu())?;
/// let b = Tensor::new(&[1f32, 2., 3.], &Device::cpu())?;
/// assert_tensor_eq(&a, &b)?;
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub fn assert_tensor_eq(t1: &Tensor, t2: &Tensor) -> Result<()> {
    assert_eq!(t1.shape(), t2.shape());
    // Default U8 may not be large enough to hold the sum (`t.sum_all` defaults to the dtype of `t`)
    let eq_tensor = t1.eq(t2)?.to_dtype(crate::DType::U32)?;
    let all_equal = eq_tensor.sum_all()?;
    assert_eq!(all_equal.to_scalar::<u32>()?, eq_tensor.elem_count() as u32);
    Ok(())
}

/// Oracle-gate comparison helper: assert two `f32` slices match within
/// absolute tolerance `atol` OR relative tolerance `rtol`.
///
/// Used by the Phase 6a CI oracle gate — every anchor model's forward
/// pass runs on both `realize_f32()` (fast) and `realize_f32()`
/// (oracle), and the two outputs must agree within tolerance. Prints
/// the first mismatching index plus max abs/rel deviations when the
/// assertion fires so divergences are easy to localize.
pub fn assert_allclose_f32(a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len(),
        "assert_allclose_f32: length mismatch {} vs {}", a.len(), b.len());
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut first_bad: Option<usize> = None;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if !x.is_finite() || !y.is_finite() {
            assert!(
                x.is_finite() == y.is_finite() && x.is_nan() == y.is_nan(),
                "assert_allclose_f32: finiteness mismatch at index {i}: {x} vs {y}"
            );
            continue;
        }
        let ad = (x - y).abs();
        let rd = ad / x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
        if ad > max_abs { max_abs = ad; }
        if rd > max_rel { max_rel = rd; }
        if ad > atol && rd > rtol && first_bad.is_none() {
            first_bad = Some(i);
        }
    }
    if let Some(i) = first_bad {
        panic!(
            "assert_allclose_f32: first mismatch at index {i}: a={} b={} \
             (diff abs={} rel={}); max abs={max_abs} max rel={max_rel} \
             over {} elements (atol={atol} rtol={rtol})",
            a[i], b[i], (a[i] - b[i]).abs(),
            (a[i] - b[i]).abs() / a[i].abs().max(b[i].abs()).max(f32::MIN_POSITIVE),
            a.len(),
        );
    }
}

/// Phase 6b CUDA oracle gate: realize `t` through the CUDA backend
/// and the reference backend, assert allclose. Skips silently when no
/// CUDA device is visible (so the same test passes on headless CI
/// hosts and dev rigs alike).
///
/// Tolerance defaults are deliberately a little looser than the CPU
/// oracle's 1e-4: a multi-op CUDA forward through cublas accumulates
/// gemm sum-order drift that's larger than the CPU fast path's, and
/// 5e-3 is the cliff beyond which we'd suspect an actual algorithmic
/// divergence rather than rounding.
#[cfg(feature = "cuda")]
pub fn assert_cuda_matches_reference(
    t: &crate::lazy::LazyTensor,
    atol: f32,
    rtol: f32,
) {
    let probe = crate::probe::ProbeReport::probe_all();
    let has_cuda = probe.devices.iter().any(|d| d.backend == fuel_ir::probe::BackendId::Cuda);
    if !has_cuda {
        eprintln!("assert_cuda_matches_reference: no CUDA device, skipping");
        return;
    }
    let reference = t.realize_f32();
    let dev = fuel_cuda_backend::CudaDevice::new(0)
        .expect("cuda device 0 available since probe found one");
    let cuda = t.realize_f32_cuda(&dev);
    assert_allclose_f32(&cuda, &reference, atol, rtol);
}

/// Extracts a scalar f32 value from a rank-0 tensor, rounded to `digits` decimal places.
///
/// # Example
///
/// ```rust
/// use fuel_core::{Tensor, Device};
/// use fuel_core::test_utils::to_vec0_round;
/// let t = Tensor::new(3.14159f32, &Device::cpu())?;
/// assert_eq!(to_vec0_round(&t, 2)?, 3.14);
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub fn to_vec0_round(t: &Tensor, digits: i32) -> Result<f32> {
    let b = 10f32.powi(digits);
    let t = t.to_vec0::<f32>()?;
    Ok(f32::round(t * b) / b)
}

/// Extracts a 1-D tensor to a `Vec<f32>`, rounding each element to `digits` decimal places.
///
/// # Example
///
/// ```rust
/// use fuel_core::{Tensor, Device};
/// use fuel_core::test_utils::to_vec1_round;
/// let t = Tensor::new(&[1.11111f32, 2.22222], &Device::cpu())?;
/// assert_eq!(to_vec1_round(&t, 2)?, vec![1.11, 2.22]);
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub fn to_vec1_round(t: &Tensor, digits: i32) -> Result<Vec<f32>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec1::<f32>()?;
    let t = t.iter().map(|t| f32::round(t * b) / b).collect();
    Ok(t)
}

/// Extracts a 2-D tensor to a `Vec<Vec<f32>>`, rounding each element to `digits` decimal places.
///
/// # Example
///
/// ```rust
/// use fuel_core::{Tensor, Device};
/// use fuel_core::test_utils::to_vec2_round;
/// let t = Tensor::new(&[[1.005f32, 2.005], [3.005, 4.005]], &Device::cpu())?;
/// let r = to_vec2_round(&t, 2)?;
/// assert_eq!(r[0][0], 1.01);
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub fn to_vec2_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec2::<f32>()?;
    let t = t
        .iter()
        .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
        .collect();
    Ok(t)
}

/// Extracts a 3-D tensor to a `Vec<Vec<Vec<f32>>>`, rounding each element to `digits` decimal places.
///
/// # Example
///
/// ```rust
/// use fuel_core::{Tensor, Device};
/// use fuel_core::test_utils::to_vec3_round;
/// let t = Tensor::zeros((2, 2, 2), fuel_core::DType::F32, &Device::cpu())?;
/// let r = to_vec3_round(&t, 2)?;
/// assert_eq!(r.len(), 2);
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub fn to_vec3_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}

/// Element-wise absolute-tolerance comparison for two flat f32 slices.
/// Panics with a descriptive message on the first cell exceeding
/// `abs_tol`. Used by tests whose precision baseline drifts by a few
/// ULPs across cuDNN algorithm choices (e.g. conv backward grads where
/// baracuda's algorithm pick differs from a prior Fuel-internal cuDNN
/// wrapper's choice; both outputs are equally IEEE-754-valid).
pub fn assert_close_vec1(actual: &[f32], expected: &[f32], abs_tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: len mismatch {} vs {}",
        actual.len(),
        expected.len(),
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= abs_tol,
            "{label}: idx {i} actual={a} expected={e} diff={diff} > {abs_tol}",
        );
    }
}
