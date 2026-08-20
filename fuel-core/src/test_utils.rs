// SPDX-License-Identifier: MIT OR Apache-2.0
//! Test helpers. The eager-`Tensor` helpers (`test_device!`, `assert_tensor_eq`,
//! `to_vec{0,1,2,3}_round`) were removed in B6; what remains is host-slice and
//! lazy infrastructure that the lazy_* port tests use.

/// Oracle-gate comparison helper: assert two `f32` slices match within
/// absolute tolerance `atol` OR relative tolerance `rtol`.
///
/// Used by the Phase 6a CI oracle gate — every anchor model's forward
/// pass runs on both `realize_f32()` (fast) and `realize_f32()`
/// (oracle), and the two outputs must agree within tolerance. Prints
/// the first mismatching index plus max abs/rel deviations when the
/// assertion fires so divergences are easy to localize.
pub fn assert_allclose_f32(a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "assert_allclose_f32: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
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
        if ad > max_abs {
            max_abs = ad;
        }
        if rd > max_rel {
            max_rel = rd;
        }
        if ad > atol && rd > rtol && first_bad.is_none() {
            first_bad = Some(i);
        }
    }
    if let Some(i) = first_bad {
        panic!(
            "assert_allclose_f32: first mismatch at index {i}: a={} b={} \
             (diff abs={} rel={}); max abs={max_abs} max rel={max_rel} \
             over {} elements (atol={atol} rtol={rtol})",
            a[i],
            b[i],
            (a[i] - b[i]).abs(),
            (a[i] - b[i]).abs() / a[i].abs().max(b[i].abs()).max(f32::MIN_POSITIVE),
            a.len(),
        );
    }
}

#[cfg(feature = "cuda")]
pub fn assert_cuda_matches_reference(t: &crate::lazy::Tensor, atol: f32, rtol: f32) {
    let probe = crate::probe::ProbeReport::probe_all();
    let has_cuda = probe
        .devices
        .iter()
        .any(|d| d.backend == fuel_ir::probe::BackendId::Cuda);
    if !has_cuda {
        eprintln!("assert_cuda_matches_reference: no CUDA device, skipping");
        return;
    }
    let reference = t.realize_f32_reference();
    let dev = fuel_cuda_backend::CudaDevice::new(0)
        .expect("cuda device 0 available since probe found one");
    let cuda = t.realize_f32_cuda(&dev);
    assert_allclose_f32(&cuda, &reference, atol, rtol);
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
