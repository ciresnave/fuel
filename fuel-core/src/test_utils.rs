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
    // GAP-243. NO SKIP PATH, BY DESIGN — and the reason is the same axis the
    // per-family defaults in `fuel_test_support::hardware` are built on.
    //
    // That module makes CUDA absence fatal by default because every CUDA call
    // site is an `#[ignore]`d test, so running one requires an explicit
    // `-- --ignored`, and *that* is the declaration. This function has no
    // `#[ignore]` to key on — but it does not need one: **naming it and passing
    // it tensors IS the declaration**, and a stronger one than `--ignored`,
    // which is a blanket flag that sweeps in tests nobody thought about. Nobody
    // calls `assert_cuda_matches_reference` by accident.
    //
    // ⚠️ It CANNOT use `fuel_test_support::hardware::skip`, and a future reader
    // will otherwise try to wire it up: `fuel-test-support` is a
    // **dev-dependency**, while this is ordinary library code (`pub mod
    // test_utils`, ungated). The mechanism is simply not in scope here, and
    // bringing it into scope would mean promoting a test-only crate into the
    // production dependency graph.
    //
    // Until 2026-08-27 this returned silently, so a caller asking for an
    // assertion got a green having asserted nothing — with zero callers it was
    // latent, which is the only reason it never lied to anyone.
    assert!(
        has_cuda,
        "assert_cuda_matches_reference REQUIRES a live CUDA device and the probe \
         found none.\n\n\
         This helper has no skip path by design: a caller that invokes it has \
         declared the requirement, so returning early would report success \
         having compared nothing. If you want to tolerate a missing device, take \
         that decision at the CALL SITE and do not call this."
    );
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
