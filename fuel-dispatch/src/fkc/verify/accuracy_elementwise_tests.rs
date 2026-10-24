// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for [`super`] (the elementwise accuracy specs and their f64 references).

use super::*;

#[test]
fn erf_series_matches_known_values() {
    // erf(1) and erf(2) to 16 digits (Abramowitz & Stegun table 7.1).
    assert!((erf_f64(1.0) - 0.842_700_792_949_714_9).abs() < 1e-14);
    assert!((erf_f64(2.0) - 0.995_322_265_018_952_7).abs() < 1e-14);
    assert!((erf_f64(-1.0) + 0.842_700_792_949_714_9).abs() < 1e-14);
    assert!((erf_f64(6.0) - 1.0).abs() < 1e-14);
    assert_eq!(erf_f64(0.0), 0.0);
}

#[test]
fn round_is_half_to_even_not_half_away() {
    let want = [
        (0.5, 0.0),
        (1.5, 2.0),
        (2.5, 2.0),
        (-0.5, -0.0),
        (-2.5, -2.0),
        (2.4, 2.0),
    ];
    for (x, r) in want {
        assert_eq!(round_half_even(x), r, "round_half_even({x})");
    }
}

#[test]
fn rem_takes_the_divisors_sign() {
    assert_eq!(rem_python(5.0, 3.0), 2.0);
    assert_eq!(rem_python(-5.0, 3.0), 1.0);
    assert_eq!(rem_python(5.0, -3.0), -1.0);
    assert_eq!(rem_python(-5.0, -3.0), -2.0);
}

/// Every spec's inputs stay inside its domain, so a NaN reference cannot slip through as
/// "no error".
#[test]
fn domains_keep_every_reference_finite() {
    for spec in UNARY_SPECS {
        let x = inputs(4096, 1, spec.domain, spec.edges);
        for v in x {
            let r = (spec.reference)(v as f64);
            assert!(r.is_finite(), "{:?}: reference({v}) = {r}", spec.op);
        }
    }
    for spec in BINARY_SPECS {
        let ea: Vec<f32> = spec.edges.iter().map(|e| e.0).collect();
        let eb: Vec<f32> = spec.edges.iter().map(|e| e.1).collect();
        let a = inputs(4096, 2, spec.domain_a, &ea);
        let b = inputs(4096, 3, spec.domain_b, &eb);
        for (x, y) in a.into_iter().zip(b) {
            let r = (spec.reference)(x as f64, y as f64);
            assert!(r.is_finite(), "{:?}: reference({x}, {y}) = {r}", spec.op);
        }
    }
}

/// The retained sabotage sibling of the measurement: a runner that returns the exact
/// reference measures zero error, and the same runner with one element perturbed measures
/// that perturbation. Without the second half a harness that compared the reference with
/// itself would pass every claim.
#[test]
fn the_measurement_sees_a_perturbed_output() {
    use super::super::bit_stability::HostTensor;
    let spec = &UNARY_SPECS[0];
    let table = {
        let mut t = crate::kernel::KernelBindingTable::new();
        crate::baracuda_dispatch::register_baracuda_cuda_kernels(&mut t);
        t
    };
    let entry = super::super::accuracy_common::find_entry(
        &table,
        OpKind::NegElementwise,
        &[DType::F32, DType::F32],
        fuel_ir::probe::BackendId::Cuda,
    )
    .expect("neg f32 entry");
    let host = |perturb: f32| {
        move |_e: &BindingEntry,
              _p: OpParams,
              ins: &[HostTensor],
              _s: Vec<usize>|
              -> Result<Vec<f32>, VerifyError> {
            let mut x = super::super::accuracy_common::f32s(&ins[0]);
            for v in x.iter_mut() {
                *v = -*v;
            }
            x[0] += perturb;
            Ok(x)
        }
    };
    let claim = Claim {
        max_relative: 1e-6,
        max_absolute: 1e-6,
    };
    let exact = measure_unary(&host(0.0), entry, spec, claim).expect("measures");
    assert_eq!(exact.worst_abs, 0.0);
    let off = measure_unary(&host(0.5), entry, spec, claim).expect("measures");
    assert!(off.worst_abs >= 0.5, "perturbation seen: {}", off.worst_abs);
    assert!(off.worst_abs > claim.max_absolute);
}
