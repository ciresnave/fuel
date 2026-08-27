// SPDX-License-Identifier: MIT OR Apache-2.0
//! Phase 7b → backend-extensions-Phase-2 (2026-06-08): oneMKL is
//! now a kernel-source extension of `BackendId::Cpu`. See the
//! parallel doc in `phase7b_aocl_anchor.rs`.

#![cfg(feature = "onemkl")]

use fuel_core::lazy::Tensor;
use fuel_ir::Shape;

fn mkl_present() -> bool {
    fuel_mkl_cpu_backend::probe_mkl_loadable().is_ok()
}

#[test]
fn mkl_loadable_check() {
    if mkl_present() {
        eprintln!("MKL loadable on this host");
    } else {
        eprintln!("MKL not loadable on this host; spike test skips");
    }
}

/// Post-backend-extensions-Phase-2 (2026-06-08): see the parallel
/// note in `phase7b_aocl_anchor.rs`. MKL is a kernel-source extension
/// of `BackendId::Cpu`; the dual-call shape is obsolete.
#[test]
fn mkl_matmul_realize_is_finite_and_sane() {
    if !mkl_present() {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Mkl,
            fuel_test_support::hardware::Missing::device("MKL not visible on this host"),
        );
    }
    let (m, k, n) = (32usize, 48, 24);
    let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let a = Tensor::from_f32(a_data, Shape::from_dims(&[m, k]), &fuel_core::Device::cpu());
    let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n]));
    let c = a.matmul(&b).expect(
        "matmul is a graph-build call; a failure here is a Fuel defect, not a hardware one",
    );

    let out = c.realize_f32();
    assert_eq!(out.len(), m * n);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "matmul output non-finite at {i}: {v}");
        assert!(
            v.abs() < 1e6,
            "matmul output magnitude unreasonable at {i}: {v}"
        );
    }
}

// `mkl_participates_in_dispatch_picks` retired 2026-06-08.
// See parallel note in `phase7b_aocl_anchor.rs`. Re-introduces
// when Judge per-alternative measurement lands (so `Pick` can
// carry kernel_source) per
// `docs/session-prompts/backend-extensions-phase-2.md`.
