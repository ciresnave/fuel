//! Phase 7b → backend-extensions-Phase-2 (2026-06-08): AOCL is now
//! a kernel-source extension of `BackendId::Cpu`, not a separate
//! backend. These tests validate that AOCL's loadability check still
//! works and that realize-through-the-picker produces sane output
//! when AOCL kernels register.

#![cfg(feature = "aocl")]

use fuel_core::lazy::LazyTensor;
use fuel_ir::Shape;

/// AOCL loadability via the crate's runtime probe. Replaces the
/// pre-retirement `BackendId::Aocl` discovery path.
fn aocl_present() -> bool {
    fuel_aocl_cpu_backend::probe_aocl_loadable().is_ok()
}

#[test]
fn aocl_loadable_check() {
    if aocl_present() {
        eprintln!("AOCL loadable on this host");
    } else {
        eprintln!("AOCL not loadable on this host; spike test skips");
    }
}

/// Post-backend-extensions-Phase-2 (2026-06-08): AOCL is now a
/// kernel-source extension of `BackendId::Cpu`, not a separate
/// backend. The dual-call shape that compared an AOCL-specific
/// executor against a "reference" is obsolete — there's one realize
/// path (the picker selects among CPU-substrate alternatives by
/// `kernel_source` tag). This test now exercises that path and
/// verifies the matmul output is numerically valid (finite, sane
/// magnitudes for the inputs in use).
#[test]
fn aocl_matmul_realize_is_finite_and_sane() {
    if !aocl_present() {
        eprintln!("skipping: AOCL not visible on this host");
        return;
    }
    let (m, k, n) = (32usize, 48, 24);
    let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[m, k]), &fuel_core::Device::cpu());
    let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n]));
    let c = a.matmul(&b);

    let out = c.realize_f32();
    assert_eq!(out.len(), m * n);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "matmul output non-finite at {i}: {v}");
        assert!(v.abs() < 1e6, "matmul output magnitude unreasonable at {i}: {v}");
    }
}

// `aocl_dispatch_table_prefers_aocl_for_matmul` retired
// 2026-06-08 (backend-extensions Phase 2). It checked that the
// DispatchTable picked `Some(BackendId::Aocl)` for MatMul, but
// AOCL kernels now share `BackendId::Cpu` with portable + MKL
// alternatives, distinguished only by `BindingEntry::kernel_source`.
// The `Pick` type doesn't carry kernel_source today; adding that
// (and the corresponding Judge per-alternative measurement) is its
// own session — backed by `docs/session-prompts/backend-extensions-phase-2.md`'s
// "Step 1 — Judge walks alternatives" sub-task. Until that lands,
// AOCL's static-cost ranking governs picker behavior; the test
// would have nothing meaningful to assert against in the interim.
