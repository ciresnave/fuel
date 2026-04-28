//! Phase 7b spike validation: AOCL backend produces correct results
//! and the dispatch table prefers AOCL over the portable CPU backend
//! on Zen-class hardware.
//!
//! Skips cleanly when the `aocl` feature is off OR the AOCL probe
//! finds no usable `libaocl_blas` (so this passes on machines where
//! AOCL isn't installed).

#![cfg(feature = "aocl")]

use fuel_core::dispatch::{Criterion, DispatchTable};
use fuel_core::judge::{Judge, OpKind, OpSize, SizeClass};
use fuel_core::lazy::LazyTensor;
use fuel_core::probe::ProbeReport;
use fuel_core_types::{probe::BackendId, DType, Shape};
use fuel_graph_executor::GraphExecutor;

fn aocl_present() -> bool {
    let probe = ProbeReport::probe_all();
    probe.devices.iter().any(|d| d.backend == BackendId::Aocl)
}

#[test]
fn aocl_probe_enumerates_when_available() {
    let probe = ProbeReport::probe_all();
    let aocl = probe.devices.iter().find(|d| d.backend == BackendId::Aocl);
    if let Some(d) = aocl {
        assert_eq!(d.device_index, 0, "spike: one AOCL descriptor expected");
        eprintln!("AOCL present: {:?}", d.hardware_sku);
    } else {
        eprintln!("AOCL not present on this host; spike test skips");
    }
}

#[test]
fn aocl_matmul_matches_reference() {
    if !aocl_present() {
        eprintln!("skipping: AOCL not visible on this host");
        return;
    }
    let backend = fuel_aocl_cpu_backend::AoclBackend::try_new()
        .expect("aocl_present == true means try_new must succeed");
    let mut exe = GraphExecutor::new(backend);

    let (m, k, n) = (32usize, 48, 24);
    let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[m, k]));
    let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n]));
    let c = a.matmul(&b);

    let reference = c.realize_f32_reference();
    let aocl_out = c.realize_f32_aocl(&mut exe);
    assert_eq!(reference.len(), aocl_out.len());
    for (i, (&r, &x)) in reference.iter().zip(aocl_out.iter()).enumerate() {
        let denom = r.abs().max(x.abs()).max(f32::MIN_POSITIVE);
        let rel = (r - x).abs() / denom;
        assert!(rel < 1e-4, "matmul mismatch at {i}: ref={r}, aocl={x} (rel {rel})");
    }
}

/// The empirical proof of the spike: run the Phase 6b judge with both
/// the portable CPU backend and AOCL, build a dispatch table, and
/// assert AOCL wins MatMul under `Fastest` for at least one size
/// class on this machine. If AOCL doesn't actually win — the test
/// reports it as a soft warning rather than a failure (Phase 7b's
/// premise is that AOCL *should* be faster on Zen, but a CI box
/// with limited AOCL build flags might tie).
#[test]
fn aocl_dispatch_table_prefers_aocl_for_matmul() {
    if !aocl_present() {
        eprintln!("skipping: AOCL not visible on this host");
        return;
    }
    let probe = ProbeReport::probe_all();
    // Shrink the size plan so the test runs in seconds, not minutes.
    let judge = Judge {
        iterations: 3,
        warmup: 1,
        size_plan_override: Some(vec![
            (OpKind::MatMul, OpSize::MatMul { m: 256, n: 256, k: 256 }),
            (OpKind::MatMul, OpSize::MatMul { m: 512, n: 512, k: 512 }),
        ]),
    };
    let report = judge.run(&probe);
    let table = DispatchTable::build(&report);

    let mut aocl_wins = 0usize;
    let mut cpu_wins  = 0usize;
    for &sz in &[256usize, 512] {
        let n = sz * sz;
        let class = SizeClass::from_elem_count(n);
        let pick = table.pick_nearest(OpKind::MatMul, DType::F32, class, Criterion::Fastest);
        match pick.map(|p| p.backend) {
            Some(BackendId::Aocl) => aocl_wins += 1,
            Some(BackendId::Cpu)  => cpu_wins += 1,
            other => eprintln!("unexpected pick at size {sz}: {other:?}"),
        }
    }
    eprintln!("dispatch picks — aocl={aocl_wins}, cpu={cpu_wins} (out of 2 size classes)");
    // Hard floor: we expect at least one AOCL win on Zen. If not, the
    // spike's working hypothesis is wrong and we want a loud failure.
    assert!(
        aocl_wins >= 1,
        "AOCL did not win MatMul at any tested size — spike hypothesis broken on this host"
    );
}
