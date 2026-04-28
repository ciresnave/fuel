//! Phase 7b second-CPU-backend validation: oneMKL backend produces
//! correct results and participates in the empirical dispatch.
//!
//! Skips cleanly when the `onemkl` feature is off OR the MKL probe
//! finds no usable `mkl_rt` (so this passes on machines where oneMKL
//! isn't installed).
//!
//! On a Zen5 with both AOCL and oneMKL installed, we don't pre-judge
//! which one wins matmul — that's the whole point of the empirical
//! layer. The third test here just confirms the dispatch table picks
//! *some* non-CPU backend when both AOCL and MKL are available; the
//! specific winner is whichever the judge measured faster.

#![cfg(feature = "onemkl")]

use fuel_core::dispatch::{Criterion, DispatchTable};
use fuel_core::judge::{Judge, OpKind, OpSize, SizeClass};
use fuel_core::lazy::LazyTensor;
use fuel_core::probe::ProbeReport;
use fuel_core_types::{probe::BackendId, DType, Shape};
use fuel_graph_executor::GraphExecutor;

fn mkl_present() -> bool {
    let probe = ProbeReport::probe_all();
    probe.devices.iter().any(|d| d.backend == BackendId::Mkl)
}

#[test]
fn mkl_probe_enumerates_when_available() {
    let probe = ProbeReport::probe_all();
    let mkl = probe.devices.iter().find(|d| d.backend == BackendId::Mkl);
    if let Some(d) = mkl {
        assert_eq!(d.device_index, 0, "spike: one MKL descriptor expected");
        eprintln!("MKL present: {:?}", d.hardware_sku);
    } else {
        eprintln!("MKL not present on this host; spike test skips");
    }
}

#[test]
fn mkl_matmul_matches_reference() {
    if !mkl_present() {
        eprintln!("skipping: MKL not visible on this host");
        return;
    }
    let backend = fuel_mkl_cpu_backend::MklBackend::try_new()
        .expect("mkl_present == true means try_new must succeed");
    let mut exe = GraphExecutor::new(backend);

    let (m, k, n) = (32usize, 48, 24);
    let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[m, k]));
    let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n]));
    let c = a.matmul(&b);

    let reference = c.realize_f32_reference();
    let mkl_out = c.realize_f32_mkl(&mut exe);
    assert_eq!(reference.len(), mkl_out.len());
    for (i, (&r, &x)) in reference.iter().zip(mkl_out.iter()).enumerate() {
        let denom = r.abs().max(x.abs()).max(f32::MIN_POSITIVE);
        let rel = (r - x).abs() / denom;
        assert!(rel < 1e-4, "matmul mismatch at {i}: ref={r}, mkl={x} (rel {rel})");
    }
}

/// The empirical part: run the Phase 6b judge and check that the
/// dispatch table picks a *non-CPU* backend (AOCL or MKL) for matmul.
/// Specific winner depends on hardware; either is acceptable.
#[test]
fn mkl_participates_in_dispatch_picks() {
    if !mkl_present() {
        eprintln!("skipping: MKL not visible on this host");
        return;
    }
    let probe = ProbeReport::probe_all();
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

    let mut mkl_wins = 0usize;
    let mut aocl_wins = 0usize;
    let mut cpu_wins = 0usize;
    let mut other = 0usize;
    for &sz in &[256usize, 512] {
        let n = sz * sz;
        let class = SizeClass::from_elem_count(n);
        let pick = table.pick_nearest(OpKind::MatMul, DType::F32, class, Criterion::Fastest);
        match pick.map(|p| p.backend) {
            Some(BackendId::Mkl)  => mkl_wins  += 1,
            Some(BackendId::Aocl) => aocl_wins += 1,
            Some(BackendId::Cpu)  => cpu_wins  += 1,
            _ => other += 1,
        }
    }
    eprintln!(
        "dispatch picks — mkl={mkl_wins}, aocl={aocl_wins}, cpu={cpu_wins}, other={other} (out of 2)",
    );
    // We don't assert *which* backend wins — the judge is empirical.
    // We only assert that the empirical layer routed away from the
    // portable CPU baseline at least once. On Zen5 with both AOCL and
    // MKL present, this should hold trivially.
    assert!(
        mkl_wins + aocl_wins >= 1,
        "expected MKL or AOCL to win at least one size class",
    );
}
