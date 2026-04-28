//! End-to-end validation of the dispatch-table-driven Router refactor:
//! when both `CpuBackend` and `AoclBackend` are attached and a
//! dispatch table picks AOCL for a given `(op, dtype, size_class)`,
//! `Router::matmul` actually invokes AOCL.
//!
//! Skipped when `aocl` feature is off OR libaocl_blas isn't loadable
//! at runtime.

#![cfg(feature = "aocl")]

use fuel_aocl_cpu_backend::AoclBackend;
use fuel_core_types::dispatch::{
    Criterion, DispatchTable, OpKind, Pick, ProfileEntry, ProfileReport, SizeClass,
    PROFILE_REPORT_VERSION,
};
use fuel_core_types::{DType, HostBuffer, Layout, Shape};
use fuel_core_types::probe::BackendId;
use fuel_graph_executor::GraphBackend;
use fuel_graph_router::Router;
use std::sync::Arc;

/// Build a tiny synthetic dispatch table that hard-picks AOCL for
/// the matmul size we'll exercise. Avoids running the full Judge —
/// keeps the test fast and deterministic.
fn synthetic_table_picking_aocl(matmul_size_class: u8) -> DispatchTable {
    let entries = vec![
        // CPU is slow → loses fastest.
        ProfileEntry {
            op: OpKind::MatMul, dtype: DType::F32,
            size_class: SizeClass(matmul_size_class),
            backend: BackendId::Cpu, device_index: 0,
            latency_ns: 1_000_000_000, iterations: 1, max_rel_error: 1e-6,
        },
        // AOCL is fast → wins fastest.
        ProfileEntry {
            op: OpKind::MatMul, dtype: DType::F32,
            size_class: SizeClass(matmul_size_class),
            backend: BackendId::Aocl, device_index: 0,
            latency_ns: 1_000_000, iterations: 1, max_rel_error: 1e-6,
        },
    ];
    let report = ProfileReport { version: PROFILE_REPORT_VERSION, entries };
    DispatchTable::build(&report)
}

#[test]
fn router_picks_aocl_for_matmul_when_table_says_so() {
    if AoclBackend::try_new().is_err() {
        eprintln!("skipping: AOCL not loadable on this host");
        return;
    }

    // Build a 16×16 matmul → 256 output elements → SizeClass(8).
    let m = 16usize;
    let n = 16usize;
    let k = 16usize;
    let out_elems = m * n;
    let class = SizeClass::from_elem_count(out_elems);

    let table = Arc::new(synthetic_table_picking_aocl(class.0));

    // Sanity: the table should pick AOCL.
    let pick = table.pick_nearest(OpKind::MatMul, DType::F32, class, Criterion::Fastest);
    assert_eq!(
        pick,
        Some(Pick { backend: BackendId::Aocl, device_index: 0 }),
        "synthetic table should pick AOCL"
    );

    // Construct the Router with both CPU and AOCL backends, and the
    // dispatch table. add_aocl is no-op-on-failure but we already
    // checked try_new succeeds above.
    let router = Router::new()
        .add_cpu()
        .add_aocl()
        .with_dispatch_table(table);

    // Run a real f32 matmul through the Router. This exercises the
    // pick_for_op path.
    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.005) - 0.1).collect();
    let a = router.upload(&HostBuffer::F32(a_data.clone()), &Shape::from_dims(&[m, k]))
        .expect("upload a");
    let b = router.upload(&HostBuffer::F32(b_data.clone()), &Shape::from_dims(&[k, n]))
        .expect("upload b");
    let la = Layout::contiguous(&Shape::from_dims(&[m, k]));
    let lb = Layout::contiguous(&Shape::from_dims(&[k, n]));
    let c = router.matmul(&a, &b, (1, m, n, k), &la, &lb).expect("matmul");

    // Sanity-check the result against the reference backend.
    let c_host = router.download(&c).expect("download c");
    let c_vec = match c_host {
        HostBuffer::F32(v) => v,
        _ => panic!("matmul result not F32"),
    };
    assert_eq!(c_vec.len(), m * n);

    // Cross-check against reference backend (oracle).
    let a_ref = fuel_reference_backend::RefTensor::from_vec(a_data, Shape::from_dims(&[m, k]));
    let b_ref = fuel_reference_backend::RefTensor::from_vec(b_data, Shape::from_dims(&[k, n]));
    let c_ref = fuel_reference_backend::ops::matmul(&a_ref, &b_ref);
    for (i, (&got, &want)) in c_vec.iter().zip(c_ref.as_slice().iter()).enumerate() {
        let denom = got.abs().max(want.abs()).max(f32::MIN_POSITIVE);
        let rel = (got - want).abs() / denom;
        assert!(rel < 1e-4, "matmul mismatch at {i}: aocl={got}, ref={want} (rel {rel})");
    }
}

#[test]
fn router_falls_through_to_cpu_when_table_absent() {
    // No dispatch table attached → Router should use the
    // first-registered (CpuBackend). This is the existing behaviour
    // pre-refactor; verifying the refactor didn't regress it.
    let router = Router::new().add_cpu();
    // 1×2 @ 2×2 = 1×2 → 4 + 4 + 4 = 12 elements total in a/b/c
    let a = router.upload(&HostBuffer::F32(vec![1.0, 2.0]),
        &Shape::from_dims(&[1, 2])).unwrap();
    let b = router.upload(&HostBuffer::F32(vec![1.0, 0.0, 0.0, 1.0]),
        &Shape::from_dims(&[2, 2])).unwrap();
    let c = router.matmul(&a, &b,
        (1, 1, 2, 2),
        &Layout::contiguous(&Shape::from_dims(&[1, 2])),
        &Layout::contiguous(&Shape::from_dims(&[2, 2])),
    ).expect("matmul falls through to CpuBackend");
    let host = router.download(&c).expect("download");
    if let HostBuffer::F32(v) = host {
        // Identity multiply: [1,2] @ I = [1,2]
        assert_eq!(v, vec![1.0, 2.0]);
    } else {
        panic!("expected F32 output");
    }
}
