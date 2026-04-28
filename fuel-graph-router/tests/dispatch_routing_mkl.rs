//! Parallel of `dispatch_routing.rs` but for the MKL backend. Verifies
//! that `Router::matmul` invokes `MklBackend` when the dispatch table
//! picks MKL.

#![cfg(feature = "onemkl")]

use fuel_core_types::dispatch::{
    Criterion, DispatchTable, OpKind, Pick, ProfileEntry, ProfileReport, SizeClass,
    PROFILE_REPORT_VERSION,
};
use fuel_core_types::{DType, HostBuffer, Layout, Shape};
use fuel_core_types::probe::BackendId;
use fuel_graph_executor::GraphBackend;
use fuel_graph_router::Router;
use fuel_mkl_cpu_backend::MklBackend;
use std::sync::Arc;

fn synthetic_table_picking_mkl(matmul_size_class: u8) -> DispatchTable {
    let entries = vec![
        ProfileEntry {
            op: OpKind::MatMul, dtype: DType::F32,
            size_class: SizeClass(matmul_size_class),
            backend: BackendId::Cpu, device_index: 0,
            latency_ns: 1_000_000_000, iterations: 1, max_rel_error: 1e-6,
        },
        ProfileEntry {
            op: OpKind::MatMul, dtype: DType::F32,
            size_class: SizeClass(matmul_size_class),
            backend: BackendId::Mkl, device_index: 0,
            latency_ns: 1_000_000, iterations: 1, max_rel_error: 1e-6,
        },
    ];
    let report = ProfileReport { version: PROFILE_REPORT_VERSION, entries };
    DispatchTable::build(&report)
}

#[test]
fn router_picks_mkl_for_matmul_when_table_says_so() {
    if MklBackend::try_new().is_err() {
        eprintln!("skipping: MKL not loadable on this host");
        return;
    }

    let m = 16usize;
    let n = 16usize;
    let k = 16usize;
    let class = SizeClass::from_elem_count(m * n);
    let table = Arc::new(synthetic_table_picking_mkl(class.0));

    let pick = table.pick_nearest(OpKind::MatMul, DType::F32, class, Criterion::Fastest);
    assert_eq!(
        pick,
        Some(Pick { backend: BackendId::Mkl, device_index: 0 }),
        "synthetic table should pick MKL"
    );

    let router = Router::new()
        .add_cpu()
        .add_mkl()
        .with_dispatch_table(table);

    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.005) - 0.1).collect();
    let a = router.upload(&HostBuffer::F32(a_data.clone()), &Shape::from_dims(&[m, k]))
        .expect("upload a");
    let b = router.upload(&HostBuffer::F32(b_data.clone()), &Shape::from_dims(&[k, n]))
        .expect("upload b");
    let la = Layout::contiguous(&Shape::from_dims(&[m, k]));
    let lb = Layout::contiguous(&Shape::from_dims(&[k, n]));
    let c = router.matmul(&a, &b, (1, m, n, k), &la, &lb).expect("matmul");

    let c_host = router.download(&c).expect("download c");
    let c_vec = match c_host {
        HostBuffer::F32(v) => v,
        _ => panic!("matmul result not F32"),
    };
    assert_eq!(c_vec.len(), m * n);

    let a_ref = fuel_reference_backend::RefTensor::from_vec(a_data, Shape::from_dims(&[m, k]));
    let b_ref = fuel_reference_backend::RefTensor::from_vec(b_data, Shape::from_dims(&[k, n]));
    let c_ref = fuel_reference_backend::ops::matmul(&a_ref, &b_ref);
    for (i, (&got, &want)) in c_vec.iter().zip(c_ref.as_slice().iter()).enumerate() {
        let denom = got.abs().max(want.abs()).max(f32::MIN_POSITIVE);
        let rel = (got - want).abs() / denom;
        assert!(rel < 1e-4, "matmul mismatch at {i}: mkl={got}, ref={want} (rel {rel})");
    }
}
