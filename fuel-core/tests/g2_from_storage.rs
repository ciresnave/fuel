//! Phase 7.5 work item G2: smoke tests for `fuel_graph::Tensor::from_storage`,
//! the slot-only Const constructor that's the foundation of B2's factory
//! migration.
//!
//! `from_storage` builds a single-node graph with `Op::Const(None)` and
//! pre-populates the graph's `storage_map` slot with the caller-supplied
//! `Arc<RwLock<Storage>>`. Realizing through any executor that's been
//! taught slot-first dispatch (G2 step 1: fuel-graph-executor,
//! fuel-graph-cpu, fuel-reference-backend) consumes the slot's bytes
//! directly — no host-side `ConstData` round-trip.

use fuel_core::{Device, Tensor};
use fuel_core_types::Shape;

/// from_storage produces a fuel_graph::Tensor whose graph contains a
/// single Op::Const node with its slot pre-populated.
#[test]
fn from_storage_builds_single_const_node_with_slot_populated() {
    let device = Device::cpu();
    // Build a Storage via the eager factory; we just want the bytes.
    let legacy = Tensor::from_slice(
        &[1.0_f32, 2.0, 3.0, 4.0],
        (2, 2),
        &device,
    ).unwrap();
    let storage_arc = legacy.realized_storage().unwrap();

    let shape = Shape::from_dims(&[2, 2]);
    let dtype = fuel_core_types::DType::F32;
    let t = fuel_graph::Tensor::from_storage(
        storage_arc.clone(), shape.clone(), dtype,
    );

    assert_eq!(t.dtype(), dtype);
    assert_eq!(t.shape().dims(), &[2, 2]);

    // The graph has exactly one node — the Const leaf.
    assert_eq!(t.graph().read().unwrap().len(), 1);

    // The slot is populated with the same Arc we handed in.
    let slot_arc = t.storage_for().expect("slot populated by from_storage");
    assert!(
        std::sync::Arc::ptr_eq(&slot_arc, &storage_arc),
        "slot Arc must be the one passed to from_storage",
    );
}

/// Realizing a from_storage tensor through fuel-graph-cpu produces the
/// slot's bytes — the slot-first dispatch consumes them directly.
#[test]
fn from_storage_realizes_through_graph_cpu() {
    let device = Device::cpu();
    let data = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let legacy = Tensor::from_slice(&data, (2, 3), &device).unwrap();
    let storage_arc = legacy.realized_storage().unwrap();

    let t = fuel_graph::Tensor::from_storage(
        storage_arc, Shape::from_dims(&[2, 3]), fuel_core_types::DType::F32,
    );

    // Realize through fuel-graph-cpu's slot-first realize loop.
    let realized = fuel_graph_cpu::realize_f32(&t);
    assert_eq!(realized.shape().dims(), &[2, 3]);
    assert_eq!(realized.as_slice(), data.as_slice());
}

/// Realizing through fuel-reference-backend's slot-first dispatch.
#[test]
fn from_storage_realizes_through_reference() {
    let device = Device::cpu();
    let data = vec![1.0_f32, -2.0, 3.5, -4.5];
    let legacy = Tensor::from_slice(&data, (4,), &device).unwrap();
    let storage_arc = legacy.realized_storage().unwrap();

    let t = fuel_graph::Tensor::from_storage(
        storage_arc, Shape::from_dims(&[4]), fuel_core_types::DType::F32,
    );

    let realized = fuel_reference_backend::exec::realize_f32(&t);
    assert_eq!(realized.shape().dims(), &[4]);
    assert_eq!(realized.as_slice(), data.as_slice());
}

/// Realizing through fuel-graph-executor + CpuBackend (the
/// generic-over-B path) — slot-first dispatch in that realize loop
/// consumes the slot identically.
#[test]
fn from_storage_realizes_through_graph_executor() {
    let device = Device::cpu();
    let data = vec![7.0_f32, 8.0, 9.0];
    let legacy = Tensor::from_slice(&data, (3,), &device).unwrap();
    let storage_arc = legacy.realized_storage().unwrap();

    let t = fuel_graph::Tensor::from_storage(
        storage_arc, Shape::from_dims(&[3]), fuel_core_types::DType::F32,
    );

    let mut exe = fuel_graph_executor::GraphExecutor::new(fuel_graph_cpu::CpuBackend);
    let realized = exe.realize_f32(&t);
    assert_eq!(realized.shape().dims(), &[3]);
    assert_eq!(realized.as_slice(), data.as_slice());
}

/// const_like_from_storage attaches a slot-only Const to the same
/// graph as `self`, allowing both ConstData-backed and slot-backed
/// leaves to coexist on one graph.
#[test]
fn const_like_from_storage_shares_graph() {
    let device = Device::cpu();
    let a_data = vec![1.0_f32, 2.0, 3.0];
    let b_data = vec![4.0_f32, 5.0, 6.0];
    let a_legacy = Tensor::from_slice(&a_data, (3,), &device).unwrap();
    let a_arc = a_legacy.realized_storage().unwrap();
    let b_legacy = Tensor::from_slice(&b_data, (3,), &device).unwrap();
    let b_arc = b_legacy.realized_storage().unwrap();

    let a = fuel_graph::Tensor::from_storage(
        a_arc, Shape::from_dims(&[3]), fuel_core_types::DType::F32,
    );
    let b = a.const_like_from_storage(
        b_arc, Shape::from_dims(&[3]), fuel_core_types::DType::F32,
    );

    // Both share the same graph.
    assert!(std::sync::Arc::ptr_eq(a.graph(), b.graph()));

    // Build a + b via the existing fuel-graph operator. Slot-first
    // dispatch covers both inputs; the executor's add op produces the
    // sum from those bytes.
    let sum = a.add(&b);
    let realized = fuel_graph_cpu::realize_f32(&sum);
    assert_eq!(realized.as_slice(), &[5.0, 7.0, 9.0]);
}
