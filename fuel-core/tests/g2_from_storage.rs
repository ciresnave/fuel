//! Phase 7.5 work item G2: smoke tests for `fuel_graph::Tensor::from_storage`,
//! the slot-only Const constructor that's the foundation of B2's factory
//! migration.
//!
//! `from_storage` builds a single-node graph with `Op::Const(None)` and
//! pre-populates the graph's `storage_map` slot with the caller-supplied
//! `Arc<RwLock<Storage>>`. Realizing consumes the slot's bytes directly —
//! no host-side `ConstData` round-trip.
//!
//! Executor-unification Session 1 (re-audit `846f7908`): the realize
//! tests originally ran one copy each through the three executors that
//! had been taught slot-first dispatch (fuel-graph-cpu,
//! fuel-reference-backend, fuel-graph-executor). Those per-executor
//! variants are collapsed onto the one production path —
//! [`fuel_core::pipelined_bridge::realize_one_as`] →
//! `PipelinedExecutor`, whose `ConstAdopt` work item is the slot-first
//! dispatch under test. Two dispatch shapes stay covered: a slot-backed
//! Const as the realize root, and slot-backed Consts feeding a kernel
//! work item.

use fuel_core::{Device, Tensor};
use fuel_ir::Shape;

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
    let dtype = fuel_ir::DType::F32;
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

/// Realizing a from_storage tensor through the pipelined bridge
/// produces the slot's bytes — the executor's `ConstAdopt` slot-first
/// dispatch consumes them directly, with the slot-backed Const as the
/// realize root.
#[test]
fn from_storage_realizes_through_pipelined_bridge() {
    let device = Device::cpu();
    let data = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let legacy = Tensor::from_slice(&data, (2, 3), &device).unwrap();
    let storage_arc = legacy.realized_storage().unwrap();

    let t = fuel_graph::Tensor::from_storage(
        storage_arc, Shape::from_dims(&[2, 3]), fuel_ir::DType::F32,
    );
    assert_eq!(t.shape().dims(), &[2, 3]);

    let realized = fuel_core::pipelined_bridge::realize_one_as::<f32>(
        t.graph(), t.id(), &device,
    ).expect("realize from_storage root via PipelinedExecutor");
    assert_eq!(realized, data);
}

/// const_like_from_storage attaches a slot-only Const to the same
/// graph as `self`, allowing multiple slot-backed leaves to coexist
/// on one graph — and feeding a kernel work item from slot bytes.
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
        a_arc, Shape::from_dims(&[3]), fuel_ir::DType::F32,
    );
    let b = a.const_like_from_storage(
        b_arc, Shape::from_dims(&[3]), fuel_ir::DType::F32,
    );

    // Both share the same graph.
    assert!(std::sync::Arc::ptr_eq(a.graph(), b.graph()));

    // Build a + b via the existing fuel-graph operator. Slot-first
    // dispatch covers both inputs; the executor's add kernel produces
    // the sum from those bytes.
    let sum = a.add(&b);
    let realized = fuel_core::pipelined_bridge::realize_one_as::<f32>(
        sum.graph(), sum.id(), &device,
    ).expect("realize slot-fed add via PipelinedExecutor");
    assert_eq!(realized, &[5.0, 7.0, 9.0]);
}
