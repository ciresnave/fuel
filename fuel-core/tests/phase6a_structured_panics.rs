//! Phase 6a structured-error gate: realize-time panics get prefixed
//! with the failing node's graph location, not just the original
//! `assert!` message.
//!
//! Today most shape mismatches are caught at build time by the
//! `Tensor::*` builders' assertions. The remaining tail (e.g. dtype
//! mismatches at op inputs, or panics that bubble up through the
//! generic executor's eval_node dispatcher) used to surface as bare
//! `assertion failed: …` panics with no graph location.
//!
//! This gate ensures the panic message includes "Node#N (Op, shape,
//! dtype, inputs=…)" — enough to grep-locate the failing op in a
//! large graph.

use fuel_core::lazy::LazyTensor;
use fuel_core_types::Shape;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Reach into the boxed payload and pull out a String. Mirrors
/// `panic_payload_to_string` in fuel-graph-executor (private there).
fn payload_text(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&'static str>() { return s.to_string(); }
    if let Some(s) = p.downcast_ref::<String>()       { return s.clone();     }
    "<non-string panic payload>".to_string()
}

#[test]
fn realize_panic_message_has_graph_location() {
    // Build a deliberately-bad graph: an IndexSelect with an
    // out-of-bounds index. The builder doesn't peek at the index
    // tensor's data, so this only fails at eval time.
    let src = LazyTensor::from_f32(vec![1.0f32, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), &fuel_core::Device::cpu());
    let bad_idx = src.const_u32_like(vec![100u32, 200u32], Shape::from_dims(&[2]));
    let result_tensor = src.index_select(0, &bad_idx);

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = result_tensor.realize_f32();
    }));
    let payload = result.expect_err("expected realize to panic on OOB index");
    let msg = payload_text(&payload);

    // The wrapped panic message should mention the executor's prefix
    // *and* the offending node's location. Reference backend wraps
    // with "fuel-reference-backend realize"; the generic executor
    // wraps with "fuel-graph-executor realize". Either is acceptable;
    // both should include the Node# marker so a user can navigate
    // back to the offending op.
    assert!(
        msg.contains("Node#"),
        "panic message missing graph location:\n{msg}",
    );
    assert!(
        msg.contains("realize"),
        "panic message missing executor wrapping prefix:\n{msg}",
    );
}
