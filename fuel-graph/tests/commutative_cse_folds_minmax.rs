// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Does CSE actually fold `min(a,b)` with `min(b,a)`?**
//!
//! **HISTORY, because the reason this guard exists is the thing worth keeping.**
//! `opt.rs::is_commutative` LISTED `Op::Minimum` and `Op::Maximum` until #65
//! (`071903c8`); this file now guards the REMOVAL, and its assertions fire if
//! anyone re-adds them to `opt.rs` or `jit.rs`. It is the only guard in the tree
//! pointed at that regression, and #65 changed BOTH functions.
//!
//! The original measurement is kept because it is why the removal happened. The
//! module doc says commutative ops "are keyed on sorted input IDs so `a + b`
//! and `b + a` fold to the same canonical node". There is an existing in-crate
//! test proving that for `Add`.
//!
//! **Nothing proved it for `Minimum`/`Maximum`** — that was inference from
//! shared membership in a `matches!`, which is exactly the step worth measuring
//! when a defect depends on it.
//!
//! It does, because measured separately at the kernel:
//!
//! ```text
//! min(+0,-0) = 0x8000 (-0)      min(-0,+0) = 0x0000 (+0)
//! ```
//!
//! `f32::min` returns *either* argument when both are zero — its own docs
//! disclaim which. So min/max on signed zeros depends on ARGUMENT ORDER, and
//! CSE is asserting an order-independence the implementation does not have.
//! Folding is therefore not a neutral dedup: the survivor's operand order wins
//! for every consumer of the folded node.
//!
//! This file measures only the FOLD. The order-dependence is measured in
//! `fuel-cpu-backend/tests/bf16_minmax_move_or_round.rs`; neither alone is the
//! defect, and the composition is.

use std::sync::Arc;

use fuel_graph::NodeHandle;
use fuel_graph::opt::optimize;
use fuel_ir::{DType, Shape};

fn cpu_dev() -> Arc<dyn fuel_backend_contract::DynBackendDevice> {
    Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice)
}

/// Two distinct tensors sharing one graph, so the operand NodeIds are shared
/// and the two orderings differ only in position.
fn two_operands() -> (NodeHandle, NodeHandle) {
    let dev = cpu_dev();
    let a = NodeHandle::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), &dev).unwrap();
    let b = a.add_scalar(5.0);
    (a, b)
}

#[test]
fn cse_folds_minimum_across_operand_order() {
    let (a, b) = two_operands();
    let ab = a.minimum(&b);
    let ba = b.minimum(&a);
    let graph = a.graph().clone();
    let roots = optimize(&graph, &[ab.id(), ba.id()]);
    println!(
        "[minmax-cse] minimum: min(a,b)->{:?}  min(b,a)->{:?}  folded={}",
        roots[0],
        roots[1],
        roots[0] == roots[1]
    );
    assert_ne!(
        roots[0], roots[1],
        "min(a,b) and min(b,a) FOLDED. `Minimum` was removed from \
         `is_commutative` by #65 (071903c8) because min/max are not bitwise \
         commutative -- NaN payload order is first-operand-biased and explicit \
         in the source. If they fold again, that removal has been undone and \
         an invalid rewrite is live."
    );
}

#[test]
fn cse_folds_maximum_across_operand_order() {
    let (a, b) = two_operands();
    let ab = a.maximum(&b);
    let ba = b.maximum(&a);
    let graph = a.graph().clone();
    let roots = optimize(&graph, &[ab.id(), ba.id()]);
    println!("[minmax-cse] maximum: folded={}", roots[0] == roots[1]);
    assert_ne!(
        roots[0], roots[1],
        "max(a,b) and max(b,a) FOLDED -- same regression as the Minimum arm. \
         BOTH `is_commutative` functions must exclude it: `opt.rs` keys CSE \
         and `jit.rs` matches order-independently; #65 changed both."
    );
}

/// **POSITIVE CONTROL FOR THE WHOLE FILE — added when the min/max assertions
/// were inverted.**
///
/// After the inversion every assertion here is a NON-equality, and a `optimize`
/// that folded NOTHING AT ALL — a dead pass, a harness that never reaches CSE,
/// a refactor that makes `optimize` a no-op on this shape — would turn the whole
/// file GREEN while proving nothing.
///
/// `control_non_commutative_sub_does_not_fold` cannot see that: it fires in the
/// SAME DIRECTION as total inertness, so it guards against over-folding and is
/// blind to under-folding. Two failure directions; without this one, two
/// controls point at one of them and none at the other.
///
/// `Add` is still in `is_commutative` at head, so this MUST fold.
#[test]
fn control_add_still_folds_across_operand_order() {
    let (a, b) = two_operands();
    let ab = a.add(&b);
    let ba = b.add(&a);
    let graph = a.graph().clone();
    let roots = optimize(&graph, &[ab.id(), ba.id()]);
    println!(
        "[minmax-cse] CONTROL add: a+b->{:?}  b+a->{:?}  folded={}",
        roots[0],
        roots[1],
        roots[0] == roots[1]
    );
    assert_eq!(
        roots[0], roots[1],
        "`Add` did NOT fold across operand order. `optimize` is not folding          anything on this shape, so every assert_ne in this file is vacuous and          its greenness means nothing. This is a larger finding than the file's          own subject -- stop and investigate rather than adjusting assertions."
    );
}

/// **The control, and it is load-bearing.** If CSE folded *everything* with the
/// same operand SET, the tests above would pass for a reason unrelated to
/// commutativity and prove nothing about `is_commutative`. `Sub` is not in that
/// list, so `a-b` and `b-a` must NOT fold.
#[test]
fn control_non_commutative_sub_does_not_fold() {
    let (a, b) = two_operands();
    let ab = a.sub(&b);
    let ba = b.sub(&a);
    let graph = a.graph().clone();
    let roots = optimize(&graph, &[ab.id(), ba.id()]);
    println!(
        "[minmax-cse] CONTROL sub: a-b->{:?}  b-a->{:?}  folded={}",
        roots[0],
        roots[1],
        roots[0] == roots[1]
    );
    assert_ne!(
        roots[0], roots[1],
        "control failed: if a-b folds with b-a then this suite is measuring \
         operand-set dedup, not commutative CSE, and the minmax results above \
         say nothing"
    );
}

/// Foundation: the two operands must actually be distinct nodes. If
/// `add_scalar` were folded away or `b` aliased `a`, both orderings would be
/// literally the same expression and folding would be trivially true.
#[test]
fn foundation_the_two_operands_are_distinct_nodes() {
    let (a, b) = two_operands();
    println!("[minmax-cse] operands: a={:?} b={:?}", a.id(), b.id());
    assert_ne!(
        a.id(),
        b.id(),
        "a and b are the same node -- every fold result above would be vacuous"
    );
    assert_eq!(
        a.dtype(),
        DType::F32,
        "control: operands are the dtype expected"
    );
}
