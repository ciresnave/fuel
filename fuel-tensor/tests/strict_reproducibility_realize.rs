// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP 13(b): strict reproducibility mode, end to end on the CPU realize
//! path. The CPU backend commits to a bit-stable kernel for every primitive,
//! so strict mode must be SATISFIABLE here and must not change the bytes.
//! (That the floor actually reaches the planner is proven by the sabotage
//! recorded in the PR: an impossible requirement makes these realizes fail.)

use fuel_ir::Shape;
use fuel_tensor::lazy::Tensor;
use fuel_tensor::reproducibility::{is_strict_reproducibility, strict_reproducibility};

fn build(dev: &fuel_tensor::Device) -> Tensor {
    let a = Tensor::from_f32(
        (0..12).map(|i| i as f32 * 0.37 - 1.5).collect::<Vec<f32>>(),
        Shape::from_dims(&[3, 4]),
        dev,
    )
    .unwrap();
    let b = Tensor::from_f32_on(
        a.graph(),
        (0..8).map(|i| (i as f32).sin()).collect::<Vec<f32>>(),
        Shape::from_dims(&[4, 2]),
        dev,
    )
    .unwrap();
    a.matmul(&b).unwrap().relu()
}

#[test]
fn strict_realize_matches_default_and_is_repeatable() {
    let dev = fuel_tensor::Device::cpu();
    let default = build(&dev).realize_f32();

    let _g = strict_reproducibility();
    assert!(is_strict_reproducibility());
    let first = build(&dev).realize_f32();
    let second = build(&dev).realize_f32();

    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&first), bits(&second), "strict runs are bit-identical");
    assert_eq!(bits(&first), bits(&default), "strict does not change bytes");
}
