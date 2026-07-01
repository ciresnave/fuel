//! Post-backend-extensions-Phase-2 (2026-06-08): AOCL and oneMKL
//! kernels are now siblings of the portable CPU conv2d at one
//! `(Conv2D, [F32×N], BackendId::Cpu)` binding-table key,
//! distinguished by `kernel_source`. The picker selects among them
//! based on cost ranking; there's no dedicated AOCL/MKL realize
//! path to dual-call against a "reference." This file now exercises
//! the picker-routed conv2d and checks output finiteness + sane
//! magnitudes across a handful of representative shapes.

#![cfg(any(feature = "aocl", feature = "onemkl"))]

use fuel_core::lazy::LazyTensor;
use fuel_ir::Shape;

fn build_conv_graph(
    n: usize, c_in: usize, h: usize, w: usize,
    c_out: usize, k: usize,
    stride: (usize, usize), padding: (usize, usize),
) -> LazyTensor {
    let x_data: Vec<f32> = (0..(n * c_in * h * w))
        .map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let w_data: Vec<f32> = (0..(c_out * c_in * k * k))
        .map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[n, c_in, h, w]), &fuel_core::Device::cpu());
    let weight = x.const_f32_like(w_data, Shape::from_dims(&[c_out, c_in, k, k]));
    x.conv2d(&weight, None, stride, padding, 1)
}

fn assert_finite_and_sane(out: &[f32], label: &str) {
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "{label}: output non-finite at {i}: {v}");
        assert!(v.abs() < 1e6, "{label}: output magnitude unreasonable at {i}: {v}");
    }
}

#[test]
fn cpu_substrate_conv2d_realize_dense() {
    for &(n, ci, h, w, co, k, s, p) in &[
        (1, 4, 8, 8, 8, 3, (1, 1), (0, 0)),
        (2, 8, 16, 16, 16, 3, (1, 1), (1, 1)),
        (1, 16, 32, 32, 32, 3, (2, 2), (1, 1)),  // YOLO-style stride-2
    ] {
        let y = build_conv_graph(n, ci, h, w, co, k, s, p);
        let out = y.realize_f32();
        assert_finite_and_sane(
            &out,
            &format!("conv2d (n={n} ci={ci} h={h} w={w} co={co} k={k} s={s:?} p={p:?})"),
        );
    }
}
