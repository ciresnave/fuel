//! Phase 6d Track 3 oracle: `Op::FusedLinear` produces the same output
//! as the unfused `MatMul + BroadcastTo + Add` sequence it replaces,
//! both via reference and via the CPU executor.

use fuel_core::lazy::LazyTensor;
use fuel_core_types::Shape;
use fuel_graph::{opt, Op};

fn rand_f32(shape: &[usize], seed: u32) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        v.push(((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.5);
    }
    v
}

#[test]
fn fused_linear_realizes_same_as_matmul_plus_bias() {
    // [2, 5] @ [5, 7] + [7]
    let a_data = rand_f32(&[2, 5], 1);
    let b_data = rand_f32(&[5, 7], 2);
    let bias_data = rand_f32(&[7], 3);

    // Unfused reference graph.
    let a = LazyTensor::from_f32(a_data.clone(), Shape::from_dims(&[2, 5]), &fuel_core::Device::cpu());
    let b = a.const_f32_like(b_data.clone(), Shape::from_dims(&[5, 7]));
    let bias = a.const_f32_like(bias_data.clone(), Shape::from_dims(&[7]));
    let mm = a.matmul(&b);
    let bias_b = bias.broadcast_to(Shape::from_dims(&[2, 7])).unwrap();
    let unfused_out = mm.add(&bias_b);

    let unfused_result = unfused_out.realize_f32();

    // Identical second graph, then run fuse_linear and realize.
    let a2 = LazyTensor::from_f32(a_data, Shape::from_dims(&[2, 5]), &fuel_core::Device::cpu());
    let b2 = a2.const_f32_like(b_data, Shape::from_dims(&[5, 7]));
    let bias2 = a2.const_f32_like(bias_data, Shape::from_dims(&[7]));
    let mm2 = a2.matmul(&b2);
    let bias2_b = bias2.broadcast_to(Shape::from_dims(&[2, 7])).unwrap();
    let fused_out = mm2.add(&bias2_b);

    let inner = fused_out.graph_tensor();
    let n_fused = opt::fuse_linear(inner.graph(), &[inner.id()]);
    assert_eq!(n_fused, 1, "exactly one fusion should fire");

    let fused_result = fused_out.realize_f32();
    assert_eq!(unfused_result.len(), fused_result.len());
    for (i, (&u, &f)) in unfused_result.iter().zip(fused_result.iter()).enumerate() {
        let diff = (u - f).abs();
        let denom = u.abs().max(f.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(diff < 1e-5 || rel < 1e-5,
            "[{i}]: unfused={u} fused={f} (abs={diff} rel={rel})");
    }
}
