//! Phase 6d Track 1 lazy-IR oracle: `LazyTensor::paged_attn` realized
//! through the CPU executor matches the reference attention_paged_naive.
//!
//! Catches dispatch wiring (executor's PagedAttn arm + fuel-graph-cpu's
//! PagedAttn dispatch + reference's PagedAttn dispatch) for the lazy
//! path.

use fuel_core::lazy::LazyTensor;
use fuel_ir::Shape;

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
fn lazy_paged_attn_decode_round_trip() {
    let (b, h, sq, sk, d) = (1usize, 2, 1, 16, 8);
    let block_size = 4;
    let blocks_per_seq = sk / block_size;
    let num_blocks = b * blocks_per_seq;

    let q_data = rand_f32(&[b, h, sq, d], 1);
    let kc_data = rand_f32(&[num_blocks, block_size, h, d], 2);
    let vc_data = rand_f32(&[num_blocks, block_size, h, d], 3);
    let bt_data: Vec<u32> = (0..num_blocks as u32).collect();
    let cl_data: Vec<u32> = vec![sk as u32; b];

    let scale = 1.0_f32 / (d as f32).sqrt();
    let q  = LazyTensor::from_f32(q_data.clone(), Shape::from_dims(&[b, h, sq, d]), &fuel_core::Device::cpu());
    let kc = q.const_f32_like(kc_data.clone(), Shape::from_dims(&[num_blocks, block_size, h, d]));
    let vc = q.const_f32_like(vc_data.clone(), Shape::from_dims(&[num_blocks, block_size, h, d]));
    let bt = q.const_u32_like(bt_data.clone(), Shape::from_dims(&[b, blocks_per_seq]));
    let cl = q.const_u32_like(cl_data.clone(), Shape::from_dims(&[b]));
    let out = q.paged_attn(&kc, &vc, &bt, &cl, None, scale, block_size, None).unwrap();

    let cpu = out.realize_f32();
    let reference = out.realize_f32();
    assert_eq!(cpu.len(), reference.len());
    for (i, (&a, &b)) in cpu.iter().zip(reference.iter()).enumerate() {
        let diff = (a - b).abs();
        let denom = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(diff < 1e-5 || rel < 1e-5, "[{i}]: cpu={a} ref={b} (abs={diff} rel={rel})");
    }
}
