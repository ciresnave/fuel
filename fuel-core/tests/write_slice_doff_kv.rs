//! Phase 1 (CapturedRun) — Op::WriteSliceDoff end-to-end through
//! PipelinedExecutor on CPU.
//!
//! Validates the device-resident-offset KV-cache append (form-B of
//! baracuda's `write_slice_*_doff`): the write start on one axis comes
//! from a rank-0 I64 `offset` operand (read host-side on CPU; device-
//! side under CUDA so a captured graph replays at the host-updated
//! position). No modulo wrap. Exercises builder validation,
//! OpKind/OpParams plumbing, the WorkItemKind::WriteSliceDoff execute
//! arm, and the CPU byte kernel (offset override + bounds check).

use fuel_core::lazy::LazyTensor;
use fuel_ir::Shape;

/// Basic append: offset 1, slab 1 row → writes to row 1 only.
#[test]
fn doff_writes_at_device_offset() {
    let device = fuel_core::Device::cpu();
    // dest [4, 2] starts at zero; write [7, 8] at device offset 1.
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let offset = dest.const_i64_like(vec![1_i64], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_doff(&src, &offset, /* axis */ 0, vec![(0, 1), (0, 2)])
        .expect("write_slice_doff builds");
    let out = post_write.realize_f32();
    assert_eq!(out, vec![0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0]);
}

/// Offset 0 lands at the leading row (no wrap, unlike rotating).
#[test]
fn doff_offset_zero_writes_leading_row() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let offset = dest.const_i64_like(vec![0_i64], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_doff(&src, &offset, 0, vec![(0, 1), (0, 2)])
        .expect("write_slice_doff builds");
    let out = post_write.realize_f32();
    assert_eq!(out, vec![7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

/// A capacity-4 decode loop appending one token per step at the live
/// `cached_len` offset — the DecodeSession/CapturedRun access pattern.
/// No wrap: each token lands at a distinct row.
#[test]
fn doff_decode_loop_appends_at_cached_len() {
    let device = fuel_core::Device::cpu();
    let max_seq = 4_usize;
    let head_dim = 2_usize;
    let mut cache = LazyTensor::from_f32(
        vec![0.0_f32; max_seq * head_dim],
        Shape::from_dims(&[max_seq, head_dim]),
        &device,
    );
    let tokens = [
        vec![1.0_f32, 1.1],
        vec![2.0_f32, 2.1],
        vec![3.0_f32, 3.1],
        vec![4.0_f32, 4.1],
    ];
    for (step, token) in tokens.iter().enumerate() {
        let token_t = cache.const_f32_like(token.clone(), Shape::from_dims(&[1, head_dim]));
        // `cached_len` = step: the append offset (device-resident under CUDA).
        let offset = cache.const_i64_like(vec![step as i64], Shape::from_dims(&[]));
        cache = cache
            .write_slice_doff(&token_t, &offset, 0, vec![(0, 1), (0, head_dim)])
            .expect("doff append");
    }
    let out = cache.realize_f32();
    // Each token lands at its own row (no wrap) — full KV history.
    assert_eq!(
        out,
        vec![
            1.0, 1.1, // step 0 → row 0
            2.0, 2.1, // step 1 → row 1
            3.0, 3.1, // step 2 → row 2
            4.0, 4.1, // step 3 → row 3
        ]
    );
}

/// Interior write on a non-leading axis: offset 2 on axis 1 of a
/// [2, 5] buffer, slab width 2.
#[test]
fn doff_writes_on_non_leading_axis() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 10], Shape::from_dims(&[2, 5]), &device);
    // slab [2, 2] written at columns [2, 4) on axis 1.
    let src = dest.const_f32_like(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        Shape::from_dims(&[2, 2]),
    );
    let offset = dest.const_i64_like(vec![2_i64], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_doff(&src, &offset, /* axis */ 1, vec![(0, 2), (0, 2)])
        .expect("write_slice_doff builds");
    let out = post_write.realize_f32();
    assert_eq!(
        out,
        vec![
            0.0, 0.0, 1.0, 2.0, 0.0, // row 0: cols 2,3 = 1,2
            0.0, 0.0, 3.0, 4.0, 0.0, // row 1: cols 2,3 = 3,4
        ]
    );
}

// ---- Build-time validation ---------------------------------------------------

/// Offset must be I64 (device-resident start); a U32 offset errors.
#[test]
fn doff_rejects_non_i64_offset() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let offset = dest.const_u32_like(vec![1_u32], Shape::from_dims(&[]));
    let r = dest.write_slice_doff(&src, &offset, 0, vec![(0, 1), (0, 2)]);
    assert!(r.is_err(), "non-I64 offset must error at build time");
}

/// Offset must be a rank-0 scalar.
#[test]
fn doff_rejects_nonscalar_offset() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let offset = dest.const_i64_like(vec![1_i64, 2_i64], Shape::from_dims(&[2]));
    let r = dest.write_slice_doff(&src, &offset, 0, vec![(0, 1), (0, 2)]);
    assert!(r.is_err(), "non-scalar offset must error at build time");
}

/// Static write width on `axis` must equal the source dim there.
#[test]
fn doff_rejects_source_axis_mismatch() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![1.0_f32; 4], Shape::from_dims(&[2, 2]));
    let offset = dest.const_i64_like(vec![0_i64], Shape::from_dims(&[]));
    let r = dest.write_slice_doff(&src, &offset, 0, vec![(0, 1), (0, 2)]);
    assert!(r.is_err(), "source/slab mismatch must error at build time");
}

/// Axis out of bounds must error.
#[test]
fn doff_rejects_axis_out_of_bounds() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let offset = dest.const_i64_like(vec![0_i64], Shape::from_dims(&[]));
    let r = dest.write_slice_doff(&src, &offset, /* axis */ 3, vec![(0, 1), (0, 2)]);
    assert!(r.is_err(), "axis out of bounds must error at build time");
}

/// Runtime overflow: offset + width > capacity on `axis`. The CPU path
/// reads the offset host-side and returns a typed error (the CUDA
/// kernel instead trusts the caller and does not clamp).
#[test]
#[should_panic(expected = "dest_shape")]
fn doff_offset_overflow_errors_at_realize_cpu() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    // offset 4 + width 1 > capacity 4 → overflow.
    let offset = dest.const_i64_like(vec![4_i64], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_doff(&src, &offset, 0, vec![(0, 1), (0, 2)])
        .expect("write_slice_doff builds (offset is dynamic, not checked at build)");
    // realize surfaces the overflow as an error → realize_f32 panics.
    let _ = post_write.realize_f32();
}
