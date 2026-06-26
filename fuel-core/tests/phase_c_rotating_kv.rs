//! Phase C — Op::WriteSliceRotating end-to-end through PipelinedExecutor.
//!
//! Validates that a sliding-window KV cache pattern (Mistral / Phi-3
//! sliding-window) realizes correctly through the production CPU
//! dispatch path: builder validation, OpKind/OpParams plumbing,
//! WorkItemKind::WriteSliceRotating execute arm, and the byte
//! kernel's two-chunk ring-boundary split.

use fuel_core::lazy::LazyTensor;
use fuel_ir::Shape;

/// Within-window write: position 1, slab 1 row, modulus 4. No
/// boundary split — writes to row 1 only.
#[test]
fn rotating_within_window() {
    let device = fuel_core::Device::cpu();
    // dest [4, 2] starts at zero; write [7, 8] at position 1.
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let position = dest.const_u32_like(vec![1_u32], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_rotating(&src, &position, /* axis */ 0, /* modulus */ 4, vec![(0, 1), (0, 2)])
        .expect("write_slice_rotating builds");
    let out = post_write.realize_f32();
    assert_eq!(out, vec![0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0]);
}

/// Position wraps modulo modulus: position 4 with modulus 4 → start 0.
#[test]
fn rotating_wraps_position_at_modulus() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let position = dest.const_u32_like(vec![4_u32], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_rotating(&src, &position, 0, 4, vec![(0, 1), (0, 2)])
        .expect("write_slice_rotating builds");
    let out = post_write.realize_f32();
    assert_eq!(out, vec![7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

/// Boundary split: position 3, slab 2 rows, modulus 4. Row 3 +
/// wrapped row 0 both written.
#[test]
fn rotating_splits_across_boundary() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(
        vec![10.0_f32, 11.0, 20.0, 21.0],
        Shape::from_dims(&[2, 2]),
    );
    let position = dest.const_u32_like(vec![3_u32], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_rotating(&src, &position, 0, 4, vec![(0, 2), (0, 2)])
        .expect("write_slice_rotating builds");
    let out = post_write.realize_f32();
    // row 3 = (10, 11), wrapped to row 0 = (20, 21).
    assert_eq!(out, vec![20.0, 21.0, 0.0, 0.0, 0.0, 0.0, 10.0, 11.0]);
}

/// Mistral-style 4-step decode loop with a window of 3. Steps 0–2
/// fit in the window; step 3 overwrites slot 0. End state holds the
/// most recent 3 rows in their post-rotation positions.
#[test]
fn rotating_mistral_style_decode_loop() {
    let device = fuel_core::Device::cpu();
    let window = 3_usize;
    let head_dim = 2_usize;
    let mut cache = LazyTensor::from_f32(
        vec![0.0_f32; window * head_dim],
        Shape::from_dims(&[window, head_dim]),
        &device,
    );
    // 4 "token" K vectors.
    let tokens = [
        vec![1.0_f32, 1.1],
        vec![2.0_f32, 2.1],
        vec![3.0_f32, 3.1],
        vec![4.0_f32, 4.1],
    ];
    for (step, token) in tokens.iter().enumerate() {
        let token_t = cache.const_f32_like(token.clone(), Shape::from_dims(&[1, head_dim]));
        let position = cache.const_u32_like(vec![step as u32], Shape::from_dims(&[]));
        cache = cache
            .write_slice_rotating(&token_t, &position, 0, window, vec![(0, 1), (0, head_dim)])
            .expect("rotating append");
    }
    let out = cache.realize_f32();
    // After 4 writes with window 3:
    //   step 0 → row 0 = token 0
    //   step 1 → row 1 = token 1
    //   step 2 → row 2 = token 2
    //   step 3 → row 0 (wraps) = token 3
    // Final cache: [token3, token1, token2]
    assert_eq!(
        out,
        vec![
            4.0, 4.1, // row 0 — overwritten by step 3
            2.0, 2.1, // row 1 — step 1
            3.0, 3.1, // row 2 — step 2
        ]
    );
}

// ---- Build-time validation tests --------------------------------------------

/// Position must be U32 scalar (rank-0).
#[test]
fn rotating_rejects_nonscalar_position() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    // position is rank-1 — should error at build time.
    let position = dest.const_u32_like(vec![1_u32, 2_u32], Shape::from_dims(&[2]));
    let r = dest.write_slice_rotating(&src, &position, 0, 4, vec![(0, 1), (0, 2)]);
    assert!(r.is_err(), "non-scalar position must error at build time");
}

/// Slab on rotating axis must equal source dim on that axis.
#[test]
fn rotating_rejects_source_axis_mismatch() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    // source has 2 rows but ranges declares slab of 1 on axis 0.
    let src = dest.const_f32_like(vec![1.0_f32; 4], Shape::from_dims(&[2, 2]));
    let position = dest.const_u32_like(vec![0_u32], Shape::from_dims(&[]));
    let r = dest.write_slice_rotating(&src, &position, 0, 4, vec![(0, 1), (0, 2)]);
    assert!(r.is_err(), "source/slab mismatch must error at build time");
}

/// Modulus > dest dim on rotating axis must error.
#[test]
fn rotating_rejects_modulus_exceeds_dest_dim() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let position = dest.const_u32_like(vec![0_u32], Shape::from_dims(&[]));
    let r = dest.write_slice_rotating(&src, &position, 0, /* modulus */ 5, vec![(0, 1), (0, 2)]);
    assert!(r.is_err(), "modulus > dest dim must error at build time");
}

/// Axis out of bounds must error.
#[test]
fn rotating_rejects_axis_out_of_bounds() {
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let position = dest.const_u32_like(vec![0_u32], Shape::from_dims(&[]));
    let r = dest.write_slice_rotating(&src, &position, /* axis */ 3, 4, vec![(0, 1), (0, 2)]);
    assert!(r.is_err(), "axis out of bounds must error at build time");
}
