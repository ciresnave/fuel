//! Unit tests for the streaming `StreamMask` + `apply_state_mask`
//! primitives added 2026-05-29 (xn-port).

use fuel_core::{Device, Result, StreamMask, Tensor, apply_state_mask};

#[test]
fn stream_mask_construction() {
    let m = StreamMask::new(vec![true, false, true, true]);
    assert!(!m.is_empty());
    assert_eq!(m.batch_size(), Some(4));
    assert!(m.is_active(0));
    assert!(!m.is_active(1));
    assert!(m.is_active(2));
    assert!(m.is_active(3));
    assert_eq!(m.as_slice().unwrap(), &[true, false, true, true]);
}

#[test]
fn stream_mask_empty_is_all_active() {
    let m = StreamMask::empty();
    assert!(m.is_empty());
    assert_eq!(m.batch_size(), None);
    // is_active returns true for any index when the mask is empty.
    for i in 0..10 {
        assert!(m.is_active(i), "empty mask should treat idx {i} as active");
    }
    assert!(m.as_slice().is_none());
}

#[test]
fn stream_mask_all_active_materialised() {
    let m = StreamMask::all_active(5);
    assert!(!m.is_empty());
    assert_eq!(m.batch_size(), Some(5));
    for i in 0..5 {
        assert!(m.is_active(i));
    }
    assert_eq!(m.as_slice().unwrap(), &[true; 5]);
}

#[test]
fn apply_state_mask_empty_mask_returns_new() -> Result<()> {
    // Empty mask → all-active → result == new_state, regardless of old.
    let dev = Device::cpu();
    let new = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &dev)?;
    let old = Tensor::new(&[[9.0f32, 9.0], [9.0, 9.0]], &dev)?;
    let mask = StreamMask::empty();
    let out = apply_state_mask(&Some(new), &Some(old), &mask)?.unwrap();
    assert_eq!(out.to_vec2::<f32>()?, [[1.0, 2.0], [3.0, 4.0]]);
    Ok(())
}

#[test]
fn apply_state_mask_blends_active_and_finished_rows() -> Result<()> {
    let dev = Device::cpu();
    // Three batch rows, scalar state each.
    let new = Tensor::new(&[[1.0f32], [2.0], [3.0]], &dev)?;
    let old = Tensor::new(&[[10.0f32], [20.0], [30.0]], &dev)?;
    // Row 0 active, row 1 finished, row 2 active.
    let mask = StreamMask::new(vec![true, false, true]);
    let out = apply_state_mask(&Some(new), &Some(old), &mask)?.unwrap();
    // Expected: row 0 → 1.0, row 1 → 20.0 (preserved), row 2 → 3.0.
    assert_eq!(out.to_vec2::<f32>()?, [[1.0], [20.0], [3.0]]);
    Ok(())
}

#[test]
fn apply_state_mask_broadcasts_over_trailing_dims() -> Result<()> {
    // The mask is per-row but the state may have extra trailing dims;
    // mask should broadcast to fill them.
    let dev = Device::cpu();
    // Shape [2, 3] — 2 batch rows × 3 features.
    let new = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &dev)?;
    let old = Tensor::new(&[[100.0f32, 200.0, 300.0], [400.0, 500.0, 600.0]], &dev)?;
    let mask = StreamMask::new(vec![true, false]);
    let out = apply_state_mask(&Some(new), &Some(old), &mask)?.unwrap();
    assert_eq!(
        out.to_vec2::<f32>()?,
        [[1.0, 2.0, 3.0], [400.0, 500.0, 600.0]],
        "row 0 active → take new; row 1 finished → take old (broadcast over 3 cols)",
    );
    Ok(())
}

#[test]
fn apply_state_mask_no_old_state_zeros_inactive_rows() -> Result<()> {
    // When old_state is None, inactive rows get 0 (masked-zero
    // behavior — there's no prior value to preserve).
    let dev = Device::cpu();
    let new = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &dev)?;
    let mask = StreamMask::new(vec![true, false, true]);
    let out = apply_state_mask(&Some(new), &None, &mask)?.unwrap();
    // Row 0 active → new; row 1 inactive + no old → 0; row 2 active.
    assert_eq!(out.to_vec2::<f32>()?, [[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]);
    Ok(())
}

#[test]
fn apply_state_mask_both_none_is_none() -> Result<()> {
    let mask = StreamMask::new(vec![true, false]);
    let out = apply_state_mask(&None, &None, &mask)?;
    assert!(out.is_none());
    Ok(())
}

#[test]
fn apply_state_mask_lost_new_state_is_err() {
    // (None, Some) violates the "constant streaming step" contract
    // (you can't go from "had state" to "no state" mid-stream).
    let dev = Device::cpu();
    let old = Tensor::new(&[[10.0f32], [20.0]], &dev).unwrap();
    let mask = StreamMask::new(vec![true, false]);
    let r = apply_state_mask(&None, &Some(old), &mask);
    assert!(r.is_err(), "expected Err on (None, Some(_)) but got {r:?}");
}

#[test]
fn apply_state_mask_preserves_dtype_bf16() -> Result<()> {
    use half::bf16;
    let dev = Device::cpu();
    let new = Tensor::new(
        &[
            [bf16::from_f32(1.0), bf16::from_f32(2.0)],
            [bf16::from_f32(3.0), bf16::from_f32(4.0)],
        ],
        &dev,
    )?;
    let old = Tensor::new(
        &[
            [bf16::from_f32(10.0), bf16::from_f32(20.0)],
            [bf16::from_f32(30.0), bf16::from_f32(40.0)],
        ],
        &dev,
    )?;
    let mask = StreamMask::new(vec![true, false]);
    let out = apply_state_mask(&Some(new), &Some(old), &mask)?.unwrap();
    assert_eq!(out.dtype(), fuel_core::DType::BF16);
    let v = out.to_vec2::<bf16>()?;
    assert_eq!(v[0][0].to_f32(), 1.0);
    assert_eq!(v[0][1].to_f32(), 2.0);
    assert_eq!(v[1][0].to_f32(), 30.0);
    assert_eq!(v[1][1].to_f32(), 40.0);
    Ok(())
}
