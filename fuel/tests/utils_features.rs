// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `fuel::utils` build-feature probes, carried over from the dissolved
//! `fuel-core`'s `utils.rs` tests (GAP-348). Each is true only under its own
//! feature, so run them with `--features accelerate|mkl|metal`.

#[test]
fn metal_is_available_tracks_the_metal_feature() {
    assert_eq!(fuel::utils::metal_is_available(), cfg!(feature = "metal"));
}

#[cfg(feature = "accelerate")]
#[test]
fn has_accelerate_is_true_under_its_own_feature() {
    assert!(fuel::utils::has_accelerate());
}

#[cfg(feature = "mkl")]
#[test]
fn has_mkl_is_true_under_its_own_feature() {
    assert!(fuel::utils::has_mkl());
}

#[cfg(not(feature = "accelerate"))]
#[test]
fn has_accelerate_is_false_without_its_feature() {
    assert!(!fuel::utils::has_accelerate());
}

#[cfg(not(feature = "mkl"))]
#[test]
fn has_mkl_is_false_without_its_feature() {
    assert!(!fuel::utils::has_mkl());
}
