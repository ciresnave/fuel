//! Arbitrary-scale `LazyTensor::interpolate2d` parity with the
//! eager UpsampleNearest2D kernel.
//!
//! The lazy primitive supports non-integer + non-uniform ratios
//! via an index_select-based composite. The indexing convention
//! must match `fuel-cpu-backend::ops::UpsampleNearest2D`:
//!     src_h[oi] = min(H - 1, floor(oi * H / H_out))
//!
//! Unblocks DepthAnythingV2's DPT head and any dense-prediction
//! head that resizes features to arbitrary targets.

use fuel_core::lazy::LazyTensor;
use fuel_core_types::Shape;

/// Reproduce the eager UpsampleNearest2D convention in plain
/// Rust so the lazy output can be checked element-wise.
fn nearest_oracle(
    src: &[f32], n: usize, c: usize, h: usize, w: usize,
    target_h: usize, target_w: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0_f32; n * c * target_h * target_w];
    let src_h_idx: Vec<usize> = (0..target_h)
        .map(|oi| ((oi * h) / target_h).min(h - 1))
        .collect();
    let src_w_idx: Vec<usize> = (0..target_w)
        .map(|oj| ((oj * w) / target_w).min(w - 1))
        .collect();
    for b in 0..n {
        for k in 0..c {
            for oi in 0..target_h {
                for oj in 0..target_w {
                    let si = src_h_idx[oi];
                    let sj = src_w_idx[oj];
                    let s_off = ((b * c + k) * h + si) * w + sj;
                    let d_off = ((b * c + k) * target_h + oi) * target_w + oj;
                    dst[d_off] = src[s_off];
                }
            }
        }
    }
    dst
}

#[test]
fn interpolate2d_integer_uniform_matches_oracle() {
    let dev = fuel_core::Device::cpu();
    let (n, c, h, w) = (1, 2, 4, 4);
    let src: Vec<f32> = (0..n*c*h*w).map(|i| i as f32 * 0.1).collect();
    let lt = LazyTensor::from_f32(src.clone(), Shape::from_dims(&[n, c, h, w]), &dev);
    // 2x uniform — should hit the fast path.
    let out = lt.interpolate2d(8, 8).unwrap();
    let shape = out.shape();
    assert_eq!(shape.dims(), &[1, 2, 8, 8]);
    let got = out.realize_f32();
    let want = nearest_oracle(&src, n, c, h, w, 8, 8);
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "uniform 2x mismatch at {i}: got {a}, expected {b}");
    }
}

#[test]
fn interpolate2d_non_integer_uniform_matches_oracle() {
    let dev = fuel_core::Device::cpu();
    let (n, c, h, w) = (1, 2, 4, 4);
    let src: Vec<f32> = (0..n*c*h*w).map(|i| (i as f32 - 8.0) * 0.25).collect();
    let lt = LazyTensor::from_f32(src.clone(), Shape::from_dims(&[n, c, h, w]), &dev);
    // 4 → 7 is non-integer ratio; takes the composite path.
    let out = lt.interpolate2d(7, 7).unwrap();
    assert_eq!(out.shape().dims(), &[1, 2, 7, 7]);
    let got = out.realize_f32();
    let want = nearest_oracle(&src, n, c, h, w, 7, 7);
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6,
            "non-integer 4→7 mismatch at {i}: got {a}, expected {b}");
    }
}

#[test]
fn interpolate2d_non_uniform_matches_oracle() {
    let dev = fuel_core::Device::cpu();
    let (n, c, h, w) = (1, 1, 3, 5);
    let src: Vec<f32> = (0..n*c*h*w).map(|i| (i as f32) * 0.3 + 0.1).collect();
    let lt = LazyTensor::from_f32(src.clone(), Shape::from_dims(&[n, c, h, w]), &dev);
    // 3→8 in H, 5→6 in W — different ratios per axis.
    let out = lt.interpolate2d(8, 6).unwrap();
    assert_eq!(out.shape().dims(), &[1, 1, 8, 6]);
    let got = out.realize_f32();
    let want = nearest_oracle(&src, n, c, h, w, 8, 6);
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6,
            "non-uniform mismatch at {i}: got {a}, expected {b}");
    }
}

#[test]
fn interpolate2d_downsample_matches_oracle() {
    let dev = fuel_core::Device::cpu();
    let (n, c, h, w) = (1, 1, 8, 8);
    let src: Vec<f32> = (0..n*c*h*w).map(|i| (i as f32) * 0.1).collect();
    let lt = LazyTensor::from_f32(src.clone(), Shape::from_dims(&[n, c, h, w]), &dev);
    // Downsampling 8→3 is still nearest under the same convention.
    let out = lt.interpolate2d(3, 3).unwrap();
    assert_eq!(out.shape().dims(), &[1, 1, 3, 3]);
    let got = out.realize_f32();
    let want = nearest_oracle(&src, n, c, h, w, 3, 3);
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6,
            "downsample 8→3 mismatch at {i}: got {a}, expected {b}");
    }
}

#[test]
fn interpolate2d_identity_returns_clone() {
    let dev = fuel_core::Device::cpu();
    let (n, c, h, w) = (1, 1, 4, 4);
    let src: Vec<f32> = (0..n*c*h*w).map(|i| i as f32).collect();
    let lt = LazyTensor::from_f32(src.clone(), Shape::from_dims(&[n, c, h, w]), &dev);
    let out = lt.interpolate2d(h, w).unwrap();
    assert_eq!(out.shape().dims(), &[1, 1, 4, 4]);
    let got = out.realize_f32();
    for (i, (a, b)) in got.iter().zip(src.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "identity mismatch at {i}: got {a}, expected {b}");
    }
}
