//! Oracle test for the per-vendor CPU conv2d paths (AOCL, oneMKL).
//! Each backend's `conv2d` should produce reference-equivalent output
//! within tolerance for dense f32 conv shapes (groups=1 — depthwise
//! still falls through to CpuBackend on these backends, same as
//! Vulkan and CUDA).

#![cfg(any(feature = "aocl", feature = "onemkl"))]

use fuel_core::lazy::LazyTensor;
use fuel_core_types::Shape;
use fuel_graph_executor::GraphExecutor;

fn build_conv_graph(
    n: usize, c_in: usize, h: usize, w: usize,
    c_out: usize, k: usize,
    stride: (usize, usize), padding: (usize, usize),
) -> LazyTensor {
    let x_data: Vec<f32> = (0..(n * c_in * h * w))
        .map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let w_data: Vec<f32> = (0..(c_out * c_in * k * k))
        .map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[n, c_in, h, w]));
    let weight = x.const_f32_like(w_data, Shape::from_dims(&[c_out, c_in, k, k]));
    x.conv2d(&weight, None, stride, padding, 1)
}

fn assert_close(reference: &[f32], got: &[f32], label: &str) {
    assert_eq!(reference.len(), got.len(), "{label}: length mismatch");
    for (i, (&r, &g)) in reference.iter().zip(got.iter()).enumerate() {
        let denom = r.abs().max(g.abs()).max(f32::MIN_POSITIVE);
        let rel = (r - g).abs() / denom;
        assert!(
            rel < 1e-4,
            "{label}: mismatch at {i}: reference={r}, got={g} (rel {rel})"
        );
    }
}

#[cfg(feature = "aocl")]
#[test]
fn aocl_conv2d_matches_reference_dense() {
    let backend = match fuel_aocl_cpu_backend::AoclBackend::try_new() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: AOCL not loadable: {e}"); return; }
    };
    let mut exe = GraphExecutor::new(backend);
    for &(n, ci, h, w, co, k, s, p) in &[
        (1, 4, 8, 8, 8, 3, (1, 1), (0, 0)),
        (2, 8, 16, 16, 16, 3, (1, 1), (1, 1)),
        (1, 16, 32, 32, 32, 3, (2, 2), (1, 1)),  // YOLO-style stride-2
    ] {
        let y = build_conv_graph(n, ci, h, w, co, k, s, p);
        let reference = y.realize_f32_reference();
        let aocl = y.realize_f32_aocl(&mut exe);
        assert_close(
            &reference, &aocl,
            &format!("aocl conv2d (n={n} ci={ci} h={h} w={w} co={co} k={k} s={s:?} p={p:?})"),
        );
    }
}

#[cfg(feature = "onemkl")]
#[test]
fn mkl_conv2d_matches_reference_dense() {
    let backend = match fuel_mkl_cpu_backend::MklBackend::try_new() {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: MKL not loadable: {e}"); return; }
    };
    let mut exe = GraphExecutor::new(backend);
    for &(n, ci, h, w, co, k, s, p) in &[
        (1, 4, 8, 8, 8, 3, (1, 1), (0, 0)),
        (2, 8, 16, 16, 16, 3, (1, 1), (1, 1)),
        (1, 16, 32, 32, 32, 3, (2, 2), (1, 1)),
    ] {
        let y = build_conv_graph(n, ci, h, w, co, k, s, p);
        let reference = y.realize_f32_reference();
        let mkl = y.realize_f32_mkl(&mut exe);
        assert_close(
            &reference, &mkl,
            &format!("mkl conv2d (n={n} ci={ci} h={h} w={w} co={co} k={k} s={s:?} p={p:?})"),
        );
    }
}
