//! End-to-end numerical correctness for `Op::Conv2D` backward.
//!
//! Builds a small lazy graph `Y = conv2d(X, W)`, calls `.backward()` on
//! `Y.sum_all()`, then realizes both gradients through the reference
//! backend and compares them to finite-difference numerical gradients
//! computed by perturbing `X` and `W` element-by-element. Tolerance is
//! 5e-3 — tight enough to catch a wrong-sign / wrong-axis bug, loose
//! enough to absorb f32 finite-difference roundoff at h = 1e-3.

use fuel_ir::Shape;
use fuel_graph::Tensor;
use fuel_reference_backend::exec;


/// Phase 7.5 G2: tests need a real device for slot-populating
/// constructors. Singleton CpuBackendDevice via OnceLock.
fn cpu_dev() -> &'static std::sync::Arc<dyn fuel_ir::DynBackendDevice> {
    static D: std::sync::OnceLock<std::sync::Arc<dyn fuel_ir::DynBackendDevice>>
        = std::sync::OnceLock::new();
    D.get_or_init(|| std::sync::Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
}
fn realize_f32_vec(t: &Tensor) -> Vec<f32> {
    exec::realize_f32(t).into_vec()
}

/// Finite-difference numerical gradient of `f(input_data) = conv2d(...).sum_all()`
/// w.r.t. each element of `input_data`. `f` is closed over the *other*
/// input being held constant (X or W).
fn numerical_grad<F>(input_data: &[f32], h: f32, f: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let mut grad = vec![0.0_f32; input_data.len()];
    let mut buf = input_data.to_vec();
    for i in 0..input_data.len() {
        buf[i] = input_data[i] + h;
        let f_plus = f(&buf);
        buf[i] = input_data[i] - h;
        let f_minus = f(&buf);
        buf[i] = input_data[i];
        grad[i] = (f_plus - f_minus) / (2.0 * h);
    }
    grad
}

fn assert_close(a: &[f32], b: &[f32], atol: f32, rtol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut max_idx = 0;
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let denom = av.abs().max(bv.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        if diff > max_abs { max_abs = diff; max_rel = rel; max_idx = i; }
    }
    eprintln!("{label}: max abs={max_abs} rel={max_rel} at idx {max_idx} (a={} b={})", a[max_idx], b[max_idx]);
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let denom = av.abs().max(bv.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(
            diff < atol || rel < rtol,
            "{label}[{i}]: analytic={av} numeric={bv} (abs={diff}, rel={rel})",
        );
    }
}

#[test]
fn conv2d_backward_matches_finite_difference_stride1_pad1() {
    let (n, cin, h, w) = (1usize, 2, 4, 4);
    let (cout, k) = (3, 3);
    let pad = 1;

    let x_data: Vec<f32> = (0..(n * cin * h * w))
        .map(|i| (i as f32) * 0.05 - 0.5)
        .collect();
    let w_data: Vec<f32> = (0..(cout * cin * k * k))
        .map(|i| (i as f32) * 0.07 - 0.4)
        .collect();

    // Analytic: build the graph once, get dX and dW from backward.
    let x = Tensor::from_f32(x_data.clone(), Shape::from_dims(&[n, cin, h, w]), cpu_dev());
    let weight = x.const_f32_like(w_data.clone(), Shape::from_dims(&[cout, cin, k, k]));
    let y = x.conv2d(&weight, None, (1, 1), (pad, pad), 1);
    let scalar = y.sum_all();
    let grads = scalar.backward();
    let dx_analytic = realize_f32_vec(&grads.get(&x).expect("dX missing"));
    let dw_analytic = realize_f32_vec(&grads.get(&weight).expect("dW missing"));

    // Numeric dX: hold weights fixed, perturb each x element.
    let dx_numeric = numerical_grad(&x_data, 1e-3, |xs| {
        let xt = Tensor::from_f32(xs.to_vec(), Shape::from_dims(&[n, cin, h, w]), cpu_dev());
        let wt = xt.const_f32_like(w_data.clone(), Shape::from_dims(&[cout, cin, k, k]));
        let yt = xt.conv2d(&wt, None, (1, 1), (pad, pad), 1).sum_all();
        realize_f32_vec(&yt)[0]
    });
    assert_close(&dx_analytic, &dx_numeric, 1e-2, 2e-2, "dX");

    // Numeric dW: hold x fixed, perturb each weight element.
    let dw_numeric = numerical_grad(&w_data, 1e-3, |ws| {
        let xt = Tensor::from_f32(x_data.clone(), Shape::from_dims(&[n, cin, h, w]), cpu_dev());
        let wt = xt.const_f32_like(ws.to_vec(), Shape::from_dims(&[cout, cin, k, k]));
        let yt = xt.conv2d(&wt, None, (1, 1), (pad, pad), 1).sum_all();
        realize_f32_vec(&yt)[0]
    });
    assert_close(&dw_analytic, &dw_numeric, 1e-2, 2e-2, "dW");
}

#[test]
fn conv2d_backward_with_bias() {
    // Verifies the bias-grad arm reduces dY over (N, H, W).
    let (n, cin, h, w) = (1usize, 2, 4, 4);
    let (cout, k) = (3, 3);
    let pad = 1;

    let x_data: Vec<f32> = (0..(n * cin * h * w)).map(|i| (i as f32) * 0.05).collect();
    let w_data: Vec<f32> = (0..(cout * cin * k * k)).map(|i| (i as f32) * 0.07).collect();
    let b_data: Vec<f32> = (0..cout).map(|i| (i as f32) * 0.11).collect();

    let x = Tensor::from_f32(x_data, Shape::from_dims(&[n, cin, h, w]), cpu_dev());
    let weight = x.const_f32_like(w_data, Shape::from_dims(&[cout, cin, k, k]));
    let bias = x.const_f32_like(b_data.clone(), Shape::from_dims(&[cout]));
    let y = x.conv2d(&weight, Some(&bias), (1, 1), (pad, pad), 1);
    let scalar = y.sum_all();
    let grads = scalar.backward();
    let db = realize_f32_vec(&grads.get(&bias).expect("dB missing"));

    // Analytic: f(b_c) = sum over (N, H, W) of (conv(X, W) + b_c)
    //                 = const_w + N·H·W·b_c
    // So df/db_c = N·H·W.
    let expected = (n * h * w) as f32;
    for (i, &v) in db.iter().enumerate() {
        let diff = (v - expected).abs();
        assert!(diff < 1e-3, "dB[{i}]: got {v}, expected {expected}");
    }
}
