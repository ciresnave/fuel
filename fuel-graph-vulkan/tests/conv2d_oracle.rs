//! Oracle test for Vulkan conv2d: synthetic conv graphs realized
//! through the Vulkan backend match `fuel-reference-backend`'s output
//! within tolerance.
//!
//! Coverage in this file:
//! - Dense conv (groups=1) with stride 1, no padding (the simplest path)
//! - Dense conv with stride 1, symmetric padding (typical "preserve
//!   spatial" 3×3 conv)
//! - Stride-2 dense conv (YOLOv8-style downsample)
//! - Asymmetric stride/padding (Vulkan supports it; CUDA bails on it
//!   today, so this is a Vulkan-only capability check)
//!
//! Depthwise (groups != 1) is NOT covered here — VulkanBackend::conv2d
//! currently bails on `groups != 1` for parity with CUDA, falling
//! through to the CPU reference backend. ConvNeXt depthwise still
//! works correctness-wise, just slowly. That parity gap closes when
//! both backends gain grouped support together.
//!
//! Tests are `#[ignore]` because they require a live Vulkan device.
//! Run explicitly:
//!
//! ```sh
//! cargo test -p fuel-graph-vulkan --test conv2d_oracle -- --ignored --nocapture
//! ```

use fuel_core_types::{HostBuffer, Layout, Shape};
use fuel_graph::Tensor;
use fuel_graph_executor::{GraphBackend, GraphExecutor};
use fuel_graph_vulkan::{DeviceSelection, VulkanBackend};

/// Lazily constructed Vulkan executor; skips tests if the backend
/// doesn't initialize on this host.
fn vulkan_exec() -> Option<GraphExecutor<VulkanBackend>> {
    match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => Some(GraphExecutor::new(b)),
        Err(e) => {
            eprintln!("skipping: Vulkan backend init failed: {e}");
            None
        }
    }
}

fn conv2d_oracle_check(
    n: usize, c_in: usize, h: usize, w: usize,
    c_out: usize, k: usize,
    stride: (usize, usize), padding: (usize, usize),
    groups: usize,
) {
    let mut exe = match vulkan_exec() { Some(e) => e, None => return };

    let x_data: Vec<f32> = (0..(n * c_in * h * w))
        .map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let cin_per_g = c_in / groups;
    let w_data: Vec<f32> = (0..(c_out * cin_per_g * k * k))
        .map(|i| ((i as f32) * 1.7e-3).cos()).collect();

    let x = Tensor::from_f32(x_data, Shape::from_dims(&[n, c_in, h, w]));
    let weight = x.const_f32_like(w_data, Shape::from_dims(&[c_out, cin_per_g, k, k]));
    let y = x.conv2d(&weight, None, stride, padding, groups);

    let reference = fuel_reference_backend::exec::realize_f32(&y).into_vec();
    let vulkan = exe.realize_f32(&y).into_vec();

    assert_eq!(reference.len(), vulkan.len(),
        "shape mismatch: reference={} vulkan={}", reference.len(), vulkan.len());
    for (i, (&r, &v)) in reference.iter().zip(vulkan.iter()).enumerate() {
        let denom = r.abs().max(v.abs()).max(f32::MIN_POSITIVE);
        let rel = (r - v).abs() / denom;
        assert!(
            rel < 5e-4,
            "conv2d (n={n} c_in={c_in} h={h} w={w} c_out={c_out} k={k} \
             stride={stride:?} pad={padding:?} groups={groups}) \
             mismatch at {i}: reference={r}, vulkan={v} (rel {rel})"
        );
    }
}

#[test]
#[ignore]
fn dense_no_pad_stride_1() {
    conv2d_oracle_check(1, 4, 8, 8, 8, 3, (1, 1), (0, 0), 1);
}

#[test]
#[ignore]
fn dense_symmetric_pad() {
    conv2d_oracle_check(2, 8, 16, 16, 16, 3, (1, 1), (1, 1), 1);
}

#[test]
#[ignore]
fn yolo_style_stride_2() {
    conv2d_oracle_check(1, 16, 32, 32, 32, 3, (2, 2), (1, 1), 1);
}

#[test]
#[ignore]
fn asymmetric_stride_padding() {
    // Vulkan supports asymmetric out of the box (separate stride.h /
    // stride.w / pad.h / pad.w in the im2col shader). CUDA today bails
    // on this case; running it through Vulkan exercises a capability
    // CUDA doesn't have yet.
    conv2d_oracle_check(1, 4, 12, 16, 4, 3, (2, 1), (1, 0), 1);
}

#[test]
#[ignore]
fn depthwise_falls_back_to_cpu() {
    // Sanity-check the bail path: groups != 1 should NOT panic. The
    // executor catches Vulkan's bail and falls back to the reference
    // backend's conv2d for correctness.
    let mut exe = match vulkan_exec() { Some(e) => e, None => return };
    let n = 1; let c = 8; let h = 8; let w_sz = 8; let k = 3;
    let x_data: Vec<f32> = (0..(n * c * h * w_sz))
        .map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let w_data: Vec<f32> = (0..(c * 1 * k * k))
        .map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let x = Tensor::from_f32(x_data, Shape::from_dims(&[n, c, h, w_sz]));
    let weight = x.const_f32_like(w_data, Shape::from_dims(&[c, 1, k, k]));
    let y = x.conv2d(&weight, None, (1, 1), (1, 1), c);  // depthwise

    // Just need to confirm it doesn't panic and produces some output.
    // Numerical match is exercised by the (correct) reference fallback.
    let out = exe.realize_f32(&y).into_vec();
    assert_eq!(out.len(), n * c * h * w_sz);
}

// Direct-call test that bypasses the LazyTensor layer and goes
// straight at the GraphBackend interface — handy for shader-level
// debugging.
#[test]
#[ignore]
fn direct_conv2d_call_vs_reference() {
    let backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => { eprintln!("skipping: Vulkan init failed: {e}"); return; }
    };
    let n = 1; let c_in = 4; let h = 6; let w_sz = 6;
    let c_out = 3; let k = 3;
    let x_data: Vec<f32> = (0..(n * c_in * h * w_sz))
        .map(|i| (i as f32) * 0.05).collect();
    let weight_data: Vec<f32> = (0..(c_out * c_in * k * k))
        .map(|i| ((i as f32) * 0.07).sin()).collect();

    let x_storage = backend.upload(
        &HostBuffer::F32(x_data.clone()),
        &Shape::from_dims(&[n, c_in, h, w_sz]),
    ).expect("upload x");
    let weight_storage = backend.upload(
        &HostBuffer::F32(weight_data.clone()),
        &Shape::from_dims(&[c_out, c_in, k, k]),
    ).expect("upload weight");
    let xl = Layout::contiguous(&Shape::from_dims(&[n, c_in, h, w_sz]));
    let wl = Layout::contiguous(&Shape::from_dims(&[c_out, c_in, k, k]));

    let out = backend.conv2d(
        &x_storage, &weight_storage,
        &xl, &wl,
        (1, 1), (1, 1), 1,
    ).expect("conv2d on Vulkan");
    let host = backend.download(&out).expect("download");
    let vulkan_out = match host {
        HostBuffer::F32(v) => v,
        _ => panic!("expected F32 output"),
    };

    // Reference oracle via fuel-conv's direct conv2d.
    let s = fuel_conv::ConvShape {
        batch: n, c_in, h, w: w_sz,
        c_out, k_h: k, k_w: k,
        stride: (1, 1), padding: (1, 1), groups: 1,
    };
    let mut reference_out = vec![0.0_f32; s.output_len()];
    fuel_conv::conv2d_direct(&x_data, &weight_data, None, &s, &mut reference_out);

    assert_eq!(vulkan_out.len(), reference_out.len());
    for (i, (&v, &r)) in vulkan_out.iter().zip(reference_out.iter()).enumerate() {
        let denom = v.abs().max(r.abs()).max(f32::MIN_POSITIVE);
        let rel = (v - r).abs() / denom;
        assert!(rel < 5e-4, "direct conv2d mismatch at {i}: vulkan={v}, ref={r} (rel {rel})");
    }
}
