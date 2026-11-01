// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 114, first step: does ONE graph with nodes on `Vulkan { gpu_id: 0 }` and
//! `Vulkan { gpu_id: 1 }` realize? The Qwen3.8-27B rig is a pipeline split across two Vulkan GPUs of
//! one backend. `seed_extra_device_handles` de-duplicates extra devices by `BackendId`, so a second
//! Vulkan handle may never reach the executor; this test measures that instead of assuming it.
//!
//! The graph: `v0 = a * b` pinned to the first Vulkan GPU, `v1 = v0 + a` pinned to the second, and
//! `out = v1 * b` back on the first, so the data crosses the GPU boundary in both directions (each
//! crossing is a two-hop copy through the host, as for CUDA and Vulkan).
//!
//! Needs two Vulkan devices; it skips (with a message) when the box has fewer than two. Run:
//!   pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -j 2 -p fuel-tensor --features vulkan --test vulkan_two_gpu_realize_live -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "vulkan")]

use std::sync::Arc;

use fuel_ir::{DeviceLocation, Shape};
use fuel_tensor::lazy::Tensor;
use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

fn place(t: &Tensor, loc: DeviceLocation) {
    let gt = t.graph_tensor();
    let id = gt.id();
    gt.graph()
        .write()
        .expect("graph lock")
        .set_placement(id, loc);
}

fn two_vulkan_devices_or_skip() -> Option<(Arc<VulkanBackend>, Arc<VulkanBackend>)> {
    let first = VulkanBackend::with_selection(DeviceSelection::Index(0)).ok()?;
    let second = VulkanBackend::with_selection(DeviceSelection::Index(1)).ok()?;
    eprintln!(
        "Vulkan device 0: gpu_id={}, device 1: gpu_id={}",
        first.gpu_id, second.gpu_id
    );
    if first.gpu_id == second.gpu_id {
        eprintln!("both selections gave gpu_id {}; skipping", first.gpu_id);
        return None;
    }
    Some((Arc::new(first), Arc::new(second)))
}

#[test]
#[ignore = "requires two live Vulkan devices"]
fn one_graph_spans_two_vulkan_gpus() {
    let Some((vk0, vk1)) = two_vulkan_devices_or_skip() else {
        eprintln!("fewer than two Vulkan devices; skipping");
        return;
    };
    let loc0 = DeviceLocation::Vulkan { gpu_id: vk0.gpu_id };
    let loc1 = DeviceLocation::Vulkan { gpu_id: vk1.gpu_id };

    let a = Tensor::from_f32(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        &fuel_tensor::Device::cpu(),
    )
    .unwrap();
    let b = a
        .const_f32_like(vec![10.0_f32, 20.0, 30.0, 40.0], Shape::from_dims(&[4]))
        .unwrap();

    let v0 = a.mul(&b).unwrap(); // [10, 40, 90, 160]   on GPU 0
    let v1 = v0.add(&a).unwrap(); // [11, 42, 93, 164]  on GPU 1
    let out = v1.mul(&b).unwrap(); // [110, 840, 2790, 6560] back on GPU 0
    place(&v0, loc0);
    place(&v1, loc1);
    place(&out, loc0);
    let out_id = out.graph_tensor().id();
    let g_arc = out.graph_tensor().graph().clone();

    let dev0: fuel_tensor::Device = vk0.clone().into();
    let dev1: fuel_tensor::Device = vk1.clone().into();
    let got = fuel_tensor::pipelined_bridge::realize_one_as_multi_device::<f32>(
        &g_arc,
        out_id,
        &dev0,
        &[&dev1],
    );
    eprintln!("two-GPU realize result: {got:?}");
    assert_eq!(
        got.expect("a graph spanning two Vulkan GPUs must realize"),
        vec![110.0_f32, 840.0, 2790.0, 6560.0]
    );
    // Both devices drained: no leaked in-flight batch on either GPU. (The executor looks a handle up
    // by exact `gpu_id` and errors when none matches, so work placed on GPU 1 cannot silently run on
    // GPU 0; the pre-fix run of this test failed at seeding, which is the sabotage control.)
    assert_eq!(fuel_dispatch::dispatch::inflight_count(loc0), 0);
    assert_eq!(fuel_dispatch::dispatch::inflight_count(loc1), 0);
}
