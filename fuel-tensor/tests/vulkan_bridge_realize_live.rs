// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-Vulkan bridge realize — proves the `optimize_graph` realize
//! path (the single path after Phase A PR-A3b-2) realizes correctly on
//! the Vulkan backend through `pipelined_bridge`
//! (`Tensor::realize_f32_vulkan` → `pipelined_bridge::realize_one_as`
//! → `PipelinedExecutor`), matching the host oracle. The path is
//! backend-agnostic (it lives in the generic `realize_one_as`), so this
//! is the Vulkan counterpart of the CPU `--lib` suite and the live-CUDA
//! `recip_abs_realize_live` / `phase_c_rotating_kv_cuda` smokes.
//!
//! Run:
//!
//!   cargo test -p fuel-tensor --features vulkan --test vulkan_bridge_realize_live -- --ignored --test-threads=1
//!
//! Gated `#[ignore]`; requires a live Vulkan device (RTX 4070 on the dev box).

#![cfg(feature = "vulkan")]

use std::sync::Arc;

use fuel_ir::{DType, DeviceLocation, Shape};
use fuel_tensor::lazy::Tensor;
use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

fn backend_or_skip() -> Option<Arc<VulkanBackend>> {
    match VulkanBackend::new() {
        Ok(b) => Some(Arc::new(b)),
        Err(e) => {
            eprintln!("no Vulkan device; skipping: {e:?}");
            None
        }
    }
}

/// Stamp an explicit per-node placement on a `Tensor`'s node (the scheduler-assignment seam).
fn place(t: &Tensor, loc: DeviceLocation) {
    let gt = t.graph_tensor();
    let id = gt.id();
    gt.graph()
        .write()
        .expect("graph lock")
        .set_placement(id, loc);
}

/// ROADMAP row 114: one graph with nodes on `Vulkan { 0 }`, `Vulkan { 1 }` and `Vulkan { 0 }` again
/// must realize. `seed_extra_device_handles` once de-duplicated the extra devices by backend, so the
/// second GPU got no handle and this failed with "no Vulkan handle to seed" (the sabotage control).
/// Runs inside an existing ignored test (the `#[ignore]` ceiling only goes down); it prints and
/// returns when the box has fewer than two Vulkan devices.
fn two_gpu_graph_realizes() {
    // Choose the two GPUs by IDENTITY, not by index: the Vulkan enumeration order is not stable on
    // this box (it flipped between two runs), so "index 0 and 1" can be a different pair of physical
    // devices from run to run. Take the devices with a usable UUID, order them by UUID (independent
    // of enumeration order), open those two by their current indices, then check the handles we got
    // really are the physical devices we asked for (ROADMAP rows 114 and 125).
    let ids = fuel_vulkan_backend::probe::device_identities().unwrap_or_default();
    let mut usable: Vec<(usize, [u8; 16])> = ids
        .iter()
        .enumerate()
        .filter_map(|(i, id)| id.and_then(|d| d.usable_uuid()).map(|u| (i, u)))
        .collect();
    usable.sort_by_key(|(_, u)| *u);
    if usable.len() < 2 {
        eprintln!("fewer than two Vulkan devices with a usable UUID; skipping the two-GPU graph");
        return;
    }
    let ((i0, u0), (i1, u1)) = (usable[0], usable[1]);
    let open = |i: usize| VulkanBackend::with_selection(DeviceSelection::Index(i)).ok();
    let (Some(vk0), Some(vk1)) = (open(i0), open(i1)) else {
        eprintln!("could not open the two chosen Vulkan devices; skipping the two-GPU graph");
        return;
    };
    assert_eq!(
        vk0.identity().and_then(|d| d.usable_uuid()),
        Some(u0),
        "the first pinned handle is not the physical device asked for"
    );
    assert_eq!(
        vk1.identity().and_then(|d| d.usable_uuid()),
        Some(u1),
        "the second pinned handle is not the physical device asked for"
    );
    assert_ne!(u0, u1, "two distinct physical devices");
    assert_ne!(vk0.gpu_id, vk1.gpu_id);
    eprintln!(
        "two-GPU graph: uuid {u0:02x?} is gpu_id {}, uuid {u1:02x?} is gpu_id {}",
        vk0.gpu_id, vk1.gpu_id
    );
    let (vk0, vk1) = (Arc::new(vk0), Arc::new(vk1));
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
    // Both devices drained. (The executor looks a handle up by exact `gpu_id` and errors when none
    // matches, so work placed on GPU 1 cannot silently run on GPU 0.)
    assert_eq!(fuel_dispatch::dispatch::inflight_count(loc0), 0);
    assert_eq!(fuel_dispatch::dispatch::inflight_count(loc1), 0);
}

/// `(a + b) * a` realized on Vulkan through the bridge matches the host
/// oracle — the exact `[11, 44, 99, 176]` the CPU suite and the CUDA
/// smoke assert, so the `optimize_graph` realize path is correct on the
/// Vulkan backend too.
#[test]
#[ignore = "requires a live Vulkan device"]
fn mul_add_realize_on_vulkan_matches_reference() {
    let Some(backend) = backend_or_skip() else {
        return;
    };

    let a = Tensor::from_f32(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        &fuel_tensor::Device::cpu(),
    )
    .unwrap();
    // `const_f32_like` keeps `b` in `a`'s graph (a bare second
    // `from_f32` would mint a separate graph and `add` across graphs
    // would fail).
    let b = a
        .const_f32_like(vec![10.0_f32, 20.0, 30.0, 40.0], Shape::from_dims(&[4]))
        .unwrap();
    // Tensor binary ops are fallible (shape/broadcast validation),
    // unlike the graph-`Tensor` ops; surface any error rather than panic
    // implicitly.
    let c = a.add(&b).expect("add").mul(&a).expect("mul");
    assert_eq!(c.dtype(), DType::F32);

    let out = c.realize_f32_vulkan(&backend);
    assert_eq!(out, vec![11.0_f32, 44.0, 99.0, 176.0]);
}

/// Step E A2 (Vulkan async): a DEEPER chain with fan-out, so several Vulkan
/// ops accumulate in the deferred batch (pipelined on the single queue) and an
/// intermediate is consumed twice (`t1`) — stressing batch accumulation,
/// same-queue producer→consumer ordering, and any in-place eviction
/// (`force_flush` before a destructive `cache.remove`). The host oracle:
///   t1 = a+b = [11,22,33,44];  t2 = t1*a = [11,44,99,176]
///   t3 = t2+t1 = [22,66,132,220];  out = t3*a = [22,132,396,880]
/// Correct bytes under deferred flush ⇒ the A2 sync points are complete.
#[test]
#[ignore = "requires a live Vulkan device"]
fn deep_chain_realize_on_vulkan_matches_reference() {
    let Some(backend) = backend_or_skip() else {
        return;
    };

    let a = Tensor::from_f32(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        &fuel_tensor::Device::cpu(),
    )
    .unwrap();
    let b = a
        .const_f32_like(vec![10.0_f32, 20.0, 30.0, 40.0], Shape::from_dims(&[4]))
        .unwrap();

    let t1 = a.add(&b).expect("add");
    let t2 = t1.mul(&a).expect("mul");
    let t3 = t2.add(&t1).expect("add2"); // reuses t1 (fan-out)
    let out_t = t3.mul(&a).expect("mul2");

    let out = out_t.realize_f32_vulkan(&backend);
    assert_eq!(out, vec![22.0_f32, 132.0, 396.0, 880.0]);

    // Also checks ROADMAP row 114 (two GPUs of one backend in one graph); see `two_gpu_graph_realizes`.
    two_gpu_graph_realizes();
}
