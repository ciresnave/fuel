#![cfg(all(feature = "cuda", feature = "vulkan"))]
//! Live tri-backend collective — **CPU + CUDA + Vulkan reducing together**.
//!
//! Every existing multi-vendor test in this repo uses CPU only as a *staging
//! intermediate*: `cuda_vulkan_multidevice_realize_live.rs` places compute on
//! CUDA and Vulkan and lets the host carry bytes between them. **No test places
//! compute on all three backends at once.** This one does — each backend holds a
//! shard that is a real operand of the reduction, not a waypoint.
//!
//! ## What it proves, and what it does not
//!
//! Proves: [`DeviceGroup`] reduces correctly when its shards are resident on
//! three *different* backends, including the cross-vendor pair (CUDA↔Vulkan)
//! that has no direct copy path and must be staged through the host by
//! `bring_to_leader`.
//!
//! Does not prove: performance, concurrency, or that the three backends execute
//! in parallel. This is a correctness gate.
//!
//! ## Hardware and selection
//!
//! Written for a box with an NVIDIA discrete GPU and an AMD integrated GPU. The
//! Vulkan handle is selected **by name** — `fuel::vulkan_backend::new_device()`
//! uses `PreferDiscrete`, which would hand back the *NVIDIA* card and make the
//! "CUDA + Vulkan" halves silently run on one vendor, proving nothing.
//!
//! ## Running
//!
//! ```text
//! cargo test -p fuel-parallel --features "cuda vulkan" \
//!     --test tri_backend_device_group_live -- --ignored --test-threads=1
//! ```
//!
//! Absent hardware makes each test return early and PASS, after printing why.

use fuel::lazy::LazyTensor;
use fuel::{Device, Shape};
use fuel_parallel::comm::ReduceOp;
use fuel_parallel::device_group::DeviceGroup;
use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

/// A live CUDA device, or `None` with an explanation.
fn cuda_or_skip() -> Option<Device> {
    match fuel::cuda_backend::new_device(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("SKIP: no usable CUDA device ({e})");
            None
        }
    }
}

/// A Vulkan device on the **integrated** GPU, or `None` with an explanation.
///
/// Prefers an AMD-named adapter; falls back to scanning `list_devices()` for one
/// reporting `"integrated"`. Deliberately never falls back to a discrete
/// adapter — a silent fallback would turn this into a same-vendor test that
/// still passes, which is worse than skipping.
fn vulkan_igpu_or_skip() -> Option<Device> {
    if let Ok(b) = VulkanBackend::with_selection(DeviceSelection::ByName("AMD".to_string())) {
        return Some(std::sync::Arc::new(b).into());
    }
    match VulkanBackend::list_devices() {
        Ok(devices) => {
            for (idx, name, kind) in devices {
                if kind == "integrated" {
                    match VulkanBackend::with_selection(DeviceSelection::Index(idx)) {
                        Ok(b) => return Some(std::sync::Arc::new(b).into()),
                        Err(e) => eprintln!("SKIP: integrated Vulkan device {idx} ({name}): {e}"),
                    }
                }
            }
            eprintln!("SKIP: no integrated Vulkan adapter found (refusing to fall back to discrete)");
            None
        }
        Err(e) => {
            eprintln!("SKIP: Vulkan enumeration failed ({e})");
            None
        }
    }
}

/// Build `n` shards on one graph. Shard `i` is `[value_i; len]`; the first is the
/// graph root and the rest hang off it, because every operand of a reduction must
/// share a graph.
fn shards_on_one_graph(values: &[f32], len: usize, dev: &Device) -> Vec<LazyTensor> {
    let shape = Shape::from_dims(&[len]);
    let mut out = Vec::with_capacity(values.len());
    let root = LazyTensor::from_f32(vec![values[0]; len], shape.clone(), dev);
    out.push(root);
    for v in &values[1..] {
        let t = LazyTensor::from_f32_on(out[0].graph(), vec![*v; len], shape.clone(), dev);
        out.push(t);
    }
    out
}

#[test]
#[ignore = "requires a live CUDA device AND an integrated Vulkan device"]
fn tri_backend_all_reduce_sums_across_cpu_cuda_and_vulkan() {
    let Some(cuda) = cuda_or_skip() else { return };
    let Some(vulkan) = vulkan_igpu_or_skip() else { return };
    let cpu = Device::cpu();

    // Leader is CPU: every hop is then a single implemented Copy, which isolates
    // "does a tri-backend reduce work at all" from the cross-vendor staging path
    // exercised by the next test.
    let group = DeviceGroup::new(vec![cpu.clone(), cuda, vulkan]).expect("3-device group");
    assert_eq!(group.size(), 3);

    const LEN: usize = 8;
    let shards = shards_on_one_graph(&[1.0, 2.0, 4.0], LEN, &cpu);
    let out = group
        .all_reduce(&shards, ReduceOp::Sum)
        .expect("tri-backend all_reduce builds");

    let got = group.realize_f32(&out).expect("multi-device realize");
    assert_eq!(got, vec![7.0f32; LEN], "1 + 2 + 4 across CPU, CUDA and Vulkan");
}

#[test]
#[ignore = "requires a live CUDA device AND an integrated Vulkan device"]
fn cross_vendor_leader_stages_through_host() {
    // Leader is CUDA and a shard lives on Vulkan, so the CUDA←Vulkan move has NO
    // direct copy path: `bring_to_leader` must author Vulkan→CPU→CUDA itself.
    // Before that staging existed this failed at execute time with a
    // foreign-GPU-output rejection, so this test is the regression guard for it.
    let Some(cuda) = cuda_or_skip() else { return };
    let Some(vulkan) = vulkan_igpu_or_skip() else { return };
    let cpu = Device::cpu();

    let group = DeviceGroup::new(vec![cuda, vulkan, cpu.clone()]).expect("3-device group");

    const LEN: usize = 4;
    let shards = shards_on_one_graph(&[10.0, 20.0, 30.0], LEN, &cpu);
    let out = group
        .all_reduce(&shards, ReduceOp::Sum)
        .expect("cross-vendor all_reduce builds");

    let got = group.realize_f32(&out).expect("multi-device realize");
    assert_eq!(got, vec![60.0f32; LEN], "CUDA leader, Vulkan shard staged via host");
}

#[test]
#[ignore = "requires a live CUDA device AND an integrated Vulkan device"]
fn tri_backend_all_reduce_max_is_elementwise() {
    // A non-commutative-looking reduce, to catch an implementation that happens
    // to be right for Sum by accident (e.g. dropping a shard would still give a
    // plausible sum, but not a plausible max).
    let Some(cuda) = cuda_or_skip() else { return };
    let Some(vulkan) = vulkan_igpu_or_skip() else { return };
    let cpu = Device::cpu();

    let group = DeviceGroup::new(vec![cpu.clone(), cuda, vulkan]).expect("3-device group");

    const LEN: usize = 4;
    let shards = shards_on_one_graph(&[3.0, 9.0, 5.0], LEN, &cpu);
    let out = group
        .all_reduce(&shards, ReduceOp::Max)
        .expect("tri-backend max builds");

    let got = group.realize_f32(&out).expect("multi-device realize");
    assert_eq!(got, vec![9.0f32; LEN], "max over the three shards");
}
