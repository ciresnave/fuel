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
//! ## Running — and why "3 passed" IS now sufficient for device presence (GAP-157)
//!
//! ```text
//! cargo test -p fuel-parallel --features "cuda vulkan" \
//!     --test tri_backend_device_group_live -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! These tests are `#[ignore]`d, so they run ONLY when asked for explicitly
//! (`--ignored`). A missing or refused device is therefore a FAILURE, not a
//! skip: `cuda_or_fail`/`vulkan_igpu_or_fail` **panic** rather than `return`, so
//! a device-less run reddens instead of reporting `ok` having asserted nothing.
//! (GAP-157: this MECHANIZES what an earlier version documented and enforced by
//! a human remembering to grep stderr for `SKIP:` — the discriminator is now a
//! control, not a habit.)
//!
//! `--nocapture` is still worth passing, but now for the adapter IDENTITY, not
//! to catch a silent skip. The run should show **three `PROOF ` lines**, one
//! per test, each naming the Vulkan adapter actually used — the remaining
//! discriminator: if the Vulkan handle resolved to the NVIDIA card, "CUDA +
//! Vulkan" would be one vendor twice and every assertion here would still hold.
//! `vulkan_igpu_or_fail` prints the full adapter enumeration, selects by index,
//! and refuses NVIDIA outright (a same-vendor fallback now panics) — so the
//! `PROOF` line distinguishes a real tri-vendor reduce from a same-vendor one
//! wearing its clothes.

use fuel::lazy::LazyTensor;
use fuel::{Device, Shape};
use fuel_parallel::comm::ReduceOp;
use fuel_parallel::device_group::DeviceGroup;
use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

// ─── GAP-157 EXEMPLAR — deliberate; do NOT "fix" these back to skip ───
// `cuda_or_fail`/`vulkan_igpu_or_fail` below hard-FAIL (panic) instead of the
// copy-pasted `*_or_skip() -> Option` idiom the other ~39 live-test files use
// (which returns `None` and lets the test `return` — an early return from a
// `#[test]` is a silent PASS). This file is the INTENDED END-STATE for GAP-157;
// the neighbours still silently pass on a device-less `--ignored` run. That
// inconsistency is intentional and temporary: this is the template, and the
// neighbours are what get changed to match it (shared-helper refactor, GAP-162),
// not the reverse. Do NOT revert these to `_or_skip`. Remove this comment only
// once a shared `dev_or_fail`/`backend_or_fail` is the single source and every
// live-test file uses it.
//
/// A live CUDA device, or a PANIC with an explanation. GAP-157: this test is
/// `#[ignore]`d — it runs only when explicitly asked for (`--ignored`), so a
/// missing device is a FAILURE, not a skip (a silent early return would report
/// `ok` having asserted nothing).
fn cuda_or_fail() -> Device {
    match fuel::cuda_backend::new_device(0) {
        Ok(d) => d,
        Err(e) => panic!(
            "no usable CUDA device ({e}) — required by this explicitly-run (#[ignore]) test; \
             a missing device is a failure, not a skip"
        ),
    }
}

/// A Vulkan device on the **integrated** GPU plus its adapter name, or a PANIC
/// with an explanation (GAP-157: absence is a failure in an explicitly-run test).
///
/// Enumerates first and selects **by index**, printing the full adapter list and
/// the chosen entry. The earlier version asked for `ByName("AMD")` and returned
/// whatever came back, which is opaque: if that selector had resolved to the
/// NVIDIA card, the "CUDA + Vulkan" halves would both run on one vendor and
/// every assertion below would still pass while proving nothing. Selecting from
/// an enumeration this function printed makes the choice auditable after the
/// fact, and an explicit NVIDIA refusal makes the bad case a panic, not a
/// silent pass.
fn vulkan_igpu_or_fail() -> (Device, String) {
    let devices = match VulkanBackend::list_devices() {
        Ok(d) => d,
        Err(e) => panic!(
            "Vulkan enumeration failed ({e}) — required by this explicitly-run (#[ignore]) test"
        ),
    };
    eprintln!("Vulkan adapters visible to this process:");
    for (idx, name, kind) in &devices {
        eprintln!("    [{idx}] {name}  ({kind})");
    }

    let pick = devices
        .iter()
        .find(|(_, name, kind)| {
            kind.as_str() == "integrated" && !name.to_uppercase().contains("NVIDIA")
        })
        .or_else(|| {
            devices
                .iter()
                .find(|(_, name, _)| name.to_uppercase().contains("AMD"))
        });

    let Some((idx, name, kind)) = pick else {
        panic!(
            "no non-NVIDIA Vulkan adapter found — refusing to fall back to the discrete card \
             (that would make this a same-vendor test); required by this explicitly-run (#[ignore]) test"
        );
    };

    match VulkanBackend::with_selection(DeviceSelection::Index(*idx)) {
        Ok(b) => {
            eprintln!("Vulkan shard will run on [{idx}] {name} ({kind})");
            (std::sync::Arc::new(b).into(), name.clone())
        }
        Err(e) => panic!(
            "could not open Vulkan adapter [{idx}] {name}: {e} — required by this explicitly-run (#[ignore]) test"
        ),
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
    let cuda = cuda_or_fail();
    let (vulkan, vk_name) = vulkan_igpu_or_fail();
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
    assert_eq!(
        got,
        vec![7.0f32; LEN],
        "1 + 2 + 4 across CPU, CUDA and Vulkan"
    );
    eprintln!(
        "PROOF tri_backend_sum: CPU + CUDA + Vulkan({vk_name}) reduced to {}",
        got[0]
    );
}

#[test]
#[ignore = "requires a live CUDA device AND an integrated Vulkan device"]
fn cross_vendor_leader_stages_through_host() {
    // Leader is CUDA and a shard lives on Vulkan, so the CUDA←Vulkan move has NO
    // direct copy path: `bring_to_leader` must author Vulkan→CPU→CUDA itself.
    // Before that staging existed this failed at execute time with a
    // foreign-GPU-output rejection, so this test is the regression guard for it.
    let cuda = cuda_or_fail();
    let (vulkan, vk_name) = vulkan_igpu_or_fail();
    let cpu = Device::cpu();

    let group = DeviceGroup::new(vec![cuda, vulkan, cpu.clone()]).expect("3-device group");

    const LEN: usize = 4;
    let shards = shards_on_one_graph(&[10.0, 20.0, 30.0], LEN, &cpu);
    let out = group
        .all_reduce(&shards, ReduceOp::Sum)
        .expect("cross-vendor all_reduce builds");

    let got = group.realize_f32(&out).expect("multi-device realize");
    assert_eq!(
        got,
        vec![60.0f32; LEN],
        "CUDA leader, Vulkan shard staged via host"
    );
    eprintln!(
        "PROOF cross_vendor_staging: CUDA leader + Vulkan({vk_name}) shard staged via host -> {}",
        got[0]
    );
}

#[test]
#[ignore = "requires a live CUDA device AND an integrated Vulkan device"]
fn tri_backend_all_reduce_max_is_elementwise() {
    // A non-commutative-looking reduce, to catch an implementation that happens
    // to be right for Sum by accident (e.g. dropping a shard would still give a
    // plausible sum, but not a plausible max).
    let cuda = cuda_or_fail();
    let (vulkan, vk_name) = vulkan_igpu_or_fail();
    let cpu = Device::cpu();

    let group = DeviceGroup::new(vec![cpu.clone(), cuda, vulkan]).expect("3-device group");

    const LEN: usize = 4;
    let shards = shards_on_one_graph(&[3.0, 9.0, 5.0], LEN, &cpu);
    let out = group
        .all_reduce(&shards, ReduceOp::Max)
        .expect("tri-backend max builds");

    let got = group.realize_f32(&out).expect("multi-device realize");
    assert_eq!(got, vec![9.0f32; LEN], "max over the three shards");
    eprintln!(
        "PROOF tri_backend_max: CPU + CUDA + Vulkan({vk_name}) max -> {}",
        got[0]
    );
}
