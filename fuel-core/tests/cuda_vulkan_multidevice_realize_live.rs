//! Step E A4c-prerequisite — multi-VENDOR (CUDA + Vulkan) realize on a SINGLE
//! pass.
//!
//! Sibling of `cuda_multidevice_realize_live.rs` (CPU+CUDA, the A4a-1 guard).
//! This one closes the CUDA<->Vulkan (cross-VENDOR GPU) gap that CPU+CUDA never
//! exercised. Two new mechanisms are validated here:
//!
//!   1. **Two-hop residency.** A CUDA<->Vulkan edge is NOT single-hop host-
//!      stageable — the source-backend Copy kernel rejects a cross-vendor GPU
//!      output. The residency pass (`insert_cross_device_copies` via
//!      `insert_residency_copies`) must therefore insert TWO `Op::Copy` hops
//!      (source-GPU -> CPU, then CPU -> target-GPU) with a CPU intermediate
//!      node, instead of one direct GPU->GPU copy. We assert the CPU
//!      intermediate is present (an `Op::Copy{target:Cpu}` feeding an
//!      `Op::Copy{target:Vulkan}` / `{target:Cuda}`).
//!   2. **Dual-device-seed.** A realize pinned to CUDA seeds only the CUDA
//!      device handle into the executor's StorageCache (via the const upload).
//!      The Vulkan sub-DAG's H2D copies then need a Vulkan backend handle too.
//!      The multi-device realize entry seeds BOTH a CUDA device handle AND a
//!      Vulkan backend handle.
//!
//! The graph: two INDEPENDENT sub-DAGs over shared consts `a`, `b`, one placed
//! on CUDA and one on Vulkan, reconverging at a final CUDA `add` that reads the
//! Vulkan result across a CUDA<->Vulkan edge (→ two-hop: Vulkan -> CPU -> CUDA).
//! Tensors are tiny + integer-valued for a byte-exact assert against the host
//! oracle.
//!
//! Gated `#[ignore]`; requires a live NVIDIA GPU + CUDA Runtime SDK AND a
//! Vulkan device for the AMD iGPU (`DeviceSelection::ByName("AMD")`). Run:
//!   cargo test -p fuel-core --features "cuda vulkan" --test cuda_vulkan_multidevice_realize_live -- --ignored --test-threads=1

#![cfg(all(feature = "cuda", feature = "vulkan"))]

use std::sync::Arc;

use fuel_core::lazy::LazyTensor;
use fuel_cuda_backend::CudaDevice;
use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
use fuel_graph::{NodeId, Op};
use fuel_ir::{probe::BackendId, DeviceLocation, Shape};

fn cuda_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("no CUDA device; skipping: {e:?}");
            None
        }
    }
}

/// Bind a Vulkan backend for the AMD iGPU. `PreferDiscrete` would pick the
/// RTX 4070 (same silicon as the CUDA device → not a cross-vendor boundary),
/// so we match the AMD integrated GPU by name. If no AMD device is present,
/// fall back to a SECOND Vulkan device distinct from index 0 if one exists;
/// otherwise skip with a clear message (a single Vulkan device that is the
/// same GPU as CUDA does not exercise the cross-vendor path meaningfully, but
/// we still try index 0 as a last resort so the test is not silently vacuous
/// on single-GPU CI).
fn vulkan_amd_or_skip() -> Option<Arc<VulkanBackend>> {
    // Preferred: the AMD iGPU by name.
    if let Ok(b) = VulkanBackend::with_selection(DeviceSelection::ByName("AMD".to_string())) {
        eprintln!("Vulkan: selected AMD device by name (gpu_id={})", b.gpu_id);
        return Some(Arc::new(b));
    }
    eprintln!("Vulkan: no AMD device by name; trying alternative selections");
    // Fallback: try a couple of enumeration indices, preferring one that is not
    // the discrete NVIDIA part. We can't introspect the name post-hoc here
    // cheaply, so just take index 1 if it exists, else index 0.
    for idx in [1usize, 0usize] {
        if let Ok(b) = VulkanBackend::with_selection(DeviceSelection::Index(idx)) {
            eprintln!("Vulkan: selected device by index {idx} (gpu_id={})", b.gpu_id);
            return Some(Arc::new(b));
        }
    }
    eprintln!("no usable Vulkan device; skipping");
    None
}

/// Stamp an explicit per-node placement on a `LazyTensor`'s node — the
/// scheduler-assignment seam (`Graph::set_placement`) that the planner honors
/// with priority over the realize-call pinned device.
fn place(t: &LazyTensor, loc: DeviceLocation) {
    let gt = t.graph_tensor();
    let id = gt.id();
    gt.graph()
        .write()
        .expect("graph lock")
        .set_placement(id, loc);
}

/// Realize a graph that genuinely spans CUDA + Vulkan in one pass and assert
/// the bytes match the host oracle exactly.
///
/// ```text
///   a   = [1, 2, 3, 4]            (const, uploaded to CUDA by the const cache)
///   b   = [10, 20, 30, 40]        (const)
///
///   -- sub-DAG 1, placed on CUDA --
///   s1  = a + b   = [11, 22, 33, 44]
///   s1b = s1 * a  = [11, 44, 99, 176]
///
///   -- sub-DAG 2, placed on VULKAN (reads consts a,b via inserted H2D copies) --
///   v2  = a * b   = [10, 40, 90, 160]
///   v2b = v2 + a  = [11, 42, 93, 164]
///
///   -- reconverge on CUDA (v2b is Vulkan → CROSS-VENDOR edge → two-hop
///      Vulkan->CPU->CUDA copy) --
///   out = s1b + v2b = [22, 86, 192, 340]
/// ```
#[test]
#[ignore = "requires a live CUDA device + a Vulkan device (AMD iGPU)"]
fn two_subdags_cuda_and_vulkan_realize_in_one_pass() {
    let Some(cuda) = cuda_or_skip() else { return };
    let Some(vk) = vulkan_amd_or_skip() else { return };

    let vk_loc = DeviceLocation::Vulkan { gpu_id: vk.gpu_id };
    let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };

    let a = LazyTensor::from_f32(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        &fuel_core::Device::cpu(),
    );
    let b = a.const_f32_like(vec![10.0_f32, 20.0, 30.0, 40.0], Shape::from_dims(&[4]));

    // Sub-DAG 1 on CUDA.
    let s1 = a.add(&b).expect("s1 = a+b");
    let s1b = s1.mul(&a).expect("s1b = s1*a");

    // Sub-DAG 2 on Vulkan — independent of sub-DAG 1, sharing only the consts.
    let v2 = a.mul(&b).expect("v2 = a*b");
    let v2b = v2.add(&a).expect("v2b = v2+a");

    // Reconverge on CUDA — reads v2b (Vulkan) across a CUDA<->Vulkan edge.
    let out = s1b.add(&v2b).expect("out = s1b + v2b");

    place(&s1, cuda0);
    place(&s1b, cuda0);
    place(&v2, vk_loc);
    place(&v2b, vk_loc);
    place(&out, cuda0);

    // Capture node ids BEFORE realize (realize mutates the graph in place).
    let v2_id = v2.graph_tensor().id();
    let v2b_id = v2b.graph_tensor().id();
    let s1_id = s1.graph_tensor().id();
    let s1b_id = s1b.graph_tensor().id();
    let out_id = out.graph_tensor().id();
    let g_arc = out.graph_tensor().graph().clone();

    // Realize pinned to CUDA, seeding the Vulkan backend handle as a SECOND
    // device. The new multi-device entry seeds both device handles into the
    // executor's StorageCache so the Vulkan sub-DAG's H2D copies resolve.
    let cuda_dev: fuel_core::Device = cuda.clone().into();
    let vk_dev: fuel_core::Device = vk.clone().into();
    let got = fuel_core::pipelined_bridge::realize_one_as_multi_device::<f32>(
        &g_arc,
        out_id,
        &cuda_dev,
        &[&vk_dev],
    )
    .expect("multi-vendor CUDA+Vulkan realize");

    assert_eq!(
        got,
        vec![22.0_f32, 86.0, 192.0, 340.0],
        "mixed CUDA+Vulkan single-pass realize must match the host oracle",
    );

    // Step E Phase C / B1 — the in-flight counter BALANCE on the REAL async
    // path. After a realize fully drains (every CUDA event waited, every Vulkan
    // batch retired), the per-device in-flight count MUST be back to 0 — no
    // leak, no underflow. This exercises the real CUDA event inc/dec
    // (produce_pending -> CudaCompletion::Drop) AND the eager Vulkan batch
    // inc/dec (eager_submit_all_vulkan -> VulkanCompletion::Drop) end-to-end on
    // a genuinely multi-device realize that both submits and drains on both
    // devices. (B1 is behavior-preserving — nothing READS this to alter the
    // result, which the byte-exact assert above proves.)
    assert_eq!(
        fuel_dispatch::dispatch::inflight_count(cuda0),
        0,
        "B1: CUDA in-flight count must return to 0 after the realize fully drains",
    );
    assert_eq!(
        fuel_dispatch::dispatch::inflight_count(vk_loc),
        0,
        "B1: Vulkan in-flight count must return to 0 after the realize fully drains",
    );

    // Self-verify the MECHANISM (not just the math).
    let g = g_arc.read().expect("graph lock");

    // (1) Per-node placement was HONORED: Vulkan-placed nodes stamped Vulkan,
    //     CUDA-placed nodes stamped Cuda.
    assert_eq!(
        g.target_backend(v2_id),
        Some(BackendId::Vulkan),
        "v2 was explicitly placed on Vulkan; its winner backend must be Vulkan",
    );
    assert_eq!(
        g.target_backend(v2b_id),
        Some(BackendId::Vulkan),
        "v2b was explicitly placed on Vulkan",
    );
    assert_eq!(
        g.target_backend(s1_id),
        Some(BackendId::Cuda),
        "s1 was explicitly placed on CUDA",
    );
    assert_eq!(
        g.target_backend(s1b_id),
        Some(BackendId::Cuda),
        "s1b was explicitly placed on CUDA",
    );
    assert_eq!(
        g.target_backend(out_id),
        Some(BackendId::Cuda),
        "the reconverge was explicitly placed on CUDA",
    );

    // (2) The CUDA<->Vulkan edge produced a TWO-HOP copy through a CPU
    //     intermediate. Find an Op::Copy{target:Cuda} whose input is an
    //     Op::Copy{target:Cpu} whose input is in turn a Vulkan-resident node —
    //     i.e. the Vulkan->CPU->CUDA bridge feeding the reconverge.
    let mut copies_to_cpu = 0usize;
    let mut copies_to_cuda = 0usize;
    let mut copies_to_vulkan = 0usize;
    let mut found_two_hop_vk_to_cuda = false;
    for i in 0..g.len() {
        if let Op::Copy { target } = g.node(NodeId(i)).op {
            match target {
                DeviceLocation::Cpu => copies_to_cpu += 1,
                DeviceLocation::Cuda { .. } => copies_to_cuda += 1,
                DeviceLocation::Vulkan { .. } => copies_to_vulkan += 1,
                _ => {}
            }
            // Two-hop detection: a Copy->Cuda fed by a Copy->Cpu.
            if matches!(target, DeviceLocation::Cuda { .. }) {
                if let Some(&src) = g.node(NodeId(i)).inputs.first() {
                    if matches!(
                        g.node(src).op,
                        Op::Copy { target: DeviceLocation::Cpu }
                    ) {
                        found_two_hop_vk_to_cuda = true;
                    }
                }
            }
        }
    }

    // H2D copies feeding the Vulkan sub-DAG (consts a,b -> Vulkan) AND the
    // second leg of the two-hop for the reconverge if any path lands on Vulkan.
    assert!(
        copies_to_vulkan >= 1,
        "expected at least one H2D Op::Copy(target=Vulkan) bridging the consts \
         into the Vulkan sub-DAG; got {copies_to_vulkan}",
    );
    // The CPU intermediate of the two-hop cross-vendor bridge (plus the
    // realize-root D2H splice).
    assert!(
        copies_to_cpu >= 2,
        "expected at least two Op::Copy(target=Cpu): the cross-vendor two-hop's \
         CPU intermediate AND the realize-root D2H splice; got {copies_to_cpu}",
    );
    assert!(
        copies_to_cuda >= 1,
        "expected at least one Op::Copy(target=Cuda) — the second leg of the \
         Vulkan->CPU->CUDA two-hop into the reconverge; got {copies_to_cuda}",
    );
    assert!(
        found_two_hop_vk_to_cuda,
        "the CUDA<->Vulkan edge must be bridged by a TWO-HOP copy: an \
         Op::Copy{{target:Cuda}} fed by an Op::Copy{{target:Cpu}} (the CPU \
         intermediate). A single direct GPU->GPU copy would be rejected by the \
         source-backend Copy kernel.",
    );
}
