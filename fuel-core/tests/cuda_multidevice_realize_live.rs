//! Step E A4a-1 — multi-device realize on a SINGLE pass (CPU + CUDA).
//!
//! The async foundation (A1 completion-handle seam, A2 Vulkan async, A3 CUDA
//! stream-ordered) only overlaps once a realize *produces* a mixed-backend
//! graph. A4a-1 is the mechanism that lets it: a single `realize` whose nodes
//! are placed on MORE THAN ONE backend, validated here with EXPLICIT per-node
//! placement (no auto-placement policy, no second GPU).
//!
//! This is the born-red TDD guard for the A4a-1 mechanism. It exercises, in ONE
//! realize pass:
//!   1. Per-node placement honored across devices — the planner does NOT prune
//!      the surviving set back to the pinned device (the "one device" framing
//!      at plan.rs:187-192 was a per-node-set invariant, never graph-global; a
//!      mixed graph is the norm).
//!   2. Cross-device `Op::Copy` inserted at the mixed-backend boundary edges by
//!      `optimize_graph`'s residency pass — both D2H (CUDA const → CPU sub-DAG)
//!      and H2D (CPU result → CUDA reconverge).
//!   3. Multi-backend dispatch in one executor walk: per-node `target_backend`
//!      routing through the per-NodeId cache, with the un-bridged-mixed-edge
//!      error never firing because the copies are present.
//!
//! The graph: two INDEPENDENT sub-DAGs over shared consts `a`, `b`, one placed
//! on CUDA and one on CPU, reconverging at a final CUDA `add`. Realize is pinned
//! to CUDA (`realize_f32_cuda`), so the const cache uploads `a`,`b` to CUDA and
//! supplies the device handle every H2D copy needs; the CPU sub-DAG reads the
//! CUDA consts via inserted D2H copies and its result feeds the CUDA reconverge
//! via an inserted H2D copy. Tensors are tiny + integer-valued for a byte-exact
//! assert against the host oracle.
//!
//! Gated `#[ignore]`; requires a live NVIDIA GPU + CUDA Runtime SDK. Run:
//!   cargo test -p fuel-core --features cuda --test cuda_multidevice_realize_live -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use fuel_core::lazy::LazyTensor;
use fuel_cuda_backend::CudaDevice;
use fuel_graph::{NodeId, Op};
use fuel_ir::{probe::BackendId, DeviceLocation, Shape};

fn dev_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("no CUDA device; skipping: {e:?}");
            None
        }
    }
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

/// Realize a graph that genuinely spans CPU + CUDA in one pass and assert the
/// bytes match the host oracle exactly.
///
/// ```text
///   a   = [1, 2, 3, 4]            (const, uploaded to CUDA by the const cache)
///   b   = [10, 20, 30, 40]        (const, uploaded to CUDA)
///
///   -- sub-DAG 1, placed on CUDA --
///   s1  = a + b   = [11, 22, 33, 44]
///   s1b = s1 * a  = [11, 44, 99, 176]
///
///   -- sub-DAG 2, placed on CPU (reads CUDA consts a,b via inserted D2H copies) --
///   s2  = a * b   = [10, 40, 90, 160]
///   s2b = s2 + a  = [11, 42, 93, 164]
///
///   -- reconverge on CUDA (s2b is CPU → inserted H2D copy) --
///   out = s1b + s2b = [22, 86, 192, 340]
/// ```
#[test]
#[ignore = "requires a live CUDA device"]
fn two_subdags_cpu_and_cuda_realize_in_one_pass() {
    let Some(dev) = dev_or_skip() else { return };

    let a = LazyTensor::from_f32(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        &fuel_core::Device::cpu(),
    );
    // `const_f32_like` keeps `b` in `a`'s graph (a bare second `from_f32` would
    // mint a separate graph and `add`/`mul` across graphs would fail).
    let b = a.const_f32_like(vec![10.0_f32, 20.0, 30.0, 40.0], Shape::from_dims(&[4]));

    // Sub-DAG 1 on CUDA.
    let s1 = a.add(&b).expect("s1 = a+b");
    let s1b = s1.mul(&a).expect("s1b = s1*a");

    // Sub-DAG 2 on CPU — independent of sub-DAG 1, sharing only the consts.
    let s2 = a.mul(&b).expect("s2 = a*b");
    let s2b = s2.add(&a).expect("s2b = s2+a");

    // Reconverge on CUDA.
    let out = s1b.add(&s2b).expect("out = s1b + s2b");

    let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
    let cpu = DeviceLocation::Cpu;
    place(&s1, cuda0);
    place(&s1b, cuda0);
    place(&s2, cpu);
    place(&s2b, cpu);
    place(&out, cuda0);

    // Capture the node ids BEFORE realize (the realize mutates the graph in
    // place — `optimize_graph` stamps backends + splices cross-device copies).
    let s2_id = s2.graph_tensor().id();
    let s2b_id = s2b.graph_tensor().id();
    let s1_id = s1.graph_tensor().id();
    let s1b_id = s1b.graph_tensor().id();
    let out_id = out.graph_tensor().id();
    let g_arc = out.graph_tensor().graph().clone();

    let got = out.realize_f32_cuda(&dev);
    assert_eq!(
        got,
        vec![22.0_f32, 86.0, 192.0, 340.0],
        "mixed CPU+CUDA single-pass realize must match the host oracle",
    );

    // Self-verify the MECHANISM (not just the math): a graph that quietly
    // collapsed every node onto CUDA would also produce the right answer, so
    // assert the realize genuinely went mixed-device.
    let g = g_arc.read().expect("graph lock");

    // (1) Per-node placement was HONORED: the explicitly CPU-placed nodes were
    //     stamped Cpu, the CUDA-placed nodes Cuda — the planner did NOT prune
    //     the surviving set back to the pinned (CUDA) device.
    assert_eq!(
        g.target_backend(s2_id),
        Some(BackendId::Cpu),
        "s2 was explicitly placed on CPU; its winner backend must be Cpu",
    );
    assert_eq!(
        g.target_backend(s2b_id),
        Some(BackendId::Cpu),
        "s2b was explicitly placed on CPU; its winner backend must be Cpu",
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

    // (2) Cross-device boundaries were BRIDGED: the residency pass inserted
    //     Op::Copy nodes for BOTH directions — D2H (CUDA const → the CPU
    //     sub-DAG) and H2D (the CPU result → the CUDA reconverge). Without these
    //     the executor errors on the un-bridged mixed edge.
    let mut copies_to_cpu = 0usize;
    let mut copies_to_cuda = 0usize;
    for i in 0..g.len() {
        if let Op::Copy { target } = g.node(NodeId(i)).op {
            match target {
                DeviceLocation::Cpu => copies_to_cpu += 1,
                DeviceLocation::Cuda { .. } => copies_to_cuda += 1,
                _ => {}
            }
        }
    }
    assert!(
        copies_to_cpu >= 1,
        "expected at least one D2H Op::Copy(target=Cpu) bridging the CUDA \
         consts into the CPU sub-DAG (plus the realize-root D2H splice); got {copies_to_cpu}",
    );
    assert!(
        copies_to_cuda >= 1,
        "expected at least one H2D Op::Copy(target=Cuda) bridging the CPU \
         sub-DAG result into the CUDA reconverge; got {copies_to_cuda}",
    );
}
