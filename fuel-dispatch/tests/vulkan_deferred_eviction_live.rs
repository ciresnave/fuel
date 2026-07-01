//! Step E A2.1 — live-Vulkan DEFERRED-DELETION of an evicted data buffer.
//!
//! End-to-end proof that the executor's destructive-eviction path
//! (`Op::Move`/`Op::Release` cleanup) on a multi-backend realize RETAINS an
//! evicted Vulkan data buffer until the reader fences signal — instead of
//! host-blocking on a drain (the pre-A2.1 behavior) — and is byte-exact.
//!
//! This drives the REAL `PipelinedExecutor::realize` through
//! `defer_evicted_vulkan_buffer` (the helper that REPLACED the eviction-time
//! `drain_inflight_vulkan` + open-batch `force_flush`). The deterministic
//! retain-until-fence MECHANISM is proven separately + without wall-clock in
//! `fuel-vulkan-backend`'s `byte_storage_live::evicted_buffer_retained_on_batch_frees_post_fence`;
//! THIS test proves the executor wiring + byte-exactness on the live device.
//!
//! Mirrors `residency_eviction_live.rs` (the CUDA evict→fault-back twin) but
//! seeds the candidate on VULKAN, so the destructive `Op::Move{Cpu}` evicts a
//! Vulkan buffer. The Vulkan→CPU reload arms the executor's `multi_backend`
//! gate, so `defer_evicted_vulkan_buffer` is genuinely reached (not the
//! single-device no-op path).
//!
//! Gated `#[ignore]`; requires a live Vulkan device. Run:
//!   cargo test -p fuel-dispatch --features vulkan \
//!       --test vulkan_deferred_eviction_live -- --ignored --nocapture --test-threads=1

#![cfg(feature = "vulkan")]

use std::sync::{Arc, RwLock};

use fuel_ir::{probe::BackendId, DType, DeviceLocation, Shape};
use fuel_dispatch::pipelined::{PipelinedExecutor, StorageCache};
use fuel_dispatch::residency::insert_residency_evictions;
use fuel_graph::{Graph, Node, NodeId, Op, SharedGraph};
use fuel_memory::{BackendStorage, Storage};
use fuel_vulkan_backend::VulkanBackend;

fn backend_or_skip() -> Option<Arc<VulkanBackend>> {
    match VulkanBackend::new() {
        Ok(b) => Some(Arc::new(b)),
        Err(e) => {
            eprintln!("no Vulkan device; skipping: {e:?}");
            None
        }
    }
}

/// Evict→fault-back of a VULKAN-resident tensor preserves the output
/// bit-exactly, exercising A2.1 deferred-deletion in the executor.
///
/// The canonical gapped graph (same as the CUDA twin):
///
///   a    = [-1, 2, -3, 4]  on Vulkan
///   b    = relu(a)         = [0, 2, 0, 4]
///   pad  = neg(b)
///   pad2 = neg(pad)
///   c    = pad2 * a        = [0, 4, 0, 16]   (reads a AFTER the gap)
///   out  = b + c           = [0, 6, 0, 20]
///
/// With budget = 1 byte the residency pass evicts `a`: a destructive
/// `Op::Move{Cpu}` (D2H + release of the Vulkan device buffer — the eviction
/// that now DEFERS the buffer's free onto the in-flight batch) + an
/// `Op::Copy{Vulkan}` fault-back feeding `c`. The realize result must equal the
/// no-eviction baseline bit-for-bit (A2.1 is a lifetime/timing change, never a
/// value change).
#[test]
#[ignore = "requires a live Vulkan device"]
fn vulkan_evict_fault_back_roundtrip_preserves_output() {
    let Some(backend) = backend_or_skip() else { return };
    let vk_loc = DeviceLocation::Vulkan { gpu_id: backend.gpu_id };

    let build = || -> (SharedGraph, NodeId, NodeId) {
        let graph: SharedGraph = Arc::new(RwLock::new(Graph::new()));
        let (a, out) = {
            let mut g = graph.write().unwrap();
            let push = |g: &mut Graph, op: Op, inputs: Vec<NodeId>| {
                g.push(Node {
                    op,
                    inputs,
                    shape: Shape::from_dims(&[4]),
                    dtype: DType::F32,
                })
            };
            let a = push(&mut g, Op::Const, vec![]);
            let b = push(&mut g, Op::Relu, vec![a]);
            let pad = push(&mut g, Op::Neg, vec![b]);
            let pad2 = push(&mut g, Op::Neg, vec![pad]);
            let c = push(&mut g, Op::Mul, vec![pad2, a]);
            let out = push(&mut g, Op::Add, vec![b, c]);
            for n in [a, b, pad, pad2, c, out] {
                g.set_target_backend(n, BackendId::Vulkan);
                g.set_placement(n, vk_loc);
            }
            (a, out)
        };
        (graph, a, out)
    };

    // Seed `a` as a backend-attached Vulkan storage (so Op::Move's D2H + the
    // executor's deferred-eviction retain can reach the backend handle).
    let seed_vulkan = |a: NodeId| -> StorageCache {
        let host = [-1.0_f32, 2.0, -3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&host);
        let vk = backend.upload_bytes_handle(bytes).expect("h2d seed");
        let mut cache = StorageCache::new();
        cache.insert(
            a,
            Arc::new(RwLock::new(Storage::new(BackendStorage::Vulkan(vk), DType::F32))),
        );
        cache
    };

    let download = |storage: &Arc<RwLock<Storage>>| -> Vec<f32> {
        let guard = storage.read().unwrap();
        let bytes = match &guard.inner {
            BackendStorage::Vulkan(v) => backend.download_bytes(v).expect("d2h"),
            BackendStorage::Cpu(c) => c.bytes().to_vec(),
            // Cpu+Vulkan are exhaustive on a vulkan-only build; the catch-all
            // only matters when other backends (cuda) are also compiled in.
            #[allow(unreachable_patterns)]
            other => panic!("unexpected output backend: {other:?}"),
        };
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    };

    // --- Baseline: no eviction --------------------------------------
    let (graph1, a1, out1) = build();
    let (s1, _) = PipelinedExecutor::realize(graph1, out1, seed_vulkan(a1))
        .expect("baseline realize on Vulkan");
    let baseline = download(&s1);
    assert_eq!(baseline, vec![0.0_f32, 6.0, 0.0, 20.0]);

    // --- With evict→fault-back (exercises A2.1 deferred-deletion) ---
    let (graph2, a2, out2) = build();
    let chains = insert_residency_evictions(&graph2, &[out2], 1, 16, move |_| Some(vk_loc))
        .expect("eviction pass");
    assert!(
        chains.iter().any(|c| c.candidate == a2),
        "budget=1 must evict the gapped Vulkan tensor `a`; got {chains:?}",
    );
    {
        let g = graph2.read().unwrap();
        let chain = chains.iter().find(|c| c.candidate == a2).unwrap();
        assert!(
            matches!(g.node(chain.move_node).op, Op::Move { target: DeviceLocation::Cpu }),
            "evict half must be a destructive D2H Op::Move (the buffer whose free A2.1 defers)",
        );
        // The Move's transfer kernel runs on the SOURCE's backend (Vulkan D2H);
        // the reload runs on CPU (H2D from the staged host copy) — the Vulkan↔CPU
        // backend switch is what arms the executor's `multi_backend` gate, so the
        // eviction genuinely takes the deferred-deletion path.
        assert_eq!(g.target_backend(chain.move_node), Some(BackendId::Vulkan));
        assert_eq!(g.target_backend(chain.reload), Some(BackendId::Cpu));
    }

    let (s2, _) = PipelinedExecutor::realize(graph2, out2, seed_vulkan(a2))
        .expect("evicted realize on Vulkan (A2.1 deferred-deletion path)");
    let evicted = download(&s2);

    assert_eq!(
        baseline, evicted,
        "A2.1: evict→fault-back with deferred-deletion must preserve the output \
         bit-exactly (lifetime/timing change, never a value change)",
    );
}

/// Step E A2.1 — the OPEN-BATCH deferred-deletion case via `Op::Release`: the
/// path where A2.1 actually REMOVES a host block, asserted deterministically.
///
/// `Op::Release(t1)` destructively evicts a Vulkan intermediate `t1` whose
/// producer is still in the OPEN (recorded-but-unsubmitted) Vulkan batch. The
/// pre-A2.1 eviction host-BLOCKED here (`force_flush` = submit+WAIT the open
/// batch). A2.1 instead EAGER-SUBMITS the open batch (no wait) and retains the
/// evicted buffer onto it, freeing it post-fence — no host block.
///
/// **Throughput assertion (deterministic — no wall-clock, noisy on this box):**
/// the executor's `deferred_eviction_retains` counter MUST advance across this
/// realize — i.e. an evicted Vulkan buffer was retained onto ≥1 in-flight batch
/// instead of waited. That counter only increments on the deferred (no-drain)
/// path, so its advance is direct proof the eviction did NOT block the host on a
/// Vulkan fence. Paired with a byte-exact assertion (the retain is a
/// lifetime/timing change, never a value change).
///
/// A CPU `out` node (reading a CPU-seeded const — no cross-device read) sits in
/// the dispatch order so the executor's `multi_backend` gate arms; the Vulkan
/// `relu` producing `t1` is in the still-open batch when `Op::Release(t1)` fires.
/// Both `out` and `rel` are realize targets so the Release marker stays reachable
/// (its 0-elem output is never read by a consumer); `t1` is not a target, so its
/// destructive eviction fires — onto the eager-submitted open batch (no drain).
///
///   a    = [-1, 2, -3, 4]   on Vulkan (seeded)
///   t1   = relu(a)          = [0, 2, 0, 4]    (Vulkan; in the OPEN batch, released)
///   rel  = Release(t1)      (destructive evict of the Vulkan buffer t1)
///   bcpu = [-1, -1, -1, -1] on CPU (seeded)
///   out  = neg(bcpu)        = [1, 1, 1, 1]    (CPU — arms multi_backend; value target)
#[test]
#[ignore = "requires a live Vulkan device"]
fn vulkan_release_open_batch_deferred_deletion_no_drain() {
    let Some(backend) = backend_or_skip() else { return };
    let vk_loc = DeviceLocation::Vulkan { gpu_id: backend.gpu_id };

    let graph: SharedGraph = Arc::new(RwLock::new(Graph::new()));
    let (a, bcpu, rel, out) = {
        let mut g = graph.write().unwrap();
        let push = |g: &mut Graph, op: Op, inputs: Vec<NodeId>, n: usize| {
            g.push(Node {
                op,
                inputs,
                shape: Shape::from_dims(&[n]),
                dtype: DType::F32,
            })
        };
        let a = push(&mut g, Op::Const, vec![], 4);
        let t1 = push(&mut g, Op::Relu, vec![a], 4);
        // Release t1 (a Vulkan buffer) — destructive eviction (destructive_input
        // == Some(0) drives the executor's eviction of t1). A realize target so
        // the marker stays reachable; t1 itself is NOT a target → evicted.
        let rel = push(&mut g, Op::Release, vec![t1], 0);
        // Independent CPU branch (reads a CPU const — NO cross-device read). Its
        // CPU placement vs the Vulkan relu arms the executor's `multi_backend` gate.
        let bcpu = push(&mut g, Op::Const, vec![], 4);
        let out = push(&mut g, Op::Neg, vec![bcpu], 4);
        for n in [a, t1] {
            g.set_target_backend(n, BackendId::Vulkan);
            g.set_placement(n, vk_loc);
        }
        g.set_target_backend(rel, BackendId::Vulkan);
        g.set_placement(rel, vk_loc);
        for n in [bcpu, out] {
            g.set_target_backend(n, BackendId::Cpu);
            g.set_placement(n, DeviceLocation::Cpu);
        }
        (a, bcpu, rel, out)
    };

    let seed = || -> StorageCache {
        let mut cache = StorageCache::new();
        let a_host = [-1.0_f32, 2.0, -3.0, 4.0];
        let a_vk = backend
            .upload_bytes_handle(bytemuck::cast_slice(&a_host))
            .expect("h2d seed a");
        cache.insert(
            a,
            Arc::new(RwLock::new(Storage::new(BackendStorage::Vulkan(a_vk), DType::F32))),
        );
        // CPU-seeded const (the independent CPU branch — no cross-device read).
        let b_cpu = fuel_memory::from_slice_cpu::<f32>(&[-1.0_f32, -1.0, -1.0, -1.0]);
        cache.insert(bcpu, Arc::new(RwLock::new(b_cpu)));
        cache
    };

    let download = |storage: &Arc<RwLock<Storage>>| -> Vec<f32> {
        let guard = storage.read().unwrap();
        let bytes = match &guard.inner {
            BackendStorage::Vulkan(v) => backend.download_bytes(v).expect("d2h"),
            BackendStorage::Cpu(c) => c.bytes().to_vec(),
            // Cpu+Vulkan are exhaustive on a vulkan-only build; the catch-all
            // only matters when other backends (cuda) are also compiled in.
            #[allow(unreachable_patterns)]
            other => panic!("unexpected output backend: {other:?}"),
        };
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    };

    let before = fuel_dispatch::pipelined::deferred_eviction_retains();

    // Targets: out (the value) + rel (keeps the Release marker reachable so the
    // destructive eviction of t1 fires).
    let results = PipelinedExecutor::realize_many(graph, &[out, rel], seed())
        .expect("Op::Release deferred-deletion realize_many");

    let after = fuel_dispatch::pipelined::deferred_eviction_retains();

    // THROUGHPUT: the deferred (no-drain) eviction path fired — an evicted Vulkan
    // buffer was retained onto an in-flight batch instead of host-waited. Pre-A2.1
    // this site host-blocked on a Vulkan fence (`force_flush`/`drain_inflight`).
    assert!(
        after > before,
        "A2.1: the destructive Vulkan eviction must take the DEFERRED-DELETION \
         path (retain-until-fence, no host block) — `deferred_eviction_retains` \
         must advance (before={before}, after={after})",
    );

    // BYTE-EXACT: out = neg([-1,-1,-1,-1]) = [1,1,1,1] (the realize succeeding
    // through the deferred eviction without a UAF/fault is the core check).
    let got = download(&results[0].0);
    assert_eq!(
        got,
        vec![1.0_f32, 1.0, 1.0, 1.0],
        "A2.1 open-batch deferred-deletion (Op::Release) must be byte-exact",
    );
}
