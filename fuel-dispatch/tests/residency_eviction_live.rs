//! Live-GPU residency eviction roundtrip (executor-unification
//! Session 6): the `fuel_dispatch::residency` pass spills a CUDA-
//! resident tensor to host via `Op::Move { target: Cpu }` and faults
//! it back via the reload `Op::Copy { target: Cuda }`, end-to-end
//! through the PRODUCTION `PipelinedExecutor`. First live coverage of
//! the `WorkItemKind::Move` arm (the deferral noted when `b93bdb82`
//! shipped the arm with CPU-only tests).
//!
//! Gated `#[ignore]` — requires an NVIDIA GPU + CUDA Runtime SDK.
//! Invoke explicitly (verify agent):
//!
//! ```sh
//! cargo test -p fuel-dispatch --features cuda \
//!     --test residency_eviction_live -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{probe::BackendId, DType, DeviceLocation, Shape};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::pipelined::{PipelinedExecutor, StorageCache};
use fuel_dispatch::residency::insert_residency_evictions;
use fuel_graph::{Graph, Node, NodeId, Op, SharedGraph};
use fuel_storage::{BackendStorage, Storage};

fn dev_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("no CUDA device; skipping: {e:?}");
            None
        }
    }
}

/// Evict→fault-back roundtrip on the live GPU. Same canonical graph
/// as the CPU unit tests (`a` has two consumers separated by a gap):
///
///   a    = [-1, 2, -3, 4]  on CUDA
///   b    = relu(a)         = [0, 2, 0, 4]
///   pad  = neg(b)
///   pad2 = neg(pad)
///   c    = pad2 * a        = [0, 4, 0, 16]   (reads a after the gap)
///   out  = b + c           = [0, 6, 0, 20]
///
/// With budget = 1 byte the pass evicts `a`: `Op::Move{Cpu}` (D2H +
/// destructive release of the device storage) + `Op::Copy{Cuda}`
/// (H2D fault-back feeding `c`). The realize result must be
/// bit-exact with the no-eviction baseline.
#[test]
#[ignore]
fn cuda_evict_fault_back_roundtrip_preserves_output() {
    let Some(dev) = dev_or_skip() else { return };

    let build = || -> (SharedGraph, NodeId, NodeId) {
        let graph: SharedGraph = Arc::new(RwLock::new(Graph::new()));
        let (a, out) = {
            let mut g = graph.write().unwrap();
            let mut push = |g: &mut Graph, op: Op, inputs: Vec<NodeId>| {
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
            for n in [b, pad, pad2, c, out] {
                g.set_target_backend(n, BackendId::Cuda);
                g.set_placement(n, DeviceLocation::Cuda { gpu_id: 0 });
            }
            g.set_placement(a, DeviceLocation::Cuda { gpu_id: 0 });
            (a, out)
        };
        (graph, a, out)
    };

    let seed_cuda = |dev: &CudaDevice, a: NodeId| -> StorageCache {
        let host = [-1.0_f32, 2.0, -3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&host);
        let cuda = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d seed");
        let mut cache = StorageCache::new();
        cache.insert(
            a,
            Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(cuda), DType::F32))),
        );
        cache
    };

    let download = |storage: &Arc<RwLock<Storage>>| -> Vec<f32> {
        let guard = storage.read().unwrap();
        let BackendStorage::Cuda(c) = &guard.inner else {
            panic!("expected output resident on CUDA");
        };
        let host = c.to_cpu_bytes().expect("d2h");
        bytemuck::cast_slice::<u8, f32>(&host).to_vec()
    };

    // --- Baseline: no eviction --------------------------------------
    let (graph1, a1, out1) = build();
    let (s1, _) = PipelinedExecutor::realize(graph1, out1, seed_cuda(&dev, a1))
        .expect("baseline realize on CUDA");
    let baseline = download(&s1);
    assert_eq!(baseline, vec![0.0_f32, 6.0, 0.0, 20.0]);

    // --- With evict→fault-back --------------------------------------
    let (graph2, a2, out2) = build();
    let chains = insert_residency_evictions(&graph2, &[out2], 1, 16, |_| {
        Some(DeviceLocation::Cuda { gpu_id: 0 })
    })
    .expect("eviction pass");
    assert!(
        chains.iter().any(|c| c.candidate == a2),
        "budget=1 must evict the gapped CUDA tensor `a`; got {chains:?}",
    );
    {
        let g = graph2.read().unwrap();
        let chain = chains.iter().find(|c| c.candidate == a2).unwrap();
        assert!(
            matches!(g.node(chain.move_node).op, Op::Move { target: DeviceLocation::Cpu }),
            "evict half must be a destructive D2H Op::Move",
        );
        assert!(
            matches!(
                g.node(chain.reload).op,
                Op::Copy { target: DeviceLocation::Cuda { gpu_id: 0 } },
            ),
            "fault-back half must be an H2D Op::Copy reload",
        );
        // The pass stamped the transfer pair: Move's kernel runs on
        // the source's backend (CUDA D2H); the reload's on CPU (H2D
        // from the staged host copy).
        assert_eq!(g.target_backend(chain.move_node), Some(BackendId::Cuda));
        assert_eq!(g.target_backend(chain.reload), Some(BackendId::Cpu));
    }

    let (s2, _) = PipelinedExecutor::realize(graph2, out2, seed_cuda(&dev, a2))
        .expect("evicted realize on CUDA");
    let evicted = download(&s2);

    assert_eq!(
        baseline, evicted,
        "evict→fault-back must preserve the output bit-exactly",
    );
}
