//! Bridge from fuel-core's user-facing `Tensor::realize_*` API to
//! fuel-storage's `PipelinedExecutor` (Phase 7.6 step 9c, sub-phases
//! E.1 + E.2).
//!
//! Pre-Phase-E, `Tensor::realize_f32` etc. constructed a
//! `fuel-graph-executor::GraphExecutor<B>` and called its typed
//! `realize_f32(&tensor)` method. The legacy executor's
//! `try_adopt_slot` walked the graph's storage map, did D2H, then
//! `B::upload(&buf, shape)` to put the data on the backend.
//!
//! Post-Phase-E, the user-facing API:
//! 1. Walks the graph from the requested targets and **pre-realizes
//!    every reachable `Op::Const`** into a `StorageCache` on the
//!    chosen target device. This is the legacy `try_adopt_slot`
//!    work, now external to the executor.
//! 2. Sets `target_backend` on every reachable computational node
//!    (the legacy executor implicitly used `self.backend`; the
//!    pipelined path reads it from the graph side-table).
//! 3. For non-CPU realize devices, splices an
//!    `Op::Copy { target: Cpu }` at each realize root so D2H runs
//!    as a graph node the optimizer can see (bridge-retirement
//!    Phase 2, post-9c). The Op::Copy node's kernel is registered
//!    at `(OpKind::Copy, [dt, dt], source_backend)`; the executor's
//!    `WorkItemKind::Copy` arm allocates the output on the target
//!    location and runs the source-backend's download wrapper.
//! 4. Calls [`PipelinedExecutor::realize_many`] for multi-target or
//!    `PipelinedExecutor::realize` for single-target on the spliced
//!    targets — the executor returns a `BackendStorage::Cpu` for
//!    each.
//! 5. Reads the CPU bytes into a typed `Vec<T>` via `bytemuck`.
//!
//! This module owns steps 1–5 so [`crate::lazy::LazyTensor`]'s
//! `realize_*` methods stay one-liners.
//!
//! ## Status post-Phase 3
//!
//! Bridge-retirement Phases 2 + 3a + 3b complete:
//! * **Phase 2** (D2H): `realize_*_as` splices `Op::Copy { target: Cpu }`
//!   at every realize root; the executor's `WorkItemKind::Copy` arm
//!   downloads bytes via the binding-table-registered source-backend
//!   wrapper. `BackendStorage::read_to_cpu_bytes` deleted.
//! * **Phase 3a** (zero-alloc): `KvCache::with_capacity` emits
//!   `Op::Alloc → Op::ZeroFill` pairs and realizes via
//!   `PipelinedExecutor::realize_many`. `alloc_zeroed_on` deleted.
//! * **Phase 3b** (H2D Const upload): [`build_const_cache`] (for
//!   non-CPU targets) builds a transient graph of `Op::Const →
//!   Op::Copy { target: device }` pairs and realizes them
//!   multi-target. The executor's `WorkItemKind::Copy` arm allocates
//!   the device-side output (uninit) and the `copy_from_cpu_wrapper`
//!   writes host bytes via per-backend H2D helpers
//!   (`CudaStorageBytes::write_from_host`,
//!   `VulkanBackend::write_bytes`). `upload_host_buffer` deleted.
//!
//! Residual bridge code: [`device_seed_storage`] (~30 LOC, just the
//! 0-byte device-handle anchor per backend) and
//! [`host_buffer_to_bytes`] (per-dtype HostBuffer → bytes
//! conversion — orthogonal to the device-dispatch concern).
//!
//! ## Not yet covered (Phase E.3+)
//!
//! - `KVCache<B>` and `forward_with_cache_on<B>` — autoregressive
//!   decoding needs a const cache that survives realize calls; the
//!   pattern is "caller holds a long-lived `StorageCache` across
//!   calls" but the API surface for that lands in Phase E.3.
//! - `generate_*` and speculative decoding loops — same.

use std::sync::{Arc, RwLock};

use fuel_core_types::{
    probe::BackendId, DeviceLocation, Error, HostBuffer, Result,
};
use fuel_graph::{Graph, Node, NodeId, Op, topo_order_multi};
use fuel_storage::{
    pipelined::{PipelinedExecutor, StorageCache},
    BackendStorage, Storage,
};

use crate::Device;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Realize a single tensor by NodeId on the given device, returning
/// the result's host bytes as a typed `Vec<T>` via `bytemuck`.
///
/// Steps:
/// 1. `ensure_target_backends` — propagate the target backend to every
///    reachable computational node.
/// 2. `prepare_const_cache` — D2H + re-upload every reachable
///    `Op::Const` slot onto `device`.
/// 3. For non-CPU `device`: splice an `Op::Copy { target: Cpu }` at
///    the realize root so D2H is a binding-table-dispatched graph
///    node (bridge-retirement Phase 2).
/// 4. `PipelinedExecutor::realize` — kick the compile + execute
///    pipeline; returns a `BackendStorage::Cpu` for the spliced root.
/// 5. `bytemuck::cast_slice` — reinterpret the CPU bytes as `T`.
pub fn realize_one_as<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    target: NodeId,
    device: &Device,
) -> Result<Vec<T>> {
    realize_one_as_with_initial::<T>(graph, target, device, StorageCache::new())
}

/// Multi-target counterpart of [`realize_one_as`]. Returns parallel
/// `Vec<Vec<T>>` in the order of `targets`.
pub fn realize_many_as<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    device: &Device,
) -> Result<Vec<Vec<T>>> {
    realize_many_as_with_initial::<T>(graph, targets, device, StorageCache::new())
}

/// Realize-one variant that seeds the executor's input cache with
/// `initial` before adding Op::Const slot uploads. Used by
/// [`crate::inference_context::InferenceContext`] to thread its
/// persistent storage Arcs through each realize call without
/// re-uploading them. NodeIds already present in `initial` are
/// not re-fetched from the graph's storage_map; their Arcs survive
/// the call.
pub fn realize_one_as_with_initial<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    target: NodeId,
    device: &Device,
    initial: StorageCache,
) -> Result<Vec<T>> {
    let (cache, _backend_id, mut effective_targets) =
        prepare(graph, &[target], device, initial)?;
    let cpu_target = effective_targets
        .pop()
        .expect("prepare returns one effective target per input target");
    let (storage, _layout) =
        PipelinedExecutor::realize(graph.clone(), cpu_target, cache)?;
    extract_cpu_bytes_typed::<T>(&storage)
}

/// Multi-target counterpart of [`realize_one_as_with_initial`].
pub fn realize_many_as_with_initial<T: bytemuck::Pod>(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    device: &Device,
    initial: StorageCache,
) -> Result<Vec<Vec<T>>> {
    if targets.is_empty() {
        return Ok(Vec::new());
    }
    let (cache, _, effective_targets) = prepare(graph, targets, device, initial)?;
    let results = PipelinedExecutor::realize_many(
        graph.clone(), &effective_targets, cache,
    )?;
    let mut out = Vec::with_capacity(results.len());
    for (storage, _layout) in results {
        out.push(extract_cpu_bytes_typed::<T>(&storage)?);
    }
    Ok(out)
}

/// Read a realize result's CPU bytes and reinterpret them as `Vec<T>`.
///
/// Post bridge-retirement Phase 2: the executor produced this Storage
/// through the spliced `Op::Copy { target: Cpu }` node (for non-CPU
/// devices) or directly on CPU (for CPU realizes). Either way, this
/// is a `BackendStorage::Cpu` — extract its bytes via the
/// CPU-variant pattern.
fn extract_cpu_bytes_typed<T: bytemuck::Pod>(
    storage: &Arc<RwLock<Storage>>,
) -> Result<Vec<T>> {
    let guard = storage
        .read()
        .map_err(|_| Error::Msg("storage lock poisoned".into()).bt())?;
    let bytes: &[u8] = match &guard.inner {
        BackendStorage::Cpu(s) => s.bytes(),
        // The other arms are feature-gated; on default-features-only
        // builds CPU is the sole variant and this arm is unreachable
        // — but suppress the lint so it still parses with `--features
        // cuda` / `--features vulkan`.
        #[allow(unreachable_patterns)]
        other => {
            return Err(Error::Msg(format!(
                "pipelined_bridge: realize root produced non-CPU storage \
                 ({other:?}) — the Op::Copy splice in `prepare()` should \
                 have made the root CPU-resident. This is a bug.",
            ))
            .bt());
        }
    };
    Ok(bytemuck::cast_slice::<u8, T>(bytes).to_vec())
}

// ---------------------------------------------------------------------------
// Prep — internal
// ---------------------------------------------------------------------------

/// One-shot prep: derive a `BackendId` from `device`, propagate it to
/// every reachable computational node, build a `StorageCache`
/// containing every reachable `Op::Const`, and (post-9c Phase 2 of
/// bridge-retirement) splice an `Op::Copy { target: Cpu }` at each
/// non-CPU realize root so the executor produces a CPU storage at the
/// returned `effective_targets`.
///
/// Returns `(cache, backend_id, effective_targets)`:
/// - `effective_targets[i]` mirrors `targets[i]`'s order. For CPU
///   realizes it equals `targets[i]`; for GPU realizes it is the
///   NodeId of the spliced Op::Copy node, whose output the executor
///   produces as a fresh `BackendStorage::Cpu`.
///
/// Mutates the graph (takes a write lock); the executor takes its own
/// read lock after this returns.
fn prepare(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    device: &Device,
    initial: StorageCache,
) -> Result<(StorageCache, BackendId, Vec<NodeId>)> {
    let backend_id = device_to_backend_id(device);

    // Phase 2 of bridge-retirement: splice an `Op::Copy { target:
    // Cpu }` at every realize root, regardless of source backend, so
    // D2H runs as a graph node the optimizer can see (architecture
    // identity check #1).
    //
    // Why always — even for CPU realizes:
    //   1. Strided / sliced / permuted realize roots are common; the
    //      executor's WorkItemKind::Copy arm runs `auto_contiguize`
    //      on the input before the kernel, so the output is the
    //      LOGICAL view's bytes, not the parent storage's full bytes.
    //      Without the splice on CPU, a `realize_f32` of a slice view
    //      returned the parent's full bytes (a long-standing bug
    //      inherited from the pre-9c `read_to_cpu_bytes`); routing
    //      through Op::Copy fixes it uniformly.
    //   2. The CPU→CPU Copy kernel is one memcpy that replaces the
    //      `.to_vec()` `read_to_cpu_bytes` used to do; no extra cost
    //      in the contiguous case.
    //   3. One code path through Op::Copy keeps the executor's
    //      semantics consistent across devices.
    //
    // The spliced node's shape + dtype match the source; the
    // executor's WorkItemKind::Copy arm allocates a fresh CPU storage
    // and runs the source-backend's registered Copy kernel.
    let effective_targets = {
        let mut g = graph
            .write()
            .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
        targets
            .iter()
            .map(|&src_id| {
                let (shape, dtype) = {
                    let n = g.node(src_id);
                    (n.shape.clone(), n.dtype)
                };
                g.push(Node {
                    op: Op::Copy { target: DeviceLocation::Cpu },
                    inputs: vec![src_id],
                    shape,
                    dtype,
                })
            })
            .collect::<Vec<_>>()
    };

    let order = {
        let g = graph
            .read()
            .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
        topo_order_multi(&g, &effective_targets)
    };

    // Build the StorageCache on top of `initial` (which may carry
    // persistent storages from an InferenceContext). build_const_cache
    // adds any reachable Op::Const NodeId not already present.
    let cache = build_const_cache(graph, &order, device, initial)?;

    // Now set target_backend on every computational node. View ops,
    // Reshape, Const, and Release inherit/don't need it — see
    // `compile_one` in fuel-storage::pipelined.
    //
    // For Op::Copy { target: Cpu } spliced at realize roots: we want
    // target_backend = backend_id (the SOURCE backend, where the
    // download kernel runs). That's exactly what this overwrite does
    // — `Op::Copy` is computational, not a view, so it gets the same
    // backend_id stamp. The executor's WorkItemKind::Copy arm reads
    // `target_location` from the op's variant field for output
    // allocation; `target_backend` drives the kernel lookup.
    //
    // We *always* overwrite rather than preserving prior values. The
    // reason: graphs are shared (`Arc<RwLock<Graph>>`) and a single
    // graph may be realized on multiple backends across a session.
    // E.g. test pattern `let cpu = t.realize_f32(); let cuda =
    // t.realize_f32_cuda(&dev);` would otherwise see the CPU pinning
    // from the first call and silently re-realize on CPU.
    //
    // When the Router migrates to PipelinedExecutor (Phase G), the
    // Router will need its own per-node-explicit-pinning protocol —
    // either Op::Copy edges that set the target on their output
    // (preserved by this overwrite because they're set ahead of the
    // realize call), or a side-table this prep pass consults.
    {
        let mut g = graph
            .write()
            .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
        for &id in &order {
            let node = g.node(id);
            if matches!(node.op, Op::Const | Op::Release)
                || node.op.is_view_op()
                || matches!(node.op, Op::Reshape(_))
            {
                continue;
            }
            g.set_target_backend(id, backend_id);
        }
    }

    Ok((cache, backend_id, effective_targets))
}

/// For each reachable `Op::Const`, take its legacy storage slot,
/// extract bytes via the dyn host-buffer interface, and produce a
/// device-resident `fuel_storage::Storage` keyed in a StorageCache by
/// the Const's NodeId.
///
/// **CPU device** (target == `DeviceLocation::Cpu`): per-Const
/// CPU-storage construction — no transient graph, no executor
/// invocation. Just `CpuStorageBytes::from_bytes(host_bytes)`.
///
/// **Non-CPU device** (Phase 3b of bridge-retirement, post-9c):
/// builds a transient graph with one `Op::Const → Op::Copy { target }`
/// pair per user Const, seeds the transient StorageCache with CPU
/// storages of host bytes (+ a device-handle anchor), and realizes
/// the Op::Copy targets via `PipelinedExecutor::realize_many`. The
/// resulting device storages are inserted at the **original** user-
/// Const NodeIds. The transient graph isn't observable to the user
/// — only the user-Const NodeIds appear in the returned cache.
///
/// This replaces the deleted `upload_host_buffer`'s per-`DeviceLocation`
/// match. The per-target match now lives in the executor's
/// `WorkItemKind::Copy` arm (output allocation) and the
/// `copy_from_cpu_wrapper` (per-target H2D).
fn build_const_cache(
    graph: &Arc<RwLock<Graph>>,
    order: &[NodeId],
    device: &Device,
    initial: StorageCache,
) -> Result<StorageCache> {
    let mut cache = initial;
    cache.reserve(order.len() / 4);

    // Pass 1: collect (user_const_id, host_bytes, dtype, need_bytes)
    // for every reachable Op::Const that isn't already in the cache
    // (persistent slots from InferenceContext take precedence).
    let consts_to_upload: Vec<(NodeId, Vec<u8>, fuel_core_types::DType)> = {
        let g = graph
            .read()
            .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
        let mut out: Vec<(NodeId, Vec<u8>, fuel_core_types::DType)> =
            Vec::with_capacity(order.len() / 4);
        for &id in order {
            if cache.contains_key(&id) {
                continue;
            }
            let node = g.node(id);
            if !matches!(node.op, Op::Const) {
                continue;
            }
            let slot_arc = g.storage_for(id).ok_or_else(|| {
                Error::Msg(format!(
                    "pipelined_bridge: Op::Const node {id:?} has no \
                     storage in graph.storage_map (constructor failed \
                     to seed the slot)",
                ))
                .bt()
            })?;
            let (host_buf, dtype) = {
                let slot = slot_arc
                    .read()
                    .map_err(|_| Error::Msg("slot lock poisoned".into()).bt())?;
                (slot.as_dyn().to_host_buffer_dyn()?, slot.dtype())
            };
            // Truncate to the node's declared shape. The slot's buffer
            // may hold more bytes than the node consumes (shared
            // storage across views, padding for alignment). Same
            // truncation contract the deleted `upload_host_buffer`'s
            // `truncate_to` parameter enforced.
            let need_bytes = node.shape.elem_count() * dtype.size_in_bytes();
            let mut bytes = host_buffer_to_bytes(&host_buf);
            if bytes.len() > need_bytes {
                bytes.truncate(need_bytes);
            }
            out.push((id, bytes, dtype));
        }
        out
    };

    if consts_to_upload.is_empty() {
        return Ok(cache);
    }

    let target_loc = device.location();
    if target_loc == DeviceLocation::Cpu {
        // CPU realize: short-circuit. CPU→CPU through the executor
        // would be one extra memcpy per Const for no architectural
        // benefit (the per-`DeviceLocation` match in the deleted
        // `upload_host_buffer` was about routing to the right
        // backend allocator; for CPU there's no routing decision).
        for (id, bytes, dtype) in consts_to_upload {
            let storage = Storage::new(
                BackendStorage::Cpu(fuel_cpu_backend::byte_storage::CpuStorageBytes::from_bytes(
                    &bytes,
                )),
                dtype,
            );
            cache.insert(id, Arc::new(RwLock::new(storage)));
        }
        return Ok(cache);
    }

    // Non-CPU realize: build a transient graph with `Op::Const →
    // Op::Copy { target: target_loc }` pairs and realize the Op::Copy
    // targets multi-target. The transient graph is internal — the
    // user's graph stays unmodified.
    let transient = Arc::new(RwLock::new(Graph::new()));
    let mut transient_cache = StorageCache::new();

    // Device-handle anchor: the executor's Op::Copy arm derives the
    // device handle by searching the cache for any storage on the
    // target backend. Without an anchor, the first Op::Copy can't
    // resolve a CUDA/Vulkan device handle. Push an Op::Const
    // placeholder first; its cache entry is the 4-byte device-seed
    // Storage.
    if let Some(seed) = device_seed_storage(device)? {
        let anchor_id = {
            let mut g = transient
                .write()
                .map_err(|_| Error::Msg("transient graph lock poisoned".into()).bt())?;
            g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: fuel_core_types::Shape::from_dims(&[4]),
                dtype: fuel_core_types::DType::U8,
            })
        };
        transient_cache.insert(anchor_id, Arc::new(RwLock::new(seed)));
    }

    // Push one Op::Const → Op::Copy pair per user Const. The
    // transient Const's cache entry is the CPU storage of the host
    // bytes; the Op::Copy reads it and produces a device-resident
    // output. Keep parallel vectors of (user_const_id, transient
    // copy_id) so we can write results into the user's cache.
    //
    // target_backend on the Op::Copy nodes = Cpu (the SOURCE
    // backend; the kernel that runs is `copy_from_cpu_wrapper`,
    // registered at `(OpKind::Copy, [dt, dt], Cpu)`). The
    // executor's WorkItemKind::Copy arm reads target_location from
    // the op's variant to know where to allocate the output.
    let mut user_to_copy: Vec<(NodeId, NodeId)> =
        Vec::with_capacity(consts_to_upload.len());
    {
        let mut g = transient
            .write()
            .map_err(|_| Error::Msg("transient graph lock poisoned".into()).bt())?;
        for (user_id, bytes, dtype) in consts_to_upload.into_iter() {
            let n_elem = if dtype.size_in_bytes() == 0 {
                0
            } else {
                bytes.len() / dtype.size_in_bytes()
            };
            let shape = fuel_core_types::Shape::from_dims(&[n_elem]);
            let trans_const_id = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: shape.clone(),
                dtype,
            });
            let cpu_storage = Storage::new(
                BackendStorage::Cpu(fuel_cpu_backend::byte_storage::CpuStorageBytes::from_bytes(
                    &bytes,
                )),
                dtype,
            );
            transient_cache.insert(trans_const_id, Arc::new(RwLock::new(cpu_storage)));
            let copy_id = g.push(Node {
                op: Op::Copy { target: target_loc },
                inputs: vec![trans_const_id],
                shape,
                dtype,
            });
            g.set_target_backend(copy_id, BackendId::Cpu);
            user_to_copy.push((user_id, copy_id));
        }
    }

    let copy_targets: Vec<NodeId> = user_to_copy.iter().map(|(_, c)| *c).collect();
    let realized = PipelinedExecutor::realize_many(
        Arc::clone(&transient), &copy_targets, transient_cache,
    )?;
    if realized.len() != user_to_copy.len() {
        return Err(Error::Msg(format!(
            "build_const_cache: realize_many returned {} storages for {} \
             Op::Copy targets — internal bug",
            realized.len(), user_to_copy.len(),
        )).bt());
    }
    for ((user_id, _), (arc, _layout)) in user_to_copy.into_iter().zip(realized) {
        cache.insert(user_id, arc);
    }
    Ok(cache)
}

/// Extract the raw bytes from a `HostBuffer` via a per-variant match
/// (`bytemuck::cast_slice` for typed numeric vecs; identity for the
/// raw-byte sub-byte variants).
fn host_buffer_to_bytes(buf: &HostBuffer) -> Vec<u8> {
    match buf {
        HostBuffer::U8(v) => v.clone(),
        HostBuffer::I8(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::U32(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::I16(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::I32(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::I64(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::BF16(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::F16(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::F32(v) => bytemuck::cast_slice(v).to_vec(),
        HostBuffer::F64(v) => bytemuck::cast_slice(v).to_vec(),
        // F8E4M3 has no `Pod` impl in the float8 crate; reinterpret
        // via std::slice::from_raw_parts. `F8E4M3` is `Copy` + 1 byte
        // wide so this is a safe transmute over &[F8E4M3] → &[u8].
        HostBuffer::F8E4M3(v) => {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<float8::F8E4M3>(),
                )
            };
            bytes.to_vec()
        }
        HostBuffer::F6E2M3(v) => v.clone(),
        HostBuffer::F6E3M2(v) => v.clone(),
        HostBuffer::F4(v) => v.clone(),
        HostBuffer::F8E8M0(v) => v.clone(),
    }
}

/// Map a `Device` (the fuel-core wrapper around `DynBackendDevice`) to
/// the `BackendId` the kernel-binding-table keys on. Mirrors the
/// `DeviceLocation` variants 1:1.
fn device_to_backend_id(device: &Device) -> BackendId {
    match device.location() {
        DeviceLocation::Cpu => BackendId::Cpu,
        DeviceLocation::Cuda { .. } => BackendId::Cuda,
        DeviceLocation::Vulkan { .. } => BackendId::Vulkan,
        DeviceLocation::Metal { .. } => BackendId::Metal,
    }
}

/// Allocate a small "device anchor" storage on `device` — enough bytes
/// to carry the device handle into the [`StorageCache`] so the
/// pipelined executor's [`WorkItemKind::Alloc`] arm can derive the
/// per-backend handle for `Op::Alloc` nodes.
///
/// Phase 3a of bridge-retirement (post-9c). This is the *residual*
/// of the deleted [`fuel-core::inference_context::alloc_zeroed_on`]:
/// it does only the per-backend "allocate-on-device" piece, not the
/// zero-fill (that moves to the executor's Alloc arm). Callers
/// (today: [`crate::inference_context::KvCache::with_capacity`])
/// insert the returned Storage into the StorageCache before realizing
/// Op::Alloc nodes; the executor finds the device handle by searching
/// the cache for any storage on the target backend.
///
/// For CPU targets returns `Ok(None)` — CPU's Op::Alloc arm doesn't
/// need a device-handle anchor (`alloc_cpu_zeroed` is allocator-free).
///
/// The 4-byte size is arbitrary: small enough to be ~free, large
/// enough that even Vulkan's strict `vkAllocateMemory` accepts it.
pub fn device_seed_storage(device: &Device) -> Result<Option<Storage>> {
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    const SEED_BYTES: usize = 4;
    match device.location() {
        DeviceLocation::Cpu => Ok(None),
        #[cfg(feature = "cuda")]
        DeviceLocation::Cuda { .. } => {
            let cuda_dev = crate::cuda_backend::as_device(device)?;
            let cuda_bytes =
                fuel_cuda_backend::CudaStorageBytes::alloc(cuda_dev, SEED_BYTES)?;
            Ok(Some(Storage::new(BackendStorage::Cuda(cuda_bytes), fuel_core_types::DType::U8)))
        }
        #[cfg(not(feature = "cuda"))]
        DeviceLocation::Cuda { .. } => Err(Error::Msg(
            "device_seed_storage: CUDA device requested but fuel-core wasn't built \
             with --features cuda".into(),
        )
        .bt()),
        #[cfg(feature = "vulkan")]
        DeviceLocation::Vulkan { .. } => {
            let backend = crate::vulkan_backend::as_device(device)?;
            let zeros = vec![0_u8; SEED_BYTES];
            let vk_bytes = backend.upload_bytes_handle(&zeros)?;
            Ok(Some(Storage::new(BackendStorage::Vulkan(vk_bytes), fuel_core_types::DType::U8)))
        }
        #[cfg(not(feature = "vulkan"))]
        DeviceLocation::Vulkan { .. } => Err(Error::Msg(
            "device_seed_storage: Vulkan device requested but fuel-core wasn't built \
             with --features vulkan".into(),
        )
        .bt()),
        other => Err(Error::Msg(format!(
            "device_seed_storage: device {other:?} not wired (CPU + CUDA + Vulkan \
             today; Metal pending its byte-storage substrate)",
        ))
        .bt()),
    }
}
