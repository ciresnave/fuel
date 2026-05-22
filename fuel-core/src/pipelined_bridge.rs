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
//! 3. Calls [`PipelinedExecutor::realize_many`] for multi-target or
//!    `PipelinedExecutor::realize` for single-target.
//! 4. D2H's the resulting `Storage` Arc back into a `Vec<T>` for the
//!    caller via [`BackendStorage::read_to_cpu_bytes`].
//!
//! This module owns steps 1–4 so [`crate::lazy::LazyTensor`]'s
//! `realize_*` methods stay one-liners.
//!
//! ## Not yet covered (Phase E.3+)
//!
//! - `KVCache<B>` and `forward_with_cache_on<B>` — autoregressive
//!   decoding needs a const cache that survives realize calls; the
//!   pattern is "caller holds a long-lived `StorageCache` across
//!   calls" but the API surface for that lands in Phase E.3.
//! - `generate_*` and speculative decoding loops — same.
//! - `realize_f32_vulkan` — `BackendStorage::read_to_cpu_bytes`
//!   for Vulkan isn't wired yet (the byte-storage substrate doesn't
//!   carry a `VulkanBackend` handle); kept on the legacy executor
//!   until that gap closes.

use std::sync::{Arc, RwLock};

use fuel_core_types::{
    probe::BackendId, DeviceLocation, Error, HostBuffer, Result,
};
use fuel_graph::{Graph, NodeId, Op, topo_order_multi};
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
/// 3. `PipelinedExecutor::realize` — kick the compile + execute
///    pipeline.
/// 4. `read_to_cpu_bytes` — D2H the result.
/// 5. `bytemuck::cast_slice` — reinterpret the bytes as `T`.
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
    let (cache, _backend_id) = prepare(graph, &[target], device, initial)?;
    let (storage, _layout) =
        PipelinedExecutor::realize(graph.clone(), target, cache)?;
    let guard = storage
        .read()
        .map_err(|_| Error::Msg("storage lock poisoned".into()).bt())?;
    let bytes = guard.inner.read_to_cpu_bytes()?;
    Ok(bytemuck::cast_slice::<u8, T>(&bytes).to_vec())
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
    let (cache, _) = prepare(graph, targets, device, initial)?;
    let results = PipelinedExecutor::realize_many(graph.clone(), targets, cache)?;
    let mut out = Vec::with_capacity(results.len());
    for (storage, _layout) in results {
        let guard = storage
            .read()
            .map_err(|_| Error::Msg("storage lock poisoned".into()).bt())?;
        let bytes = guard.inner.read_to_cpu_bytes()?;
        out.push(bytemuck::cast_slice::<u8, T>(&bytes).to_vec());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Prep — internal
// ---------------------------------------------------------------------------

/// One-shot prep: derive a `BackendId` from `device`, propagate it to
/// every reachable computational node, and build a `StorageCache`
/// containing every reachable `Op::Const`. Mutates the graph (takes a
/// write lock); the executor takes its own read lock after this
/// returns.
fn prepare(
    graph: &Arc<RwLock<Graph>>,
    targets: &[NodeId],
    device: &Device,
    initial: StorageCache,
) -> Result<(StorageCache, BackendId)> {
    let backend_id = device_to_backend_id(device);
    let order = {
        let g = graph
            .read()
            .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
        topo_order_multi(&g, targets)
    };

    // Build the StorageCache on top of `initial` (which may carry
    // persistent storages from an InferenceContext). build_const_cache
    // adds any reachable Op::Const NodeId not already present.
    let cache = build_const_cache(graph, &order, device, initial)?;

    // Now set target_backend on every computational node. View ops,
    // Reshape, Const, and Release inherit/don't need it — see
    // `compile_one` in fuel-storage::pipelined.
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

    Ok((cache, backend_id))
}

/// For each reachable `Op::Const`, take its legacy storage slot,
/// extract bytes via the dyn host-buffer interface, and upload to
/// `device` as a fresh `fuel_storage::Storage`. Insert into a
/// StorageCache keyed by the Const's NodeId.
fn build_const_cache(
    graph: &Arc<RwLock<Graph>>,
    order: &[NodeId],
    device: &Device,
    initial: StorageCache,
) -> Result<StorageCache> {
    let g = graph
        .read()
        .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
    let mut cache = initial;
    cache.reserve(order.len() / 4);
    for &id in order {
        // Persistent slots from InferenceContext (or already-uploaded
        // Consts from a prior pass) take precedence — don't re-fetch
        // from the graph's storage_map.
        if cache.contains_key(&id) {
            continue;
        }
        let node = g.node(id);
        if !matches!(node.op, Op::Const) {
            continue;
        }
        let slot_arc = match g.storage_for(id) {
            Some(s) => s,
            None => {
                return Err(Error::Msg(format!(
                    "pipelined_bridge: Op::Const node {id:?} has no \
                     storage in graph.storage_map (constructor failed \
                     to seed the slot)",
                ))
                .bt());
            }
        };
        let (host_buf, dtype) = {
            let slot = slot_arc
                .read()
                .map_err(|_| Error::Msg("slot lock poisoned".into()).bt())?;
            (slot.as_dyn().to_host_buffer_dyn()?, slot.dtype())
        };
        // Truncate to the node's declared shape. The slot's buffer
        // may hold more bytes than the node consumes (e.g. when the
        // slot is shared across multiple views or padded for
        // alignment). Mirrors the legacy executor's
        // `backend.upload(&buf, shape)` which is shape-bounded.
        let need_elem = node.shape.elem_count();
        let need_bytes = need_elem * dtype.size_in_bytes();
        let storage = upload_host_buffer(&host_buf, dtype, device, Some(need_bytes))?;
        cache.insert(id, Arc::new(RwLock::new(storage)));
    }
    Ok(cache)
}

/// Upload a `HostBuffer` to a `Device`, producing the new
/// `fuel_storage::Storage` shape. Bytes are extracted via a per-dtype
/// match (no `HostBuffer::as_bytes` helper exists yet — should land
/// in fuel-core-types when other call sites need it). `truncate_to`
/// caps the bytes uploaded — used when a Const slot is shared across
/// views and only the leading `shape.elem_count() * dtype.size`
/// bytes are this node's view.
fn upload_host_buffer(
    buf: &HostBuffer,
    dtype: fuel_core_types::DType,
    device: &Device,
    truncate_to: Option<usize>,
) -> Result<Storage> {
    let mut bytes = host_buffer_to_bytes(buf);
    if let Some(n) = truncate_to {
        if bytes.len() > n {
            bytes.truncate(n);
        }
    }
    match device.location() {
        DeviceLocation::Cpu => Ok(Storage::new(
            BackendStorage::Cpu(fuel_cpu_backend::byte_storage::CpuStorageBytes::from_bytes(
                &bytes,
            )),
            dtype,
        )),
        #[cfg(feature = "cuda")]
        DeviceLocation::Cuda { .. } => {
            // Downcast the caller's Device handle to reuse their
            // CudaDevice (context, stream, cuBLAS handle, etc.).
            // Constructing a fresh CudaDevice per realize would tear
            // down + rebuild the context — way too expensive.
            let cuda_dev = crate::cuda_backend::as_device(device)?;
            let cuda_bytes =
                fuel_cuda_backend::CudaStorageBytes::from_cpu_bytes(cuda_dev, &bytes)?;
            Ok(Storage::new(BackendStorage::Cuda(cuda_bytes), dtype))
        }
        #[cfg(not(feature = "cuda"))]
        DeviceLocation::Cuda { .. } => Err(Error::Msg(
            "pipelined_bridge: CUDA device requested but fuel-core wasn't built \
             with --features cuda"
                .into(),
        )
        .bt()),
        #[cfg(feature = "vulkan")]
        DeviceLocation::Vulkan { .. } => {
            // `upload_bytes_handle` allocates a fresh device buffer +
            // attaches the `Arc<VulkanBackend>` handle so the resulting
            // `VulkanStorageBytes` flows through the pipelined-executor
            // binding-table dispatch (kernels reach the backend through
            // an input's storage).
            let backend = crate::vulkan_backend::as_device(device)?;
            let vk_bytes = backend.upload_bytes_handle(&bytes)?;
            Ok(Storage::new(BackendStorage::Vulkan(vk_bytes), dtype))
        }
        #[cfg(not(feature = "vulkan"))]
        DeviceLocation::Vulkan { .. } => Err(Error::Msg(
            "pipelined_bridge: Vulkan device requested but fuel-core wasn't built \
             with --features vulkan"
                .into(),
        )
        .bt()),
        other => Err(Error::Msg(format!(
            "pipelined_bridge: upload to {other:?} not yet wired (Metal D2H \
             integration pending — these stay on the legacy executor for now)",
        ))
        .bt()),
    }
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
