//! Persistent inference state for the pipelined executor (Phase 7.6
//! step 9c, sub-phase E.3.0).
//!
//! ## What this module owns
//!
//! - **[`InferenceContext`]**: a host-side per-session struct holding
//!   long-lived `Arc<RwLock<Storage>>` references that survive across
//!   realize calls. Replaces the legacy executor's `const_pool`
//!   for the pipelined dispatch world. Today's backing is a simple
//!   in-memory `HashMap<NodeId, Arc<...>>`; future work may swap this
//!   for an mmap-backed coherent store per
//!   [`project_unified_durable_tensor_store.md`].
//!
//! - **[`KvCache`]**, **[`KvLayer`]**, **[`KvLayerId`]**, **[`KvSlot`]**,
//!   **[`AuthorityState`]**: the backend-erased KV cache primitive that
//!   replaces `lazy::KVCache<B>` and `lazy::LlamaKVCache`. Single
//!   concrete type with no generic-over-B parameter; each layer holds
//!   `Arc<RwLock<Storage>>` for K and V plus side-table metadata
//!   (layout, monotonic version, coherence authority).
//!
//! ## What this module deliberately does NOT do (yet)
//!
//! - **Pre-allocated buffers** (added Phase E.3.3.A, 2026-05-20):
//!   [`KvCache::with_capacity`] eagerly allocates `[1, n_kv_heads,
//!   max_seq_len, head_dim]` zero-initialized K + V buffers on a
//!   target device. Subsequent forward steps write into these via
//!   `Op::WriteSlice` rather than growing-by-replacement. The legacy
//!   [`KvCache::with_dims`] constructor stays available for callers
//!   not yet on the pre-allocated path (grow-by-replace works without
//!   changes).
//! - **Multi-device coherence protocol**: the `authority` and
//!   `version` fields exist as placeholders. No protocol consults
//!   them yet — every layer starts and stays `AuthorityState::Host`
//!   for the lifetime of the session in single-device usage.
//!   Phase J (multi-GPU) activates the protocol.
//! - **`forward_with_cache_on` migration**: the legacy `KVCache<B>`
//!   and `LlamaKVCache` from `lazy.rs` are still the active types
//!   for autoregressive decoding. Phase E.3.3 ports them to use
//!   `KvCache` + `InferenceContext`.
//! - **`generate_*` + speculative decoding**: Phase E.3.4.
//! - **Weight persistence**: weights stay in the graph's storage_map
//!   per the design discussion; the persistent map handles KV layers
//!   and transient cross-step state only.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use fuel_core_types::{DType, DeviceLocation, Error, Layout, Result, Shape};
use fuel_graph::{Graph, Node, NodeId, Op};
use fuel_dispatch::{pipelined::{PipelinedExecutor, StorageCache}};
use fuel_storage::{BackendStorage, Storage};

use crate::Device;

// ===========================================================================
// KV cache primitive types
// ===========================================================================

/// Identifies a single key-or-value tensor within a [`KvCache`].
/// Stable across realize calls — the graph-level NodeId may churn
/// per step (each `Op::WriteSlice` produces a fresh NodeId pointing
/// at the same Storage Arc) but the logical KV slot identity stays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvLayerId {
    pub layer_idx: usize,
    pub slot: KvSlot,
}

/// Which half of a KV pair a [`KvLayerId`] addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvSlot {
    K,
    V,
}

/// Coherence authority for one KV slot's storage. Placeholder for the
/// multi-device protocol; today every slot starts and stays
/// [`AuthorityState::Host`] for the lifetime of the session unless
/// future code explicitly transitions it.
///
/// The intended protocol (deferred until multi-GPU lands):
/// - **Host**: the host has the most recent version. Devices may
///   hold stale copies. The default at session start.
/// - **Device(loc)**: a device owns the most recent version. The
///   host's copy may be stale until an explicit flush.
#[derive(Debug, Clone)]
pub enum AuthorityState {
    Host,
    Device(DeviceLocation),
}

/// One layer's K + V slots in a backend-erased KV cache.
///
/// Storage Arcs are dispatch-erased: each Arc points at a
/// [`fuel_storage::BackendStorage`] enum variant (`Cpu`, `Cuda`,
/// `Vulkan`, `Metal`). The graph references these Arcs via the
/// persistent map in [`InferenceContext`]; readers see the bytes
/// through whichever Layout the graph carries for the consuming
/// NodeId.
pub struct KvLayer {
    pub k: Arc<RwLock<Storage>>,
    pub v: Arc<RwLock<Storage>>,
    /// View layout into `k` describing the live extent. For the grow-
    /// by-replacement pattern (today) the layout matches the storage's
    /// full shape. For pre-allocated buffers (Phase E.3.2) the layout
    /// describes the leading `[..., cached_len, ...]` slice.
    pub k_layout: Layout,
    pub v_layout: Layout,
    /// Monotonic write version for K. Bumps on every successful
    /// `Op::WriteSlice` targeting this slot. Used by the future
    /// multi-device coherence protocol to detect staleness.
    pub k_version: u64,
    pub v_version: u64,
    /// Coherence authority per slot. K and V can in principle diverge
    /// (a future optimization that decouples their update cadence
    /// could exploit this), though standard transformer decoding
    /// updates both together every step.
    pub k_authority: AuthorityState,
    pub v_authority: AuthorityState,
}

/// Backend-erased KV cache. Replaces `lazy::KVCache<B>` and
/// `lazy::LlamaKVCache`. Indexed by global layer index; `layers[i]`
/// is `None` if layer `i` hasn't been populated yet.
///
/// Pipeline-parallel friendly: when a model is split across devices,
/// each device's cache holds entries only for its layer range; other
/// indices are `None`. The model code queries `cache.layer(global_i)`
/// regardless of which device owns it; the storage Arc inside
/// [`KvLayer`] carries the device identity via its
/// [`fuel_storage::BackendStorage`] variant.
pub struct KvCache {
    pub layers: Vec<Option<KvLayer>>,
    pub cached_len: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Pre-allocated capacity along the sequence axis. `Some(n)` when
    /// constructed via [`KvCache::with_capacity`] — every layer's K + V
    /// storage holds `[1, n_kv_heads, n, head_dim]` of zeros and the
    /// forward path writes into it via `Op::WriteSlice`. `None` for the
    /// legacy [`KvCache::with_dims`] grow-by-replacement constructor.
    pub max_seq_len: Option<usize>,
    /// Dtype the pre-allocated buffers were allocated with. `Some(dt)`
    /// in the [`KvCache::with_capacity`] path, `None` in the legacy
    /// `with_dims` path (which leaves dtype to whatever the first
    /// inserted layer specifies).
    pub dtype: Option<DType>,
}

impl KvCache {
    /// Legacy grow-by-replacement constructor. Layers start `None`; the
    /// caller calls [`Self::set_layer`] each forward step with a freshly
    /// allocated `KvLayer` whose storage is sized to the current cached
    /// length. Subsequent steps replace the layer entirely with a
    /// larger buffer.
    ///
    /// Replaced by [`Self::with_capacity`] when the caller knows the
    /// generation's max sequence length up-front (the common case for
    /// autoregressive decoding).
    pub fn with_dims(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| None).collect(),
            cached_len: 0,
            n_kv_heads,
            head_dim,
            max_seq_len: None,
            dtype: None,
        }
    }

    /// Pre-allocated KV cache. Every layer's K + V storage is allocated
    /// up-front as a `[1, n_kv_heads, max_seq_len, head_dim]` zero
    /// buffer on `device` with `dtype`. Subsequent forward steps write
    /// fresh K/V slabs into these buffers via `Op::WriteSlice` rather
    /// than growing-by-replacement.
    ///
    /// Memory footprint: `n_layers * 2 * n_kv_heads * max_seq_len *
    /// head_dim * dtype_size` bytes. For Llama-7B at max_seq_len=4096,
    /// bf16: 32 * 2 * 8 * 4096 * 128 * 2 ≈ 512 MiB.
    ///
    /// Returns `Err` if any per-layer allocation fails (e.g. CUDA OOM)
    /// or the requested device hasn't been wired up in
    /// [`pipelined_bridge`] (Vulkan / Metal — D2H/H2D for those still
    /// goes through the legacy executor).
    pub fn with_capacity(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // Bridge-retirement Phase 3a (post-9c): allocate the 2N K/V
        // buffers via `Op::Alloc` graph emission, not direct backend
        // calls. The executor's `WorkItemKind::Alloc` arm dispatches
        // per-backend; the per-`DeviceLocation` match that used to
        // live in `alloc_zeroed_on` is now in the executor (the
        // architectural dispatch layer).
        let shape = Shape::from_dims(&[1, n_kv_heads, max_seq_len, head_dim]);
        let layout = Layout::contiguous(shape.clone());
        let target_loc = device.location();

        // Build the transient graph. For non-CPU targets the first
        // node is an `Op::Const` placeholder whose StorageCache entry
        // carries a small "device anchor" storage — the executor's
        // Alloc arm searches the cache for any storage on the target
        // backend to derive the device handle, so this entry makes
        // the first Op::Alloc's device lookup succeed. The anchor's
        // NodeId must exist in the graph so `realize_many`'s
        // `layout_cache` seeding (which calls `g.layout(id)` on every
        // cache key) doesn't panic.
        let graph = Arc::new(RwLock::new(Graph::new()));
        let mut cache = StorageCache::new();
        if let Some(seed) = crate::pipelined_bridge::device_seed_storage(device)? {
            let anchor_id = {
                let mut g = graph
                    .write()
                    .map_err(|_| Error::Msg("graph lock poisoned during KvCache build".into()).bt())?;
                g.push(Node {
                    op: Op::Const,
                    inputs: vec![],
                    shape: Shape::from_dims(&[4]),
                    dtype: DType::U8,
                })
            };
            cache.insert(anchor_id, Arc::new(RwLock::new(seed)));
        }
        // Emit pairs of (Op::Alloc, Op::ZeroFill) per K/V slot. The
        // ZeroFill consumes the Alloc destructively (in-place fill,
        // output adopts the Alloc's Storage Arc). Realizing the
        // ZeroFill IDs produces zero-initialized device storages.
        //
        // Phase 3a follow-up architecture: Op::Alloc gives uninit
        // memory; Op::ZeroFill explicitly fills with zero. On Vulkan
        // this replaces the host-staged-zeros path with device-side
        // `vkCmdFillBuffer` (~2× the bandwidth saved). On CUDA the
        // pair becomes uninit `cuMemAlloc` + async `cuMemsetD8Async`
        // — same total cost as the old `alloc_zeros` but exposes the
        // ZeroFill as a first-class graph node the optimizer can
        // see (and skip if a downstream op covers the full buffer).
        let zero_fill_ids: Vec<NodeId> = {
            let mut g = graph
                .write()
                .map_err(|_| Error::Msg("graph lock poisoned during KvCache build".into()).bt())?;
            (0..(2 * n_layers))
                .map(|_| {
                    let alloc_id = g.push(Node {
                        op: Op::Alloc { target: target_loc },
                        inputs: vec![],
                        shape: shape.clone(),
                        dtype,
                    });
                    g.push(Node {
                        op: Op::ZeroFill,
                        inputs: vec![alloc_id],
                        shape: shape.clone(),
                        dtype,
                    })
                })
                .collect()
        };

        // Realize all 2*n_layers Op::ZeroFill targets in one pass —
        // PipelinedExecutor::realize_many shares the compile/execute
        // pipeline so device-handle reuse is automatic, and the
        // executor's destructive_input cleanup evicts the Op::Alloc
        // intermediate NodeIds while the ZeroFill targets adopt the
        // same Arcs (post-fill bytes).
        let realized = PipelinedExecutor::realize_many(
            Arc::clone(&graph), &zero_fill_ids, cache,
        )?;
        if realized.len() != 2 * n_layers {
            return Err(Error::Msg(format!(
                "KvCache::with_capacity: realize_many returned {} storages \
                 for {} Op::ZeroFill targets — internal bug",
                realized.len(), 2 * n_layers,
            )).bt());
        }

        let mut realized_iter = realized.into_iter();
        let mut layers: Vec<Option<KvLayer>> = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let (k_arc, _) = realized_iter.next().expect("checked above");
            let (v_arc, _) = realized_iter.next().expect("checked above");
            layers.push(Some(KvLayer {
                k: k_arc,
                v: v_arc,
                k_layout: layout.clone(),
                v_layout: layout.clone(),
                k_version: 0,
                v_version: 0,
                k_authority: AuthorityState::Host,
                v_authority: AuthorityState::Host,
            }));
        }

        // Silence the unused-bind lint for elem_count when this
        // path no longer needs the legacy alloc helper.
        let _ = (n_kv_heads, max_seq_len, head_dim);

        Ok(Self {
            layers,
            cached_len: 0,
            n_kv_heads,
            head_dim,
            max_seq_len: Some(max_seq_len),
            dtype: Some(dtype),
        })
    }

    /// Borrow the K-or-V storage Arc for a given layer + slot. Used by
    /// the forward path to bind cache storage to per-step Const nodes
    /// via [`InferenceContext::insert`]. Returns `None` if the layer
    /// isn't populated (with_dims path before first
    /// [`Self::set_layer`]) or `layer_idx` is out of range.
    pub fn slot_storage(
        &self,
        layer_idx: usize,
        slot: KvSlot,
    ) -> Option<Arc<RwLock<Storage>>> {
        let layer = self.layer(layer_idx)?;
        Some(match slot {
            KvSlot::K => Arc::clone(&layer.k),
            KvSlot::V => Arc::clone(&layer.v),
        })
    }

    /// Bump a slot's monotonic version counter. Called after every
    /// successful `Op::WriteSlice` targeting the slot — the forward
    /// path invokes this once per (layer, slot) per step.
    ///
    /// The version is consumed by the future multi-device coherence
    /// protocol (placeholder today — see [`AuthorityState`]).
    pub fn bump_version(&mut self, layer_idx: usize, slot: KvSlot) {
        if let Some(layer) = self.layer_mut(layer_idx) {
            match slot {
                KvSlot::K => layer.k_version += 1,
                KvSlot::V => layer.v_version += 1,
            }
        }
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, li: usize) -> Option<&KvLayer> {
        self.layers.get(li).and_then(|o| o.as_ref())
    }

    pub fn layer_mut(&mut self, li: usize) -> Option<&mut KvLayer> {
        self.layers.get_mut(li).and_then(|o| o.as_mut())
    }

    pub fn set_layer(&mut self, li: usize, layer: KvLayer) {
        self.layers[li] = Some(layer);
    }

    /// Drop every layer; reset `cached_len` to zero. Use between
    /// independent generations (the cache's K/V shapes are tied to
    /// a specific prompt prefix).
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            *layer = None;
        }
        self.cached_len = 0;
    }

    /// Shrink the cache to `new_len` cached positions. Speculative-
    /// decoding's reject path calls this after a draft batch is
    /// rejected by the target model.
    ///
    /// **Phase E.3.0 limitation**: this only updates `cached_len`;
    /// the underlying storages and their layouts are *not* shrunk.
    /// The grow-by-replacement pattern (today) means the next forward
    /// pass would still see the old layout's full extent. Phase E.3.2
    /// pairs this with pre-allocated buffers + `Op::WriteSlice` so
    /// truncate becomes a pure metadata update — the trailing rows
    /// of the pre-allocated buffer simply stop being read.
    pub fn truncate_to(&mut self, new_len: usize) {
        if new_len < self.cached_len {
            self.cached_len = new_len;
        }
    }
}

// `alloc_zeroed_on` retired 2026-05-22 (bridge-retirement Phase 3a).
// Zero-init device allocation now flows through `Op::Alloc` graph
// emission + the executor's `WorkItemKind::Alloc` arm. The per-
// `DeviceLocation` match this function used to carry lives in
// `fuel-storage::pipelined::execute_work_item`'s Alloc arm; the
// residual "0-byte device anchor" helper lives in
// `crate::pipelined_bridge::device_seed_storage`.

// ===========================================================================
// InferenceContext
// ===========================================================================

/// Per-session host-side context holding long-lived storage Arcs
/// that survive across realize calls.
///
/// The persistent map is the seam the unified-storage / mmap-coherence
/// backplane work later replaces (see
/// [`project_unified_durable_tensor_store.md`]); today it's a simple
/// in-memory `HashMap`. Each realize call clones the Arcs into the
/// executor's input cache; persistent entries reuse those Arcs
/// across calls instead of re-uploading.
///
/// ## Lifecycle
///
/// 1. Construct with a target device: `InferenceContext::new(device)`.
/// 2. Insert long-lived storages (KV layer storages, anything else
///    that should survive across realize calls):
///    `ctx.insert(node_id, arc)`.
/// 3. Realize: `ctx.realize_one_as::<f32>(&graph, target_node)`. The
///    persistent map is seeded into the executor's input cache; new
///    `Op::Const` nodes the graph references that aren't in the
///    persistent map get uploaded fresh from the graph's storage map.
/// 4. Subsequent realize calls reuse the persistent Arcs — no
///    re-upload, no D2H/H2D round-trip.
pub struct InferenceContext {
    device: Device,
    persistent: HashMap<NodeId, Arc<RwLock<Storage>>>,
}

impl InferenceContext {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            persistent: HashMap::new(),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Insert a long-lived storage at `node_id`. The Arc survives
    /// across subsequent realize calls and is treated as pre-realized
    /// for any graph that references `node_id` as a Const-shaped leaf.
    ///
    /// Replaces any prior entry at `node_id`.
    pub fn insert(&mut self, node_id: NodeId, storage: Arc<RwLock<Storage>>) {
        self.persistent.insert(node_id, storage);
    }

    /// Remove the persistent entry at `node_id`. Returns the Arc if
    /// one was present. After this, the next realize that touches
    /// `node_id` will re-fetch from the graph's storage_map (and fail
    /// if the slot isn't populated there either).
    pub fn remove(&mut self, node_id: NodeId) -> Option<Arc<RwLock<Storage>>> {
        self.persistent.remove(&node_id)
    }

    /// Whether the persistent map has an entry at `node_id`.
    pub fn contains(&self, node_id: NodeId) -> bool {
        self.persistent.contains_key(&node_id)
    }

    /// Borrow the persistent entry at `node_id` if present.
    pub fn get(&self, node_id: NodeId) -> Option<&Arc<RwLock<Storage>>> {
        self.persistent.get(&node_id)
    }

    /// Number of persistent slots.
    pub fn len(&self) -> usize {
        self.persistent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.persistent.is_empty()
    }

    /// Drop every persistent entry. The Arcs are released; storages
    /// whose only remaining reference was the context's are freed
    /// (host memory for CPU, deferred-on-stream for async backends).
    pub fn clear_persistent(&mut self) {
        self.persistent.clear();
    }

    /// Realize a single target. The persistent map is seeded into
    /// the executor's input cache via `Arc::clone` (the persistent
    /// entries survive the call). Op::Const NodeIds in the graph
    /// that aren't already in the persistent map get uploaded fresh
    /// from `graph.storage_for(id)` per the existing pipelined-bridge
    /// pattern.
    pub fn realize_one_as<T: bytemuck::Pod>(
        &self,
        graph: &Arc<RwLock<Graph>>,
        target: NodeId,
    ) -> Result<Vec<T>> {
        crate::pipelined_bridge::realize_one_as_with_initial::<T>(
            graph,
            target,
            &self.device,
            self.cloned_persistent(),
        )
    }

    /// Multi-target counterpart of [`realize_one_as`].
    pub fn realize_many_as<T: bytemuck::Pod>(
        &self,
        graph: &Arc<RwLock<Graph>>,
        targets: &[NodeId],
    ) -> Result<Vec<Vec<T>>> {
        crate::pipelined_bridge::realize_many_as_with_initial::<T>(
            graph,
            targets,
            &self.device,
            self.cloned_persistent(),
        )
    }

    /// Build a [`StorageCache`] containing Arc-clones of every
    /// persistent entry. The clone is cheap (Arc refcount bumps); the
    /// returned `StorageCache` is consumed by the realize call but
    /// the original Arcs in `self.persistent` survive.
    fn cloned_persistent(&self) -> StorageCache {
        let mut out = StorageCache::with_capacity(self.persistent.len());
        for (id, arc) in &self.persistent {
            out.insert(*id, Arc::clone(arc));
        }
        out
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::{DType, Shape};
    use fuel_graph::{Node, Op};

    /// `InferenceContext::insert` + immediate retrieval round-trips.
    #[test]
    fn context_insert_retrieve() {
        let mut ctx = InferenceContext::new(Device::cpu());
        let storage = fuel_storage::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
        let arc = Arc::new(RwLock::new(storage));
        let id = NodeId(42);

        assert!(!ctx.contains(id));
        assert_eq!(ctx.len(), 0);

        ctx.insert(id, Arc::clone(&arc));

        assert!(ctx.contains(id));
        assert_eq!(ctx.len(), 1);
        // The returned Arc is the same as what we inserted.
        let retrieved = ctx.get(id).expect("just inserted");
        assert!(Arc::ptr_eq(retrieved, &arc));
    }

    /// Inserted Arcs survive across realize calls. Build a minimal
    /// 3-node graph (Const + Const + Add). Both Const slots come
    /// from the context's persistent map (no graph.storage_map
    /// involvement). Realize twice; verify the persistent Arcs are
    /// still held by the context after each call.
    #[test]
    fn context_persistent_arc_survives_realize() {
        let lhs_arc = Arc::new(RwLock::new(fuel_storage::from_slice_cpu(&[1.0_f32, 2.0, 3.0])));
        let rhs_arc = Arc::new(RwLock::new(fuel_storage::from_slice_cpu(&[10.0_f32, 20.0, 30.0])));

        let graph = Arc::new(RwLock::new(Graph::new()));
        let (lhs_id, rhs_id, add_id) = {
            let mut g = graph.write().unwrap();
            let lhs = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3]),
                dtype: DType::F32,
            });
            let rhs = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3]),
                dtype: DType::F32,
            });
            let add = g.push(Node {
                op: Op::Add,
                inputs: vec![lhs, rhs],
                shape: Shape::from_dims(&[3]),
                dtype: DType::F32,
            });
            (lhs, rhs, add)
        };

        let mut ctx = InferenceContext::new(Device::cpu());
        ctx.insert(lhs_id, Arc::clone(&lhs_arc));
        ctx.insert(rhs_id, Arc::clone(&rhs_arc));

        // Before realize: each Arc held by the test (1) + context (1) = 2.
        assert_eq!(Arc::strong_count(&lhs_arc), 2);
        assert_eq!(Arc::strong_count(&rhs_arc), 2);

        let out1 = ctx
            .realize_one_as::<f32>(&graph, add_id)
            .expect("first realize");
        assert_eq!(out1, vec![11.0, 22.0, 33.0]);

        // After first realize: same count. The realize call's
        // cloned_persistent() bumped the counts to 3 during the call
        // and dropped them back to 2 when the call returned.
        assert_eq!(Arc::strong_count(&lhs_arc), 2);
        assert_eq!(Arc::strong_count(&rhs_arc), 2);

        // Realize again — same Arcs get reused without re-upload.
        // This is the autoregressive-decoding pattern: weights stay
        // resident across forward passes.
        let out2 = ctx
            .realize_one_as::<f32>(&graph, add_id)
            .expect("second realize");
        assert_eq!(out2, vec![11.0, 22.0, 33.0]);
        assert_eq!(Arc::strong_count(&lhs_arc), 2);
        assert_eq!(Arc::strong_count(&rhs_arc), 2);
    }

    /// `KvCache::with_dims` produces a fresh cache of the right
    /// shape with all layers `None` and `cached_len = 0`.
    #[test]
    fn kv_cache_with_dims_constructs_fresh() {
        let cache = KvCache::with_dims(4, 8, 64);
        assert_eq!(cache.n_layers(), 4);
        assert_eq!(cache.cached_len, 0);
        assert_eq!(cache.n_kv_heads, 8);
        assert_eq!(cache.head_dim, 64);
        for li in 0..4 {
            assert!(cache.layer(li).is_none());
        }
    }

    /// `KvCache::set_layer` + `layer` round-trip.
    #[test]
    fn kv_cache_set_and_get_layer() {
        let mut cache = KvCache::with_dims(2, 4, 8);
        let k_arc = Arc::new(RwLock::new(fuel_storage::from_slice_cpu(&[0.0_f32; 32])));
        let v_arc = Arc::new(RwLock::new(fuel_storage::from_slice_cpu(&[0.0_f32; 32])));
        let layout = Layout::contiguous(Shape::from_dims(&[1, 4, 1, 8]));
        cache.set_layer(
            0,
            KvLayer {
                k: Arc::clone(&k_arc),
                v: Arc::clone(&v_arc),
                k_layout: layout.clone(),
                v_layout: layout.clone(),
                k_version: 0,
                v_version: 0,
                k_authority: AuthorityState::Host,
                v_authority: AuthorityState::Host,
            },
        );
        let layer = cache.layer(0).expect("just set");
        assert!(Arc::ptr_eq(&layer.k, &k_arc));
        assert!(Arc::ptr_eq(&layer.v, &v_arc));
        assert_eq!(layer.k_version, 0);
        assert!(matches!(layer.k_authority, AuthorityState::Host));
        assert!(cache.layer(1).is_none());
    }

    /// `KvCache::clear` drops all layers and resets `cached_len`.
    #[test]
    fn kv_cache_clear_drops_layers() {
        let mut cache = KvCache::with_dims(2, 4, 8);
        let k = Arc::new(RwLock::new(fuel_storage::from_slice_cpu(&[0.0_f32; 32])));
        let v = Arc::new(RwLock::new(fuel_storage::from_slice_cpu(&[0.0_f32; 32])));
        let layout = Layout::contiguous(Shape::from_dims(&[1, 4, 1, 8]));
        cache.set_layer(
            0,
            KvLayer {
                k: Arc::clone(&k),
                v: Arc::clone(&v),
                k_layout: layout.clone(),
                v_layout: layout,
                k_version: 1,
                v_version: 1,
                k_authority: AuthorityState::Host,
                v_authority: AuthorityState::Host,
            },
        );
        cache.cached_len = 7;
        // Before clear: cache holds an Arc to k → strong_count 2.
        assert_eq!(Arc::strong_count(&k), 2);

        cache.clear();

        assert!(cache.layer(0).is_none());
        assert_eq!(cache.cached_len, 0);
        // After clear: cache's Arc dropped → strong_count 1.
        assert_eq!(Arc::strong_count(&k), 1);
    }

    /// `KvCache::truncate_to` updates `cached_len` only (Phase E.3.0
    /// limitation; pre-allocated buffer + WriteSlice handles the
    /// in-place truncate in Phase E.3.2).
    #[test]
    fn kv_cache_truncate_updates_cached_len() {
        let mut cache = KvCache::with_dims(2, 4, 8);
        cache.cached_len = 16;
        cache.truncate_to(10);
        assert_eq!(cache.cached_len, 10);
        // Truncating to a value >= current is a no-op.
        cache.truncate_to(15);
        assert_eq!(cache.cached_len, 10);
        // Truncating to zero clears the live extent.
        cache.truncate_to(0);
        assert_eq!(cache.cached_len, 0);
    }

    /// Authority + version fields are placeholders today; just
    /// confirm they default sensibly via `KvLayer` construction.
    #[test]
    fn kv_layer_default_authority_is_host() {
        let arc = Arc::new(RwLock::new(fuel_storage::from_slice_cpu(&[0.0_f32; 4])));
        let layout = Layout::contiguous(Shape::from_dims(&[4]));
        let layer = KvLayer {
            k: Arc::clone(&arc),
            v: Arc::clone(&arc),
            k_layout: layout.clone(),
            v_layout: layout,
            k_version: 0,
            v_version: 0,
            k_authority: AuthorityState::Host,
            v_authority: AuthorityState::Host,
        };
        assert!(matches!(layer.k_authority, AuthorityState::Host));
        assert!(matches!(layer.v_authority, AuthorityState::Host));
        assert_eq!(layer.k_version, 0);
        assert_eq!(layer.v_version, 0);
    }

    // ---- KvCache::with_capacity + accessors (Phase E.3.3.A) -----------------

    /// `with_capacity` allocates n_layers fresh K + V buffers on the
    /// CPU device. Each buffer is `n_kv_heads * max_seq_len * head_dim`
    /// elements of the requested dtype, zero-initialized, with the
    /// `[1, n_kv_heads, max_seq_len, head_dim]` layout pre-populated.
    #[test]
    fn kv_cache_with_capacity_allocates_all_layers_on_cpu() {
        let device = Device::cpu();
        let cache = KvCache::with_capacity(
            /* n_layers     */ 3,
            /* n_kv_heads   */ 4,
            /* head_dim     */ 16,
            /* max_seq_len  */ 32,
            DType::F32,
            &device,
        ).expect("with_capacity");

        assert_eq!(cache.n_layers(), 3);
        assert_eq!(cache.cached_len, 0);
        assert_eq!(cache.n_kv_heads, 4);
        assert_eq!(cache.head_dim, 16);
        assert_eq!(cache.max_seq_len, Some(32));
        assert_eq!(cache.dtype, Some(DType::F32));

        // Every layer is populated and has the expected layout.
        for li in 0..3 {
            let layer = cache.layer(li).expect("layer populated");
            assert_eq!(layer.k_layout.shape().dims(), &[1, 4, 32, 16]);
            assert_eq!(layer.v_layout.shape().dims(), &[1, 4, 32, 16]);
            // Bytes are zero-initialized: 4 (n_kv) * 32 (seq) * 16 (head)
            // * 4 (f32) = 8192 bytes per slot.
            let k_guard = layer.k.read().unwrap();
            assert_eq!(k_guard.inner.len_bytes(), 8192);
            assert_eq!(k_guard.dtype, DType::F32);
            // Spot-check zero init.
            if let BackendStorage::Cpu(c) = &k_guard.inner {
                let typed: &[f32] = c.as_slice().unwrap();
                assert!(typed.iter().all(|&x| x == 0.0));
            } else {
                panic!("expected CPU storage");
            }
        }
    }

    /// `slot_storage(li, K)` returns the same Arc as `layer(li).k`.
    /// Used by the forward path to bind cache storage to per-step
    /// Const nodes.
    #[test]
    fn kv_cache_slot_storage_returns_layer_arc() {
        let device = Device::cpu();
        let cache = KvCache::with_capacity(2, 4, 8, 16, DType::F32, &device)
            .expect("with_capacity");

        let k_via_layer = Arc::clone(&cache.layer(0).unwrap().k);
        let k_via_slot = cache.slot_storage(0, KvSlot::K).expect("layer 0 K");
        assert!(
            Arc::ptr_eq(&k_via_layer, &k_via_slot),
            "slot_storage should return the same Arc as layer().k",
        );

        let v_via_layer = Arc::clone(&cache.layer(1).unwrap().v);
        let v_via_slot = cache.slot_storage(1, KvSlot::V).expect("layer 1 V");
        assert!(Arc::ptr_eq(&v_via_layer, &v_via_slot));
    }

    /// `slot_storage` returns `None` for an unpopulated layer (the
    /// with_dims path before the first set_layer) and for an out-of-
    /// range layer index.
    #[test]
    fn kv_cache_slot_storage_returns_none_for_unpopulated() {
        let cache = KvCache::with_dims(2, 4, 8);
        assert!(cache.slot_storage(0, KvSlot::K).is_none());
        assert!(cache.slot_storage(1, KvSlot::V).is_none());
        // Out of range.
        assert!(cache.slot_storage(5, KvSlot::K).is_none());
    }

    /// `bump_version` advances the per-slot version counter
    /// independently for K and V.
    #[test]
    fn kv_cache_bump_version_advances_per_slot() {
        let device = Device::cpu();
        let mut cache = KvCache::with_capacity(1, 4, 8, 16, DType::F32, &device)
            .expect("with_capacity");
        assert_eq!(cache.layer(0).unwrap().k_version, 0);
        assert_eq!(cache.layer(0).unwrap().v_version, 0);

        cache.bump_version(0, KvSlot::K);
        cache.bump_version(0, KvSlot::K);
        cache.bump_version(0, KvSlot::V);

        assert_eq!(cache.layer(0).unwrap().k_version, 2);
        assert_eq!(cache.layer(0).unwrap().v_version, 1);
    }

    /// Bumping a version on an unpopulated layer is a no-op (not a
    /// panic). The `with_dims` path leaves layers `None`; the
    /// forward code shouldn't blow up if it calls bump_version
    /// before the layer is wired.
    #[test]
    fn kv_cache_bump_version_unpopulated_is_noop() {
        let mut cache = KvCache::with_dims(2, 4, 8);
        cache.bump_version(0, KvSlot::K);
        cache.bump_version(99, KvSlot::V); // out of range, also no-op
        assert!(cache.layer(0).is_none());
    }

    /// Sanity: bf16 capacity allocation produces 2-byte elements.
    #[test]
    fn kv_cache_with_capacity_bf16_byte_count() {
        let device = Device::cpu();
        let cache = KvCache::with_capacity(1, 2, 4, 8, DType::BF16, &device)
            .expect("with_capacity bf16");
        let layer = cache.layer(0).unwrap();
        // 2 (n_kv) * 8 (seq) * 4 (head) * 2 (bf16) = 128 bytes per slot.
        assert_eq!(layer.k.read().unwrap().inner.len_bytes(), 128);
        assert_eq!(cache.dtype, Some(DType::BF16));
    }
}
