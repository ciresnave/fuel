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
//!   replaced `lazy::KVCache<B>` and `lazy::LlamaKVCache` (both
//!   retired — E.3.3.D and Unification Session 4 E.3.4). Single
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
//! - **Weight persistence**: weights stay in the graph's storage_map
//!   per the design discussion; the persistent map handles KV layers
//!   and transient cross-step state only.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use fuel_ir::{DType, DeviceLocation, Error, Layout, Result, Shape, SymEnv, SymId};
use fuel_graph::{Graph, Node, NodeId, Op};
use fuel_dispatch::{optimize::OptimizedGraph, pipelined::{PipelinedExecutor, StorageCache}};
use fuel_memory::{BackendStorage, Storage};

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
/// [`fuel_memory::BackendStorage`] enum variant (`Cpu`, `Cuda`,
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
/// [`fuel_memory::BackendStorage`] variant.
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

// ===========================================================================
// LatentKvCache — N-slot generalization of KvCache (MLA increment 4)
// ===========================================================================

/// One slot's persistent latent buffer within a [`LatentKvCache`] layer.
/// The per-slot analogue of [`KvLayer`]'s `k`/`v` pair, generalized to an
/// arbitrary slot count and independent per-slot trailing shape.
pub struct LatentSlot {
    pub storage: Arc<RwLock<Storage>>,
    /// View layout into `storage`. For the pre-allocated (capacity)
    /// buffers [`LatentKvCache::with_capacity`] allocates, the layout
    /// matches the storage's full `[max_seq_len, ...trailing]` shape.
    pub layout: Layout,
    /// Monotonic write version. Bumps on every successful
    /// `Op::WriteSlice` targeting this slot. Placeholder for the future
    /// multi-device coherence protocol — mirrors [`KvLayer::k_version`] /
    /// `v_version`.
    pub version: u64,
    /// Coherence authority. Placeholder — see [`AuthorityState`].
    pub authority: AuthorityState,
}

/// Cross-graph, cross-forward-pass persistent **N-slot** decode cache —
/// the [`KvCache`] generalization for latent-caching attention
/// architectures.
///
/// [`KvCache`] hardwires a symmetric K/V pair (two buffers per layer,
/// same shape). That's wrong for compression architectures:
///
///   - **Multi-head Latent Attention (DeepSeek-V2 MLA)**: per layer, a
///     low-rank compressed latent `[kv_lora_rank]` (slot 0) **and** a
///     single-head RoPE key `[qk_rope_head_dim]` (slot 1) — two slots of
///     *different* trailing shape, neither a `[n_kv_heads, head_dim]` K/V.
///   - **Two-projection attention / QKV pruning**: a single retained
///     projection per layer — one slot.
///
/// `LatentKvCache` is this module's counterpart to
/// [`crate::lazy_latent_cache::LazyLatentCache`] — that type makes the
/// exact same **per-layer, ordered list of latent buffers with
/// independent trailing shapes** generalization for the **per-forward-
/// pass** (single [`fuel_graph::Graph`]-anchored, functional
/// append-and-thread) lifecycle. `LatentKvCache` is the **persistent**
/// sibling: device-resident `Arc<RwLock<Storage>>` buffers that survive
/// across graphs/forward calls and are mutated in place via
/// `Op::WriteSlice`, exactly the relationship [`KvCache`] has to a plain
/// per-pass K/V cache. Read both modules' docs together for the two
/// lifecycles side by side.
///
/// # Shape contract
///
/// Slot `s` of every layer is a buffer `[max_seq_len, ...slot_trailing[s]]`
/// — the capacity axis is dim **0**. This differs from [`KvCache`]'s `[1,
/// n_kv_heads, max_seq_len, head_dim]` convention (capacity axis 2):
/// a latent is a per-token *vector*, not a per-head *plane*, so there is
/// no head axis to lead with. Matches [`crate::lazy_latent_cache::
/// LazyLatentCache`]'s dim-0 convention exactly — this type's per-slot
/// buffer is byte-for-byte what that type's `slot_buffer_full` realizes,
/// modulo device residency.
pub struct LatentKvCache {
    /// `layers[l]` is `None` until populated; `Some(slots)` holds one
    /// [`LatentSlot`] per slot index (pipeline-parallel friendly — mirrors
    /// [`KvCache::layers`]).
    pub layers: Vec<Option<Vec<LatentSlot>>>,
    pub cached_len: usize,
    /// Per-slot trailing shape (past the leading capacity axis); its
    /// length is the per-layer slot count.
    pub slot_trailing: Vec<Vec<usize>>,
    pub max_seq_len: usize,
    pub dtype: DType,
}

impl LatentKvCache {
    /// Pre-allocated N-slot latent cache. Every layer's slot buffers are
    /// allocated up-front as zero buffers on `device` with `dtype`,
    /// shaped `[max_seq_len, ...slot_trailing[s]]`. Mirrors [`KvCache::
    /// with_capacity`]'s `Op::Alloc` → `Op::ZeroFill` graph-emission
    /// pattern exactly, generalized from a fixed 2-buffer (K, V) layer to
    /// `slot_trailing.len()` buffers of independent shape.
    ///
    /// Returns `Err` for degenerate geometry (`n_layers == 0`,
    /// `max_seq_len == 0`, or no slots) or if any per-layer allocation
    /// fails (e.g. CUDA OOM / an unwired device).
    pub fn with_capacity(
        n_layers: usize,
        max_seq_len: usize,
        slot_trailing: Vec<Vec<usize>>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if n_layers == 0 {
            return Err(Error::Msg(
                "LatentKvCache::with_capacity: n_layers must be >= 1".into(),
            ).bt());
        }
        if max_seq_len == 0 {
            return Err(Error::Msg(
                "LatentKvCache::with_capacity: max_seq_len must be >= 1".into(),
            ).bt());
        }
        if slot_trailing.is_empty() {
            return Err(Error::Msg(
                "LatentKvCache::with_capacity: need at least one slot (slot_trailing empty)".into(),
            ).bt());
        }
        let n_slots = slot_trailing.len();
        let shapes: Vec<Shape> = slot_trailing.iter().map(|trailing| {
            let mut dims = Vec::with_capacity(1 + trailing.len());
            dims.push(max_seq_len);
            dims.extend_from_slice(trailing);
            Shape::from_dims(&dims)
        }).collect();
        let layouts: Vec<Layout> = shapes.iter().map(|s| Layout::contiguous(s.clone())).collect();
        let target_loc = device.location();

        // Transient graph, same device-anchor trick + Alloc/ZeroFill
        // emission pattern as KvCache::with_capacity — see that
        // constructor's doc for the full rationale.
        let graph = Arc::new(RwLock::new(Graph::new()));
        let mut cache = StorageCache::new();
        if let Some(seed) = crate::pipelined_bridge::device_seed_storage(device)? {
            let anchor_id = {
                let mut g = graph.write().map_err(|_| {
                    Error::Msg("graph lock poisoned during LatentKvCache build".into()).bt()
                })?;
                g.push(Node {
                    op: Op::Const,
                    inputs: vec![],
                    shape: Shape::from_dims(&[4]),
                    dtype: DType::U8,
                })
            };
            cache.insert(anchor_id, Arc::new(RwLock::new(seed)));
        }

        // Emit (Op::Alloc, Op::ZeroFill) pairs for every (layer, slot),
        // each shaped per its own slot's shape.
        let mut zero_fill_ids: Vec<NodeId> = Vec::with_capacity(n_layers * n_slots);
        {
            let mut g = graph.write().map_err(|_| {
                Error::Msg("graph lock poisoned during LatentKvCache build".into()).bt()
            })?;
            for _ in 0..n_layers {
                for shape in &shapes {
                    let alloc_id = g.push(Node {
                        op: Op::Alloc { target: target_loc },
                        inputs: vec![],
                        shape: shape.clone(),
                        dtype,
                    });
                    let zf_id = g.push(Node {
                        op: Op::ZeroFill,
                        inputs: vec![alloc_id],
                        shape: shape.clone(),
                        dtype,
                    });
                    zero_fill_ids.push(zf_id);
                }
            }
        }

        // Realize all n_layers * n_slots Op::ZeroFill targets in one pass
        // — see KvCache::with_capacity's doc for why a single realize_many
        // call is used instead of one realize per buffer.
        let realized = PipelinedExecutor::realize_many(
            Arc::clone(&graph), &zero_fill_ids, cache,
        )?;
        if realized.len() != n_layers * n_slots {
            return Err(Error::Msg(format!(
                "LatentKvCache::with_capacity: realize_many returned {} storages \
                 for {} Op::ZeroFill targets — internal bug",
                realized.len(), n_layers * n_slots,
            )).bt());
        }

        let mut realized_iter = realized.into_iter();
        let mut layers: Vec<Option<Vec<LatentSlot>>> = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let mut slots = Vec::with_capacity(n_slots);
            for layout in &layouts {
                let (arc, _) = realized_iter.next().expect("checked above");
                slots.push(LatentSlot {
                    storage: arc,
                    layout: layout.clone(),
                    version: 0,
                    authority: AuthorityState::Host,
                });
            }
            layers.push(Some(slots));
        }

        Ok(Self {
            layers,
            cached_len: 0,
            slot_trailing,
            max_seq_len,
            dtype,
        })
    }

    /// Borrow the slot `s` storage Arc for layer `layer_idx`. `None` if
    /// the layer is unpopulated or either index is out of range. Used by
    /// the forward path to bind cache storage to per-step Const nodes via
    /// [`InferenceContext::insert`]. Mirrors [`KvCache::slot_storage`].
    pub fn slot_storage(&self, layer_idx: usize, slot: usize) -> Option<Arc<RwLock<Storage>>> {
        let layer = self.layers.get(layer_idx)?.as_ref()?;
        layer.get(slot).map(|s| Arc::clone(&s.storage))
    }

    /// Bump slot `s`'s monotonic version counter for `layer_idx`. No-op
    /// (never panics) if either index is out of range. Mirrors
    /// [`KvCache::bump_version`].
    pub fn bump_version(&mut self, layer_idx: usize, slot: usize) {
        if let Some(Some(layer)) = self.layers.get_mut(layer_idx) {
            if let Some(s) = layer.get_mut(slot) {
                s.version += 1;
            }
        }
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Number of slots per layer.
    pub fn n_slots(&self) -> usize {
        self.slot_trailing.len()
    }

    /// Trailing shape of slot `s` (past the leading capacity axis).
    pub fn slot_trailing(&self, s: usize) -> &[usize] {
        &self.slot_trailing[s]
    }

    /// Drop every layer; reset `cached_len` to zero. Mirrors [`KvCache::clear`].
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            *layer = None;
        }
        self.cached_len = 0;
    }

    /// Shrink the cache to `new_len` cached positions — metadata-only,
    /// same Phase E.3.0 limitation and rationale as [`KvCache::
    /// truncate_to`]: the underlying storages are not shrunk; the
    /// pre-allocated buffers' trailing rows simply stop being read until
    /// overwritten by a later `Op::WriteSlice` at the same positions.
    pub fn truncate_to(&mut self, new_len: usize) {
        if new_len < self.cached_len {
            self.cached_len = new_len;
        }
    }
}

// ===========================================================================
// DecodeSession (Phase D · D2b — plan-once persistent decode)
// ===========================================================================

/// Per-token re-bound data-Const Arcs for a persistent decode step
/// (Phase D · D2b). Each is a device-resident `fuel_memory::Storage`
/// built from freshly recomputed host bytes (token-ids = the new token,
/// RoPE cos+sin at `position = cached_len`, mask with the shifted `-inf`
/// boundary). [`DecodeSession::realize_token`] overwrites the held
/// `base_cache`'s stable data-Const entries with these for the token.
pub struct DecodeTokenData {
    pub token_ids: Arc<RwLock<Storage>>,
    pub rope_cos: Arc<RwLock<Storage>>,
    pub rope_sin: Arc<RwLock<Storage>>,
    pub mask: Arc<RwLock<Storage>>,
    /// The device-resident KV-write offset (`cached_len` as a rank-0
    /// `I64`), present ONLY on the device-offset decode path (CUDA/CPU
    /// targets, where `Op::WriteSliceDoff` reads the start from this
    /// buffer at kernel launch — capture-ready). `None` on the SymEnv
    /// path (Vulkan), where the KV write is `Op::WriteSlice` with a
    /// `DynScalar::Sym(cached_len)` offset resolved host-side per token.
    pub offset: Option<Arc<RwLock<Storage>>>,
}

/// Plan-once persistent decode state for one `LlamaModel` + one
/// [`KvCache`] capacity/dtype. Built on the first `seq == 1` decode token;
/// reused (graph + optimize view HELD, only the per-token data Consts +
/// `SymEnv` re-bound) for every subsequent token.
///
/// The ~1.8×/token win lives here: the decode-step graph is built +
/// `prepare`d (D2H `Op::Copy` spliced at the root) + `optimize_graph`'d
/// (placement stamps + residency `Op::Copy` + layout `Op::Contiguize`
/// baked in place) ONCE, then re-realized via the D2a prebuilt seam
/// ([`InferenceContext::realize_prebuilt_as_with_env`]) which SKIPS both
/// `prepare` and `optimize_graph`.
///
/// Held by the generate-loop owner (the caller of
/// `forward_with_kv_context_persistent`), NOT by the immutable
/// `LlamaModel` (the model is read-only weights; a session is per-
/// generation state). Constructed lazily as `Option<DecodeSession>` and
/// dropped-then-rebuilt on any validity-key mismatch / `TopologyChanged`
/// / a non-`seq==1` step (see `forward_with_kv_context_persistent`).
///
/// ## Why the graph is HELD here
///
/// D1 rebuilds a fresh `LazyTensor` graph every token and drops it after
/// realize. D2 keeps the `Arc<RwLock<Graph>>` alive on the session so the
/// already-optimized structure (the stamps + inserted copies) survives.
/// The cached [`OptimizedGraph`] holds only `{roots, generation}` — it
/// bakes NO Const data / storage / `SymEnv`, so reusing it is sound
/// **iff the graph structure + topology generation are unchanged** (D1's
/// input-independent-graph guarantee).
pub struct DecodeSession {
    /// The held decode-step graph, ALREADY optimized in place (stamps +
    /// residency/layout copies + D2H root splice baked in). Structure is
    /// stable across tokens (D1 guarantee).
    graph: Arc<RwLock<Graph>>,
    /// The cached optimize view from the first realize. Holds only
    /// `{roots, generation}`; valid while `graph` structure + topology
    /// generation are unchanged.
    optimized: OptimizedGraph,
    /// The realize root the executor was asked for — the D2H `Op::Copy`
    /// NodeId `prepare` spliced (NOT the logits node itself). Stable
    /// across tokens.
    effective_target: NodeId,
    /// The logits node (pre-D2H-splice) — `effective_target`'s input.
    /// Retained for D3 attribution / debugging; the executor is asked
    /// for `effective_target`.
    #[allow(dead_code)]
    logits_node: NodeId,
    /// Stable re-bindable token-ids Const (`[seq]` U32).
    token_ids_node: NodeId,
    /// Stable re-bindable RoPE cos table Const (`[seq, head_dim]` F32).
    rope_cos_node: NodeId,
    /// Stable re-bindable RoPE sin table Const (`[seq, head_dim]` F32).
    rope_sin_node: NodeId,
    /// Stable re-bindable causal mask Const (`[1, 1, seq, max_seq_len]`
    /// F32) — hoisted to ONE shared Const (was per-layer in D1).
    mask_node: NodeId,
    /// Per-layer `(k_const, v_const)` stable KV placeholder NodeIds. The
    /// KV Arcs are re-bound once at build time and mutated in place by
    /// `Op::WriteSlice`/`Op::WriteSliceDoff` each token (never re-inserted
    /// per token).
    kv_nodes: Vec<(NodeId, NodeId)>,
    /// The stable rank-0 `I64` KV-write offset Const, present ONLY on the
    /// **device-offset** decode path (CUDA/CPU — `Op::WriteSliceDoff`
    /// reads `cached_len` from this buffer device-side at launch, making
    /// the decode step CUDA-graph-capturable). `None` on the SymEnv path
    /// (Vulkan — the offset is host-resolved via `cached_len_sym`). When
    /// `Some`, `realize_token` re-binds the per-token offset Arc into the
    /// base-cache clone alongside token-ids/RoPE/mask.
    offset_node: Option<NodeId>,
    /// The symbol the per-pass `SymEnv` binds to `cached_len` each token.
    cached_len_sym: SymId,
    /// The symbol the per-pass `SymEnv` binds to the live **attended
    /// prefix length** (`cached_len + seq`) each token. This is the
    /// `k_len` the optimizer-emitted CUDA flash-decode arm resolves
    /// against (`decode_flash::DecodeFlashSpec::k_len`); distinct from
    /// `cached_len_sym` because the flash kernel attends `[0, k_len)`
    /// (the whole prefix including this token), while the KV-write lands
    /// at `[cached_len, cached_len + seq)`. Unreferenced on today's f32
    /// decode graph (no flash arm offered) — a harmless extra binding.
    attended_len_sym: SymId,
    /// The full realized [`StorageCache`] from the first realize — every
    /// reachable `Op::Const` (weights + the KV Arcs + the initial data
    /// Consts) that `build_const_cache` uploaded. Held because the
    /// prebuilt-realize path SKIPS the const-cache walk: each token we
    /// clone this and overwrite ONLY the per-token data-Const entries
    /// (token-ids / RoPE / mask). The weight Arcs + KV Arcs are stable
    /// (KV mutates in place via `Op::WriteSlice`).
    base_cache: StorageCache,
    /// The `seq` the held graph is shape-keyed to (always 1 for D2 decode).
    seq: usize,
    // ---- validity keys — rebuild if any change vs. the live cache/model ----
    max_seq_len: usize,
    n_layers: usize,
    cache_dtype: DType,
}

impl DecodeSession {
    /// Assemble a held session from the artifacts a first-realize
    /// prebuild produced. `graph` is the (already prepared + optimized)
    /// held decode graph; `effective_target` / `optimized` come from
    /// [`InferenceContext::prebuild_optimized_as_with_env`]; the node ids
    /// are the STABLE data-Const NodeIds the builder minted.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        graph: Arc<RwLock<Graph>>,
        optimized: OptimizedGraph,
        effective_target: NodeId,
        logits_node: NodeId,
        token_ids_node: NodeId,
        rope_cos_node: NodeId,
        rope_sin_node: NodeId,
        mask_node: NodeId,
        kv_nodes: Vec<(NodeId, NodeId)>,
        offset_node: Option<NodeId>,
        cached_len_sym: SymId,
        attended_len_sym: SymId,
        base_cache: StorageCache,
        seq: usize,
        max_seq_len: usize,
        n_layers: usize,
        cache_dtype: DType,
    ) -> Self {
        Self {
            graph,
            optimized,
            effective_target,
            logits_node,
            token_ids_node,
            rope_cos_node,
            rope_sin_node,
            mask_node,
            kv_nodes,
            offset_node,
            cached_len_sym,
            attended_len_sym,
            base_cache,
            seq,
            max_seq_len,
            n_layers,
            cache_dtype,
        }
    }

    /// Realize the held graph for one decode token via the D2a prebuilt
    /// seam (SKIP `prepare` + `optimize_graph`). `data` supplies the
    /// per-token re-bound data-Const Arcs (token-ids / RoPE cos+sin /
    /// mask); they overwrite the held `base_cache`'s entries for this
    /// token (the weight + KV Arcs stay from `base_cache`). `sym_env`
    /// binds `cached_len`. `TopologyChanged` surfaces typed (caller
    /// invalidates the session).
    pub fn realize_token(
        &self,
        device: &Device,
        data: DecodeTokenData,
        sym_env: &SymEnv,
    ) -> Result<Vec<f32>> {
        let mut cache = self.base_cache.clone();
        cache.insert(self.token_ids_node, data.token_ids);
        cache.insert(self.rope_cos_node, data.rope_cos);
        cache.insert(self.rope_sin_node, data.rope_sin);
        cache.insert(self.mask_node, data.mask);
        // Device-offset path (CUDA/CPU): re-bind the per-token KV-write
        // offset (`cached_len` as I64) into the base-cache clone so the
        // held `Op::WriteSliceDoff` nodes read the live position from a
        // fixed device buffer. On the SymEnv path both are `None` (the
        // offset rides `cached_len_sym` in `sym_env` instead).
        if let (Some(offset_node), Some(offset)) = (self.offset_node, data.offset) {
            cache.insert(offset_node, offset);
        }
        crate::pipelined_bridge::realize_one_prebuilt_env::<f32>(
            &self.graph,
            self.effective_target,
            &self.optimized,
            device,
            cache,
            sym_env,
        )
    }

    /// Whether this held session is valid for the given decode step.
    /// Rebuild (drop + build fresh) on any mismatch. `seq` must be 1
    /// (the held graph is the seq==1 decode graph); a change in
    /// `max_seq_len` / `n_layers` / `cache_dtype` means a different
    /// model/cache → the held graph's shapes are stale.
    pub fn is_valid_for(
        &self,
        seq: usize,
        max_seq_len: usize,
        n_layers: usize,
        cache_dtype: DType,
    ) -> bool {
        self.seq == seq
            && self.max_seq_len == max_seq_len
            && self.n_layers == n_layers
            && self.cache_dtype == cache_dtype
    }

    /// The held graph handle (the caller re-binds data Consts + realizes
    /// through it).
    pub fn graph(&self) -> &Arc<RwLock<Graph>> {
        &self.graph
    }

    /// The cached optimize view (fed to the prebuilt-realize seam).
    pub fn optimized(&self) -> &OptimizedGraph {
        &self.optimized
    }

    /// The D2H `Op::Copy` root the executor is asked for.
    pub fn effective_target(&self) -> NodeId {
        self.effective_target
    }

    pub fn token_ids_node(&self) -> NodeId { self.token_ids_node }
    pub fn rope_cos_node(&self) -> NodeId { self.rope_cos_node }
    pub fn rope_sin_node(&self) -> NodeId { self.rope_sin_node }
    pub fn mask_node(&self) -> NodeId { self.mask_node }
    pub fn kv_nodes(&self) -> &[(NodeId, NodeId)] { &self.kv_nodes }
    /// The device-resident KV-write offset Const NodeId, `Some` only on
    /// the device-offset decode path (CUDA/CPU); `None` on Vulkan's
    /// SymEnv path. Used by the CapturedRun replay wiring (Phase 3) and
    /// the per-token offset re-bind.
    pub fn offset_node(&self) -> Option<NodeId> { self.offset_node }
    pub fn cached_len_sym(&self) -> SymId { self.cached_len_sym }
    pub fn attended_len_sym(&self) -> SymId { self.attended_len_sym }
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }

    /// Build the per-token [`SymEnv`] for one decode step: bind
    /// `cached_len_sym = cached_len` (the KV-write offset) AND
    /// `attended_len_sym = cached_len + seq` (the flash-arm `k_len` — the
    /// live attended prefix including this token). Both are bound each
    /// token; the attended-length binding is unreferenced on today's f32
    /// decode graph (no flash arm) and becomes load-bearing the moment a
    /// bf16/f16 CUDA decode offers the arm. Write-once per pass (a
    /// conflicting rebind surfaces a typed error, never a panic).
    pub fn per_token_sym_env(&self, cached_len: usize) -> Result<SymEnv> {
        let mut env = SymEnv::new();
        env.bind(self.cached_len_sym, cached_len)?;
        env.bind(self.attended_len_sym, cached_len + self.seq)?;
        Ok(env)
    }

    /// The held graph's node count. The D2b born-red test asserts this
    /// is stable from token 2 onward (no per-token node growth — the
    /// guard against an accidental re-splice / re-insert / a builder
    /// sneaking a `cached_len`-dependent shape back in).
    pub fn graph_node_count(&self) -> usize {
        self.graph.read().map(|g| g.len()).unwrap_or(0)
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
        self.realize_one_as_with_env::<T>(graph, target, &SymEnv::default())
    }

    /// Env-carrying counterpart of [`Self::realize_one_as`]: supplies a
    /// per-pass [`SymEnv`] binding the runtime values of any `DynScalar`
    /// op params (Phase D symbolic extents — e.g. the decode KV-cache
    /// write offset `cached_len`). The env is **per-pass** (re-supplied
    /// every forward step) while the persistent map is **per-session**;
    /// an empty env is byte-identical to [`Self::realize_one_as`].
    pub fn realize_one_as_with_env<T: bytemuck::Pod>(
        &self,
        graph: &Arc<RwLock<Graph>>,
        target: NodeId,
        env: &SymEnv,
    ) -> Result<Vec<T>> {
        crate::pipelined_bridge::realize_one_as_with_initial_env::<T>(
            graph,
            target,
            &self.device,
            self.cloned_persistent(),
            env,
        )
    }

    /// Phase D · D2a — first-realize that ALSO captures the reusable
    /// optimize artifacts for a later prebuilt (plan-once) re-realize.
    ///
    /// Runs the full `prepare` + `optimize_graph` + dispatch path ONCE (like
    /// [`Self::realize_one_as_with_env`]) and returns
    /// `(effective_target, OptimizedGraph, result)`. The caller (D2b's
    /// `DecodeSession`) holds the `effective_target` (the spliced D2H
    /// `Op::Copy` root) + the cached `OptimizedGraph` and feeds them to
    /// [`Self::realize_prebuilt_as_with_env`] on later tokens to SKIP the
    /// re-plan. See [`crate::pipelined_bridge::prebuild_optimized_env`].
    pub fn prebuild_optimized_as_with_env<T: bytemuck::Pod>(
        &self,
        graph: &Arc<RwLock<Graph>>,
        target: NodeId,
        env: &SymEnv,
    ) -> Result<(NodeId, fuel_dispatch::optimize::OptimizedGraph, Vec<T>)> {
        crate::pipelined_bridge::prebuild_optimized_env::<T>(
            graph,
            target,
            &self.device,
            self.cloned_persistent(),
            env,
        )
    }

    /// Phase D · D2b — first-realize that captures the reusable optimize
    /// artifacts AND the full realized [`StorageCache`] (all weight
    /// Consts uploaded by `build_const_cache`, merged over this context's
    /// persistent Arcs). The [`DecodeSession`] holds the cache so later
    /// prebuilt realizes (which SKIP the const-cache walk) still resolve
    /// every weight Const. Returns
    /// `(effective_target, OptimizedGraph, full_cache, result)`.
    pub fn prebuild_optimized_capturing_as_with_env<T: bytemuck::Pod>(
        &self,
        graph: &Arc<RwLock<Graph>>,
        target: NodeId,
        env: &SymEnv,
    ) -> Result<(NodeId, fuel_dispatch::optimize::OptimizedGraph, StorageCache, Vec<T>)> {
        crate::pipelined_bridge::prebuild_optimized_env_capturing_cache::<T>(
            graph,
            target,
            &self.device,
            self.cloned_persistent(),
            env,
        )
    }

    /// Phase D · D2a — plan-once re-realize over a graph already prepared +
    /// optimized by a prior [`Self::prebuild_optimized_as_with_env`]. Skips
    /// BOTH `prepare` (no D2H re-splice / const-cache walk) AND
    /// `optimize_graph` (no re-plan / double-insert). Re-binds only the
    /// per-call `StorageCache` (this context's persistent Arcs, incl. the
    /// re-bound per-token data Consts) + `env`, then dispatches straight to
    /// the executor. `effective_target` + `optimized` come from the prebuild.
    ///
    /// A `TopologyChanged` error surfaces to the caller (typed, not
    /// retried) — the cached view is stale; invalidate + rebuild the session.
    pub fn realize_prebuilt_as_with_env<T: bytemuck::Pod>(
        &self,
        graph: &Arc<RwLock<Graph>>,
        effective_target: NodeId,
        optimized: &fuel_dispatch::optimize::OptimizedGraph,
        env: &SymEnv,
    ) -> Result<Vec<T>> {
        crate::pipelined_bridge::realize_one_prebuilt_env::<T>(
            graph,
            effective_target,
            optimized,
            &self.device,
            self.cloned_persistent(),
            env,
        )
    }

    /// Multi-target counterpart of [`realize_one_as`].
    pub fn realize_many_as<T: bytemuck::Pod>(
        &self,
        graph: &Arc<RwLock<Graph>>,
        targets: &[NodeId],
    ) -> Result<Vec<Vec<T>>> {
        self.realize_many_as_with_env::<T>(graph, targets, &SymEnv::default())
    }

    /// Env-carrying counterpart of [`Self::realize_many_as`] (Phase D
    /// symbolic extents). An empty env is byte-identical to
    /// [`Self::realize_many_as`].
    pub fn realize_many_as_with_env<T: bytemuck::Pod>(
        &self,
        graph: &Arc<RwLock<Graph>>,
        targets: &[NodeId],
        env: &SymEnv,
    ) -> Result<Vec<Vec<T>>> {
        crate::pipelined_bridge::realize_many_as_with_initial_env::<T>(
            graph,
            targets,
            &self.device,
            self.cloned_persistent(),
            env,
        )
    }

    /// Realize-split counterpart of [`Self::realize_many_as`]: the
    /// first `n_host` targets are downloaded to host `Vec<T>`s, the
    /// rest come back as device-resident `(storage, layout)` pairs —
    /// no D2H for results that feed the next step's graph. See
    /// [`crate::pipelined_bridge::realize_split_as_with_initial`].
    pub fn realize_split_as<T: bytemuck::Pod>(
        &self,
        graph: &Arc<RwLock<Graph>>,
        targets: &[NodeId],
        n_host: usize,
    ) -> Result<(Vec<Vec<T>>, Vec<(Arc<RwLock<Storage>>, Layout)>)> {
        crate::pipelined_bridge::realize_split_as_with_initial::<T>(
            graph,
            targets,
            n_host,
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
    use fuel_ir::{DType, Shape};
    use fuel_graph::{Node, Op};

    /// `InferenceContext::insert` + immediate retrieval round-trips.
    #[test]
    fn context_insert_retrieve() {
        let mut ctx = InferenceContext::new(Device::cpu());
        let storage = fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0]);
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
        let lhs_arc = Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[1.0_f32, 2.0, 3.0])));
        let rhs_arc = Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[10.0_f32, 20.0, 30.0])));

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

    /// Phase D symbolic extents — the decode-shaped path: a persistent
    /// fixed-capacity buffer + an `Op::WriteSlice` whose start offset is
    /// a per-pass `SymEnv` binding (the KV-cache append at `cached_len`).
    /// `realize_one_as_with_env` threads the env from the session through
    /// the bridge to the executor; the width-2 slab lands at the bound
    /// offset (3), not the static placeholder (0). Re-realizing the SAME
    /// graph with a different binding lands the slab at the new offset —
    /// the input-independent-graph property persistent decode relies on.
    #[test]
    fn realize_one_as_with_env_resolves_write_slice_offset() {
        use fuel_ir::{DynScalar, SymId};
        let dest_arc = Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[0.0_f32; 6])));
        let src_arc = Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[7.0_f32, 8.0])));
        let graph = Arc::new(RwLock::new(Graph::new()));
        let sym = SymId(0);
        let (dest_id, src_id, ws_id) = {
            let mut g = graph.write().unwrap();
            let dest = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[6]), dtype: DType::F32,
            });
            let src = g.push(Node {
                op: Op::Const, inputs: vec![],
                shape: Shape::from_dims(&[2]), dtype: DType::F32,
            });
            let ws = g.push(Node {
                op: Op::WriteSlice {
                    ranges: vec![(0, 2)],
                    dyn_offset: Some((0, DynScalar::Sym(sym))),
                },
                inputs: vec![dest, src],
                shape: Shape::from_dims(&[6]),
                dtype: DType::F32,
            });
            (dest, src, ws)
        };
        let mut ctx = InferenceContext::new(Device::cpu());
        ctx.insert(dest_id, Arc::clone(&dest_arc));
        ctx.insert(src_id, Arc::clone(&src_arc));

        // Bind cached_len = 3: the slab must append at indices [3, 4].
        let mut env = SymEnv::new();
        env.bind(sym, 3).unwrap();
        let out = ctx
            .realize_one_as_with_env::<f32>(&graph, ws_id, &env)
            .expect("realize_one_as_with_env");
        assert_eq!(
            out, vec![0.0, 0.0, 0.0, 7.0, 8.0, 0.0],
            "KV slab must land at the SymEnv-bound cached_len=3, not the placeholder 0",
        );

        // An empty env on a graph with an unbound dyn_offset surfaces a
        // typed error (never a panic) — the write-once contract's
        // "presence ⇒ produced" read at realize.
        let err = ctx.realize_one_as_with_env::<f32>(&graph, ws_id, &SymEnv::new());
        assert!(err.is_err(), "unbound cached_len must surface a typed error");
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
        let k_arc = Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[0.0_f32; 32])));
        let v_arc = Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[0.0_f32; 32])));
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
        let k = Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[0.0_f32; 32])));
        let v = Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[0.0_f32; 32])));
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
        let arc = Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[0.0_f32; 4])));
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

    // ---- LatentKvCache (MLA increment 4 — persistent N-slot decode cache) --

    /// `LatentKvCache::with_capacity` allocates `n_layers * n_slots` fresh
    /// zero-initialized buffers on the CPU device, one per (layer, slot),
    /// shaped `[max_seq_len, ...slot_trailing[s]]`.
    #[test]
    fn latent_kv_cache_with_capacity_allocates_all_layers_on_cpu() {
        let device = Device::cpu();
        let cache = LatentKvCache::with_capacity(
            /* n_layers    */ 3,
            /* max_seq_len */ 8,
            // MLA-shaped: slot 0 latent trailing [5], slot 1 k_pe trailing [2].
            vec![vec![5], vec![2]],
            DType::F32,
            &device,
        ).expect("with_capacity");

        assert_eq!(cache.n_layers(), 3);
        assert_eq!(cache.n_slots(), 2);
        assert_eq!(cache.cached_len, 0);
        assert_eq!(cache.max_seq_len, 8);
        assert_eq!(cache.dtype, DType::F32);
        assert_eq!(cache.slot_trailing(0), &[5]);
        assert_eq!(cache.slot_trailing(1), &[2]);

        for li in 0..3 {
            let latent = cache.slot_storage(li, 0).expect("latent slot populated");
            let kpe = cache.slot_storage(li, 1).expect("k_pe slot populated");
            // 8 (max_seq) * 5 (trailing) * 4 (f32) = 160 bytes.
            assert_eq!(latent.read().unwrap().inner.len_bytes(), 160);
            // 8 (max_seq) * 2 (trailing) * 4 (f32) = 64 bytes.
            assert_eq!(kpe.read().unwrap().inner.len_bytes(), 64);
            let guard = latent.read().unwrap();
            if let BackendStorage::Cpu(c) = &guard.inner {
                let typed: &[f32] = c.as_slice().unwrap();
                assert!(typed.iter().all(|&x| x == 0.0));
            } else {
                panic!("expected CPU storage");
            }
        }
    }

    /// `slot_storage` returns `None` for an out-of-range layer or slot
    /// index (never panics).
    #[test]
    fn latent_kv_cache_slot_storage_returns_none_for_oob() {
        let device = Device::cpu();
        let cache = LatentKvCache::with_capacity(2, 4, vec![vec![3]], DType::F32, &device)
            .expect("with_capacity");
        assert!(cache.slot_storage(5, 0).is_none()); // layer OOB
        assert!(cache.slot_storage(0, 5).is_none()); // slot OOB
    }

    /// `bump_version` advances the per-slot version counter independently
    /// per slot; out-of-range (layer, slot) is a no-op, never a panic.
    #[test]
    fn latent_kv_cache_bump_version_advances_per_slot_and_oob_is_noop() {
        let device = Device::cpu();
        let mut cache = LatentKvCache::with_capacity(
            1, 4, vec![vec![2], vec![1]], DType::F32, &device,
        ).expect("with_capacity");

        cache.bump_version(0, 0);
        cache.bump_version(0, 0);
        cache.bump_version(0, 1);
        cache.bump_version(99, 0); // layer OOB, no-op
        cache.bump_version(0, 99); // slot OOB, no-op

        let layer0 = cache.layers[0].as_ref().expect("layer 0 populated");
        assert_eq!(layer0[0].version, 2);
        assert_eq!(layer0[1].version, 1);
    }

    /// Degenerate geometry (`n_layers == 0`, `max_seq_len == 0`, or no
    /// slots) surfaces as a typed `Err`, never a panic.
    #[test]
    fn latent_kv_cache_with_capacity_rejects_bad_geometry() {
        let device = Device::cpu();
        assert!(LatentKvCache::with_capacity(0, 4, vec![vec![2]], DType::F32, &device).is_err());
        assert!(LatentKvCache::with_capacity(1, 0, vec![vec![2]], DType::F32, &device).is_err());
        assert!(LatentKvCache::with_capacity(1, 4, vec![], DType::F32, &device).is_err());
    }
}
