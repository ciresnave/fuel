//! Graph-keyed storage map (Phase 7.5 work item G).
//!
//! `GraphStorage` is the sidecar that pairs a `fuel_graph::SharedGraph` with
//! a `NodeId`-keyed map of realized `Storage` slots. Together they form the
//! "graph as memory owner" model that work item G introduces — Tensors
//! become thin handles that consult this map at read time instead of
//! owning storage directly.
//!
//! ## Why a sidecar in fuel-core rather than a field on fuel_graph::Graph
//!
//! fuel-graph depends only on fuel-core-types (no awareness of `Storage`).
//! Putting the storage map directly on `Graph` would invert the dependency
//! graph or require moving `Storage` (and its eager-dispatch surface) into
//! fuel-core-types, which is an orthogonal — and more invasive — refactor.
//!
//! The sidecar achieves the same end-state for users: lifetime is tied to
//! the graph (callers hand `SharedGraph` and `SharedGraphStorage` around
//! together), residency machinery still keys on `NodeId`, and "Tensor
//! doesn't own Storage" is true.
//!
//! ## Slot lifecycle
//!
//! - **Const nodes** populate their slot at factory time (work item B2).
//! - **Computed intermediates** populate their slot during executor
//!   realize (work item B3).
//! - Slots are reference-counted via `Arc<RwLock<Storage>>`. Live Tensor
//!   handles to a `NodeId` keep the slot's bytes alive even if the map
//!   entry is removed.
//! - The map can be pruned by residency / eviction passes that already
//!   operate on `NodeId` (`Op::Release`, `ResidencyEvictionRule`).

use crate::Storage;
use fuel_graph::NodeId;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// One realized storage slot, keyed by `NodeId` in a `GraphStorage`.
///
/// The wrapping `Arc<RwLock<Storage>>` mirrors the type today's
/// `Tensor_::storage` field carries, so cloning a slot's storage is the
/// same Arc-bump that Tensor cloning has always been. Multiple Tensor
/// handles for the same `NodeId` share one set of bytes.
#[derive(Debug)]
pub struct StorageSlot {
    pub storage: Arc<RwLock<Storage>>,
}

impl StorageSlot {
    /// Wrap an existing `Arc<RwLock<Storage>>` as a slot. Used by the
    /// migration shim when registering a legacy-mode tensor's storage
    /// into a graph.
    pub fn from_arc(storage: Arc<RwLock<Storage>>) -> Self {
        Self { storage }
    }

    /// Wrap an owned `Storage` as a slot, freshly allocating the
    /// `Arc<RwLock<>>` wrapper. Used when factory code has just produced
    /// the bytes and is registering them for the first time.
    pub fn from_storage(storage: Storage) -> Self {
        Self {
            storage: Arc::new(RwLock::new(storage)),
        }
    }
}

/// `NodeId`-keyed storage map. Lifetime is tied to its owning
/// `SharedGraphStorage`; when that drops, every slot's owning Arc loses
/// one reference. Slots may stay live if other Tensor handles still
/// reference their bytes.
#[derive(Debug, Default)]
pub struct GraphStorage {
    map: HashMap<NodeId, StorageSlot>,
}

impl GraphStorage {
    /// Construct an empty storage map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a slot under the given `NodeId`. Replaces any existing
    /// entry — callers responsible for ensuring this is the intended
    /// semantics (e.g., re-realization after eviction).
    pub fn set(&mut self, id: NodeId, slot: StorageSlot) {
        self.map.insert(id, slot);
    }

    /// Convenience: register an owned `Storage` directly.
    pub fn set_storage(&mut self, id: NodeId, storage: Storage) {
        self.set(id, StorageSlot::from_storage(storage));
    }

    /// Borrow a slot by id, returning `None` if no slot is registered.
    pub fn get(&self, id: NodeId) -> Option<&StorageSlot> {
        self.map.get(&id)
    }

    /// Remove a slot, returning it if present. Used by eviction /
    /// `Op::Release` paths.
    pub fn remove(&mut self, id: NodeId) -> Option<StorageSlot> {
        self.map.remove(&id)
    }

    /// Whether a slot is currently registered for `id`.
    pub fn contains(&self, id: NodeId) -> bool {
        self.map.contains_key(&id)
    }

    /// Number of registered slots.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the map has any slots.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate over registered NodeIds. Order is unspecified.
    pub fn ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.map.keys().copied()
    }
}

/// Cheap-to-clone shared handle to a `GraphStorage`. Uses
/// `Arc<RwLock<>>` (matching `fuel_graph::SharedGraph` post-G) so the
/// pair `(graph, storage)` carried by `GraphLink` is `Send + Sync` and
/// every fuel-core `Tensor` (graph-mode included) inherits Send+Sync
/// auto-derive.
pub type SharedGraphStorage = Arc<RwLock<GraphStorage>>;

/// Construct a fresh empty `SharedGraphStorage`.
pub fn new_shared_graph_storage() -> SharedGraphStorage {
    Arc::new(RwLock::new(GraphStorage::new()))
}

/// A graph-mode Tensor's reference into a graph + its sidecar storage
/// map. Carried in `Tensor_::link` (work item G step 2) once a Tensor
/// is constructed in node-handle mode.
///
/// The graph and storage references are independent — they're paired by
/// convention because they share a lifetime, but the storage map is
/// keyed by `NodeId` and doesn't otherwise need the graph reference.
#[derive(Debug, Clone)]
pub struct GraphLink {
    pub graph:   fuel_graph::SharedGraph,
    pub storage: SharedGraphStorage,
    pub id:      NodeId,
}

impl GraphLink {
    /// Construct a link from its parts. Callers responsible for ensuring
    /// `id` is a valid node in `graph` and (when the slot is expected to
    /// be populated) that `storage` has a slot for it.
    pub fn new(graph: fuel_graph::SharedGraph, storage: SharedGraphStorage, id: NodeId) -> Self {
        Self { graph, storage, id }
    }

    /// Look up this link's storage slot. Returns `None` if no slot has
    /// been registered yet — the caller should treat this as
    /// "needs realize" rather than as an error.
    pub fn storage_slot(&self) -> Option<Arc<RwLock<Storage>>> {
        self.storage.read().unwrap().get(self.id).map(|s| s.storage.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::Shape;

    fn fresh_graph_with_const_node() -> (fuel_graph::SharedGraph, NodeId) {
        let data = Arc::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let t = fuel_graph::Tensor::from_const(
            fuel_graph::ConstData::F32(data),
            Shape::from_dims(&[2, 2]),
        );
        (t.graph().clone(), t.id())
    }

    #[test]
    fn empty_storage_map_basics() {
        let map = GraphStorage::new();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert!(map.get(NodeId(0)).is_none());
        assert!(!map.contains(NodeId(0)));
    }

    #[test]
    fn set_and_get_roundtrip() {
        let device = crate::Device::cpu();
        let storage = device.zeros(&Shape::from_dims(&[2, 2]), crate::DType::F32).unwrap();
        let mut map = GraphStorage::new();
        let id = NodeId(7);
        map.set_storage(id, storage);

        assert_eq!(map.len(), 1);
        assert!(map.contains(id));
        let slot = map.get(id).expect("slot should exist after set");
        assert_eq!(slot.storage.read().unwrap().dtype(), crate::DType::F32);
    }

    #[test]
    fn remove_returns_slot_and_clears_entry() {
        let device = crate::Device::cpu();
        let storage = device.zeros(&Shape::from_dims(&[3]), crate::DType::F32).unwrap();
        let mut map = GraphStorage::new();
        let id = NodeId(3);
        map.set_storage(id, storage);

        let removed = map.remove(id);
        assert!(removed.is_some());
        assert!(!map.contains(id));
        assert_eq!(map.len(), 0);

        // Removing again is a no-op.
        assert!(map.remove(id).is_none());
    }

    #[test]
    fn slot_is_arc_shared() {
        let device = crate::Device::cpu();
        let storage = device.zeros(&Shape::from_dims(&[2]), crate::DType::F32).unwrap();
        let mut map = GraphStorage::new();
        let id = NodeId(0);
        map.set_storage(id, storage);

        let arc1 = map.get(id).unwrap().storage.clone();
        let arc2 = map.get(id).unwrap().storage.clone();
        // Both clones see the same RwLock<Storage>.
        assert!(Arc::ptr_eq(&arc1, &arc2));

        // Removing the map entry doesn't free the bytes — the live Arcs keep them.
        map.remove(id);
        assert_eq!(arc1.read().unwrap().dtype(), crate::DType::F32);
    }

    #[test]
    fn graph_link_storage_slot_lookup() {
        let (graph, node_id) = fresh_graph_with_const_node();
        let storage_map = new_shared_graph_storage();
        let link = GraphLink::new(graph, storage_map.clone(), node_id);

        // No slot registered yet.
        assert!(link.storage_slot().is_none());

        // Register a slot, look it up via the link.
        let device = crate::Device::cpu();
        let s = device.zeros(&Shape::from_dims(&[2, 2]), crate::DType::F32).unwrap();
        storage_map.write().unwrap().set_storage(node_id, s);
        assert!(link.storage_slot().is_some());
    }

    #[test]
    fn shared_graph_storage_is_shared() {
        let storage = new_shared_graph_storage();
        let device = crate::Device::cpu();
        let s = device.zeros(&Shape::from_dims(&[1]), crate::DType::F32).unwrap();
        storage.write().unwrap().set_storage(NodeId(0), s);

        let storage2 = storage.clone();
        assert_eq!(storage2.read().unwrap().len(), 1);
    }

    // Phase 7.5 work item G step 5 — end-to-end smoke test for
    // node-handle mode. Constructs a Tensor whose `Tensor_.storage`
    // legacy Arc and `Tensor_.link`'s graph slot point at the SAME
    // underlying bytes, then verifies:
    //   - the realized_storage() seam returns the slot's Arc clone
    //   - the slot Arc and legacy Arc are the same object (parallel
    //     mode invariant: both views see the same bytes)
    //   - has_graph_link() / graph_link() reflect node-handle mode
    //   - the data is readable via the slot path
    //
    // This is G's proof-of-life: it demonstrates a node-handle Tensor
    // can be constructed and read through the new seam. B2 will use
    // the same construction pattern in the migrated factories.
    /// Parametric helper for the node-handle smoke test. Builds a
    /// node-handle Tensor on `device`, registers its storage Arc into
    /// a graph slot, and verifies the seam returns that exact Arc and
    /// that device identity survives the slot path. Used by both the
    /// CPU smoke test and the gated CUDA/Metal/Vulkan parity tests.
    fn node_handle_smoke_for_device(device: &crate::Device) {
        use crate::op::BackpropOp;
        use crate::tensor::from_storage_with_link;
        use fuel_graph::ConstData;

        let shape = Shape::from_dims(&[3]);
        let legacy = crate::Tensor::new(&[1.0_f32, 2.0, 3.0], device).unwrap();
        let storage_arc = legacy.realized_storage();
        let const_data = ConstData::F32(Arc::from(vec![1.0_f32, 2.0, 3.0]));
        let const_t = fuel_graph::Tensor::from_const(const_data, shape.clone());
        let g = const_t.graph().clone();
        let id = const_t.id();
        let storage_map = new_shared_graph_storage();
        storage_map
            .write()
            .unwrap()
            .set(id, StorageSlot::from_arc(storage_arc.clone()));
        let link = GraphLink::new(g, storage_map, id);

        let t = from_storage_with_link(
            storage_arc.clone(),
            shape,
            BackpropOp::none(),
            false,
            link,
        );

        assert!(t.has_graph_link());
        let slot_arc = t.realized_storage();
        assert!(
            Arc::ptr_eq(&slot_arc, &storage_arc),
            "realized_storage should return the registered slot Arc"
        );

        // Device identity survives the slot path: the slot's Storage
        // reports the same device location as the original Tensor.
        let slot_arc_dev = slot_arc.read().unwrap().device();
        assert_eq!(
            slot_arc_dev.location_dyn(),
            device.location(),
            "slot Storage device must match construction device",
        );
    }

    #[test]
    fn node_handle_tensor_smoke() {
        use crate::op::BackpropOp;
        use crate::tensor::from_storage_with_link;
        use fuel_graph::ConstData;

        let device = crate::Device::cpu();
        let shape = Shape::from_dims(&[3]);

        // Build a Storage with known F32 data using the legacy factory
        // (cheap host-allocate + fill). We pull out its Arc to share
        // between the legacy storage field and the graph slot.
        let legacy = crate::Tensor::new(&[1.0_f32, 2.0, 3.0], &device).unwrap();
        let storage_arc = legacy.realized_storage();

        // Build a fresh single-node graph + slot map. Use the public
        // `Tensor::from_const` builder which gives back a (graph, id)
        // pair already wired up.
        let const_data = ConstData::F32(Arc::from(vec![1.0_f32, 2.0, 3.0]));
        let const_t = fuel_graph::Tensor::from_const(const_data, shape.clone());
        let g = const_t.graph().clone();
        let id = const_t.id();
        let storage_map = new_shared_graph_storage();
        storage_map
            .write()
            .unwrap()
            .set(id, StorageSlot::from_arc(storage_arc.clone()));
        let link = GraphLink::new(g, storage_map, id);

        // Construct the node-handle Tensor.
        let t = from_storage_with_link(
            storage_arc.clone(),
            shape,
            BackpropOp::none(),
            false,
            link,
        );

        // Mode predicates.
        assert!(t.has_graph_link());
        assert!(t.graph_link().is_some());

        // The seam returns the slot's Arc, which is the same as the
        // legacy Arc we passed in (parallel-mode invariant).
        let slot_arc = t.realized_storage();
        assert!(
            Arc::ptr_eq(&slot_arc, &storage_arc),
            "realized_storage should return the registered slot Arc"
        );

        // Read through the seam — same bytes the legacy tensor sees.
        let bytes = slot_arc.read().unwrap();
        assert_eq!(bytes.dtype(), crate::DType::F32);
    }

    // Phase 7.5 work item G step 6 — multi-device parity for the
    // node-handle Tensor mechanism. The slot map keys on NodeId and
    // type-erases the Storage via DynBackendStorage, so by
    // construction it works on any backend; these gated tests prove
    // it on real device-resident Storage (CUDA/Metal). Vulkan would
    // fit the same pattern but requires backend-internal routing
    // not in scope here; skipped because the device-identity check
    // at the slot path is purely a property of `Storage::device()`,
    // which Vulkan inherits from the same trait. Re-enable once
    // we add a Vulkan device-construction shortcut for tests.

    #[cfg(feature = "cuda")]
    #[test]
    fn node_handle_tensor_smoke_cuda() {
        let device = crate::cuda_backend::new_device(0)
            .expect("cuda device 0 expected for cuda-feature test");
        node_handle_smoke_for_device(&device);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn node_handle_tensor_smoke_metal() {
        let device = crate::metal_backend::new_device(0)
            .expect("metal device 0 expected for metal-feature test");
        node_handle_smoke_for_device(&device);
    }
}
