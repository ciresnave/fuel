//! Device-backed KV block pool — the thin integration layer that binds the pure
//! [`KvBlockPool`] host-side allocator to real device `Storage` (multi-session
//! serving, Increment 2, Part 2).
//!
//! The pure core ([`crate::kv_block_pool`]) owns block *metadata* — free list,
//! refcounts, per-session block tables — and deliberately touches no device, no
//! tensors, no model (which is what keeps its eventual `fuel-inference` move
//! cheap). This module is the counterpart that owns the *bytes*:
//!
//! - **Real pool buffers.** `Op::PagedAttn` reads physical K/V caches shaped
//!   `[num_blocks, block_size, Hkv, D]`. A pool holds `n_layers × 2` of them (a
//!   physical block is the same slot in *every* layer's K and V buffer — the
//!   vLLM shared-block-table model), allocated once via the same
//!   `Op::Alloc → Op::ZeroFill → realize_many` path [`crate::inference_context::
//!   KvCache::with_capacity`] uses, so the executor's device-handle reuse and
//!   destructive-fill cleanup come for free.
//! - **`block_table` / `context_lens` materialization.** `Op::PagedAttn`
//!   consumes a `[B, max_blocks]` u32 page table + a `[B]` u32 length vector.
//!   [`DeviceKvPool::materialize_block_table`] projects the core's per-session
//!   block tables into exactly that host layout, ready to upload as `Op::Const`
//!   operands.
//!
//! Byte movement into blocks (`Op::WriteSlice` at a physical block offset) and
//! evict/restore's device↔host copy land in the following sub-increments; this
//! one delivers the buffers + the page-table projection and gates them on a
//! CPU-realizable structural test.

use std::sync::{Arc, RwLock};

use fuel_ir::{DType, Error, Layout, Result, Shape};
use fuel_graph::{Graph, Node, NodeId, Op};
use fuel_dispatch::pipelined::{PipelinedExecutor, StorageCache};
use fuel_memory::Storage;

use crate::Device;
use crate::kv_block_pool::{KvAllocError, KvBlockPool, KvGeometry, PhysBlockId, SessionHandle};

/// A materialized `Op::PagedAttn` page table — the host-side `block_table` +
/// `context_lens` for a batch of `B` sessions, ready to upload as `Op::Const`
/// operands. `block_table` is row-major `[B, max_blocks]`; `context_lens` is
/// `[B]`. Both are `u32` (the op's declared operand dtype).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageTableHost {
    /// Row-major `[B, max_blocks]` logical→physical map. Rows shorter than
    /// `max_blocks` are **zero-padded**: index 0 is always an in-bounds physical
    /// block, and every padded key position is `≥ context_len`, so the op's
    /// variable-length mask (`key_pos ≥ context_len → −inf`) neutralizes it
    /// before softmax. Padding is therefore never *read*, only never *invalid*.
    pub block_table: Vec<u32>,
    /// Per-sequence context length (`filled_tokens`), `[B]`.
    pub context_lens: Vec<u32>,
    /// `B` — number of sessions (rows).
    pub batch: usize,
    /// `max_blocks` — the padded row width (the widest session's block count,
    /// at least 1 so the tensor stays a well-formed rank-2 `[B, max_blocks]`).
    pub max_blocks: usize,
}

impl PageTableHost {
    /// Shape of the `block_table` `Op::Const` — `[B, max_blocks]`.
    pub fn block_table_shape(&self) -> Shape {
        Shape::from_dims(&[self.batch, self.max_blocks])
    }
    /// Shape of the `context_lens` `Op::Const` — `[B]`.
    pub fn context_lens_shape(&self) -> Shape {
        Shape::from_dims(&[self.batch])
    }
}

/// Device-backed KV block pool: the pure [`KvBlockPool`] core plus the physical
/// K/V pool buffers it hands out slots into. Owns the core (so device-aware
/// evict/restore can move bytes and update metadata atomically); the core is
/// reachable via [`core`](Self::core) / [`core_mut`](Self::core_mut) for the
/// pure capacity/lifecycle verbs.
pub struct DeviceKvPool {
    core: KvBlockPool,
    #[allow(dead_code)] // consumed by the WriteSlice / evict-restore sub-increments
    device: Device,
    dtype: DType,
    /// `[num_blocks, block_size, Hkv, D]` — the shape of every layer buffer.
    pool_shape: Shape,
    /// Per-layer K pool buffers (`k_pools[l]` is layer `l`'s `[num_blocks, …]`).
    k_pools: Vec<Arc<RwLock<Storage>>>,
    /// Per-layer V pool buffers.
    v_pools: Vec<Arc<RwLock<Storage>>>,
}

impl DeviceKvPool {
    /// Build a device-backed pool: allocate `n_layers × 2` zero-initialized
    /// `[num_blocks, block_size, Hkv, D]` buffers of `dtype` on `device`, and
    /// wrap a pure [`KvBlockPool`] over the same geometry.
    ///
    /// `geom.elem_size` must equal `dtype.size_in_bytes()` — the geometry's byte
    /// accounting (`kv_bytes_resident`) and the physical buffers must agree, so a
    /// mismatch is a build-time `Err`, never a silently wrong budget signal.
    pub fn new(geom: KvGeometry, dtype: DType, device: &Device) -> Result<Self> {
        if geom.elem_size != dtype.size_in_bytes() {
            return Err(Error::Msg(format!(
                "DeviceKvPool::new: geometry elem_size {} ≠ dtype {:?} size {} — the \
                 pool's byte accounting would disagree with its physical buffers",
                geom.elem_size, dtype, dtype.size_in_bytes(),
            )).bt());
        }
        let n_layers = geom.n_layers;
        let pool_shape =
            Shape::from_dims(&[geom.num_blocks, geom.block_size, geom.n_kv_heads, geom.head_dim]);
        let (k_pools, v_pools) = alloc_layer_buffers(&pool_shape, dtype, device, n_layers)?;
        Ok(Self {
            core: KvBlockPool::new(geom),
            device: device.clone(),
            dtype,
            pool_shape,
            k_pools,
            v_pools,
        })
    }

    // --- pure-core access -------------------------------------------------

    /// The pure metadata core (capacity/lifecycle verbs live here).
    pub fn core(&self) -> &KvBlockPool {
        &self.core
    }
    /// Mutable core — `open`/`append`/`evict`/`splice`/… route through this.
    pub fn core_mut(&mut self) -> &mut KvBlockPool {
        &mut self.core
    }
    /// Pool geometry (delegates to the core).
    pub fn geometry(&self) -> KvGeometry {
        self.core.geometry()
    }

    // --- physical buffers -------------------------------------------------

    /// Number of transformer layers (= number of K buffers = number of V
    /// buffers).
    pub fn n_layers(&self) -> usize {
        self.k_pools.len()
    }
    /// `[num_blocks, block_size, Hkv, D]` — every layer buffer's shape.
    pub fn pool_shape(&self) -> &Shape {
        &self.pool_shape
    }
    /// Pool buffer element dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// Layer `l`'s K pool buffer, if `l < n_layers`.
    pub fn k_pool(&self, layer: usize) -> Option<&Arc<RwLock<Storage>>> {
        self.k_pools.get(layer)
    }
    /// Layer `l`'s V pool buffer, if `l < n_layers`.
    pub fn v_pool(&self, layer: usize) -> Option<&Arc<RwLock<Storage>>> {
        self.v_pools.get(layer)
    }

    // --- page-table materialization (C: Op::PagedAttn operands) -----------

    /// Project the core's per-session block tables into the `Op::PagedAttn`
    /// `[B, max_blocks]` u32 `block_table` + `[B]` u32 `context_lens`, one row
    /// per session in `sessions` order (that order defines the op's batch axis).
    ///
    /// Fails (typed, never a panic) if any session is unknown or has an
    /// externalized slot — a session must be resident to back live attention, so
    /// a mis-sequenced call surfaces as [`KvAllocError`] rather than routing
    /// through a reclaimed block. Shorter rows are zero-padded (see
    /// [`PageTableHost::block_table`]).
    pub fn materialize_block_table(
        &self,
        sessions: &[SessionHandle],
    ) -> std::result::Result<PageTableHost, KvAllocError> {
        let batch = sessions.len();
        let mut rows: Vec<Vec<PhysBlockId>> = Vec::with_capacity(batch);
        let mut context_lens: Vec<u32> = Vec::with_capacity(batch);
        for &s in sessions {
            rows.push(self.core.session_block_table(s)?);
            let filled = self.core.filled_tokens(s).ok_or(KvAllocError::UnknownSession)?;
            context_lens.push(filled as u32);
        }
        // At least 1 column so the tensor is a well-formed rank-2 [B, max_blocks].
        let max_blocks = rows.iter().map(|r| r.len()).max().unwrap_or(0).max(1);
        let mut block_table = vec![0u32; batch * max_blocks];
        for (bi, row) in rows.iter().enumerate() {
            for (i, &p) in row.iter().enumerate() {
                block_table[bi * max_blocks + i] = p;
            }
        }
        Ok(PageTableHost { block_table, context_lens, batch, max_blocks })
    }
}

/// Allocate `n_layers` pairs of zero-initialized `shape` buffers on `device`,
/// returning `(k_pools, v_pools)`. Mirrors [`crate::inference_context::KvCache::
/// with_capacity`]'s `Op::Alloc → Op::ZeroFill → realize_many` path: one Alloc
/// (uninit) + ZeroFill (destructive in-place zero) pair per buffer, all realized
/// in a single pass so device-handle reuse is automatic.
fn alloc_layer_buffers(
    shape: &Shape,
    dtype: DType,
    device: &Device,
    n_layers: usize,
) -> Result<(Vec<Arc<RwLock<Storage>>>, Vec<Arc<RwLock<Storage>>>)> {
    let target_loc = device.location();
    let graph = Arc::new(RwLock::new(Graph::new()));
    let mut cache = StorageCache::new();

    // Non-CPU targets need a device anchor in the cache so the first Op::Alloc's
    // device lookup succeeds (see with_capacity's note). CPU returns None.
    if let Some(seed) = crate::pipelined_bridge::device_seed_storage(device)? {
        let anchor_id = graph
            .write()
            .map_err(|_| Error::Msg("graph lock poisoned during DeviceKvPool build".into()).bt())?
            .push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[4]), dtype: DType::U8 });
        cache.insert(anchor_id, Arc::new(RwLock::new(seed)));
    }

    // Emit 2*n_layers (Alloc, ZeroFill) pairs — K then V per layer.
    let zero_fill_ids: Vec<NodeId> = {
        let mut g = graph
            .write()
            .map_err(|_| Error::Msg("graph lock poisoned during DeviceKvPool build".into()).bt())?;
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

    let realized = PipelinedExecutor::realize_many(Arc::clone(&graph), &zero_fill_ids, cache)?;
    if realized.len() != 2 * n_layers {
        return Err(Error::Msg(format!(
            "DeviceKvPool::new: realize_many returned {} storages for {} Op::ZeroFill \
             targets — internal bug",
            realized.len(), 2 * n_layers,
        )).bt());
    }

    let mut it = realized.into_iter();
    let mut k_pools = Vec::with_capacity(n_layers);
    let mut v_pools = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let (k_arc, _) = it.next().expect("checked len == 2*n_layers");
        let (v_arc, _) = it.next().expect("checked len == 2*n_layers");
        k_pools.push(k_arc);
        v_pools.push(v_arc);
    }
    let _ = Layout::contiguous(shape.clone()); // pool_shape is the layout; kept explicit for clarity
    Ok((k_pools, v_pools))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom(n_layers: usize, num_blocks: usize, block_size: usize) -> KvGeometry {
        KvGeometry {
            n_layers,
            num_blocks,
            block_size,
            n_kv_heads: 2,
            head_dim: 8,
            elem_size: 4, // f32
        }
    }

    /// The pool allocates exactly `n_layers` K buffers + `n_layers` V buffers,
    /// each a real, correctly-sized, zero-initialized `[num_blocks, block_size,
    /// Hkv, D]` device buffer.
    #[test]
    fn allocates_two_pool_buffers_per_layer_with_the_paged_cache_shape() {
        let g = geom(/*n_layers*/ 3, /*num_blocks*/ 8, /*block_size*/ 4);
        let pool = DeviceKvPool::new(g, DType::F32, &Device::cpu()).unwrap();

        assert_eq!(pool.n_layers(), 3, "one K + one V buffer per layer");
        assert_eq!(pool.pool_shape().dims(), &[8, 4, 2, 8], "[num_blocks, block_size, Hkv, D]");
        let want_elems = 8 * 4 * 2 * 8;
        for l in 0..3 {
            for (name, buf) in [("k", pool.k_pool(l)), ("v", pool.v_pool(l))] {
                let s = buf.unwrap_or_else(|| panic!("{name}_pool({l}) exists")).read().unwrap();
                assert_eq!(s.dtype(), DType::F32, "{name}[{l}] dtype");
                assert_eq!(s.elem_count(), want_elems, "{name}[{l}] element count");
            }
        }
        assert!(pool.k_pool(3).is_none(), "no layer 3 in a 3-layer pool");
    }

    /// A geometry whose `elem_size` disagrees with the requested dtype is a
    /// build-time typed error — the byte-accounting/buffer mismatch never ships.
    #[test]
    fn dtype_vs_elem_size_mismatch_is_a_build_time_error() {
        let mut g = geom(1, 4, 4);
        g.elem_size = 2; // claims bf16-sized elements…
        let err = DeviceKvPool::new(g, DType::F32, &Device::cpu()); // …but asks for f32 buffers
        assert!(err.is_err(), "elem_size 2 vs f32 (4 bytes) must be rejected");
    }

    /// The page table projects the core's *actual* per-session physical block
    /// ids (a non-identity permutation here) into row-major `[B, max_blocks]`,
    /// zero-pads short rows, and reports each session's `filled_tokens` as its
    /// context length. A stub that emitted an identity `[0,1,2,…]` table would
    /// fail against the permuted layout.
    #[test]
    fn materialize_block_table_projects_the_cores_permuted_layout() {
        let mut pool = DeviceKvPool::new(geom(1, 16, 4), DType::F32, &Device::cpu()).unwrap();

        // Force a NON-identity physical assignment: open a filler session, then
        // a real one, discard the filler, and open more — so the second real
        // session's blocks are not 0,1,2,… in order.
        let filler = pool.core_mut().open();
        pool.core_mut().append(filler, 8).unwrap(); // grabs physical 0,1
        let sa = pool.core_mut().open();
        pool.core_mut().append(sa, 12).unwrap(); // grabs physical 2,3,4 (3 blocks, 12 tokens)
        pool.core_mut().discard(filler); // returns 0,1 to the free list
        let sb = pool.core_mut().open();
        pool.core_mut().append(sb, 6).unwrap(); // grabs freed blocks (6 tokens → 2 blocks)

        // Ground truth straight from the core.
        let a_blocks = pool.core().session_block_table(sa).unwrap();
        let b_blocks = pool.core().session_block_table(sb).unwrap();
        assert_eq!(a_blocks.len(), 3);
        assert_eq!(b_blocks.len(), 2);

        let pt = pool.materialize_block_table(&[sa, sb]).unwrap();
        assert_eq!(pt.batch, 2);
        assert_eq!(pt.max_blocks, 3, "widest session (sa) sets the row width");
        assert_eq!(pt.context_lens, vec![12u32, 6u32], "context_len = filled_tokens per session");

        // Row 0 = sa's three physical ids; row 1 = sb's two ids then zero pad.
        let row_a = &pt.block_table[0..3];
        let row_b = &pt.block_table[3..6];
        assert_eq!(row_a, &[a_blocks[0], a_blocks[1], a_blocks[2]], "sa row = its real physical ids");
        assert_eq!(row_b[0..2], [b_blocks[0], b_blocks[1]], "sb row = its real physical ids");
        assert_eq!(row_b[2], 0, "sb's short row is zero-padded");
        assert_eq!(pt.block_table_shape().dims(), &[2, 3]);
        assert_eq!(pt.context_lens_shape().dims(), &[2]);
    }

    /// Materializing over an unknown or externalized session is a typed error
    /// (delegates the core's resident-only guard), never a panic.
    #[test]
    fn materialize_rejects_a_non_resident_session() {
        let mut pool = DeviceKvPool::new(geom(1, 16, 4), DType::F32, &Device::cpu()).unwrap();
        let s = pool.core_mut().open();
        pool.core_mut().append(s, 9).unwrap();
        pool.core_mut().evict(s).unwrap(); // slots now externalized
        let err = pool.materialize_block_table(&[s]);
        assert_eq!(err, Err(KvAllocError::SessionNotResident { slot: 0 }));
    }
}
