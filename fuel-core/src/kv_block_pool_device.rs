// SPDX-License-Identifier: MIT OR Apache-2.0
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

use fuel_dispatch::pipelined::{PipelinedExecutor, StorageCache};
use fuel_graph::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Error, Layout, Result, Shape};
use fuel_memory::Storage;

use crate::Device;
use crate::decode_shape::KvAllocId;
use crate::kv_block_pool::{
    Externalized, KvAllocError, KvBlockPool, KvGeometry, PhysBlockId, SessionHandle,
};
use crate::lazy::Tensor;

/// Which of a layer's two pool buffers a block operation targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    K,
    V,
}

fn msg_err(s: String) -> crate::Error {
    fuel_ir::Error::Msg(s).bt()
}

fn alloc_err(e: KvAllocError) -> crate::Error {
    msg_err(format!("KV allocator: {e:?}"))
}

/// The device-backed externalization of a session's evicted state (C-3, lossy).
/// Wraps the pure core's structural [`Externalized`] handle with the **bytes**
/// of the detached blocks — one `[block_size · Hkv · D]` f32 vector per
/// externalized logical slot, per layer, for K and V. The core is byte-free by
/// design (move-ready), so the device layer is where the block contents live
/// while a session is suspended. Never carries a recompute instruction (Q9): it
/// is externalized *state*, restorable as-is into fresh physical blocks.
pub struct DeviceEvicted {
    core_handle: Externalized,
    saved: Vec<SavedBlock>,
}

/// The captured contents of one detached (exclusive) logical block across all
/// layers. `k[l]` / `v[l]` are layer `l`'s `[block_size, Hkv, D]` slab as raw
/// pool-dtype bytes (dtype-agnostic — f32 or bf16, via the byte movement core).
struct SavedBlock {
    logical_slot: usize,
    k: Vec<Vec<u8>>,
    v: Vec<Vec<u8>>,
}

impl DeviceEvicted {
    /// The categories of state this handle covers (delegates the core handle's
    /// enumeration — the Q9 completeness gate).
    pub fn covers(&self) -> &[crate::kv_block_pool::StateKind] {
        self.core_handle.covers()
    }
    /// This handle's fidelity guarantee (`Lossy` this increment).
    pub fn fidelity(&self) -> crate::kv_block_pool::Fidelity {
        self.core_handle.fidelity()
    }
    /// Number of blocks whose bytes this handle carries (the exclusive,
    /// actually-detached blocks — shared blocks stay resident and aren't here).
    pub fn saved_block_count(&self) -> usize {
        self.saved.len()
    }
}

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
    device: Device,
    dtype: DType,
    /// `[num_blocks, block_size, Hkv, D]` — the shape of every layer buffer.
    pool_shape: Shape,
    /// Per-layer K pool buffers (`k_pools[l]` is layer `l`'s `[num_blocks, …]`).
    k_pools: Vec<Arc<RwLock<Storage>>>,
    /// Per-layer V pool buffers.
    v_pools: Vec<Arc<RwLock<Storage>>>,
    /// Identity of these pool buffers — the paged twin of
    /// [`KvCache::alloc_id`](crate::inference_context::KvCache::alloc_id).
    /// A held [`PagedDecodeSession`](crate::inference_context::PagedDecodeSession)
    /// bakes these `Arc`s into its `base_cache`, and
    /// `rebind_and_realize_paged_prebuilt` re-binds only the per-token data
    /// (token id, RoPE, block_table, context_lens, offset) — never the pool.
    /// So the held plan reads the pool it was BUILT against no matter which
    /// `&mut DeviceKvPool` the caller passes, and geometry alone cannot tell
    /// two same-shaped pools apart. See [`KvAllocId`].
    alloc_id: KvAllocId,
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
                geom.elem_size,
                dtype,
                dtype.size_in_bytes(),
            ))
            .bt());
        }
        let n_layers = geom.n_layers;
        let pool_shape = Shape::from_dims(&[
            geom.num_blocks,
            geom.block_size,
            geom.n_kv_heads,
            geom.head_dim,
        ]);
        let (k_pools, v_pools) = alloc_layer_buffers(&pool_shape, dtype, device, n_layers)?;
        Ok(Self {
            core: KvBlockPool::new(geom),
            device: device.clone(),
            dtype,
            pool_shape,
            k_pools,
            v_pools,
            alloc_id: KvAllocId::next(),
        })
    }

    /// Identity of the physical pool buffers a held paged plan is welded to.
    /// See [`KvAllocId`].
    ///
    /// There is no re-mint path: `DeviceKvPool` allocates once in [`Self::new`]
    /// and never replaces a layer buffer. Block *allocation* churn lives in
    /// `core` (a `KvBlockPool` index) and reaches the graph through the
    /// per-token `block_table` rebind, so it is correctly NOT part of this
    /// identity — over-keying it would rebuild the plan on every new block.
    pub fn alloc_id(&self) -> KvAllocId {
        self.alloc_id
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
    /// `[num_blocks · block_size, Hkv, D]` — the pool buffer viewed with its
    /// leading (block, slot) axes collapsed into ONE flat slot axis. A physical
    /// `(phys, slot)` then addresses a single row `phys · block_size + slot`, so
    /// the paged decode's new-token K/V write is one dynamic start on axis 0
    /// (`Op::WriteSliceDyn` / `Op::WriteSliceDoff`) instead of a concrete
    /// two-axis `[(phys,phys+1),(slot,slot+1),…]` slab whose ranges change every
    /// step. [`Self::build_decode_attn_off`] binds a placeholder of THIS shape to
    /// the same pool buffer — a pure metadata reshape of [`Self::pool_shape`]
    /// (same element count, row-major order preserved), so the write still
    /// mutates the persistent buffer in place.
    pub fn pool_shape_flat(&self) -> Shape {
        let g = self.geometry();
        Shape::from_dims(&[g.num_blocks * g.block_size, g.n_kv_heads, g.head_dim])
    }
    /// Pool buffer element dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// The device the pool buffers live on — a paged forward builds + realizes on
    /// this device (so a CUDA pool runs on CUDA, not CPU).
    pub fn device(&self) -> &Device {
        &self.device
    }
    /// Layer `l`'s K pool buffer, if `l < n_layers`.
    pub fn k_pool(&self, layer: usize) -> Option<&Arc<RwLock<Storage>>> {
        self.k_pools.get(layer)
    }
    /// Layer `l`'s V pool buffer, if `l < n_layers`.
    pub fn v_pool(&self, layer: usize) -> Option<&Arc<RwLock<Storage>>> {
        self.v_pools.get(layer)
    }

    // --- block byte movement (device-general via the executor) ------------

    /// Layer `layer`'s K-or-V pool buffer, or a typed error if `layer` is out
    /// of range.
    fn buffer(&self, layer: usize, kind: BlockKind) -> crate::Result<&Arc<RwLock<Storage>>> {
        let v = match kind {
            BlockKind::K => &self.k_pools,
            BlockKind::V => &self.v_pools,
        };
        v.get(layer).ok_or_else(|| {
            msg_err(format!(
                "DeviceKvPool: no layer {layer} (n_layers={})",
                self.n_layers()
            ))
        })
    }

    /// Elements in one physical block for one buffer (`block_size · Hkv · D`).
    fn block_elems(&self) -> usize {
        let g = self.geometry();
        g.block_size * g.n_kv_heads * g.head_dim
    }

    /// The `Op::WriteSlice` rectangular ranges that select physical block
    /// `phys` in a `[num_blocks, block_size, Hkv, D]` pool buffer:
    /// `[(phys, phys+1), (0, block_size), (0, Hkv), (0, D)]`. Public so a
    /// consumer building its own decode graph can write a token's K/V slab
    /// straight into a block without re-deriving the offset.
    pub fn block_write_ranges(&self, phys: PhysBlockId) -> Vec<(usize, usize)> {
        let g = self.geometry();
        let p = phys as usize;
        vec![
            (p, p + 1),
            (0, g.block_size),
            (0, g.n_kv_heads),
            (0, g.head_dim),
        ]
    }

    /// Write `data` (`block_size · Hkv · D` f32, row-major `[block_size, Hkv,
    /// D]`) into physical block `phys` of layer `layer`'s K-or-V buffer, in
    /// place, through the executor (`Op::WriteSlice`) — so it is device-general
    /// (the same path a CUDA/Vulkan pool uses), not a host poke.
    ///
    /// f32-only for now (the CPU-gated increment); the pool's real dtype is
    /// carried in [`Self::dtype`] and a byte/dtype-generic form is the follow-up
    /// tracked for the CUDA bf16 pool.
    pub fn write_block(
        &self,
        layer: usize,
        kind: BlockKind,
        phys: PhysBlockId,
        data: &[f32],
    ) -> crate::Result<()> {
        if self.dtype != DType::F32 {
            return Err(msg_err(format!(
                "DeviceKvPool::write_block: f32-typed API, pool dtype is {:?} — use write_block_bytes",
                self.dtype,
            )));
        }
        if data.len() != self.block_elems() {
            return Err(msg_err(format!(
                "DeviceKvPool::write_block: data len {} ≠ block elems {}",
                data.len(),
                self.block_elems(),
            )));
        }
        // f32 typed wrapper over the dtype-agnostic byte core.
        self.write_block_bytes(layer, kind, phys, bytemuck::cast_slice::<f32, u8>(data))
    }

    /// Read physical block `phys` of layer `layer`'s K-or-V buffer back to host
    /// as `block_size · Hkv · D` f32 (row-major `[block_size, Hkv, D]`). Reads
    /// *through the executor* (`Slice` of the cache-bound buffer, realized) so
    /// it is device-general — a CUDA/Vulkan pool D2H-copies the block the same
    /// way. f32-only for now (see [`Self::write_block`]).
    pub fn read_block(
        &self,
        layer: usize,
        kind: BlockKind,
        phys: PhysBlockId,
    ) -> crate::Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(msg_err(format!(
                "DeviceKvPool::read_block: f32-typed API, pool dtype is {:?} — use read_block_bytes",
                self.dtype,
            )));
        }
        let bytes = self.read_block_bytes(layer, kind, phys)?;
        let v: &[f32] = bytemuck::try_cast_slice(&bytes)
            .map_err(|e| msg_err(format!("DeviceKvPool::read_block: f32 cast: {e:?}")))?;
        Ok(v.to_vec())
    }

    /// **rung-2 product** — materialize a registered prefix at a block-aligned,
    /// NON-ZERO position offset in `dst` via a uniform θ·M RoPE delta-rotation of
    /// its cached keys. Unlike the pure-core zero-copy
    /// [`splice_prefix_from`](crate::kv_block_pool::KvBlockPool::splice_prefix_from)
    /// (rung-1, prefix at position 0), this COPIES: the prefix's keys were rotated
    /// for positions `0..N` and are numerically wrong at `M..M+N`, so fresh blocks
    /// are allocated and written with rotated K + copied V. `rope_base` is the
    /// model's θ — a plain param, so a pool-driving consumer supplies its own (the
    /// §15 seam). Returns the shared token count `N`. `M = filled_tokens(dst)` must
    /// be block-aligned; validation + fresh-block bookkeeping is
    /// [`KvBlockPool::alloc_shifted_prefix_slots`](crate::kv_block_pool::KvBlockPool::alloc_shifted_prefix_slots),
    /// which is transactional (a refusal allocates nothing). f32 pool.
    pub fn splice_prefix_shifted(
        &mut self,
        prefix: crate::kv_block_pool::PrefixId,
        dst: crate::kv_block_pool::SessionHandle,
        rope_base: f64,
    ) -> crate::Result<usize> {
        let g = self.geometry();
        let (m, pairs) = self
            .core_mut()
            .alloc_shifted_prefix_slots(prefix, dst)
            .map_err(|e| msg_err(format!("DeviceKvPool::splice_prefix_shifted: {e:?}")))?;
        for (src, fresh) in &pairs {
            for l in 0..g.n_layers {
                // K: delta-rotate the cached (post-RoPE) block by θ·M into the fresh block.
                let k = self.read_block(l, BlockKind::K, *src)?;
                let rot = crate::lazy::Tensor::rope_delta_rotate_block_f32(
                    self.device(),
                    &k,
                    rope_base,
                    m,
                    g.block_size,
                    g.n_kv_heads,
                    g.head_dim,
                );
                self.write_block(l, BlockKind::K, *fresh, &rot)?;
                // V: position-independent, copied verbatim (dtype-agnostic bytes).
                let v = self.read_block_bytes(l, BlockKind::V, *src)?;
                self.write_block_bytes(l, BlockKind::V, *fresh, &v)?;
            }
        }
        Ok(pairs.len() * g.block_size)
    }

    /// Dtype-agnostic byte-level block WRITE — the movement core that the f32
    /// [`Self::write_block`] wrapper and the evict/restore/CoW paths build on.
    /// `bytes` is the block's raw content in the pool's dtype
    /// (`block_elems · dtype.size_in_bytes()` bytes, row-major `[block_size, Hkv,
    /// D]`). Backend-neutral: builds a dtype-tagged `Op::Const` from the bytes and
    /// `Op::WriteSlice`s it into the persistent pool buffer through the executor —
    /// the same path a CUDA/Vulkan (or future) pool uses; no host poke, no
    /// per-backend code.
    pub fn write_block_bytes(
        &self,
        layer: usize,
        kind: BlockKind,
        phys: PhysBlockId,
        bytes: &[u8],
    ) -> crate::Result<()> {
        let g = self.geometry();
        if phys as usize >= g.num_blocks {
            return Err(msg_err(format!(
                "DeviceKvPool::write_block_bytes: physical block {phys} out of range (num_blocks={})",
                g.num_blocks,
            )));
        }
        let want = self.block_elems() * self.dtype_elem_bytes()?;
        if bytes.len() != want {
            return Err(msg_err(format!(
                "DeviceKvPool::write_block_bytes: {} bytes ≠ block bytes {} \
                 ([block_size {}, Hkv {}, D {}] for {:?})",
                bytes.len(),
                want,
                g.block_size,
                g.n_kv_heads,
                g.head_dim,
                self.dtype,
            )));
        }
        let buf = Arc::clone(self.buffer(layer, kind)?);
        // Dtype-tagged Const from the raw bytes (backend-neutral) → WriteSlice into
        // the persistent pool buffer; realized as u8 (a byte reinterpret, discarded).
        let src = self.const_from_bytes(
            bytes,
            Shape::from_dims(&[1, g.block_size, g.n_kv_heads, g.head_dim]),
        )?;
        let dest = src.const_placeholder_like(self.pool_shape.clone(), self.dtype);
        let post = dest.write_slice(&src, self.block_write_ranges(phys))?;
        let mut cache = StorageCache::new();
        cache.insert(dest.node_id(), buf);
        let _ = crate::pipelined_bridge::realize_one_as_with_initial::<u8>(
            post.graph_handle(),
            post.node_id(),
            &self.device,
            cache,
        )?;
        Ok(())
    }

    /// Dtype-agnostic byte-level block READ — sibling of [`Self::write_block_bytes`].
    /// Returns the block's raw bytes in the pool's dtype via a uniform
    /// `realize_one_as::<u8>` (a byte reinterpret of the realized `Slice`, correct
    /// for ANY dtype — there is no per-dtype read path).
    pub fn read_block_bytes(
        &self,
        layer: usize,
        kind: BlockKind,
        phys: PhysBlockId,
    ) -> crate::Result<Vec<u8>> {
        let g = self.geometry();
        if phys as usize >= g.num_blocks {
            return Err(msg_err(format!(
                "DeviceKvPool::read_block_bytes: physical block {phys} out of range (num_blocks={})",
                g.num_blocks,
            )));
        }
        let buf = Arc::clone(self.buffer(layer, kind)?);
        // Bind the pool buffer to a placeholder, slice the one block out, realize
        // as u8 — a byte reinterpret, so the read is correct for ANY dtype. The
        // f32 anchor only mints the graph (mirrors the historical read_block).
        let anchor = Tensor::from_f32(vec![0.0], Shape::from_dims(&[1]), &self.device);
        let dest = anchor.const_placeholder_like(self.pool_shape.clone(), self.dtype);
        let block = dest
            .slice(0, phys as usize, 1)?
            .reshape(Shape::from_dims(&[self.block_elems()]))?;
        let mut cache = StorageCache::new();
        cache.insert(dest.node_id(), buf);
        crate::pipelined_bridge::realize_one_as_with_initial::<u8>(
            block.graph_handle(),
            block.node_id(),
            &self.device,
            cache,
        )
    }

    /// Byte width of one pool element for byte-level movement. F32/BF16 only (the
    /// activation dtypes the CUDA bf16 leg needs); F16 movement is a trivial
    /// follow-up (needs a `Tensor::from_f16_on` const source).
    fn dtype_elem_bytes(&self) -> crate::Result<usize> {
        match self.dtype {
            DType::F32 => Ok(4),
            DType::BF16 => Ok(2),
            other => Err(msg_err(format!(
                "DeviceKvPool: byte movement unsupported for dtype {other:?} (F32/BF16)",
            ))),
        }
    }

    /// Build a fresh-graph `Op::Const` of the pool's dtype from raw little-endian
    /// element bytes — the backend-neutral source [`Self::write_block_bytes`]
    /// WriteSlices. Dispatches `self.dtype` to the matching typed graph
    /// constructor via alignment-safe `try_cast_slice` (never panics). For BF16 a
    /// tiny f32 anchor mints the graph (mirrors `read_block_bytes`'s anchor); the
    /// bf16 const is a sibling on it.
    fn const_from_bytes(&self, bytes: &[u8], shape: Shape) -> crate::Result<Tensor> {
        let dev = &self.device;
        match self.dtype {
            DType::F32 => {
                let v: &[f32] = bytemuck::try_cast_slice(bytes)
                    .map_err(|e| msg_err(format!("const_from_bytes: f32 cast: {e:?}")))?;
                Ok(Tensor::from_f32(v.to_vec(), shape, dev))
            }
            DType::BF16 => {
                let v: &[half::bf16] = bytemuck::try_cast_slice(bytes)
                    .map_err(|e| msg_err(format!("const_from_bytes: bf16 cast: {e:?}")))?;
                let anchor = Tensor::from_f32(vec![0.0f32], Shape::from_dims(&[1]), dev);
                Ok(Tensor::from_bf16_on(anchor.graph(), v.to_vec(), shape, dev))
            }
            other => Err(msg_err(format!(
                "DeviceKvPool::const_from_bytes: unsupported dtype {other:?} (F32/BF16)",
            ))),
        }
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
            let filled = self
                .core
                .filled_tokens(s)
                .ok_or(KvAllocError::UnknownSession)?;
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
        Ok(PageTableHost {
            block_table,
            context_lens,
            batch,
            max_blocks,
        })
    }

    /// Capacity-padded sibling of [`Self::materialize_block_table`]: every row is
    /// right-padded with 0 to a FIXED width `max_blocks_cap`, so the block_table
    /// shape `[B, max_blocks_cap]` is constant across decode steps — the paged
    /// plan-once prerequisite. Padded entries (0) are in-bounds and neutralized
    /// by `Op::PagedAttn`'s `context_len` mask (a padded key position is always
    /// `>= context_len`). Errors if a session already occupies more than
    /// `max_blocks_cap` blocks.
    pub fn materialize_block_table_padded(
        &self,
        sessions: &[SessionHandle],
        max_blocks_cap: usize,
    ) -> std::result::Result<PageTableHost, KvAllocError> {
        let batch = sessions.len();
        let cap = max_blocks_cap.max(1); // >= 1 so the tensor is well-formed rank-2
        let mut block_table = vec![0u32; batch * cap];
        let mut context_lens: Vec<u32> = Vec::with_capacity(batch);
        for (bi, &s) in sessions.iter().enumerate() {
            let row = self.core.session_block_table(s)?;
            if row.len() > cap {
                // The fixed capacity can't hold this session's blocks — a caller
                // bug (pass a cap >= ceil(max_seq_len / block_size)). Never truncate.
                return Err(KvAllocError::OutOfBlocks {
                    need: row.len(),
                    have: cap,
                });
            }
            for (i, &p) in row.iter().enumerate() {
                block_table[bi * cap + i] = p;
            }
            let filled = self
                .core
                .filled_tokens(s)
                .ok_or(KvAllocError::UnknownSession)?;
            context_lens.push(filled as u32);
        }
        Ok(PageTableHost {
            block_table,
            context_lens,
            batch,
            max_blocks: cap,
        })
    }

    // --- paged decode step (the storage integration building block) -------

    /// Build (does NOT realize) one decode step's paged attention for one layer:
    /// write the new token's `[Hkv, D]` K/V into its physical pool slot, then run
    /// `Op::PagedAttn` over the (post-write) pool via the session's block table.
    /// The graph-builder half of paged decode — the model's forward calls this
    /// per layer, binds `k_pool(layer)`/`v_pool(layer)` into the realize cache
    /// (the caller owns the binding + the single realize for the whole forward),
    /// and threads the result into the o-projection.
    ///
    /// All tensors are on `q`'s graph: `k_pool_ph`/`v_pool_ph` are
    /// [`const_placeholder_like`](Tensor::const_placeholder_like) placeholders
    /// the caller binds to this layer's pool buffers; `block_table`/`context_lens`
    /// are the [`materialize_block_table`](Self::materialize_block_table) tensors;
    /// `q` is `[1, Hq, 1, D]`; `k_new`/`v_new` are `[1, Hkv, 1, D]` (the new
    /// token, post-RoPE for K). `(phys_block, slot)` is where the new token lands
    /// — `phys_block = resident_block(session, tok_pos / block_size)`,
    /// `slot = tok_pos % block_size`, after the session has been `append`ed to
    /// cover `tok_pos`. Returns the attention output `[1, Hq, 1, D]`.
    ///
    /// The write mutates the bound pool buffer in place (so it persists for the
    /// next step) AND feeds this step's `Op::PagedAttn` — one graph, one realize,
    /// exactly like the contiguous `write_slice`-then-attend path it replaces.
    #[allow(clippy::too_many_arguments)]
    pub fn build_decode_attn(
        &self,
        k_pool_ph: &Tensor,
        v_pool_ph: &Tensor,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
        phys_block: PhysBlockId,
        slot: usize,
        scale: f32,
    ) -> crate::Result<Tensor> {
        let g = self.geometry();
        let p = phys_block as usize;
        // The new token occupies one slot of one physical block: [1, 1, Hkv, D].
        let slot_ranges = vec![
            (p, p + 1),
            (slot, slot + 1),
            (0, g.n_kv_heads),
            (0, g.head_dim),
        ];
        let slot_shape = Shape::from_dims(&[1, 1, g.n_kv_heads, g.head_dim]);
        // [1, Hkv, 1, D] → [1, 1, Hkv, D] is a pure reshape (h-major order is
        // identical), aligning the new token to the pool block's slot layout.
        let k_slot = k_new.reshape(slot_shape.clone())?;
        let v_slot = v_new.reshape(slot_shape)?;
        let post_k = k_pool_ph.write_slice(&k_slot, slot_ranges.clone())?;
        let post_v = v_pool_ph.write_slice(&v_slot, slot_ranges)?;
        q.paged_attn(
            &post_k,
            &post_v,
            block_table,
            context_lens,
            None,
            scale,
            g.block_size,
            None,
        )
    }

    /// Batched (`B = K`) sibling of [`build_decode_attn`] — one decode step over
    /// K sessions sharing this layer's pool buffers (paged-storage PS4a). Each
    /// session `i` contributes one new token: its `[Hkv, D]` K/V (row `i` of
    /// `k_new`/`v_new`, shaped `[K, Hkv, 1, D]`) is written into its own physical
    /// slot `writes[i] = (phys, slot)`; then a single `Op::PagedAttn` at batch
    /// `K` reads all K sessions via the `[K, max_blocks]` `block_table` +
    /// `[K]` `context_lens`. `q` is `[K, Hq, 1, D]`; returns `[K, Hq, 1, D]`.
    ///
    /// The K per-session writes chain on the same bound pool buffer (each
    /// `write_slice` returns the post-write handle) so all land before the op
    /// reads — one graph, one realize, exactly like the single-session path.
    /// `writes` must be non-empty and in `block_table`/`q` batch-row order.
    #[allow(clippy::too_many_arguments)]
    pub fn build_decode_attn_batched(
        &self,
        k_pool_ph: &Tensor,
        v_pool_ph: &Tensor,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
        writes: &[(PhysBlockId, usize)],
        scale: f32,
    ) -> crate::Result<Tensor> {
        let g = self.geometry();
        let slot_shape = Shape::from_dims(&[1, 1, g.n_kv_heads, g.head_dim]);
        let mut post_k: Option<Tensor> = None;
        let mut post_v: Option<Tensor> = None;
        for (i, &(phys, slot)) in writes.iter().enumerate() {
            let p = phys as usize;
            let ranges = vec![
                (p, p + 1),
                (slot, slot + 1),
                (0, g.n_kv_heads),
                (0, g.head_dim),
            ];
            // Row i of the batched new K/V: [K,Hkv,1,D] --slice0--> [1,Hkv,1,D]
            // --reshape--> [1,1,Hkv,D] (the block-slot layout).
            let k_row = k_new.slice(0, i, 1)?.reshape(slot_shape.clone())?;
            let v_row = v_new.slice(0, i, 1)?.reshape(slot_shape.clone())?;
            let dest_k = post_k.as_ref().unwrap_or(k_pool_ph);
            let dest_v = post_v.as_ref().unwrap_or(v_pool_ph);
            post_k = Some(dest_k.write_slice(&k_row, ranges.clone())?);
            post_v = Some(dest_v.write_slice(&v_row, ranges)?);
        }
        let post_k = post_k.ok_or_else(|| {
            msg_err("build_decode_attn_batched: writes is empty (need ≥ 1 session)".into())
        })?;
        // `post_k`/`post_v` are assigned together in the loop above, so this is
        // unreachable — but a serving path should degrade to an error rather
        // than abort the process if that pairing ever drifts.
        let post_v = post_v.ok_or_else(|| {
            msg_err("build_decode_attn_batched: post_v unset while post_k is set".into())
        })?;
        q.paged_attn(
            &post_k,
            &post_v,
            block_table,
            context_lens,
            None,
            scale,
            g.block_size,
            None,
        )
    }

    /// **Plan-once sibling of [`Self::build_decode_attn`]** — the new token's K/V
    /// is written at a FLATTENED, runtime-resolved offset instead of a concrete
    /// two-axis slab, so the write op is structurally identical across decode
    /// steps (only the bound offset changes). The pool buffer is viewed as
    /// [`Self::pool_shape_flat`] `[num_blocks·block_size, Hkv, D]`; the new token
    /// `[1, Hkv, D]` lands at row `linear = phys·block_size + slot` via a single
    /// dynamic start on axis 0. The post-write flat buffer is reshaped back to
    /// `[num_blocks, block_size, Hkv, D]` for `Op::PagedAttn`.
    ///
    /// `k_pool_ph`/`v_pool_ph` must be `const_placeholder_like(pool_shape_flat())`
    /// placeholders the caller binds to this layer's pool buffers (NOTE: the flat
    /// shape, not [`Self::pool_shape`]). The offset is supplied one of two ways,
    /// mirroring the contiguous KV-write split in `apply_layer_with_kv_writes`:
    /// - `write_off = Some(off)`: a rank-0 `I64` tensor read device-side at kernel
    ///   launch (`Op::WriteSliceDoff`) — the CUDA-graph-capturable arm.
    /// - `write_off = None`: `Op::WriteSliceDyn` with `DynScalar::Sym(write_sym)`,
    ///   the caller binding `write_sym → linear` in the per-pass `SymEnv` — the
    ///   backend-generic (CPU/Vulkan) arm.
    ///
    /// Both land the same slab at the same offset — bit-identical to the concrete
    /// [`Self::build_decode_attn`]. Returns the attention output `[1, Hq, 1, D]`.
    #[allow(clippy::too_many_arguments)]
    pub fn build_decode_attn_off(
        &self,
        k_pool_ph: &Tensor,
        v_pool_ph: &Tensor,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
        write_off: Option<&Tensor>,
        write_sym: fuel_ir::SymId,
        scale: f32,
    ) -> crate::Result<Tensor> {
        let g = self.geometry();
        // New token as a single flat row: [1, Hkv, 1, D] → [1, Hkv, D] (h-major
        // order preserved), matching one row of the flattened pool's slot axis.
        let flat_row = Shape::from_dims(&[1, g.n_kv_heads, g.head_dim]);
        let k_row = k_new.reshape(flat_row.clone())?;
        let v_row = v_new.reshape(flat_row)?;
        // Width-1 slab on axis 0; the start is the dynamic offset
        // (linear = phys·block_size + slot). ranges[0].0 is ignored on the
        // dynamic axis — the width (1) must equal the source's axis-0 dim.
        let ranges = vec![(0usize, 1usize), (0, g.n_kv_heads), (0, g.head_dim)];
        // Write directly into the flat-shaped bound placeholder (no reshape
        // BETWEEN the binding and the write) so the in-place pool mutation is
        // preserved exactly like the contiguous `write_slice_doff` precedent.
        let (post_flat_k, post_flat_v) = match write_off {
            Some(off) => (
                k_pool_ph.write_slice_doff(&k_row, off, 0, ranges.clone())?,
                v_pool_ph.write_slice_doff(&v_row, off, 0, ranges)?,
            ),
            None => {
                let dyn_off = fuel_ir::DynScalar::Sym(write_sym);
                (
                    k_pool_ph.write_slice_dyn(&k_row, ranges.clone(), 0, dyn_off)?,
                    v_pool_ph.write_slice_dyn(&v_row, ranges, 0, dyn_off)?,
                )
            }
        };
        // Reshape the post-write flat buffers back to the paged-cache shape for
        // Op::PagedAttn — a consumer-side view; the write already mutated the
        // persistent pool buffer in place.
        let pool4 = Shape::from_dims(&[g.num_blocks, g.block_size, g.n_kv_heads, g.head_dim]);
        let post_k = post_flat_k.reshape(pool4.clone())?;
        let post_v = post_flat_v.reshape(pool4)?;
        q.paged_attn(
            &post_k,
            &post_v,
            block_table,
            context_lens,
            None,
            scale,
            g.block_size,
            None,
        )
    }

    /// **Copy-on-write guard** for the paged decode write path. Ensure the
    /// physical block backing logical slot `logical_block` of `session` is
    /// exclusively owned before a new token is written into it. If the block is
    /// **shared** (spliced, refcount > 1), break the share (`cow_break` → a fresh
    /// block) and COPY the shared block's bytes (all layers, K + V) into it, so
    /// the session keeps the shared-prefix content while its write no longer
    /// mutates a block another session still references. Returns the (now
    /// private) physical block to write into; an exclusive block is returned
    /// unchanged. **Must be called after `append` and before writing the token**
    /// — otherwise decoding a spliced session silently corrupts its co-sharers.
    pub fn ensure_writable_block(
        &mut self,
        session: SessionHandle,
        logical_block: usize,
    ) -> crate::Result<PhysBlockId> {
        let old = self
            .core
            .resident_block(session, logical_block)
            .ok_or_else(|| {
                msg_err(format!(
                    "ensure_writable_block: logical block {logical_block} not resident"
                ))
            })?;
        if self.core.block_refcount(old) <= 1 {
            return Ok(old); // exclusive — writable in place, no copy needed
        }
        // Shared → copy-on-write: fresh private block, then copy old → new so the
        // session's prefix content survives (the donor's block is untouched).
        let new = self
            .core
            .cow_break(session, logical_block)
            .map_err(alloc_err)?;
        for l in 0..self.n_layers() {
            let k = self.read_block_bytes(l, BlockKind::K, old)?;
            self.write_block_bytes(l, BlockKind::K, new, &k)?;
            let v = self.read_block_bytes(l, BlockKind::V, old)?;
            self.write_block_bytes(l, BlockKind::V, new, &v)?;
        }
        Ok(new)
    }

    // --- C-3: device-backed evict / restore (bytes move device↔host) ------

    /// Evict a session's entire resident state: capture the bytes of its
    /// exclusive blocks (D2H, all layers, K and V) into a [`DeviceEvicted`]
    /// handle, then hand the physical blocks back to the pool via the core.
    /// `restore` re-materializes it byte-for-byte. Shared (spliced) blocks are
    /// never detached and carry no bytes here (they stay resident).
    pub fn evict(&mut self, s: SessionHandle) -> crate::Result<DeviceEvicted> {
        let n = self
            .core
            .session_blocks(s)
            .ok_or_else(|| alloc_err(KvAllocError::UnknownSession))?;
        let all: Vec<usize> = (0..n).collect();
        self.evict_blocks(s, &all)
    }

    /// Evict a SET of a session's logical blocks (sub-session tiering). Captures
    /// the bytes of exactly the *exclusive* blocks in `indices` (a shared block
    /// stays resident — its sharer still needs it), then calls the core's
    /// refcount-aware [`KvBlockPool::evict_blocks`]. The returned handle carries
    /// the detached bytes; the rest of the session keeps decoding.
    pub fn evict_blocks(
        &mut self,
        s: SessionHandle,
        indices: &[usize],
    ) -> crate::Result<DeviceEvicted> {
        let n_layers = self.n_layers();
        // Capture bytes BEFORE the core detaches anything (an evicted block is
        // returned to the free pool and may be overwritten immediately). Only
        // exclusive resident blocks are actually detached, so only those are
        // read back — a shared or already-externalized slot moves no bytes.
        let to_save: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&i| match self.core.resident_block(s, i) {
                Some(p) => self.core.block_refcount(p) == 1,
                None => false,
            })
            .collect();
        let mut saved = Vec::with_capacity(to_save.len());
        for &i in &to_save {
            let p = self.core.resident_block(s, i).ok_or_else(|| {
                msg_err(format!(
                    "evict: slot {i} stopped being resident between the \
                     residency filter and the save loop"
                ))
            })?;
            let mut k = Vec::with_capacity(n_layers);
            let mut v = Vec::with_capacity(n_layers);
            for l in 0..n_layers {
                k.push(self.read_block_bytes(l, BlockKind::K, p)?);
                v.push(self.read_block_bytes(l, BlockKind::V, p)?);
            }
            saved.push(SavedBlock {
                logical_slot: i,
                k,
                v,
            });
        }
        let report = self.core.evict_blocks(s, indices).map_err(alloc_err)?;
        Ok(DeviceEvicted {
            core_handle: report.handle,
            saved,
        })
    }

    /// Re-materialize an evicted session: the core allocates fresh physical
    /// blocks for the externalized slots, then the saved bytes are written back
    /// (H2D) into whichever blocks the pool handed out — so the restored content
    /// is byte-identical regardless of physical reuse. Fails (typed) if the pool
    /// can't fit the session's blocks again (`OutOfBlocks`).
    pub fn restore(&mut self, s: SessionHandle, handle: DeviceEvicted) -> crate::Result<()> {
        let DeviceEvicted { core_handle, saved } = handle;
        self.core.restore(s, core_handle).map_err(alloc_err)?;
        for sb in &saved {
            let p = self
                .core
                .resident_block(s, sb.logical_slot)
                .ok_or_else(|| {
                    msg_err(format!(
                        "DeviceKvPool::restore: slot {} not resident after core restore — \
                     the core and device handle disagree (internal bug)",
                        sb.logical_slot,
                    ))
                })?;
            for l in 0..self.n_layers() {
                self.write_block_bytes(l, BlockKind::K, p, &sb.k[l])?;
                self.write_block_bytes(l, BlockKind::V, p, &sb.v[l])?;
            }
        }
        Ok(())
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
            .push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[4]),
                dtype: DType::U8,
            });
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
            realized.len(),
            2 * n_layers,
        ))
        .bt());
    }

    let mut it = realized.into_iter();
    let mut k_pools = Vec::with_capacity(n_layers);
    let mut v_pools = Vec::with_capacity(n_layers);
    // The length was checked to be exactly `2 * n_layers` just above, so the
    // iterator cannot run short — but pairing K with V by consuming two items
    // per turn is the kind of invariant that breaks silently under a later
    // edit, and getting it wrong would mis-pair pools rather than fail loudly.
    let short = |what: &str, l: usize| {
        crate::Error::Msg(format!(
            "kv pool realize: iterator ran short at layer {l} fetching {what}, \
             despite a checked length of {} — internal bug",
            2 * n_layers
        ))
        .bt()
    };
    for l in 0..n_layers {
        let (k_arc, _) = it.next().ok_or_else(|| short("K", l))?;
        let (v_arc, _) = it.next().ok_or_else(|| short("V", l))?;
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
        let g = geom(
            /*n_layers*/ 3, /*num_blocks*/ 8, /*block_size*/ 4,
        );
        let pool = DeviceKvPool::new(g, DType::F32, &Device::cpu()).unwrap();

        assert_eq!(pool.n_layers(), 3, "one K + one V buffer per layer");
        assert_eq!(
            pool.pool_shape().dims(),
            &[8, 4, 2, 8],
            "[num_blocks, block_size, Hkv, D]"
        );
        let want_elems = 8 * 4 * 2 * 8;
        for l in 0..3 {
            for (name, buf) in [("k", pool.k_pool(l)), ("v", pool.v_pool(l))] {
                let s = buf
                    .unwrap_or_else(|| panic!("{name}_pool({l}) exists"))
                    .read()
                    .unwrap();
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
        assert!(
            err.is_err(),
            "elem_size 2 vs f32 (4 bytes) must be rejected"
        );
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
        assert_eq!(
            pt.context_lens,
            vec![12u32, 6u32],
            "context_len = filled_tokens per session"
        );

        // Row 0 = sa's three physical ids; row 1 = sb's two ids then zero pad.
        let row_a = &pt.block_table[0..3];
        let row_b = &pt.block_table[3..6];
        assert_eq!(
            row_a,
            &[a_blocks[0], a_blocks[1], a_blocks[2]],
            "sa row = its real physical ids"
        );
        assert_eq!(
            row_b[0..2],
            [b_blocks[0], b_blocks[1]],
            "sb row = its real physical ids"
        );
        assert_eq!(row_b[2], 0, "sb's short row is zero-padded");
        assert_eq!(pt.block_table_shape().dims(), &[2, 3]);
        assert_eq!(pt.context_lens_shape().dims(), &[2]);
    }

    /// `write_block` mutates the *persistent* pool buffer in place (through the
    /// executor's `Op::WriteSlice`), and `read_block` reads it straight back —
    /// the device byte-movement round trip. If the WriteSlice produced a fresh
    /// output instead of mutating the bound buffer, read_block would still see
    /// the zero-init block and this fails.
    #[test]
    fn write_block_then_read_block_round_trips_in_place() {
        let pool = DeviceKvPool::new(geom(2, 8, 4), DType::F32, &Device::cpu()).unwrap();
        let block_elems = 4 * 2 * 8; // block_size · Hkv · D
        let data: Vec<f32> = (0..block_elems).map(|i| i as f32 + 0.5).collect();

        pool.write_block(1, BlockKind::K, 5, &data).unwrap();
        let back = pool.read_block(1, BlockKind::K, 5).unwrap();
        assert_eq!(
            back, data,
            "block content survives the device write→read round trip"
        );
    }

    /// rung-2 product: `splice_prefix_shifted` materializes the shared prefix at a
    /// block-aligned offset M as FRESH blocks whose K is the cached K delta-rotated
    /// by θ·M and whose V is copied verbatim. Verifies the orchestration — correct
    /// src→fresh mapping, per-layer loop, the offset threaded as the delta, and V
    /// left unrotated — against the Task-1 primitive.
    #[test]
    fn splice_prefix_shifted_rotates_k_copies_v() {
        let g = KvGeometry {
            n_layers: 2,
            n_kv_heads: 2,
            head_dim: 4,
            num_blocks: 64,
            block_size: 4,
            elem_size: DType::F32.size_in_bytes(),
        };
        let mut pool = DeviceKvPool::new(g, DType::F32, &Device::cpu()).unwrap();
        let rope_base = 10000.0f64;
        let block_elems = 4 * 2 * 4; // block_size · Hkv · D

        // Donor: 2 full blocks with deterministic, distinct K and V per (block, layer).
        let donor = pool.core_mut().open();
        pool.core_mut().append(donor, 8).unwrap();
        let d0 = pool.core().resident_block(donor, 0).unwrap();
        let d1 = pool.core().resident_block(donor, 1).unwrap();
        for (bi, phys) in [(0usize, d0), (1usize, d1)] {
            for l in 0..2usize {
                let base = (bi * 100 + l * 10) as f32;
                let k: Vec<f32> = (0..block_elems).map(|i| (base + i as f32) * 0.01).collect();
                let v: Vec<f32> = (0..block_elems)
                    .map(|i| (base + i as f32) * 0.02 + 1.0)
                    .collect();
                pool.write_block(l, BlockKind::K, phys, &k).unwrap();
                pool.write_block(l, BlockKind::V, phys, &v).unwrap();
            }
        }
        let pid = pool.core_mut().register_prefix(donor, 2).unwrap();

        // Sharer prefilled to a block-aligned offset M = 8.
        let dst = pool.core_mut().open();
        pool.core_mut().append(dst, 8).unwrap();
        let n = pool.splice_prefix_shifted(pid, dst, rope_base).unwrap();
        assert_eq!(n, 8, "shared token count = 2 blocks * block_size");

        // The shifted prefix is dst's slots 2,3 (after its 2 preamble blocks).
        for (dst_slot, src_phys) in [(2usize, d0), (3usize, d1)] {
            let fresh = pool.core().resident_block(dst, dst_slot).unwrap();
            for l in 0..2usize {
                let src_k = pool.read_block(l, BlockKind::K, src_phys).unwrap();
                let want = crate::lazy::Tensor::rope_delta_rotate_block_f32(
                    &Device::cpu(),
                    &src_k,
                    rope_base,
                    8,
                    4,
                    2,
                    4,
                );
                let got = pool.read_block(l, BlockKind::K, fresh).unwrap();
                let md = want
                    .iter()
                    .zip(&got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    md < 1e-6,
                    "K delta-rotated by θ·8 (layer {l}, slot {dst_slot}, maxdiff {md})"
                );
                assert_eq!(
                    pool.read_block(l, BlockKind::V, fresh).unwrap(),
                    pool.read_block(l, BlockKind::V, src_phys).unwrap(),
                    "V copied verbatim (layer {l}, slot {dst_slot})",
                );
            }
        }
    }

    /// A block write touches ONLY the addressed (layer, kind, phys) — every
    /// other block/buffer stays zero-initialized. Catches an offset/stride bug
    /// that would smear the write across blocks or the wrong buffer.
    #[test]
    fn write_block_is_isolated_to_its_layer_kind_and_physical_block() {
        let pool = DeviceKvPool::new(geom(2, 8, 4), DType::F32, &Device::cpu()).unwrap();
        let block_elems = 4 * 2 * 8;
        let ones = vec![1.0_f32; block_elems];
        let zeros = vec![0.0_f32; block_elems];

        pool.write_block(0, BlockKind::K, 3, &ones).unwrap();

        assert_eq!(
            pool.read_block(0, BlockKind::K, 3).unwrap(),
            ones,
            "target block written"
        );
        assert_eq!(
            pool.read_block(0, BlockKind::K, 2).unwrap(),
            zeros,
            "neighbor block untouched"
        );
        assert_eq!(
            pool.read_block(0, BlockKind::K, 4).unwrap(),
            zeros,
            "neighbor block untouched"
        );
        assert_eq!(
            pool.read_block(0, BlockKind::V, 3).unwrap(),
            zeros,
            "same-layer V untouched"
        );
        assert_eq!(
            pool.read_block(1, BlockKind::K, 3).unwrap(),
            zeros,
            "other-layer K untouched"
        );
    }

    /// Out-of-range physical block / wrong data length are typed errors, never
    /// panics or out-of-bounds writes.
    #[test]
    fn block_io_bounds_are_typed_errors() {
        let pool = DeviceKvPool::new(geom(1, 4, 4), DType::F32, &Device::cpu()).unwrap();
        let block_elems = 4 * 2 * 8;
        assert!(
            pool.write_block(0, BlockKind::K, 4, &vec![0.0; block_elems])
                .is_err(),
            "phys 4 ≥ num_blocks 4"
        );
        assert!(
            pool.read_block(0, BlockKind::K, 9).is_err(),
            "read phys 9 out of range"
        );
        assert!(
            pool.write_block(0, BlockKind::K, 0, &[1.0, 2.0]).is_err(),
            "short data rejected"
        );
        assert!(
            pool.write_block(3, BlockKind::K, 0, &vec![0.0; block_elems])
                .is_err(),
            "no layer 3"
        );
    }

    /// PC-1: the dtype-agnostic byte movement core round-trips an f32 pool's
    /// bytes byte-identically — the f32 baseline for the byte API. Movement, not
    /// compute, so fully CPU-verifiable.
    #[test]
    fn write_read_block_bytes_round_trips_f32() {
        let pool = DeviceKvPool::new(geom(2, 8, 4), DType::F32, &Device::cpu()).unwrap();
        let block_elems = 4 * 2 * 8; // block_size · Hkv · D
        let vals: Vec<f32> = (0..block_elems).map(|i| i as f32 + 0.5).collect();
        let bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&vals).to_vec();

        pool.write_block_bytes(1, BlockKind::K, 5, &bytes).unwrap();
        let back = pool.read_block_bytes(1, BlockKind::K, 5).unwrap();
        assert_eq!(
            back, bytes,
            "f32 block bytes survive the byte-level round trip"
        );
    }

    /// PC-1: the byte movement core is DTYPE-AGNOSTIC — a bf16 pool's 2-byte
    /// elements round-trip byte-identically with NO bf16 compute kernel (pure
    /// data movement: dtype-tagged `Op::Const` → `Op::WriteSlice` → `Op::Slice` →
    /// `realize::<u8>`). This is the real new capability the CUDA bf16 pool's
    /// evict/restore/CoW ride on.
    #[test]
    fn write_read_block_bytes_round_trips_bf16() {
        let mut g = geom(2, 8, 4);
        g.elem_size = 2; // bf16 = 2 bytes/elem
        let pool = DeviceKvPool::new(g, DType::BF16, &Device::cpu()).unwrap();
        let block_elems = 4 * 2 * 8;
        // Valid bf16 content (avoid NaN-canonicalization ambiguity), as raw bytes.
        let vals: Vec<half::bf16> = (0..block_elems)
            .map(|i| half::bf16::from_f32(i as f32 * 0.25 - 3.0))
            .collect();
        let bytes: Vec<u8> = bytemuck::cast_slice::<half::bf16, u8>(&vals).to_vec();
        assert_eq!(bytes.len(), block_elems * 2, "bf16 block = 2 bytes/elem");

        pool.write_block_bytes(1, BlockKind::K, 5, &bytes).unwrap();
        let back = pool.read_block_bytes(1, BlockKind::K, 5).unwrap();
        assert_eq!(
            back, bytes,
            "bf16 block bytes survive the byte-level round trip"
        );
    }

    /// PC-1: evict→restore is byte-exact on a BF16 pool — the dtype-agnostic
    /// movement makes suspend/resume (C-3) real for the CUDA bf16 pool, not just
    /// f32. CPU-verifiable (movement, not compute). Pre-PC-1 this errors: evict
    /// reads via the f32 `read_block`, which rejects a bf16 pool.
    #[test]
    fn bf16_pool_evict_restore_is_byte_exact() {
        let mut g = geom(2, 8, 4);
        g.elem_size = 2;
        let mut pool = DeviceKvPool::new(g, DType::BF16, &Device::cpu()).unwrap();
        let block_elems = 4 * 2 * 8;
        let vals: Vec<half::bf16> = (0..block_elems)
            .map(|i| half::bf16::from_f32(i as f32 * 0.5 - 5.0))
            .collect();
        let bytes: Vec<u8> = bytemuck::cast_slice::<half::bf16, u8>(&vals).to_vec();

        let s = pool.core_mut().open();
        pool.core_mut().append(s, 1).unwrap(); // one exclusive block resident
        let phys = pool.core().resident_block(s, 0).unwrap();
        for l in 0..pool.n_layers() {
            pool.write_block_bytes(l, BlockKind::K, phys, &bytes)
                .unwrap();
            pool.write_block_bytes(l, BlockKind::V, phys, &bytes)
                .unwrap();
        }

        let handle = pool.evict(s).unwrap();
        pool.restore(s, handle).unwrap();

        let phys2 = pool.core().resident_block(s, 0).unwrap();
        for l in 0..pool.n_layers() {
            assert_eq!(
                pool.read_block_bytes(l, BlockKind::K, phys2).unwrap(),
                bytes,
                "K byte-exact after bf16 evict/restore (layer {l})",
            );
            assert_eq!(
                pool.read_block_bytes(l, BlockKind::V, phys2).unwrap(),
                bytes,
                "V byte-exact after bf16 evict/restore (layer {l})",
            );
        }
    }

    /// Small deterministic pseudo-random f32 fill (mirrors paged_attn_oracle).
    fn rand_f32(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.5
            })
            .collect()
    }

    /// Paged plan-once Task 1: a capacity-PADDED block_table (`[B, CAP]`,
    /// 0-padded) plus the `context_len` mask produces byte-identical `paged_attn`
    /// output to the L-fitted block_table (`[B, N]`). Padded key positions are
    /// `>= context_len`, so the op's mask drives them to `−inf` before softmax
    /// (`exp(−inf)=0`, added exactly), leaving the result bit-identical. That
    /// means the block_table SHAPE can be a FIXED capacity across decode steps
    /// instead of growing with L — the structural prerequisite for one cached plan.
    #[test]
    fn padded_block_table_paged_attn_matches_unpadded() {
        let (hq, hkv, d, block_size, sk) = (2usize, 2usize, 4usize, 4usize, 12usize);
        let num_blocks = 8;
        let cap = 6; // fixed capacity > the session's 3 resident blocks
        let scale = 1.0f32 / (d as f32).sqrt();
        let dev = Device::cpu();

        let g = KvGeometry {
            n_layers: 1,
            num_blocks,
            block_size,
            n_kv_heads: hkv,
            head_dim: d,
            elem_size: 4,
        };
        let mut pool = DeviceKvPool::new(g, DType::F32, &dev).unwrap();
        let s = pool.core_mut().open();
        pool.core_mut().append(s, sk).unwrap(); // 12 tokens → 3 blocks
        let phys = pool.core().session_block_table(s).unwrap();

        let k_all = rand_f32(sk * hkv * d, 2);
        let v_all = rand_f32(sk * hkv * d, 3);
        let q_data = rand_f32(hq * d, 1);
        let per_block = block_size * hkv * d;
        for (b, &p) in phys.iter().enumerate() {
            pool.write_block(
                0,
                BlockKind::K,
                p,
                &k_all[b * per_block..(b + 1) * per_block],
            )
            .unwrap();
            pool.write_block(
                0,
                BlockKind::V,
                p,
                &v_all[b * per_block..(b + 1) * per_block],
            )
            .unwrap();
        }

        let pt_fit = pool.materialize_block_table(&[s]).unwrap();
        let pt_pad = pool.materialize_block_table_padded(&[s], cap).unwrap();
        assert_eq!(
            pt_fit.block_table_shape().dims(),
            &[1, phys.len()],
            "L-fitted width"
        );
        assert_eq!(
            pt_pad.block_table_shape().dims(),
            &[1, cap],
            "padded width is the fixed capacity"
        );
        assert_eq!(
            pt_pad.context_lens, pt_fit.context_lens,
            "same context_lens"
        );

        let run = |pt: &PageTableHost| -> Vec<f32> {
            let q = Tensor::from_f32(q_data.clone(), Shape::from_dims(&[1, hq, 1, d]), &dev);
            let kc = q.const_placeholder_like(pool.pool_shape().clone(), DType::F32);
            let vc = q.const_placeholder_like(pool.pool_shape().clone(), DType::F32);
            let bt = q.const_u32_like(pt.block_table.clone(), pt.block_table_shape());
            let cl = q.const_u32_like(pt.context_lens.clone(), pt.context_lens_shape());
            let out = q
                .paged_attn(&kc, &vc, &bt, &cl, None, scale, block_size, None)
                .unwrap();
            let mut cache = StorageCache::new();
            cache.insert(kc.node_id(), Arc::clone(pool.k_pool(0).unwrap()));
            cache.insert(vc.node_id(), Arc::clone(pool.v_pool(0).unwrap()));
            crate::pipelined_bridge::realize_one_as_with_initial::<f32>(
                out.graph_handle(),
                out.node_id(),
                &dev,
                cache,
            )
            .unwrap()
        };

        assert_eq!(
            run(&pt_pad),
            run(&pt_fit),
            "capacity-padded block_table (mask-neutralized pad) == L-fitted, byte-identical",
        );
    }

    /// Paged plan-once Task 2: writing the new token's K/V via a FLATTENED
    /// runtime offset (`linear = phys·block_size + slot`, one dynamic start on
    /// the pool's collapsed slot axis) lands byte-identically to the concrete
    /// two-axis `write_slice` — for BOTH the `SymEnv` (`write_slice_dyn`) and the
    /// device-offset (`write_slice_doff`) arms — and the resulting `paged_attn`
    /// output is bit-identical. That makes the K/V write op structurally
    /// invariant across decode steps (only the bound offset changes), removing
    /// the last per-step graph-shape variability blocking one cached plan.
    ///
    /// A filler forces a permuted physical layout so the new token's physical
    /// block (2) ≠ its slot (2): an offset that dropped the `phys·block_size`
    /// term (or the `slot`) would write the WRONG physical block and diverge on
    /// both the read-back bytes AND the attention output.
    #[test]
    fn symbolic_offset_write_matches_concrete_write() {
        let (hq, hkv, d, block_size) = (2usize, 2usize, 8usize, 4usize);
        let num_blocks = 8;
        let sk = 6usize; // context tokens; new token is position 6 → slot 2, block 2
        let scale = 1.0f32 / (d as f32).sqrt();
        let dev = Device::cpu();
        let g = KvGeometry {
            n_layers: 1,
            num_blocks,
            block_size,
            n_kv_heads: hkv,
            head_dim: d,
            elem_size: 4,
        };

        // Deterministic shared inputs (identical across all three pools).
        let q_data = rand_f32(hq * d, 1);
        let k_new_data = rand_f32(hkv * d, 4);
        let v_new_data = rand_f32(hkv * d, 5);
        let q_shape = Shape::from_dims(&[1, hq, 1, d]);
        let kv_new_shape = Shape::from_dims(&[1, hkv, 1, d]);

        // Build a fresh, identically-prepared pool: a filler grabs physical
        // block 0, then the session gets blocks 1,2; append 1 for the new token.
        let setup = || -> (DeviceKvPool, PhysBlockId, usize, PageTableHost) {
            let mut pool = DeviceKvPool::new(g, DType::F32, &dev).unwrap();
            let filler = pool.core_mut().open();
            pool.core_mut().append(filler, block_size).unwrap(); // physical block 0
            let s = pool.core_mut().open();
            pool.core_mut().append(s, sk).unwrap(); // physical blocks 1,2 (6 tokens)
            let tok_pos = sk; // new token position (6)
            pool.core_mut().append(s, 1).unwrap(); // filled = 7 (covers position 6)
            let phys = pool.core().resident_block(s, tok_pos / block_size).unwrap();
            let slot = tok_pos % block_size;
            let pt = pool.materialize_block_table(&[s]).unwrap();
            (pool, phys, slot, pt)
        };

        // ---- reference: the concrete two-axis write (build_decode_attn) ----
        let (pool_ref, phys, slot, pt) = setup();
        assert_eq!(
            phys, 2,
            "permuted layout — new token's physical block is 2, not 0"
        );
        assert_eq!(slot, 2, "new token lands at slot 2");
        let linear = phys as usize * block_size + slot; // 10
        let attn_ref = {
            let q = Tensor::from_f32(q_data.clone(), q_shape.clone(), &dev);
            // k_new/v_new must be siblings of q (build_decode_attn's same-graph
            // contract) — Tensor::from_f32 would mint separate graphs.
            let k_new = q.const_f32_like(k_new_data.clone(), kv_new_shape.clone());
            let v_new = q.const_f32_like(v_new_data.clone(), kv_new_shape.clone());
            let k_ph = q.const_placeholder_like(pool_ref.pool_shape().clone(), DType::F32);
            let v_ph = q.const_placeholder_like(pool_ref.pool_shape().clone(), DType::F32);
            let bt = q.const_u32_like(pt.block_table.clone(), pt.block_table_shape());
            let cl = q.const_u32_like(pt.context_lens.clone(), pt.context_lens_shape());
            let out = pool_ref
                .build_decode_attn(
                    &k_ph, &v_ph, &q, &k_new, &v_new, &bt, &cl, phys, slot, scale,
                )
                .unwrap();
            let mut cache = StorageCache::new();
            cache.insert(k_ph.node_id(), Arc::clone(pool_ref.k_pool(0).unwrap()));
            cache.insert(v_ph.node_id(), Arc::clone(pool_ref.v_pool(0).unwrap()));
            crate::pipelined_bridge::realize_one_as_with_initial::<f32>(
                out.graph_handle(),
                out.node_id(),
                &dev,
                cache,
            )
            .unwrap()
        };
        let ref_k_bytes = pool_ref.read_block_bytes(0, BlockKind::K, phys).unwrap();
        let ref_v_bytes = pool_ref.read_block_bytes(0, BlockKind::V, phys).unwrap();

        // ---- dyn arm: SymEnv-bound flattened offset (write_slice_dyn) ----
        let sym = fuel_ir::SymId(0);
        let (pool_dyn, _p, _s, pt_dyn) = setup();
        let attn_dyn = {
            let q = Tensor::from_f32(q_data.clone(), q_shape.clone(), &dev);
            // k_new/v_new must be siblings of q (build_decode_attn's same-graph
            // contract) — Tensor::from_f32 would mint separate graphs.
            let k_new = q.const_f32_like(k_new_data.clone(), kv_new_shape.clone());
            let v_new = q.const_f32_like(v_new_data.clone(), kv_new_shape.clone());
            let k_ph = q.const_placeholder_like(pool_dyn.pool_shape_flat(), DType::F32);
            let v_ph = q.const_placeholder_like(pool_dyn.pool_shape_flat(), DType::F32);
            let bt = q.const_u32_like(pt_dyn.block_table.clone(), pt_dyn.block_table_shape());
            let cl = q.const_u32_like(pt_dyn.context_lens.clone(), pt_dyn.context_lens_shape());
            let out = pool_dyn
                .build_decode_attn_off(&k_ph, &v_ph, &q, &k_new, &v_new, &bt, &cl, None, sym, scale)
                .unwrap();
            let mut cache = StorageCache::new();
            cache.insert(k_ph.node_id(), Arc::clone(pool_dyn.k_pool(0).unwrap()));
            cache.insert(v_ph.node_id(), Arc::clone(pool_dyn.v_pool(0).unwrap()));
            let mut env = fuel_ir::SymEnv::new();
            env.bind(sym, linear).unwrap();
            crate::pipelined_bridge::realize_one_as_with_initial_env::<f32>(
                out.graph_handle(),
                out.node_id(),
                &dev,
                cache,
                &env,
            )
            .unwrap()
        };
        assert_eq!(
            pool_dyn.read_block_bytes(0, BlockKind::K, phys).unwrap(),
            ref_k_bytes,
            "dyn write: K block bytes match the concrete write",
        );
        assert_eq!(
            pool_dyn.read_block_bytes(0, BlockKind::V, phys).unwrap(),
            ref_v_bytes,
            "dyn write: V block bytes match the concrete write",
        );
        assert_eq!(
            attn_dyn, attn_ref,
            "dyn arm paged_attn output is bit-identical to concrete"
        );

        // ---- doff arm: device rank-0 I64 offset (write_slice_doff) ----
        let (pool_doff, _p, _s, pt_doff) = setup();
        let attn_doff = {
            let q = Tensor::from_f32(q_data.clone(), q_shape.clone(), &dev);
            // k_new/v_new must be siblings of q (build_decode_attn's same-graph
            // contract) — Tensor::from_f32 would mint separate graphs.
            let k_new = q.const_f32_like(k_new_data.clone(), kv_new_shape.clone());
            let v_new = q.const_f32_like(v_new_data.clone(), kv_new_shape.clone());
            let k_ph = q.const_placeholder_like(pool_doff.pool_shape_flat(), DType::F32);
            let v_ph = q.const_placeholder_like(pool_doff.pool_shape_flat(), DType::F32);
            let bt = q.const_u32_like(pt_doff.block_table.clone(), pt_doff.block_table_shape());
            let cl = q.const_u32_like(pt_doff.context_lens.clone(), pt_doff.context_lens_shape());
            let off = q.const_i64_like(vec![linear as i64], Shape::from_dims(&[]));
            let out = pool_doff
                .build_decode_attn_off(
                    &k_ph,
                    &v_ph,
                    &q,
                    &k_new,
                    &v_new,
                    &bt,
                    &cl,
                    Some(&off),
                    sym,
                    scale,
                )
                .unwrap();
            let mut cache = StorageCache::new();
            cache.insert(k_ph.node_id(), Arc::clone(pool_doff.k_pool(0).unwrap()));
            cache.insert(v_ph.node_id(), Arc::clone(pool_doff.v_pool(0).unwrap()));
            crate::pipelined_bridge::realize_one_as_with_initial::<f32>(
                out.graph_handle(),
                out.node_id(),
                &dev,
                cache,
            )
            .unwrap()
        };
        assert_eq!(
            pool_doff.read_block_bytes(0, BlockKind::K, phys).unwrap(),
            ref_k_bytes,
            "doff write: K block bytes match the concrete write",
        );
        assert_eq!(
            pool_doff.read_block_bytes(0, BlockKind::V, phys).unwrap(),
            ref_v_bytes,
            "doff write: V block bytes match the concrete write",
        );
        assert_eq!(
            attn_doff, attn_ref,
            "doff arm paged_attn output is bit-identical to concrete"
        );
    }

    /// THE INTEGRATION GATE. A session's K/V, written into the pool's physical
    /// blocks via `write_block` at the *core-assigned* (non-identity) physical
    /// positions, then read by `Op::PagedAttn` through the *materialized*
    /// block_table, must equal a hand-computed dense attention over the logical
    /// sequence. This composes every Part-2 piece — core placement + write_block +
    /// materialize_block_table + the real fused op bound to the real pool
    /// buffers. The physical layout is a permutation (blocks 2,3,4, not 0,1,2),
    /// so a paged_attn that ignored the block_table would read the wrong blocks
    /// and fail.
    #[test]
    fn pool_routed_paged_attn_matches_dense_reference_over_a_permuted_layout() {
        let (hq, hkv, d, block_size, sk) = (2usize, 2usize, 4usize, 4usize, 12usize);
        let num_blocks = 8;
        let scale = 1.0f32 / (d as f32).sqrt();
        let dev = Device::cpu();

        let g = KvGeometry {
            n_layers: 1,
            num_blocks,
            block_size,
            n_kv_heads: hkv,
            head_dim: d,
            elem_size: 4,
        };
        let mut pool = DeviceKvPool::new(g, DType::F32, &dev).unwrap();

        // Force a NON-identity physical assignment: a filler grabs blocks 0,1;
        // the real session then gets blocks 2,3,4.
        let filler = pool.core_mut().open();
        pool.core_mut().append(filler, 8).unwrap();
        let s = pool.core_mut().open();
        pool.core_mut().append(s, sk).unwrap(); // 12 tokens → 3 blocks: 2,3,4
        let phys = pool.core().session_block_table(s).unwrap();
        assert_eq!(phys, vec![2, 3, 4], "permuted layout — not identity 0,1,2");

        // Logical K/V for the session, row-major [sk, Hkv, D]; q is [1, Hq, 1, D].
        let k_all = rand_f32(sk * hkv * d, 2);
        let v_all = rand_f32(sk * hkv * d, 3);
        let q_data = rand_f32(hq * d, 1);

        // Write each logical block into its core-assigned physical block.
        let per_block = block_size * hkv * d;
        for (b, &p) in phys.iter().enumerate() {
            let kb = &k_all[b * per_block..(b + 1) * per_block];
            let vb = &v_all[b * per_block..(b + 1) * per_block];
            pool.write_block(0, BlockKind::K, p, kb).unwrap();
            pool.write_block(0, BlockKind::V, p, vb).unwrap();
        }

        // Materialize the page table straight from the core.
        let pt = pool.materialize_block_table(&[s]).unwrap();
        assert_eq!(pt.context_lens, vec![sk as u32]);

        // Build the paged_attn graph, binding the REAL pool buffers as k/v cache.
        let q = Tensor::from_f32(q_data.clone(), Shape::from_dims(&[1, hq, 1, d]), &dev);
        let kc = q.const_placeholder_like(pool.pool_shape().clone(), DType::F32);
        let vc = q.const_placeholder_like(pool.pool_shape().clone(), DType::F32);
        let bt = q.const_u32_like(pt.block_table.clone(), pt.block_table_shape());
        let cl = q.const_u32_like(pt.context_lens.clone(), pt.context_lens_shape());
        let out = q
            .paged_attn(&kc, &vc, &bt, &cl, None, scale, block_size, None)
            .unwrap();

        let mut cache = StorageCache::new();
        cache.insert(kc.node_id(), Arc::clone(pool.k_pool(0).unwrap()));
        cache.insert(vc.node_id(), Arc::clone(pool.v_pool(0).unwrap()));
        let got = crate::pipelined_bridge::realize_one_as_with_initial::<f32>(
            out.graph_handle(),
            out.node_id(),
            &dev,
            cache,
        )
        .unwrap();

        // Dense reference: softmax(scale · q·kᵀ) · v per head, over all sk keys.
        let mut want = vec![0.0f32; hq * d];
        for h in 0..hq {
            let mut scores = vec![0.0f32; sk];
            for t in 0..sk {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q_data[h * d + dd] * k_all[(t * hkv + h) * d + dd];
                }
                scores[t] = dot * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                denom += *sc;
            }
            for t in 0..sk {
                let p = scores[t] / denom;
                for dd in 0..d {
                    want[h * d + dd] += p * v_all[(t * hkv + h) * d + dd];
                }
            }
        }

        assert_eq!(got.len(), want.len());
        for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
            let diff = (a - b).abs();
            let denom = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
            assert!(
                diff < 1e-5 || diff / denom < 1e-5,
                "[{i}]: pool={a} dense={b} (abs={diff})"
            );
        }
    }

    /// C-3 device round trip: a session's K/V, evicted (bytes captured D2H, its
    /// physical blocks returned to the pool) and then restored (bytes written
    /// H2D into freshly-assigned blocks), is byte-identical — even after every
    /// physical block was scribbled with garbage while the session was
    /// suspended. Distinct content per (slot, layer, kind) catches any mixup.
    #[test]
    fn device_evict_then_restore_is_a_byte_exact_round_trip_across_block_reuse() {
        let (n_layers, num_blocks, block_size) = (2usize, 8usize, 4usize);
        let mut pool = DeviceKvPool::new(
            geom(n_layers, num_blocks, block_size),
            DType::F32,
            &Device::cpu(),
        )
        .unwrap();
        let block_elems = block_size * 2 * 8; // Hkv=2, D=8 from geom()

        // Distinct content per (logical slot, layer, kind).
        let content = |slot: usize, layer: usize, kind: BlockKind| -> Vec<f32> {
            let base =
                (slot * 1000 + layer * 100 + if kind == BlockKind::K { 0 } else { 50 }) as f32;
            (0..block_elems).map(|e| base + e as f32 * 0.001).collect()
        };

        let s = pool.core_mut().open();
        pool.core_mut().append(s, 8).unwrap(); // 2 blocks
        for slot in 0..2 {
            let p = pool.core().resident_block(s, slot).unwrap();
            for l in 0..n_layers {
                pool.write_block(l, BlockKind::K, p, &content(slot, l, BlockKind::K))
                    .unwrap();
                pool.write_block(l, BlockKind::V, p, &content(slot, l, BlockKind::V))
                    .unwrap();
            }
        }

        let free_before = pool.core().free_blocks();
        let handle = pool.evict(s).unwrap();
        assert_eq!(
            handle.saved_block_count(),
            2,
            "both exclusive blocks captured"
        );
        assert_eq!(handle.fidelity(), crate::kv_block_pool::Fidelity::Lossy);
        assert_eq!(
            pool.core().free_blocks(),
            free_before + 2,
            "2 blocks returned to the pool"
        );

        // Scribble EVERY physical block — the session's old blocks are free and
        // could be reused; restore must overwrite whatever it grabs.
        for p in 0..num_blocks as PhysBlockId {
            for l in 0..n_layers {
                pool.write_block(l, BlockKind::K, p, &vec![-9.0; block_elems])
                    .unwrap();
                pool.write_block(l, BlockKind::V, p, &vec![-9.0; block_elems])
                    .unwrap();
            }
        }

        pool.restore(s, handle).unwrap();
        assert_eq!(
            pool.core().session_blocks(s),
            Some(2),
            "session resident again"
        );

        // Every (slot, layer, kind) reads back its original content.
        for slot in 0..2 {
            let p = pool.core().resident_block(s, slot).unwrap();
            for l in 0..n_layers {
                assert_eq!(
                    pool.read_block(l, BlockKind::K, p).unwrap(),
                    content(slot, l, BlockKind::K),
                    "K slot {slot} layer {l} restored byte-exact",
                );
                assert_eq!(
                    pool.read_block(l, BlockKind::V, p).unwrap(),
                    content(slot, l, BlockKind::V),
                    "V slot {slot} layer {l} restored byte-exact",
                );
            }
        }
    }

    /// A partial (sub-session) evict of a spliced session moves bytes only for
    /// the EXCLUSIVE blocks; the shared prefix stays resident and carries no
    /// bytes in the handle (the sharer still references it).
    #[test]
    fn device_partial_evict_captures_only_exclusive_blocks() {
        let mut pool = DeviceKvPool::new(geom(1, 16, 4), DType::F32, &Device::cpu()).unwrap();
        let a = pool.core_mut().open();
        pool.core_mut().append(a, 9).unwrap(); // 3 blocks
        let b = pool.core_mut().open();
        pool.core_mut().splice(a, b, 0, 2).unwrap(); // b shares A's first 2 blocks

        // Evict all of A: blocks 0,1 are shared (kept), block 2 is exclusive.
        let handle = pool.evict(a).unwrap();
        assert_eq!(
            handle.saved_block_count(),
            1,
            "only the 1 exclusive block's bytes are captured; the 2 shared stay resident",
        );
        // B is untouched and still resident.
        assert_eq!(pool.core().session_blocks(b), Some(2));
    }

    /// THE PAGED-DECODE GATE. Decode T tokens ONE AT A TIME into the pool —
    /// append a token, write its K/V into the freshly-grown block slot, and
    /// attend via `Op::PagedAttn` — and at every step the attention output must
    /// equal a dense `softmax(scale·q·kᵀ)·v` over the tokens accumulated so far.
    /// This is the multi-step write/accumulate path the storage integration
    /// rests on: block-boundary growth (block_size 4, 10 steps → 3 blocks), the
    /// running context_len mask, and the in-place pool writes persisting across
    /// steps. (2c proved single-shot paged_attn over a pool; this proves the
    /// token-by-token decode loop.)
    #[test]
    fn paged_decode_multistep_matches_dense_reference_across_block_growth() {
        let (hq, hkv, d, block_size) = (2usize, 2usize, 4usize, 4usize);
        let (num_blocks, n_steps) = (8usize, 10usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let dev = Device::cpu();
        let g = KvGeometry {
            n_layers: 1,
            num_blocks,
            block_size,
            n_kv_heads: hkv,
            head_dim: d,
            elem_size: 4,
        };
        let mut pool = DeviceKvPool::new(g, DType::F32, &dev).unwrap();
        let session = pool.core_mut().open();

        // Accumulated logical K/V for the dense reference (each entry [Hkv·D]).
        let mut k_hist: Vec<Vec<f32>> = Vec::new();
        let mut v_hist: Vec<Vec<f32>> = Vec::new();

        for t in 0..n_steps {
            let q_data = rand_f32(hq * d, 100 + t as u32);
            let k_data = rand_f32(hkv * d, 200 + t as u32);
            let v_data = rand_f32(hkv * d, 300 + t as u32);
            k_hist.push(k_data.clone());
            v_hist.push(v_data.clone());

            // Grow the session by one token; find where token t lands.
            pool.core_mut().append(session, 1).unwrap();
            let phys = pool.core().resident_block(session, t / block_size).unwrap();
            let slot = t % block_size;
            let pt = pool.materialize_block_table(&[session]).unwrap();
            assert_eq!(
                pt.context_lens,
                vec![(t + 1) as u32],
                "context_len tracks tokens so far"
            );

            // Build the paged decode-step graph on q's graph, bind the pool.
            let q = Tensor::from_f32(q_data.clone(), Shape::from_dims(&[1, hq, 1, d]), &dev);
            let kph = q.const_placeholder_like(pool.pool_shape().clone(), DType::F32);
            let vph = q.const_placeholder_like(pool.pool_shape().clone(), DType::F32);
            let knew = q.const_f32_like(k_data, Shape::from_dims(&[1, hkv, 1, d]));
            let vnew = q.const_f32_like(v_data, Shape::from_dims(&[1, hkv, 1, d]));
            let bt = q.const_u32_like(pt.block_table.clone(), pt.block_table_shape());
            let cl = q.const_u32_like(pt.context_lens.clone(), pt.context_lens_shape());
            let attn = pool
                .build_decode_attn(&kph, &vph, &q, &knew, &vnew, &bt, &cl, phys, slot, scale)
                .unwrap();

            let mut cache = StorageCache::new();
            cache.insert(kph.node_id(), Arc::clone(pool.k_pool(0).unwrap()));
            cache.insert(vph.node_id(), Arc::clone(pool.v_pool(0).unwrap()));
            let got = crate::pipelined_bridge::realize_one_as_with_initial::<f32>(
                attn.graph_handle(),
                attn.node_id(),
                &dev,
                cache,
            )
            .unwrap();

            // Dense reference: q attends to tokens 0..=t.
            let mut want = vec![0.0f32; hq * d];
            for h in 0..hq {
                let mut scores = vec![0.0f32; t + 1];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for dd in 0..d {
                        dot += q_data[h * d + dd] * k_hist[j][h * d + dd];
                    }
                    *sc = dot * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - m).exp();
                    denom += *sc;
                }
                for (j, &p) in scores.iter().enumerate() {
                    let w = p / denom;
                    for dd in 0..d {
                        want[h * d + dd] += w * v_hist[j][h * d + dd];
                    }
                }
            }

            assert_eq!(got.len(), want.len());
            for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
                let diff = (a - b).abs();
                let den = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
                assert!(
                    diff < 1e-5 || diff / den < 1e-5,
                    "step {t} [{i}]: paged={a} dense={b} (abs={diff})",
                );
            }
        }
        assert_eq!(
            pool.core().session_blocks(session),
            Some(3),
            "10 tokens / block_size 4 = 3 blocks"
        );
    }

    /// PS4c substrate — SPLICE on the paged path (prefix sharing / residual-
    /// stream donation). Session A fills blocks with known K/V; session B splices
    /// A's blocks (shared, refcount 2); a paged attention for B reads the SHARED
    /// blocks via B's materialized block_table and equals a dense reference over
    /// A's K/V. Proves the core's refcounted splice composes with the paged read
    /// path — B decodes over A's prefix with zero copy.
    #[test]
    fn spliced_shared_blocks_are_read_by_paged_attention() {
        let (hq, hkv, d, block_size) = (2usize, 2usize, 4usize, 4usize);
        let (num_blocks, sk) = (8usize, 8usize); // sk=8 → 2 blocks
        let scale = 1.0f32 / (d as f32).sqrt();
        let dev = Device::cpu();
        let g = KvGeometry {
            n_layers: 1,
            num_blocks,
            block_size,
            n_kv_heads: hkv,
            head_dim: d,
            elem_size: 4,
        };
        let mut pool = DeviceKvPool::new(g, DType::F32, &dev).unwrap();

        // A fills 2 blocks with known K/V.
        let a = pool.core_mut().open();
        pool.core_mut().append(a, sk).unwrap();
        let k_all = rand_f32(sk * hkv * d, 2);
        let v_all = rand_f32(sk * hkv * d, 3);
        let per_block = block_size * hkv * d;
        let a_blocks = pool.core().session_block_table(a).unwrap();
        for (bi, &p) in a_blocks.iter().enumerate() {
            pool.write_block(
                0,
                BlockKind::K,
                p,
                &k_all[bi * per_block..(bi + 1) * per_block],
            )
            .unwrap();
            pool.write_block(
                0,
                BlockKind::V,
                p,
                &v_all[bi * per_block..(bi + 1) * per_block],
            )
            .unwrap();
        }

        // B splices A's 2 blocks — shared, not copied.
        let b = pool.core_mut().open();
        pool.core_mut().splice(a, b, 0, 2).unwrap();
        assert_eq!(
            pool.core().session_block_table(b).unwrap(),
            a_blocks,
            "B shares A's exact blocks"
        );
        assert_eq!(
            pool.core().block_refcount(a_blocks[0]),
            2,
            "shared → refcount 2"
        );

        // A query for B attends over the shared prefix (context_len = B's filled = 8).
        let pt = pool.materialize_block_table(&[b]).unwrap();
        assert_eq!(
            pt.context_lens,
            vec![sk as u32],
            "spliced prefix length carried to B"
        );
        let q_data = rand_f32(hq * d, 1);
        let q = Tensor::from_f32(q_data.clone(), Shape::from_dims(&[1, hq, 1, d]), &dev);
        let kph = q.const_placeholder_like(pool.pool_shape().clone(), DType::F32);
        let vph = q.const_placeholder_like(pool.pool_shape().clone(), DType::F32);
        let bt = q.const_u32_like(pt.block_table.clone(), pt.block_table_shape());
        let cl = q.const_u32_like(pt.context_lens.clone(), pt.context_lens_shape());
        let out = q
            .paged_attn(&kph, &vph, &bt, &cl, None, scale, block_size, None)
            .unwrap();
        let mut cache = StorageCache::new();
        cache.insert(kph.node_id(), Arc::clone(pool.k_pool(0).unwrap()));
        cache.insert(vph.node_id(), Arc::clone(pool.v_pool(0).unwrap()));
        let got = crate::pipelined_bridge::realize_one_as_with_initial::<f32>(
            out.graph_handle(),
            out.node_id(),
            &dev,
            cache,
        )
        .unwrap();

        // Dense reference: q attends to A's sk shared tokens.
        let mut want = vec![0.0f32; hq * d];
        for h in 0..hq {
            let mut scores = vec![0.0f32; sk];
            for (t, sc) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q_data[h * d + dd] * k_all[(t * hkv + h) * d + dd];
                }
                *sc = dot * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                denom += *sc;
            }
            for (t, &p) in scores.iter().enumerate() {
                let w = p / denom;
                for dd in 0..d {
                    want[h * d + dd] += w * v_all[(t * hkv + h) * d + dd];
                }
            }
        }
        for (i, (&x, &y)) in got.iter().zip(want.iter()).enumerate() {
            let diff = (x - y).abs();
            let den = x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
            assert!(
                diff < 1e-5 || diff / den < 1e-5,
                "[{i}]: spliced={x} dense={y} (abs={diff})"
            );
        }
    }

    /// The copy-on-write guard (regression for the adversarial-verification
    /// finding): `ensure_writable_block` on a SHARED (spliced) block breaks the
    /// share into a fresh private block holding a COPY of the donor's bytes,
    /// drops the donor's refcount, and a subsequent write to the new block does
    /// NOT touch the donor. Without this, decoding a spliced session corrupts its
    /// co-sharers.
    #[test]
    fn ensure_writable_block_copy_on_writes_a_shared_block() {
        let (hkv, d, block_size) = (2usize, 4usize, 4usize);
        let g = KvGeometry {
            n_layers: 1,
            num_blocks: 8,
            block_size,
            n_kv_heads: hkv,
            head_dim: d,
            elem_size: 4,
        };
        let mut pool = DeviceKvPool::new(g, DType::F32, &Device::cpu()).unwrap();
        let block_elems = block_size * hkv * d;

        // A fills one block with known K.
        let a = pool.core_mut().open();
        pool.core_mut().append(a, block_size).unwrap();
        let a_phys = pool.core().resident_block(a, 0).unwrap();
        let a_data: Vec<f32> = (0..block_elems).map(|i| i as f32 + 1.0).collect();
        pool.write_block(0, BlockKind::K, a_phys, &a_data).unwrap();

        // B splices A's block (shared, refcount 2).
        let b = pool.core_mut().open();
        pool.core_mut().splice(a, b, 0, 1).unwrap();
        assert_eq!(
            pool.core().resident_block(b, 0),
            Some(a_phys),
            "B shares A's block"
        );
        assert_eq!(pool.core().block_refcount(a_phys), 2);

        // CoW: B gets a fresh private block holding a COPY of A's bytes.
        let b_phys = pool.ensure_writable_block(b, 0).unwrap();
        assert_ne!(b_phys, a_phys, "B got a fresh private block");
        assert_eq!(
            pool.core().block_refcount(a_phys),
            1,
            "A's block no longer shared"
        );
        assert_eq!(
            pool.read_block(0, BlockKind::K, b_phys).unwrap(),
            a_data,
            "B's block is a copy of A's"
        );

        // Writing B's copy must not touch A's block.
        pool.write_block(0, BlockKind::K, b_phys, &vec![-1.0; block_elems])
            .unwrap();
        assert_eq!(
            pool.read_block(0, BlockKind::K, a_phys).unwrap(),
            a_data,
            "A's block intact after B writes its own copy",
        );

        // An exclusive block is returned unchanged (no needless copy).
        assert_eq!(
            pool.ensure_writable_block(a, 0).unwrap(),
            a_phys,
            "exclusive block: no CoW"
        );
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
