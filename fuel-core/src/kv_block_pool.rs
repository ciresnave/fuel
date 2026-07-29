//! KV block-pool allocator — the host-side mechanism behind `Op::PagedAttn`
//! (multi-session serving, Increment 2).
//!
//! `Op::PagedAttn` already defines the graph-side paged substrate: physical K/V
//! caches shaped `[num_blocks, block_size, Hkv, D]` and a `block_table`
//! `[B, max_blocks]` (u32) mapping logical → physical blocks. What was absent
//! (ROADMAP §4) is the **host-side manager** that owns the physical pool, hands
//! out blocks, refcounts them (for copy-on-write splice), builds each session's
//! block table, and externalizes state. This module is that manager.
//!
//! ## Contract placement ([15-consumer-contract](../../docs/architecture/15-consumer-contract.md))
//!
//! This is **mechanism, not policy** — Fuel advertises capacity and provides
//! evict/restore/splice; the *consumer* decides admission, which session to
//! evict, and when to splice. The allocator delivers, as mechanism:
//!
//! - **C-1 (capacity advertisement):** [`KvBlockPool::free_blocks`] +
//!   [`KvBlockPool::blocks_required`] — the consumer answers "will this session
//!   fit?" itself, rather than discovering it via an OOM on construction.
//! - **C-3 (state externalization), the *lossy* arm:** [`KvBlockPool::evict`] /
//!   [`KvBlockPool::restore`] / [`KvBlockPool::discard`] + the cross-branch
//!   [`KvBlockPool::splice`] (refcounted copy-on-write).
//! - **C-4 (measured cost), one bite:** [`KvBlockPool::kv_bytes_resident`].
//!
//! ## C-3 fidelity (Q9, settled)
//!
//! State externalization has a fidelity axis. This module implements
//! [`Fidelity::Lossy`] (KV blocks, recomputable from tokens). The single
//! load-bearing rule that keeps the future [`Fidelity::Exact`] arm (training /
//! RL state — params + optimizer moments + RNG stream position) expressible:
//! **[`restore`](KvBlockPool::restore) takes externalized *state* (an opaque
//! [`Externalized`] handle), NEVER a "recompute-from-tokens" instruction.** Bake
//! the inference recompute path into the signature and the exact arm can never
//! be built.
//!
//! **Exact-arm completeness gate (specified now; enforced by the later
//! increment):** a `Fidelity::Exact` restore MUST diverge from an uninterrupted
//! run by *exactly zero*, and its handle MUST enumerate everything it covers
//! (see [`Externalized::covers`]) — Fuel-owned RNG stream position, cached plan,
//! captured run, any backend-side accumulator. Anything outside the handle makes
//! "exact" a silent lie. **The RNG coverage is bounded to Fuel-owned stochastic
//! state** (training-side sampling, dropout) — *never* the consumer's sampler
//! RNG, which the consumer owns and re-seeds itself (a real inference consumer,
//! Lightbulb, re-seeds its `StdRng` per sample call). Over-scoping the handle
//! into consumer sampler state is as wrong as under-scoping Fuel's. This
//! increment ships only `Lossy` but shapes the interface to pass that gate
//! unchanged (Exact is gated on the RNG/generator seam).
//!
//! ## Single-geometry boundary
//!
//! A pool is **single-geometry** (one `block_size` / head config), matching
//! `Op::PagedAttn`'s single block size. This does **not** foreclose compressed /
//! heterogeneous KV: the geometry is a *parameter*, so a compressed-KV session
//! uses a pool sized for its geometry. Per-session heterogeneous sizing *within
//! one pool* is a separate mechanism (multiple pools / a future variant), never
//! baked into the contract here. Where there can be N pools, admission is a
//! cross-pool question — see [`PoolCapacity`].
//!
//! ## Scope
//!
//! Pure host-side block-metadata logic (free list, refcounts, block tables) — no
//! device, no tensors, no model. K/V byte movement (evict/restore actually
//! copying block contents device↔host) and materializing the u32 `block_table`
//! tensor for `Op::PagedAttn` are the thin device-backed integration layer, built
//! on top of this once the core is green.

use std::collections::HashMap;

/// Opaque physical block id (index into the `[num_blocks, …]` pool).
pub type PhysBlockId = u32;

/// Opaque per-session handle minted by [`KvBlockPool::open`]. The allocator owns
/// its own keyspace so it never depends on a consumer's session identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SessionHandle(u64);

/// Fidelity of a [`Fidelity::Lossy`] KV externalization vs the future
/// [`Fidelity::Exact`] training/RL arm. See the module docs (Q9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fidelity {
    /// Recomputable-from-source state; cheapness dominates. This increment.
    Lossy,
    /// Bit-identical-including-RNG state. Later increment (RNG-seam-gated).
    Exact,
}

/// A category of engine-held state a handle covers. For `Lossy` KV the only
/// entry is [`StateKind::KvBlocks`]; the enum exists so the future `Exact` arm's
/// handle can *enumerate* its coverage (the completeness gate) rather than
/// silently omit RNG/plan/capture state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateKind {
    KvBlocks,
    // Exact-arm entries (RngStream, CachedPlan, CapturedRun, …) land with that
    // increment; enumerated here so "what's covered" is never implicit.
}

/// KV pool geometry — the model-agnostic shape the pool manages. It speaks KV
/// block geometry and **never a model**, which is what keeps the eventual
/// `fuel-inference` move cheap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvGeometry {
    /// Total physical blocks in the pool.
    pub num_blocks: usize,
    /// Tokens per block.
    pub block_size: usize,
    /// KV heads.
    pub n_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Bytes per KV element (e.g. 2 for bf16/f16, 4 for f32).
    pub elem_size: usize,
}

impl KvGeometry {
    /// Bytes one physical block occupies across BOTH K and V.
    fn bytes_per_block(&self) -> u64 {
        (self.block_size * self.n_kv_heads * self.head_dim * 2 * self.elem_size) as u64
    }
}

/// A pool's capacity (C-1), **keyed by geometry** so a consumer holding N pools
/// (one per geometry) can compare "admit into pool A / admit into pool B / grow a
/// pool" without reimplementing block math. Returning a descriptor here rather
/// than a bare `usize` is what keeps a future multi-pool world from forcing a
/// breaking change on the one API the consumer builds admission on.
///
/// Cross-pool arbitration and the **unpooled device-memory remainder** (a
/// consumer can be out of blocks in every pool yet have free VRAM — pools
/// fragment memory) are a higher layer: a future pool-set manager over N pools +
/// the backend's free-memory query. NOT this single pool's job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolCapacity {
    pub geometry: KvGeometry,
    pub free_blocks: usize,
    pub total_blocks: usize,
}

/// One logical slot in a session's block table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    /// Backed by a resident physical block.
    Resident(PhysBlockId),
    /// Externalized by [`KvBlockPool::evict`] — its physical block was returned
    /// to the pool; contents live in the [`Externalized`] handle until
    /// [`KvBlockPool::restore`] re-materializes it.
    Externalized,
}

/// Per-session logical→physical block map + fill level.
#[derive(Clone, Debug)]
struct BlockTable {
    slots: Vec<Slot>,
    filled_tokens: usize,
}

/// What [`KvBlockPool::evict`] actually did — honest counts so the consumer's
/// admission math never drifts. **`freed` is what was actually detached, not
/// what was requested**: blocks shared with another session (refcount > 1) are
/// never touched (that would corrupt the sharer), so a heavily-spliced session
/// may free little and report the rest as `still_shared`.
#[derive(Debug)]
pub struct EvictReport {
    /// Physical blocks actually returned to the pool (were refcount == 1).
    pub freed: usize,
    /// Blocks left in place because another session still references them.
    pub still_shared: usize,
    /// The reversible handle for [`KvBlockPool::restore`].
    pub handle: Externalized,
}

/// An opaque, reversible externalization of a session's evictable state. In the
/// pure core it carries the logical structure; the device-backed integration
/// carries the block bytes keyed by the same handle. **Never carries a
/// "recompute" instruction** — that is what keeps `Fidelity::Exact` expressible.
#[derive(Debug)]
pub struct Externalized {
    fidelity: Fidelity,
    covers: Vec<StateKind>,
    filled_tokens: usize,
    /// Logical slot indices that were externalized (need re-materializing).
    externalized_slots: Vec<usize>,
    /// Logical slot indices still backed by a resident (shared) block, and the
    /// physical block they point at — restored by re-reference, not re-alloc.
    resident_slots: Vec<(usize, PhysBlockId)>,
}

impl Externalized {
    /// The categories of state this handle covers. For `Lossy` KV this is
    /// exactly `[KvBlocks]`; the `Exact` arm's completeness gate requires this
    /// to enumerate every covered category so nothing is silently omitted.
    pub fn covers(&self) -> &[StateKind] {
        &self.covers
    }
    /// This handle's fidelity guarantee.
    pub fn fidelity(&self) -> Fidelity {
        self.fidelity
    }
}

/// Errors from the allocator. Never panics on the production path (C-1 is the
/// consumer's pre-check; these surface a mis-sequenced call as a typed error).
#[derive(Debug, PartialEq, Eq)]
pub enum KvAllocError {
    /// Not enough free blocks to satisfy the request. `need`/`have` let the
    /// consumer size its load-shedding. This is *not* the intended admission
    /// path — the consumer should ask [`KvBlockPool::blocks_required`] +
    /// [`KvBlockPool::free_blocks`] first.
    OutOfBlocks { need: usize, have: usize },
    /// The session handle isn't open.
    UnknownSession,
    /// A splice referenced a logical block range that doesn't exist / isn't
    /// resident in the source.
    BadSpliceRange,
}

/// The host-side KV block-pool allocator. See the module docs.
pub struct KvBlockPool {
    geom: KvGeometry,
    /// Free physical block ids (LIFO; order is irrelevant to correctness).
    free: Vec<PhysBlockId>,
    /// `refcount[p]` = number of session slots referencing physical block `p`.
    /// `0` ⇒ free.
    refcount: Vec<u32>,
    tables: HashMap<SessionHandle, BlockTable>,
    next_session: u64,
}

fn blocks_for(tokens: usize, block_size: usize) -> usize {
    tokens.div_ceil(block_size)
}

impl KvBlockPool {
    /// Build a pool over `num_blocks` physical blocks of the given geometry.
    pub fn new(geom: KvGeometry) -> Self {
        let n = geom.num_blocks;
        Self {
            geom,
            free: (0..n as PhysBlockId).rev().collect(),
            refcount: vec![0; n],
            tables: HashMap::new(),
            next_session: 0,
        }
    }

    /// Pool geometry.
    pub fn geometry(&self) -> KvGeometry {
        self.geom
    }

    // --- C-1: capacity advertisement -------------------------------------

    /// Physical blocks currently available. A convenience over
    /// [`capacity`](Self::capacity); prefer `capacity()` for admission so the
    /// query stays composable across N pools (see [`PoolCapacity`]).
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// This pool's capacity descriptor (C-1), geometry-keyed and composable
    /// across pools — the admission primitive. See [`PoolCapacity`].
    pub fn capacity(&self) -> PoolCapacity {
        PoolCapacity {
            geometry: self.geom,
            free_blocks: self.free.len(),
            total_blocks: self.geom.num_blocks,
        }
    }

    /// Blocks a session currently holding `cur_filled` tokens needs in order to
    /// admit `add_tokens` more. Fresh session ⇒ `blocks_required(0, total)`.
    /// The consumer asks this against [`free_blocks`](Self::free_blocks) to
    /// decide admission — so its block math matches the pool's exactly, never
    /// reimplemented and disagreeing at the boundary.
    pub fn blocks_required(&self, cur_filled: usize, add_tokens: usize) -> usize {
        let bs = self.geom.block_size;
        blocks_for(cur_filled + add_tokens, bs) - blocks_for(cur_filled, bs)
    }

    /// Batch admissibility helper (C-1, pure convenience): the total blocks
    /// needed to admit a whole batch, one `(cur_filled, add_tokens)` per
    /// sequence. The consumer compares this to [`free_blocks`](Self::free_blocks)
    /// — the **admit decision stays the consumer's** (policy, e.g. the uniformity
    /// gate); this only keeps the summation in one place, so it stays correct if
    /// geometry ever makes the per-sequence cost non-additive. Requested by the
    /// first real consumer's batched-decode admit path.
    pub fn blocks_required_batch(&self, seqs: &[(usize, usize)]) -> usize {
        seqs.iter().map(|&(filled, add)| self.blocks_required(filled, add)).sum()
    }

    // --- C-4: one measured-cost bite -------------------------------------

    /// Resident KV bytes across K + V — the consumer's budget signal. Shared
    /// blocks are counted **once** (they are one physical block), so this is
    /// true resident memory, not summed-per-session.
    pub fn kv_bytes_resident(&self) -> u64 {
        let used = (self.geom.num_blocks - self.free.len()) as u64;
        used * self.geom.bytes_per_block()
    }

    // --- session lifecycle ------------------------------------------------

    /// Register a new empty session; returns its handle.
    pub fn open(&mut self) -> SessionHandle {
        let h = SessionHandle(self.next_session);
        self.next_session += 1;
        self.tables.insert(h, BlockTable { slots: Vec::new(), filled_tokens: 0 });
        h
    }

    /// Grow a session by `add_tokens`, allocating physical blocks as needed.
    /// Returns `Err(OutOfBlocks)` if the pool can't satisfy it (the consumer
    /// should have pre-checked via C-1).
    pub fn append(&mut self, s: SessionHandle, add_tokens: usize) -> Result<(), KvAllocError> {
        let bs = self.geom.block_size;
        let (cur_filled, cur_blocks) = {
            let t = self.tables.get(&s).ok_or(KvAllocError::UnknownSession)?;
            (t.filled_tokens, t.slots.len())
        };
        let want_blocks = blocks_for(cur_filled + add_tokens, bs);
        let new_blocks = want_blocks.saturating_sub(cur_blocks);
        if new_blocks > self.free.len() {
            return Err(KvAllocError::OutOfBlocks { need: new_blocks, have: self.free.len() });
        }
        for _ in 0..new_blocks {
            let p = self.free.pop().expect("checked free ≥ new_blocks");
            self.refcount[p as usize] = 1;
            self.tables.get_mut(&s).unwrap().slots.push(Slot::Resident(p));
        }
        self.tables.get_mut(&s).unwrap().filled_tokens = cur_filled + add_tokens;
        Ok(())
    }

    /// Number of logical blocks a session holds (resident + externalized).
    pub fn session_blocks(&self, s: SessionHandle) -> Option<usize> {
        self.tables.get(&s).map(|t| t.slots.len())
    }

    /// Physical block backing logical slot `i`, if resident (test/integration
    /// accessor; the integration reads/writes the pool buffer at this offset).
    pub fn resident_block(&self, s: SessionHandle, i: usize) -> Option<PhysBlockId> {
        match self.tables.get(&s)?.slots.get(i)? {
            Slot::Resident(p) => Some(*p),
            Slot::Externalized => None,
        }
    }

    /// Refcount of a physical block (test/introspection).
    pub fn block_refcount(&self, p: PhysBlockId) -> u32 {
        self.refcount[p as usize]
    }

    // --- C-3: state externalization (lossy) ------------------------------

    /// Evict a session's **exclusively-held** (refcount == 1) blocks, returning
    /// them to the pool and recording them in a reversible [`Externalized`]
    /// handle. **Blocks shared with another session are never touched** — that
    /// would corrupt the sharer — and are reported via
    /// [`EvictReport::still_shared`]. `freed` is the *actual* count detached.
    pub fn evict(&mut self, s: SessionHandle) -> Result<EvictReport, KvAllocError> {
        let slots = self.tables.get(&s).ok_or(KvAllocError::UnknownSession)?.slots.clone();
        let filled = self.tables[&s].filled_tokens;
        let mut externalized_slots = Vec::new();
        let mut resident_slots = Vec::new();
        let mut freed = 0;
        let mut still_shared = 0;
        for (i, slot) in slots.iter().enumerate() {
            match slot {
                Slot::Resident(p) => {
                    let p = *p;
                    if self.refcount[p as usize] == 1 {
                        // Exclusive → detach: return to pool, mark externalized.
                        self.refcount[p as usize] = 0;
                        self.free.push(p);
                        self.tables.get_mut(&s).unwrap().slots[i] = Slot::Externalized;
                        externalized_slots.push(i);
                        freed += 1;
                    } else {
                        // Shared → leave it; someone else references it.
                        resident_slots.push((i, p));
                        still_shared += 1;
                    }
                }
                Slot::Externalized => externalized_slots.push(i),
            }
        }
        Ok(EvictReport {
            freed,
            still_shared,
            handle: Externalized {
                fidelity: Fidelity::Lossy,
                covers: vec![StateKind::KvBlocks],
                filled_tokens: filled,
                externalized_slots,
                resident_slots,
            },
        })
    }

    /// Re-materialize an evicted session from its handle: allocate fresh
    /// physical blocks for the externalized slots (content copy-back is the
    /// integration layer's job), re-reference the still-resident shared blocks.
    /// Returns the restored session handle.
    pub fn restore(&mut self, s: SessionHandle, handle: Externalized) -> Result<(), KvAllocError> {
        if !self.tables.contains_key(&s) {
            return Err(KvAllocError::UnknownSession);
        }
        let need = handle.externalized_slots.len();
        if need > self.free.len() {
            return Err(KvAllocError::OutOfBlocks { need, have: self.free.len() });
        }
        for &i in &handle.externalized_slots {
            let p = self.free.pop().expect("checked");
            self.refcount[p as usize] = 1;
            let t = self.tables.get_mut(&s).unwrap();
            if i >= t.slots.len() {
                t.slots.resize(i + 1, Slot::Externalized);
            }
            t.slots[i] = Slot::Resident(p);
        }
        for &(i, p) in &handle.resident_slots {
            let t = self.tables.get_mut(&s).unwrap();
            if i >= t.slots.len() {
                t.slots.resize(i + 1, Slot::Externalized);
            }
            t.slots[i] = Slot::Resident(p);
        }
        self.tables.get_mut(&s).unwrap().filled_tokens = handle.filled_tokens;
        Ok(())
    }

    /// Irreversibly drop a session and free its blocks now (C-3 discard — the
    /// consumer is dropping the session and will re-prefill from tokens if it
    /// ever wants it back; no restore path, no bytes retained). Also the normal
    /// end-of-session close.
    pub fn discard(&mut self, s: SessionHandle) {
        if let Some(t) = self.tables.remove(&s) {
            for slot in t.slots {
                if let Slot::Resident(p) = slot {
                    let rc = &mut self.refcount[p as usize];
                    *rc -= 1;
                    if *rc == 0 {
                        self.free.push(p);
                    }
                }
            }
        }
    }

    // --- C-3: cross-branch splice (copy-on-write) ------------------------

    /// Share `src`'s resident blocks over logical range `[from, to)` into `dst`
    /// (appended to `dst`'s table) copy-on-write: refcounts bump, both point at
    /// the same physical blocks. A later write to a shared block must go through
    /// [`cow_break`](Self::cow_break) first. This is the "parallel trains of
    /// thought" mechanism — the consumer decides *when* to splice.
    pub fn splice(
        &mut self,
        src: SessionHandle,
        dst: SessionHandle,
        from: usize,
        to: usize,
    ) -> Result<(), KvAllocError> {
        if !self.tables.contains_key(&dst) {
            return Err(KvAllocError::UnknownSession);
        }
        let (shared, src_filled) = {
            let src_t = self.tables.get(&src).ok_or(KvAllocError::UnknownSession)?;
            if from > to || to > src_t.slots.len() {
                return Err(KvAllocError::BadSpliceRange);
            }
            let mut shared = Vec::with_capacity(to - from);
            for slot in &src_t.slots[from..to] {
                match slot {
                    Slot::Resident(p) => shared.push(*p),
                    Slot::Externalized => return Err(KvAllocError::BadSpliceRange),
                }
            }
            (shared, src_t.filled_tokens)
        };
        // Tokens the shared block range actually carries — the last shared block
        // may be partially filled in `src`, so clamp to `src`'s fill; this keeps
        // `dst`'s fill level consistent with its block count for later `append`.
        let bs = self.geom.block_size;
        let shared_tokens = ((to - from) * bs).min(src_filled.saturating_sub(from * bs));
        for p in shared {
            self.refcount[p as usize] += 1;
            self.tables.get_mut(&dst).unwrap().slots.push(Slot::Resident(p));
        }
        self.tables.get_mut(&dst).unwrap().filled_tokens += shared_tokens;
        Ok(())
    }

    /// Break the copy-on-write share on logical slot `i` of session `s` before a
    /// write: if the block is shared (refcount > 1), allocate a fresh block for
    /// `s` (content copy is the integration's job), drop `s`'s reference to the
    /// shared original, and point the slot at the fresh block. Returns the
    /// physical block the integration should write into. A no-op (returns the
    /// existing block) when already exclusive.
    pub fn cow_break(&mut self, s: SessionHandle, i: usize) -> Result<PhysBlockId, KvAllocError> {
        let p = match self.tables.get(&s).ok_or(KvAllocError::UnknownSession)?.slots.get(i) {
            Some(Slot::Resident(p)) => *p,
            _ => return Err(KvAllocError::BadSpliceRange),
        };
        if self.refcount[p as usize] <= 1 {
            return Ok(p); // exclusive — no break needed
        }
        if self.free.is_empty() {
            return Err(KvAllocError::OutOfBlocks { need: 1, have: 0 });
        }
        let q = self.free.pop().unwrap();
        self.refcount[q as usize] = 1;
        self.refcount[p as usize] -= 1;
        self.tables.get_mut(&s).unwrap().slots[i] = Slot::Resident(q);
        Ok(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom(num_blocks: usize, block_size: usize) -> KvGeometry {
        KvGeometry { num_blocks, block_size, n_kv_heads: 2, head_dim: 4, elem_size: 2 }
    }

    /// THE HAZARD (peer-flagged, born-red before the design set): a naive
    /// "detach all of a session's blocks" evict would corrupt a session that
    /// splice-shares those blocks. Refcount-aware evict must NEVER touch a
    /// shared block, must free only the exclusive tail, and must honestly report
    /// `still_shared` so the consumer's admission math stays exact.
    #[test]
    fn evict_of_spliced_session_does_not_corrupt_sharer() {
        let mut pool = KvBlockPool::new(geom(16, 4));
        let a = pool.open();
        // A holds 3 blocks (9 tokens over block_size 4 → 3 blocks).
        pool.append(a, 9).unwrap();
        assert_eq!(pool.session_blocks(a), Some(3));
        let (p0, p1, p2) = (
            pool.resident_block(a, 0).unwrap(),
            pool.resident_block(a, 1).unwrap(),
            pool.resident_block(a, 2).unwrap(),
        );

        // B shares A's first two blocks (the common prefix); p0,p1 → refcount 2.
        let b = pool.open();
        pool.splice(a, b, 0, 2).unwrap();
        assert_eq!(pool.block_refcount(p0), 2);
        assert_eq!(pool.block_refcount(p1), 2);
        assert_eq!(pool.block_refcount(p2), 1);
        let free_before = pool.free_blocks();

        // Evict A. Only p2 (exclusive) is detachable; p0,p1 are shared → kept.
        let rep = pool.evict(a).unwrap();
        assert_eq!(rep.freed, 1, "only the exclusive block frees");
        assert_eq!(rep.still_shared, 2, "the two shared blocks are honestly reported");
        assert_eq!(pool.free_blocks(), free_before + 1, "exactly one block returned");

        // The sharer B is intact: its blocks still resolve to the SAME physical
        // blocks, which are still allocated (refcount dropped 2→1, not freed).
        assert_eq!(pool.resident_block(b, 0), Some(p0), "B's shared prefix intact");
        assert_eq!(pool.resident_block(b, 1), Some(p1));
        // A retains its refs on the shared blocks: evict is PARTIAL for shared
        // blocks — they aren't A's alone to reclaim, and the Q9 self-contained-
        // restore rule forbids copying a shared block's bytes into A's handle. So
        // p0/p1 stay allocated at refcount 2 (A + B). A naive detach-all evict
        // would drop them to 0 and free them out from under B — the hazard.
        assert_eq!(pool.block_refcount(p0), 2, "still shared by A and B, not freed");
        assert_eq!(pool.block_refcount(p1), 2);
        // The freed block p2 is genuinely reusable and not referenced by B.
        assert_ne!(pool.resident_block(b, 0), Some(p2));
        assert_ne!(pool.resident_block(b, 1), Some(p2));
    }

    #[test]
    fn capacity_is_geometry_keyed_and_tracks_the_free_list() {
        let mut pool = KvBlockPool::new(geom(10, 4));
        let cap = pool.capacity();
        assert_eq!(cap.geometry, geom(10, 4), "geometry-keyed for cross-pool admission");
        assert_eq!(cap.total_blocks, 10);
        assert_eq!(cap.free_blocks, 10);
        let s = pool.open();
        pool.append(s, 10).unwrap(); // 3 blocks
        assert_eq!(pool.capacity().free_blocks, 7);
        assert_eq!(pool.capacity().total_blocks, 10, "total is fixed");
    }

    #[test]
    fn append_free_blocks_and_blocks_required_agree() {
        let mut pool = KvBlockPool::new(geom(10, 4));
        assert_eq!(pool.free_blocks(), 10);
        // A fresh session needs ceil(10/4)=3 blocks for 10 tokens.
        assert_eq!(pool.blocks_required(0, 10), 3);
        let s = pool.open();
        pool.append(s, 10).unwrap();
        assert_eq!(pool.free_blocks(), 7);
        // Growing from 10 by 3 tokens: 10→13 spans ceil(13/4)-ceil(10/4)=4-3=1.
        assert_eq!(pool.blocks_required(10, 3), 1);
        pool.append(s, 3).unwrap();
        assert_eq!(pool.free_blocks(), 6);
    }

    #[test]
    fn blocks_required_batch_sums_per_sequence() {
        let pool = KvBlockPool::new(geom(100, 4));
        // Three fresh 10-token sequences: ceil(10/4)=3 blocks each → 9.
        assert_eq!(pool.blocks_required_batch(&[(0, 10), (0, 10), (0, 10)]), 9);
        // Mixed grow: (5,+3) needs ceil(8/4)-ceil(5/4)=0; (0,+8) needs 2 → 2.
        assert_eq!(pool.blocks_required_batch(&[(5, 3), (0, 8)]), 2);
        // Empty batch admits with zero blocks.
        assert_eq!(pool.blocks_required_batch(&[]), 0);
    }

    #[test]
    fn over_append_is_a_typed_error_never_a_panic() {
        let mut pool = KvBlockPool::new(geom(2, 4));
        let s = pool.open();
        let need = pool.blocks_required(0, 100); // ceil(100/4) = 25
        assert!(need > pool.free_blocks());
        let err = pool.append(s, 100).unwrap_err();
        assert!(matches!(err, KvAllocError::OutOfBlocks { .. }));
        // Nothing was partially allocated.
        assert_eq!(pool.free_blocks(), 2);
        assert_eq!(pool.session_blocks(s), Some(0));
    }

    #[test]
    fn evict_then_restore_round_trips_the_structure() {
        let mut pool = KvBlockPool::new(geom(16, 4));
        let s = pool.open();
        pool.append(s, 10).unwrap(); // 3 blocks, all exclusive
        assert_eq!(pool.session_blocks(s), Some(3));
        let free_after_alloc = pool.free_blocks();

        let rep = pool.evict(s).unwrap();
        assert_eq!(rep.freed, 3, "all exclusive → all freed");
        assert_eq!(rep.still_shared, 0);
        assert_eq!(pool.free_blocks(), free_after_alloc + 3);
        assert_eq!(rep.handle.fidelity(), Fidelity::Lossy);
        assert_eq!(rep.handle.covers(), &[StateKind::KvBlocks]);

        pool.restore(s, rep.handle).unwrap();
        assert_eq!(pool.session_blocks(s), Some(3), "structure restored");
        assert!(pool.resident_block(s, 0).is_some());
        assert!(pool.resident_block(s, 2).is_some());
        assert_eq!(pool.free_blocks(), free_after_alloc, "3 re-allocated");
    }

    #[test]
    fn discard_frees_irreversibly_and_reclaims() {
        let mut pool = KvBlockPool::new(geom(8, 4));
        let s = pool.open();
        pool.append(s, 8).unwrap(); // 2 blocks
        assert_eq!(pool.free_blocks(), 6);
        pool.discard(s);
        assert_eq!(pool.free_blocks(), 8, "all reclaimed");
        assert_eq!(pool.session_blocks(s), None, "session gone");
    }

    #[test]
    fn discard_of_a_sharer_keeps_the_other_sessions_blocks() {
        let mut pool = KvBlockPool::new(geom(8, 4));
        let a = pool.open();
        pool.append(a, 8).unwrap(); // p0,p1
        let (p0, p1) = (pool.resident_block(a, 0).unwrap(), pool.resident_block(a, 1).unwrap());
        let b = pool.open();
        pool.splice(a, b, 0, 2).unwrap();
        let free_before = pool.free_blocks();
        pool.discard(a); // A gone, but B still references p0,p1
        assert_eq!(pool.free_blocks(), free_before, "shared blocks NOT freed — B holds them");
        assert_eq!(pool.resident_block(b, 0), Some(p0));
        assert_eq!(pool.resident_block(b, 1), Some(p1));
        assert_eq!(pool.block_refcount(p0), 1);
    }

    #[test]
    fn cow_break_gives_a_fresh_block_and_leaves_the_sharer_unchanged() {
        let mut pool = KvBlockPool::new(geom(8, 4));
        let a = pool.open();
        pool.append(a, 4).unwrap(); // p0
        let p0 = pool.resident_block(a, 0).unwrap();
        let b = pool.open();
        pool.splice(a, b, 0, 1).unwrap(); // B shares p0 (rc 2)
        assert_eq!(pool.block_refcount(p0), 2);

        // B is about to write its slot 0 → must break the share first.
        let q = pool.cow_break(b, 0).unwrap();
        assert_ne!(q, p0, "fresh block, not the shared one");
        assert_eq!(pool.resident_block(b, 0), Some(q));
        assert_eq!(pool.resident_block(a, 0), Some(p0), "A (the sharer) unchanged");
        assert_eq!(pool.block_refcount(p0), 1, "back to exclusive for A");
        assert_eq!(pool.block_refcount(q), 1);

        // Breaking an already-exclusive block is a no-op (returns it).
        assert_eq!(pool.cow_break(a, 0).unwrap(), p0);
    }

    #[test]
    fn kv_bytes_resident_counts_shared_blocks_once() {
        let mut pool = KvBlockPool::new(geom(16, 4));
        let per = pool.geometry().bytes_per_block();
        let a = pool.open();
        pool.append(a, 8).unwrap(); // 2 blocks
        assert_eq!(pool.kv_bytes_resident(), 2 * per);
        let b = pool.open();
        pool.splice(a, b, 0, 2).unwrap(); // shares — no new physical blocks
        assert_eq!(pool.kv_bytes_resident(), 2 * per, "sharing adds no resident bytes");
        pool.append(b, 4).unwrap(); // B grows by 1 exclusive block
        assert_eq!(pool.kv_bytes_resident(), 3 * per);
    }
}
