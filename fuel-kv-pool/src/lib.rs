// SPDX-License-Identifier: MIT OR Apache-2.0
//! KV block-pool allocator — the host-side mechanism behind `Op::PagedAttn`
//! (multi-session serving, Increment 2).
//!
//! Extracted from `fuel-core::kv_block_pool` into its own leaf crate
//! (fuel-core dissolution, plan item 9): a pure host-side data structure
//! with zero coupling to `Tensor`/`Device` or any other fuel-internal
//! crate (its only dependency is `std::collections::HashMap`). The
//! original module's proposed destination, `fuel-inference`, does not
//! work for this file — `fuel-model-llama` is a real, substantial
//! consumer (`SessionHandle`/`PhysBlockId`/`KvGeometry`) and is itself a
//! dependency OF `fuel-inference`, so `fuel-inference` can't host the
//! types its own dependency needs without creating a cycle. A standalone
//! leaf crate, upstream of both, has no such constraint.
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
    /// Transformer layers. A physical block addresses the SAME slot in *every*
    /// layer's K/V buffer (the vLLM shared-block-table model: one block table,
    /// physical block `p` = slot `p` in all layers), so the pool's resident bytes
    /// and the device layer's `n_layers × 2` pool buffers both scale with this.
    pub n_layers: usize,
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
    /// Bytes one physical block occupies across ALL layers and BOTH K and V — a
    /// block is a slot in every layer's K buffer and every layer's V buffer.
    fn bytes_per_block(&self) -> u64 {
        (self.n_layers * self.block_size * self.n_kv_heads * self.head_dim * 2 * self.elem_size)
            as u64
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
///
/// **Reserved axis — device residence (multi-device KV).** A pool is
/// single-device today, so this descriptor implicitly belongs to its pool's
/// device. When multi-device KV lands (CireSnave's call, 2026-07-29: it is
/// mechanism and belongs in Fuel — distributed KV = this block pool + *which
/// device a block lives on* + the device↔device transfer path; the
/// replicate-hot/shard-cold *strategy* stays consumer policy), the **device**
/// becomes a second arbitration axis alongside geometry — the pool-set manager
/// arbitrates over N pools keyed by *(device, geometry)*. Whether that surfaces
/// as a per-pool device tag here or a per-block `DeviceLocation` lower down is
/// deferred to that increment's design (the field shape follows the design, not
/// a guess) — the field stays absent until that design lands, per "establish
/// it belongs before building it".
///
/// **Contract requirement both shapes must satisfy (C-5, corrected 2026-07-29
/// §15 v0.7):** device placement is a **per-device budget**, not a device set.
/// Fuel *observes* availability (topology + the memory-pressure/slot/queue
/// telemetry `05-backend-contract` already requires downward); the consumer owns
/// **entitlement** — what it is *permitted* to consume ("don't use more than 50%
/// of the 4070's VRAM"; "GPU 0 drives the display — leave it alone"), which Fuel
/// cannot observe. So residence must be **constrainable from outside** as a
/// budget with two axes — *admissibility* (may this device be used at all) and
/// *quantity* (VRAM fraction/absolute; potentially slot/compute share) — not
/// merely observable. The acceptance test: the descriptor must answer "how MUCH
/// may I use on device X", not just "is X permitted". Headroom then becomes
/// **per-device and budget-relative** — `free_blocks` against a device the
/// consumer capped at 50% reports against the *cap*, not the hardware; a pool
/// budgeted out of a device it could physically fit declines *as though full*,
/// and must distinguish "no room" from "not permitted" (different states a
/// consumer acts on differently). A pool that picks residence *internally* fails
/// this. Directionality: unconstrained = all visible devices at full capacity (a
/// single-process script enumerates nothing); Fuel may narrow for cost, **never**
/// expand past the budget; the consumer may always narrow further. The pool-set
/// manager that arbitrates **fit** is Fuel's; one that arbitrates **priority** is
/// policy.
///
/// **Cross-vendor block movement must author two hops (2026-07-30, seam owner).**
/// Cross-backend `Op::Copy` is solved *for optimizer-inserted transfers* — the
/// `insert_cross_device_copies` pass splits e.g. CUDA→Vulkan into CUDA→CPU→Vulkan
/// (host-staged, CSE-shared), byte-exact-tested on CPU+CUDA+Vulkan
/// (`cuda_vulkan_multidevice_realize_live.rs`). But that split is deliberately NOT
/// applied to a consumer that is itself a Copy/Move (`fuel-graph/opt.rs` skips
/// them — infinite-regress guard), so a *hand-authored* cross-vendor
/// `copy_to_device` is never split and the foreign-GPU backends reject it. This
/// pool moves bytes DELIBERATELY (evict/restore = D2H/H2D realizes today, all
/// single-hop through the host — safe). When the device coordinate lands, a
/// device↔device block move ACROSS vendors must author both hops itself
/// (dev→CPU→dev), not emit one `Op::Copy` and expect the optimizer to split it.
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
    /// Logical block indices actually detached this call (were refcount == 1).
    /// Per-block rather than a count so a consumer evicting a SET (H2O /
    /// multi-span segmented eviction) can attribute the outcome back to its own
    /// spans: a block that came back in `still_shared` means its span is NOT
    /// fully demoted. `.len()` recovers the count.
    pub freed: Vec<usize>,
    /// Logical block indices left in place because another session still
    /// references them (never freed — that would corrupt the sharer).
    pub still_shared: Vec<usize>,
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
    /// An `evict_blocks`/`evict_range` request named a logical block index
    /// outside the session (nothing is evicted — the call is atomic).
    BadBlockIndex { index: usize, session_blocks: usize },
    /// A device-layer materialization (block-table / byte movement) named a
    /// session whose logical slot `slot` is externalized, not resident — it
    /// must be [`restore`](KvBlockPool::restore)d before it can back a live
    /// `Op::PagedAttn`. Surfaces a mis-sequenced call as a typed error rather
    /// than silently routing attention through a reclaimed physical block.
    SessionNotResident { slot: usize },
    /// [`splice_prefix`](KvBlockPool::splice_prefix) into a non-empty target. A
    /// shared prefix must be the target's FIRST blocks (the underlying `splice`
    /// appends), so a non-empty target is refused — before any mutation.
    PrefixTargetNotEmpty,
    /// [`splice_prefix`](KvBlockPool::splice_prefix) asked to share a prefix
    /// whose last block is only partially filled (`prefix_blocks * block_size >
    /// donor_filled`). Only FULLY-filled whole blocks may be shared, so the
    /// sharer's fill stays block-aligned and its first suffix write lands on a
    /// fresh (unshared) block. Refused before any mutation.
    PrefixNotFullyFilled {
        prefix_blocks: usize,
        donor_filled: usize,
    },
    /// A [`PrefixId`](PrefixId) named a prefix that isn't registered (never
    /// minted, or already [`release_prefix`](KvBlockPool::release_prefix)d).
    UnknownPrefix,
    /// [`alloc_shifted_prefix_slots`](KvBlockPool::alloc_shifted_prefix_slots)
    /// (rung-2) into a target whose fill is NOT block-aligned. A shifted prefix
    /// must land on fresh WHOLE blocks appended after the target's full blocks;
    /// a partial last block would put the prefix mid-block (partial-block merge is
    /// a separate follow-on). Refused before any mutation.
    OffsetNotBlockAligned { filled: usize, block_size: usize },
}

/// A registry-minted handle for a shared prefix owner (a named, refcounted KV
/// prefix — e.g. a shared system prompt computed once and read by many sessions).
/// Its lifetime is controlled by the pool's prefix registry, not by the session
/// that originally computed the prefix: see
/// [`register_prefix`](KvBlockPool::register_prefix).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PrefixId(u64);

/// A registered prefix owner: an internal `SessionHandle` that holds the shared
/// blocks alive (never decodes) plus the block count it owns. The owner's own
/// refcount keeps the prefix resident even after the original donor is discarded.
struct PrefixOwner {
    owner: SessionHandle,
    prefix_blocks: usize,
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
    /// Registered prefix owners, keyed by their minted [`PrefixId`]. Each owner is
    /// a `SessionHandle` holding the shared prefix blocks; the registry — not the
    /// donor — controls when they're released ([`release_prefix`](Self::release_prefix)).
    prefixes: HashMap<PrefixId, PrefixOwner>,
    next_prefix: u64,
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
            prefixes: HashMap::new(),
            next_prefix: 0,
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
        seqs.iter()
            .map(|&(filled, add)| self.blocks_required(filled, add))
            .sum()
    }

    // --- C-4: one measured-cost bite -------------------------------------

    /// Resident KV bytes across K + V — the consumer's C-4 budget signal. Shared
    /// blocks are counted **once** (they are one physical block), so this is true
    /// resident memory, not summed-per-session.
    ///
    /// **`resident` means *bytes physically held by this pool*** — exact today
    /// because the device-tier pool buffers are real VRAM (`resident` ==
    /// `mapped`). If a memory-mapped host tier is ever added (the decided-but-
    /// unbuilt unified durable store — mapping succeeds beyond physical RAM,
    /// residency becomes the kernel's page-cache decision), `resident` and
    /// `mapped` diverge arbitrarily. This name is then **resident** (what C-4
    /// admission must budget on); the mapped count would want a SEPARATE
    /// accessor (`kv_bytes_mapped`), never a meaning-by-tier overload of this one.
    /// Reserving the *distinction* in docs now (renaming a live C-4 surface later
    /// is the expensive version); device-tier accounting is unaffected either way.
    pub fn kv_bytes_resident(&self) -> u64 {
        let used = (self.geom.num_blocks - self.free.len()) as u64;
        used * self.geom.bytes_per_block()
    }

    // --- session lifecycle ------------------------------------------------

    /// Register a new empty session; returns its handle.
    pub fn open(&mut self) -> SessionHandle {
        let h = SessionHandle(self.next_session);
        self.next_session += 1;
        self.tables.insert(
            h,
            BlockTable {
                slots: Vec::new(),
                filled_tokens: 0,
            },
        );
        h
    }

    /// `&mut` access to a session's block table as a typed error rather than a
    /// panic.
    ///
    /// Several call sites re-look-up the table after an earlier
    /// `contains_key` / `get` check and used to `.unwrap()` the result. Those
    /// unwraps were correct — nothing interleaves between the check and the use
    /// — but "correct" there was a property of the current call graph, not of
    /// the function, and this is the multi-agent serving allocator: a panic
    /// here takes down every session in the process, not just the one that
    /// tripped it. The `?` costs nothing and cannot rot.
    fn table_mut(&mut self, s: SessionHandle) -> Result<&mut BlockTable, KvAllocError> {
        self.tables.get_mut(&s).ok_or(KvAllocError::UnknownSession)
    }

    /// Pop one free physical block, or report exhaustion.
    ///
    /// Callers still pre-check `free.len()` when they need *several* blocks so
    /// the error can carry the real `need`; this is the last-mile guard, so a
    /// miscounted pre-check degrades to `OutOfBlocks` instead of panicking
    /// halfway through mutating the pool.
    fn take_free(&mut self) -> Result<PhysBlockId, KvAllocError> {
        self.free
            .pop()
            .ok_or(KvAllocError::OutOfBlocks { need: 1, have: 0 })
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
            return Err(KvAllocError::OutOfBlocks {
                need: new_blocks,
                have: self.free.len(),
            });
        }
        for _ in 0..new_blocks {
            let p = self.take_free()?;
            self.refcount[p as usize] = 1;
            self.table_mut(s)?.slots.push(Slot::Resident(p));
        }
        self.table_mut(s)?.filled_tokens = cur_filled + add_tokens;
        Ok(())
    }

    /// Number of logical blocks a session holds (resident + externalized).
    pub fn session_blocks(&self, s: SessionHandle) -> Option<usize> {
        self.tables.get(&s).map(|t| t.slots.len())
    }

    /// Tokens currently filled in a session — its `Op::PagedAttn`
    /// `context_len`. The device layer reads this to build the `context_lens`
    /// tensor, so it stays the single source of truth (never recomputed from
    /// `session_blocks × block_size`, which would over-count the partial last
    /// block). `None` if the session isn't open.
    pub fn filled_tokens(&self, s: SessionHandle) -> Option<usize> {
        self.tables.get(&s).map(|t| t.filled_tokens)
    }

    /// The session's resident logical→physical block table — the physical id
    /// backing each logical slot, in order. This is exactly what the device
    /// layer flattens into the `Op::PagedAttn` `block_table` row for the
    /// session. `Err(UnknownSession)` if not open; `Err(SessionNotResident)`
    /// if any slot is externalized (the caller must `restore` first — routing
    /// attention through a reclaimed block would be silent corruption).
    pub fn session_block_table(&self, s: SessionHandle) -> Result<Vec<PhysBlockId>, KvAllocError> {
        let t = self.tables.get(&s).ok_or(KvAllocError::UnknownSession)?;
        let mut out = Vec::with_capacity(t.slots.len());
        for (i, slot) in t.slots.iter().enumerate() {
            match slot {
                Slot::Resident(p) => out.push(*p),
                Slot::Externalized => return Err(KvAllocError::SessionNotResident { slot: i }),
            }
        }
        Ok(out)
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

    /// Checks every index in `indices` is within session `s`'s current slot
    /// count. Shared by every entry point that takes caller-supplied block
    /// indices, so the bounds check reads the same way everywhere.
    fn validate_block_indices(
        &self,
        s: SessionHandle,
        indices: &[usize],
    ) -> Result<(), KvAllocError> {
        let n_slots = self
            .tables
            .get(&s)
            .ok_or(KvAllocError::UnknownSession)?
            .slots
            .len();
        for &i in indices {
            if i >= n_slots {
                return Err(KvAllocError::BadBlockIndex {
                    index: i,
                    session_blocks: n_slots,
                });
            }
        }
        Ok(())
    }

    /// Evict a SET of a live session's blocks — the primitive `evict` and
    /// `evict_range` build on. The rest of the session stays resident and keeps
    /// decoding (this is sub-session tiering: shed the cold middle of a long
    /// conversation without dropping it). The block set may be **non-contiguous**
    /// (H2O keeps scattered heavy-hitter blocks; a segmented policy evicts a
    /// union of spans). Refcount-aware PER BLOCK: a block shared with another
    /// session (a spliced prefix) is never detached — that would corrupt the
    /// sharer — and comes back in [`EvictReport::still_shared`]; exclusively-held
    /// blocks are detached into the reversible handle and listed in
    /// [`EvictReport::freed`]. Atomic in its validation: an out-of-range index
    /// evicts nothing and returns [`KvAllocError::BadBlockIndex`].
    pub fn evict_blocks(
        &mut self,
        s: SessionHandle,
        indices: &[usize],
    ) -> Result<EvictReport, KvAllocError> {
        self.validate_block_indices(s, indices)?;
        let mut externalized_slots = Vec::new();
        let mut resident_slots = Vec::new();
        let mut freed = Vec::new();
        let mut still_shared = Vec::new();
        // Owner-only `still_shared` invariant: a registered prefix owner's blocks
        // ARE the shared prefix — eviction must NEVER free them (only
        // `release_prefix` does), even when the owner is their sole reference
        // (refcount 1). Report them `still_shared` instead, so no evict path can
        // silently reclaim a live prefix out from under its consumers. Computed
        // once here (immutable borrow) before the mutating loop.
        let querying_is_prefix_owner = self.is_prefix_owner(s);
        for &i in indices {
            match self.tables[&s].slots[i] {
                Slot::Resident(p) => {
                    if self.refcount[p as usize] == 1 && !querying_is_prefix_owner {
                        // Exclusive → detach: return to pool, mark externalized.
                        self.refcount[p as usize] = 0;
                        self.free.push(p);
                        self.table_mut(s)?.slots[i] = Slot::Externalized;
                        externalized_slots.push(i);
                        freed.push(i);
                    } else {
                        // Shared → leave it; someone else references it.
                        resident_slots.push((i, p));
                        still_shared.push(i);
                    }
                }
                // Already externalized by an earlier partial evict — its bytes
                // live in that call's handle, not this one; a no-op here.
                Slot::Externalized => {}
            }
        }
        Ok(EvictReport {
            freed,
            still_shared,
            handle: Externalized {
                fidelity: Fidelity::Lossy,
                covers: vec![StateKind::KvBlocks],
                externalized_slots,
                resident_slots,
            },
        })
    }

    /// Evict a session's ENTIRE resident state (the whole-session convenience —
    /// part 1's shape, unchanged). Suspends the session; `restore` un-suspends.
    pub fn evict(&mut self, s: SessionHandle) -> Result<EvictReport, KvAllocError> {
        let n = self
            .tables
            .get(&s)
            .ok_or(KvAllocError::UnknownSession)?
            .slots
            .len();
        let all: Vec<usize> = (0..n).collect();
        self.evict_blocks(s, &all)
    }

    /// Evict a contiguous logical block range `[from, to)` of a live session (the
    /// single-span convenience over [`evict_blocks`](Self::evict_blocks)). The
    /// session stays live outside the range.
    pub fn evict_range(
        &mut self,
        s: SessionHandle,
        from: usize,
        to: usize,
    ) -> Result<EvictReport, KvAllocError> {
        if from > to {
            return Err(KvAllocError::BadBlockIndex {
                index: from,
                session_blocks: to,
            });
        }
        let range: Vec<usize> = (from..to).collect();
        self.evict_blocks(s, &range)
    }

    /// Places physical block `p` at logical slot `i` of session `s`'s table,
    /// growing the table if `i` is past its current end. Shared by both
    /// `restore` loops (externalized slots getting a freshly-allocated
    /// block, resident slots getting their already-shared one back) --
    /// same grow-then-set shape either way.
    fn place_resident_slot(
        &mut self,
        s: SessionHandle,
        i: usize,
        p: PhysBlockId,
    ) -> Result<(), KvAllocError> {
        let t = self.table_mut(s)?;
        if i >= t.slots.len() {
            t.slots.resize(i + 1, Slot::Externalized);
        }
        t.slots[i] = Slot::Resident(p);
        Ok(())
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
            return Err(KvAllocError::OutOfBlocks {
                need,
                have: self.free.len(),
            });
        }
        for &i in &handle.externalized_slots {
            let p = self.take_free()?;
            self.refcount[p as usize] = 1;
            self.place_resident_slot(s, i, p)?;
        }
        for &(i, p) in &handle.resident_slots {
            self.place_resident_slot(s, i, p)?;
        }
        // Deliberately does NOT touch the session's `filled_tokens`: `evict`
        // never changes it (a session's logical length is unchanged — only block
        // residency is). A PARTIAL evict of a still-decoding session that grew
        // its fill after the evict must keep the live value.
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

    /// Validates `[from, to)` against `src`'s table and collects the
    /// physical blocks it names (erroring if any slot in range is
    /// externalized rather than resident), returning them alongside
    /// `src`'s current fill. Pure read -- no mutation, so a refused
    /// `splice` call via this helper leaves the pool untouched.
    fn validate_and_collect_shared(
        &self,
        src: SessionHandle,
        from: usize,
        to: usize,
    ) -> Result<(Vec<PhysBlockId>, usize), KvAllocError> {
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
        Ok((shared, src_t.filled_tokens))
    }

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
        let (shared, src_filled) = self.validate_and_collect_shared(src, from, to)?;
        // Tokens the shared block range actually carries — the last shared block
        // may be partially filled in `src`, so clamp to `src`'s fill; this keeps
        // `dst`'s fill level consistent with its block count for later `append`.
        let bs = self.geom.block_size;
        let shared_tokens = ((to - from) * bs).min(src_filled.saturating_sub(from * bs));
        for p in shared {
            self.refcount[p as usize] += 1;
            self.table_mut(dst)?.slots.push(Slot::Resident(p));
        }
        self.table_mut(dst)?.filled_tokens += shared_tokens;
        Ok(())
    }

    /// Break the copy-on-write share on logical slot `i` of session `s` before a
    /// write: if the block is shared (refcount > 1), allocate a fresh block for
    /// `s` (content copy is the integration's job), drop `s`'s reference to the
    /// shared original, and point the slot at the fresh block. Returns the
    /// physical block the integration should write into. A no-op (returns the
    /// existing block) when already exclusive.
    pub fn cow_break(&mut self, s: SessionHandle, i: usize) -> Result<PhysBlockId, KvAllocError> {
        let p = match self
            .tables
            .get(&s)
            .ok_or(KvAllocError::UnknownSession)?
            .slots
            .get(i)
        {
            Some(Slot::Resident(p)) => *p,
            _ => return Err(KvAllocError::BadSpliceRange),
        };
        if self.refcount[p as usize] <= 1 {
            return Ok(p); // exclusive — no break needed
        }
        if self.free.is_empty() {
            return Err(KvAllocError::OutOfBlocks { need: 1, have: 0 });
        }
        let q = self.take_free()?;
        self.refcount[q as usize] = 1;
        self.refcount[p as usize] -= 1;
        self.table_mut(s)?.slots[i] = Slot::Resident(q);
        Ok(q)
    }

    /// Share a prefix of `prefix_blocks` FULLY-FILLED blocks from `src` into an
    /// EMPTY `dst` (rung-1: same absolute positions `0..shared_tokens`). Returns
    /// the shared token count (`= prefix_blocks * block_size`) so the consumer
    /// prefills ONLY the suffix — the "prefill only the suffix" contract is a
    /// pool invariant, not a per-consumer convention. The transactional wrapper
    /// over [`splice`](Self::splice): ALL preconditions are validated BEFORE any
    /// mutation, so a refused call leaves the pool byte-identical (no half-spliced
    /// `dst` with bumped refcounts and no unsplice).
    ///
    /// Preconditions (each refused before mutation):
    /// - `dst` open + EMPTY — a shared prefix must be `dst`'s FIRST blocks, but
    ///   `splice` APPENDS, so a non-empty `dst` would place the prefix at a shifted
    ///   position (numerically wrong; rung-2's job). [`PrefixTargetNotEmpty`].
    /// - `prefix_blocks <= src` block count. [`BadSpliceRange`].
    /// - every shared block FULLY filled (`prefix_blocks * block_size <=
    ///   src_filled`) so `shared_tokens` is block-aligned and the sharer's first
    ///   suffix write lands on a fresh (unshared) block. [`PrefixNotFullyFilled`].
    ///
    /// NOTE: distinct from `lightbulb::model_fuel::policies::splice_prefix` — that
    /// takes a `PrefixMatch` and layers its own (stricter) policy; this is the pool
    /// mechanism it can build on. Different signature + semantics.
    ///
    /// [`PrefixTargetNotEmpty`]: KvAllocError::PrefixTargetNotEmpty
    /// [`BadSpliceRange`]: KvAllocError::BadSpliceRange
    /// [`PrefixNotFullyFilled`]: KvAllocError::PrefixNotFullyFilled
    pub fn splice_prefix(
        &mut self,
        src: SessionHandle,
        dst: SessionHandle,
        prefix_blocks: usize,
    ) -> Result<usize, KvAllocError> {
        // ---- Validate BEFORE any mutation (transactional). ----
        // `dst` must be open and EMPTY.
        if !self
            .tables
            .get(&dst)
            .ok_or(KvAllocError::UnknownSession)?
            .slots
            .is_empty()
        {
            return Err(KvAllocError::PrefixTargetNotEmpty);
        }
        // `src` must be open; the shared range must fit and be fully filled.
        let src_filled = {
            let src_t = self.tables.get(&src).ok_or(KvAllocError::UnknownSession)?;
            if prefix_blocks > src_t.slots.len() {
                return Err(KvAllocError::BadSpliceRange);
            }
            src_t.filled_tokens
        };
        let shared_tokens = prefix_blocks * self.geom.block_size;
        if shared_tokens > src_filled {
            return Err(KvAllocError::PrefixNotFullyFilled {
                prefix_blocks,
                donor_filled: src_filled,
            });
        }
        // ---- Mutate. `splice` is itself transactional for its own range/residency
        // checks; all our preconditions passed, so nothing is left partial. ----
        self.splice(src, dst, 0, prefix_blocks)?;
        Ok(shared_tokens)
    }

    // --- Prefix registry (Task 2): named refcounted shared-prefix owners -----

    /// Register a shared prefix: mint a [`PrefixId`] whose owner independently
    /// keeps the donor's first `prefix_blocks` blocks alive. Opens an internal
    /// owner session and transactionally splices those blocks into it (reusing
    /// [`splice_prefix`](Self::splice_prefix)'s guard — the donor's shared blocks
    /// must be fully filled; the fresh owner is empty). After this the prefix's
    /// lifetime is the REGISTRY's to control: the donor may be discarded and the
    /// blocks stay resident (held by the owner), so a consumer's prefix reference
    /// no longer races the donor's teardown.
    ///
    /// On refusal the just-opened owner is rolled back, so a failed registration
    /// leaves no trace (no dangling empty session, no refcount change).
    pub fn register_prefix(
        &mut self,
        donor: SessionHandle,
        prefix_blocks: usize,
    ) -> Result<PrefixId, KvAllocError> {
        let owner = self.open();
        match self.splice_prefix(donor, owner, prefix_blocks) {
            Ok(_shared_tokens) => {
                let id = PrefixId(self.next_prefix);
                self.next_prefix += 1;
                self.prefixes.insert(
                    id,
                    PrefixOwner {
                        owner,
                        prefix_blocks,
                    },
                );
                Ok(id)
            }
            Err(e) => {
                // Roll back the empty owner (it never took a block, so this frees
                // nothing) — a failed registration leaves the pool untouched.
                self.discard(owner);
                Err(e)
            }
        }
    }

    /// Release a registered prefix: discard its owner handle. Each shared block's
    /// refcount drops by one; a block frees ONLY if it reaches refcount 0 — i.e.
    /// no live sharer still references it. Sharers that spliced the prefix keep
    /// their copies alive independently.
    pub fn release_prefix(&mut self, id: PrefixId) -> Result<(), KvAllocError> {
        let owner = self
            .prefixes
            .remove(&id)
            .ok_or(KvAllocError::UnknownPrefix)?;
        self.discard(owner.owner);
        Ok(())
    }

    /// The number of blocks a registered prefix owns (introspection).
    pub fn prefix_blocks(&self, id: PrefixId) -> Result<usize, KvAllocError> {
        self.prefixes
            .get(&id)
            .map(|o| o.prefix_blocks)
            .ok_or(KvAllocError::UnknownPrefix)
    }

    /// Splice a REGISTERED prefix (by [`PrefixId`]) into `dst` — the
    /// consumer-facing splice that does NOT require the original donor session to
    /// still exist (the owner keeps the blocks alive). Same contract as
    /// [`splice_prefix`](Self::splice_prefix): `dst` must be empty, and the shared
    /// blocks are the owner's (already fully-filled by construction); returns the
    /// shared token count `prefix_blocks * block_size` so the consumer prefills
    /// only `prompt[shared_tokens..]`. `Err(UnknownPrefix)` if `prefix` isn't
    /// registered — never a panic.
    pub fn splice_prefix_from(
        &mut self,
        prefix: PrefixId,
        dst: SessionHandle,
    ) -> Result<usize, KvAllocError> {
        let (owner, prefix_blocks) = {
            let o = self
                .prefixes
                .get(&prefix)
                .ok_or(KvAllocError::UnknownPrefix)?;
            (o.owner, o.prefix_blocks)
        };
        self.splice_prefix(owner, dst, prefix_blocks)
    }

    /// Is `s` the owner session of a registered prefix? An owner session holds
    /// ONLY the shared prefix blocks (it never appends/decodes), so all of its
    /// blocks are eviction-immune — [`evict_blocks`](Self::evict_blocks) reports
    /// them `still_shared` regardless of refcount (the owner-only invariant). A
    /// non-owner session holding a block at refcount 1 genuinely owns it
    /// exclusively (any owner reference would push the count to ≥ 2), so this
    /// per-session check is exactly the discriminator evict needs.
    fn is_prefix_owner(&self, s: SessionHandle) -> bool {
        self.prefixes.values().any(|o| o.owner == s)
    }

    /// All four preconditions `alloc_shifted_prefix_slots` must check before
    /// mutating anything: `dst`'s fill is block-aligned, `prefix` resolves,
    /// its owner's prefix is fully filled, and enough free blocks exist.
    /// Returns `(m_offset, owner, prefix_blocks)` on success -- pure read,
    /// so a refusal here leaves the pool untouched.
    fn validate_shifted_prefix_alloc(
        &self,
        prefix: PrefixId,
        dst: SessionHandle,
    ) -> Result<(usize, SessionHandle, usize), KvAllocError> {
        let bs = self.geom.block_size;
        let m = self
            .tables
            .get(&dst)
            .ok_or(KvAllocError::UnknownSession)?
            .filled_tokens;
        if m % bs != 0 {
            return Err(KvAllocError::OffsetNotBlockAligned {
                filled: m,
                block_size: bs,
            });
        }
        let (owner, prefix_blocks) = {
            let o = self
                .prefixes
                .get(&prefix)
                .ok_or(KvAllocError::UnknownPrefix)?;
            (o.owner, o.prefix_blocks)
        };
        let owner_filled = self
            .tables
            .get(&owner)
            .ok_or(KvAllocError::UnknownSession)?
            .filled_tokens;
        if prefix_blocks * bs > owner_filled {
            return Err(KvAllocError::PrefixNotFullyFilled {
                prefix_blocks,
                donor_filled: owner_filled,
            });
        }
        if prefix_blocks > self.free.len() {
            return Err(KvAllocError::OutOfBlocks {
                need: prefix_blocks,
                have: self.free.len(),
            });
        }
        Ok((m, owner, prefix_blocks))
    }

    /// rung-2 bookkeeping for a SHIFTED-prefix splice: validate that `dst`'s fill
    /// is block-aligned, then allocate one FRESH block per prefix block — a copy
    /// target, NOT a refcount share (a shifted prefix's keys must be re-rotated,
    /// so it cannot alias the owner's) — append them to `dst`, and return
    /// `(m_offset, [(owner_src_phys, fresh_dst_phys); prefix_blocks])`. The device
    /// layer (`DeviceKvPool::splice_prefix_shifted` (fuel_core::kv_block_pool_device))
    /// then rotates K src→fresh and copies V src→fresh. Validate-before-mutate: a
    /// refusal allocates nothing and leaves `dst` untouched.
    pub fn alloc_shifted_prefix_slots(
        &mut self,
        prefix: PrefixId,
        dst: SessionHandle,
    ) -> Result<(usize, Vec<(PhysBlockId, PhysBlockId)>), KvAllocError> {
        let (m, owner, prefix_blocks) = self.validate_shifted_prefix_alloc(prefix, dst)?;
        // All preconditions passed — allocate + append (infallible from here).
        let src: Vec<PhysBlockId> = (0..prefix_blocks)
            .map(|i| {
                self.resident_block(owner, i)
                    .expect("registered prefix owner block is resident")
            })
            .collect();
        let mut pairs = Vec::with_capacity(prefix_blocks);
        for s in src {
            let fresh = self.take_free()?;
            self.refcount[fresh as usize] = 1;
            self.table_mut(dst)?.slots.push(Slot::Resident(fresh));
            pairs.push((s, fresh));
        }
        self.table_mut(dst)?.filled_tokens += prefix_blocks * self.geom.block_size;
        Ok((m, pairs))
    }
}

#[cfg(test)]
mod tests;
