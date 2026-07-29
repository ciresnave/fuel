# Multi-session serving — Increment 2: the KV block-pool allocator

**Status:** design-of-record (2026-07-28). Builds on serving Increment 1
(`fuel-core/src/multi_session.rs`) and the newly-constitutional consumer contract
([`docs/architecture/15-consumer-contract.md`](../../architecture/15-consumer-contract.md), v0.1).
Coordinated with the consumer-seam author (peer `trpe1mc5`, docs-only — no code overlap; `KvCache`
+ the new allocator are this session's).

## Goal

Build the **host-side KV block-pool allocator** behind `Op::PagedAttn` — the mechanism that is the
confirmed-absent keystone for multi-session serving at scale (ROADMAP §4: "no allocator/refcounting
exists behind `Op::PagedAttn` today"). It delivers two consumer-contract clauses the Increment-1
audit flagged as absent, **as mechanism only** (policy — admission, which session to evict, when to
splice — stays with the consumer, per 15's mechanism/policy line):

- **C-1 (capacity advertisement)** — `free_blocks()` + `blocks_required(geometry)` so the consumer
  can answer "will this session fit?" itself. An OOM-on-construction is the wrong shape.
- **C-3 (state externalization), the *lossy* arm** — evict / restore / discard KV blocks, plus
  cross-branch **splice** (refcounted copy-on-write block sharing for parallel "trains of thought").

## What already exists (grounding)

`Op::PagedAttn` (`fuel-graph/src/lib.rs:4211`) already defines the *graph-side* paged substrate:
- `k_cache` / `v_cache`: `[num_blocks, block_size, Hkv, D]` — a physical pool of `num_blocks`.
- `block_table`: `[B, max_blocks_per_seq]` u32 — logical→physical block map (the page table).
- `context_lens`: `[B]` u32 — per-sequence lengths.

So the op consumes a block pool + block table; what is missing is the **host-side manager** that owns
the pool, hands out physical blocks, refcounts them, builds each session's `block_table`, and
externalizes state. This increment is that manager. K/V writes into physical blocks continue to go
through the executor (`Op::WriteSlice` at the block offset) — unchanged.

## Design

Two pieces, split so the **pure logic is unit-testable without a device**:

### 1. `KvBlockPool` — the pure allocator core (no device, no tensor deps)

Owns only *metadata* over a fixed pool of `num_blocks` physical blocks:
- geometry: `block_size`, `n_kv_heads`, `head_dim`, `dtype`, `num_blocks` (model-agnostic — it speaks
  KV geometry, **never a model**; this is what keeps the eventual `fuel-inference` move cheap, Q2).
- a **free list** of physical block ids.
- a **per-block refcount** (`Vec<u32>`, index = physical block id) — enables COW splice.
- per-session **block tables** (`SessionId -> BlockTable { blocks: Vec<PhysBlockId>, filled: usize }`).

API (all mechanism):

| Verb | Contract | Semantics |
| --- | --- | --- |
| `open(session) ` | — | register an empty block table |
| `append(session, n_tokens) -> Result<()>` | — | grow the table by ⌈…⌉ blocks; `Err` iff `n_tokens` needs more than `free_blocks()` (the consumer should have asked `blocks_required` first) |
| `free_blocks() -> usize` | **C-1** | physical blocks currently on the free list |
| `blocks_required(cur_filled, add_tokens) -> usize` | **C-1** | blocks a session at `cur_filled` needs to admit `add_tokens` more — so the consumer's admission math matches the pool's exactly (never reimplemented) |
| `kv_bytes_resident() -> u64` | **C-4** (one bite) | `used_blocks · block_size · n_kv_heads · head_dim · 2 · dtype_size` — the consumer's only budget signal; near-free from the pool |
| `evict(session) -> EvictReport` | **C-3 lossy** | detach **refcount==1** blocks to host bytes; **never touches shared blocks**; returns `{ freed: usize, still_shared: usize, handle: Externalized }` |
| `discard(session)` | **C-3** | irreversible free (decrement refcounts, reclaim refcount→0). NOT a restore path — the consumer is dropping+re-prefilling from tokens |
| `restore(Externalized) -> Result<SessionId>` | **C-3 lossy** | re-attach externalized blocks (allocate fresh physical blocks, copy bytes back) |
| `splice(src, dst, block_range) -> Result<()>` | **C-3 lossy** | share `src`'s blocks into `dst` COW: bump refcounts, `dst`'s table points at the shared physical blocks; a later write to a shared block copies-on-write (bump→alloc-fresh→copy→decrement) |
| `close(session)` | — | drop the table, decrement refcounts, reclaim any refcount→0 |

### 2. Device-backed integration (thin, in `fuel-core`)

Binds the core to real device Storage: the `[num_blocks, block_size, Hkv, D]` K + V pool buffers,
and materializes each session's `block_table` + `context_lens` as u32 tensors for `Op::PagedAttn`.
`evict`/`restore` move block bytes device↔host. Built **after** the core is green.

## The two decisions that were mine to close (contract Q3 / Q9)

**Q3 — is C-3 in scope? YES, and it IS the allocator.** Paged blocks + refcounting *are* the
evict/restore/splice mechanism; C-1 falls out of the same free list. One coherent piece.

**Q9 — one C-3 mechanism with a fidelity flag, or two impls? One INTERFACE + a `Fidelity`
discriminator; two implementations backed by different state; the *lossy* one now.** The load-bearing
rule (sharper than "fidelity flag", per the coordination): **`restore` takes externalized *state* — an
opaque handle — NEVER a "recompute-from-tokens" instruction.** Bake the inference recompute path into
the signature and the exact arm can never be expressed. So:
- `Fidelity::Lossy` (this increment): KV blocks, recomputable from tokens; cheapness dominates.
- `Fidelity::Exact` (**later increment**, training/RL state — params+moments+RNG stream position;
  gated on the RNG/generator seam that 15 itself says C-3-exact depends on).

**Exact-arm completeness gate (specified now, so "specify against ≥2 classes before it hardens" is
real, not nominal):** a future `Fidelity::Exact` restore MUST diverge from an uninterrupted run by
**exactly zero**, and its handle MUST enumerate everything it covers (RNG stream position, cached
plan, captured run, any backend-side accumulator). Anything outside the handle makes "exact" a silent
lie. This increment implements only Lossy but shapes the interface to pass that gate unchanged.

## The correctness hazard: splice × evict (build the guard first)

COW shared blocks and evict collide: **you cannot detach a block another session still references.**
Resolution: **refcount-aware partial evict** — only `refcount==1` blocks are detachable; shared
blocks are never touched; `evict` reports the **actual** freed count + the still-shared count so the
consumer's admission math stays exact and it can decide to break a splice / discard / shed a different
session. Never corrupts a sharer, never force-copies (preserves splice's point), never
refuse-and-starves. **Born-red test first:** `evict_of_spliced_session_does_not_corrupt_sharer`.

## Test plan (TDD; the core is pure logic)

1. **`evict_of_spliced_session_does_not_corrupt_sharer`** (born-red, the hazard): splice src→dst,
   evict src, assert dst's shared blocks are intact + `EvictReport.still_shared` is honest.
2. `append`/`free_blocks`/`blocks_required` round-trip: admission math matches the pool.
3. `evict`→`restore` round-trip: bit-identical block contents (lossy handle carries bytes).
4. `discard` frees irreversibly; refcounts reclaim.
5. `splice` COW: shared reads alias; a write to a shared block copies-on-write, leaves the sharer
   unchanged.
6. `kv_bytes_resident` tracks used blocks.
7. Totality: over-`append` past `free_blocks` is a typed `Err`, never a panic / never an OOM-shaped
   surprise (C-1 is the consumer's pre-check).
8. (later) device-backed integration + a `block_table` that `Op::PagedAttn` accepts.

## Explicitly out of scope (sequenced elsewhere)

- **Scheduler-surface reshape** (`advance(ready, order, quantum, cancel)`, `add_session`→admission
  split, `SchedulePolicy`→`DecodeArm` rename, sampling location Q5) — rides the `fuel-inference` move,
  not this increment.
- **The `Fidelity::Exact` arm** — later increment, gated on the RNG/generator seam.
- **The `multi_session.rs` → `fuel-inference` layer move (Q2)** — a verified layer-drift defect;
  agreed it happens *after* this increment and is this session's to do, but it is constitution-adjacent
  and needs CireSnave's explicit go-ahead. **Flag it to CireSnave when this increment lands.** Until
  then: build the allocator fully model-agnostic so the move stays cheap; do NOT deepen the
  `fuel-core`→`LlamaModel` coupling.

## Crate placement

The pure core is dependency-light + model-agnostic. It starts as a new `fuel-core` module
(`kv_block_pool.rs`), co-located with `KvCache` (`inference_context.rs`), move-ready for Q2. Revisit
`fuel-memory` as a home if the device-backed pool wants to sit lower.

## Refinements from the consumer-seam review (peer `trpe1mc5`, Lightbulb survey)

Folded into the code + module doc after the design set:

- **Capacity is geometry-keyed (`PoolCapacity`), not a bare `usize`.** A compressed/heterogeneous-KV
  world implies N pools (one per geometry); a consumer choosing "admit into pool A / B / grow a pool"
  needs per-pool headroom keyed by geometry. `free_blocks() -> usize` stays as a convenience but
  `capacity() -> PoolCapacity { geometry, free_blocks, total_blocks }` is the admission primitive, so a
  second geometry never forces a breaking change. Cross-pool arbitration + the **unpooled
  device-memory remainder** (pools fragment VRAM — a consumer can be out of blocks in every pool yet
  have free memory) are a future pool-set manager over N pools + the backend free-memory query, not
  this single pool.
- **Single-geometry boundary documented** (does not foreclose compressed KV — the geometry is a
  parameter; heterogeneous-within-one-pool is a separate mechanism).
- **Exact-arm RNG coverage bounded to Fuel-owned RNG.** The completeness gate's RNG entry is
  Fuel-owned stochastic state (training-side sampling, dropout) — *never* the consumer's sampler RNG
  (a real consumer, Lightbulb, re-seeds its `StdRng` per sample; that is consumer policy). Over-scoping
  the Exact handle into consumer sampler state is as wrong as under-scoping Fuel's.
- **COW splice validated against a real consumer** — Lightbulb's `cache/prefix_cache.rs` +
  `cache_span.rs` are shared-prefix blocks across sessions, i.e. exactly this refcounted COW splice
  seen from the consumer side. The scheduler-surface deferral is confirmed (Lightbulb has its own
  engine/admission/batching and wants the allocator underneath its own policies; it will almost
  certainly never adopt `SessionScheduler`).

## Open-contract items closed by this increment

To mark resolved in `docs/fuel-consumer-seam.md`'s open-questions list when this lands (Q1-style,
with the commit ref), per the whoever-closes-edits convention:
- **Q3** (is C-3 in scope for Increment 2?) → **YES**, and it is the same piece as the block-pool
  allocator (paged blocks + refcounting *are* evict/restore/splice; C-1 falls out of the free list).
- **Q9** (one C-3 mechanism with a fidelity flag, or two?) → **one interface + a `Fidelity`
  discriminator, two implementations backed by different state; the Lossy-KV arm now.** The
  load-bearing rule: `restore` takes externalized *state*, never a recompute-from-tokens instruction.

## Consumer-driven hardening (Lightbulb, the first real consumer)

A live design loop with the Lightbulb port session (`0guhhdqj`) — Eric put *batched decode from the
start* on the port, making this allocator its critical path — hardened the evict surface **before it
set**, against the §15 two-consumer bar (Lightbulb's `segmented_eviction_policy` + `fuel-inference`'s
`segmented_eviction`, both "logical segments evicted as complete units"):

- **`evict` is whole-session; the real eviction unit is a block SET within a live session.** Tiering
  sheds the cold middle of a long conversation while it keeps decoding. So the primitive is now
  **`evict_blocks(s, &[logical_index])`** (may be non-contiguous — H2O keeps scattered heavy-hitters;
  a segmented policy evicts a union of spans), with `evict_range(s, from, to)` the single-span
  convenience and `evict(s)` the whole-session convenience (part 1's surface, unchanged). `restore`
  needs no range variant — it re-materializes whatever the handle covers at each block's original
  logical index (→ RoPE ranges reconstruct), so it's granularity-agnostic; and it no longer touches
  `filled_tokens` (a partial evict of a still-growing session must not be reset).
- **`EvictReport` carries per-block attribution, not counts** (`freed: Vec<usize>`,
  `still_shared: Vec<usize>`). After a scattered set-evict, a *count* ("3 still_shared") hides *which*
  blocks stayed shared, so a consumer marking a span demoted on the count would silently diverge from
  the pool. Same honesty property `{freed, still_shared}` already carried, at the set primitive's
  granularity. Test coverage includes the exact shape a shared-system-prompt conversation generates:
  a set straddling a spliced prefix and an exclusive tail → both lists non-empty.
- **RoPE positions** are preserved by construction (restore places each block at its original logical
  index → `logical_index·block_size + offset` reconstructs the position range on promote). No
  handle-riding metadata stash was added — the consumer initiates every eviction (§15) so it holds
  its own span→blocks map; adding an opaque blob for one consumer is the speculative generality §15
  warns against.

### Deferred: a named, refcounted block-group handle (future increment, two-consumer-specified)

Lightbulb noted (and self-corrected toward the mechanism/policy line) that the per-block attribution
above exists *only* because a consumer reassembles group outcomes from block outcomes — evidence that
a **named, refcounted block group with atomic evict/restore** may itself be Fuel mechanism (splice
already shares a contiguous run anonymously; the two-consumer bar is met by the two segmented
policies). The **extent** half of a "span" (contiguous positions + identity, evicted/restored as a
unit) is the mechanism candidate; the **meaning** half (importance, tags) is unambiguously consumer
policy. Deferred deliberately: design it against both segmented policies *from the start*, never
retrofitted around one consumer's span semantics (the C-3 failure mode §15 documents). Not built now,
not built for one consumer. → ROADMAP.

## Part 2 + follow-ups

- **Part 2 (device-backed integration):** bind the core to real `[num_blocks, block_size, Hkv, D]`
  K/V pool buffers; materialize the `block_table`/`context_lens` tensors for `Op::PagedAttn`; wire
  evict/restore byte movement device↔host. Read `fuel-inference`'s `tiered_storage.rs` (a C-3-lossy
  consumer written before C-3) + Lightbulb's counterpart before finalizing the device evict/restore
  signature — it's the real-ergonomics surface, and `fuel-inference`'s policy layer
  (eviction/prefix_cache/tiered_storage, zero consumers today) is the natural first thing to wire the
  allocator under.
- **Q2 — `multi_session.rs` → `fuel-inference` — AUTHORIZED by CireSnave** (2026-07-29, "move it when
  convenient"). This session's, *after* part 2. NOT a pure file move: it forces `&LlamaModel` into a
  model-agnostic trait (SamplingStrategy sheds rather than moves — sampling is consumer policy, Q5).
  The trait is the real design change and the reason to move before an inference consumer arrives.
