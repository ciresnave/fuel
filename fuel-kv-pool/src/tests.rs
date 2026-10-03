// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unit tests for `KvBlockPool` -- split from `lib.rs` (Codacy's file-size
//! threshold) into its own `#[cfg(test)]` module file. No content change.

use super::*;

fn geom(num_blocks: usize, block_size: usize) -> KvGeometry {
    KvGeometry {
        n_layers: 1,
        num_blocks,
        block_size,
        n_kv_heads: 2,
        head_dim: 4,
        elem_size: 2,
    }
}

#[test]
fn kv_bytes_resident_scales_with_n_layers() {
    // A block is a slot in every layer's K and V buffer, so resident bytes
    // scale linearly with n_layers (the device layer will own n_layers×2
    // pool buffers).
    let one = geom(16, 4);
    let mut many = one;
    many.n_layers = 32;
    let mut pool1 = KvBlockPool::new(one);
    let mut pool32 = KvBlockPool::new(many);
    let s1 = pool1.open();
    let s32 = pool32.open();
    pool1.append(s1, 8).unwrap(); // 2 blocks
    pool32.append(s32, 8).unwrap(); // 2 blocks
    assert_eq!(
        pool32.kv_bytes_resident(),
        32 * pool1.kv_bytes_resident(),
        "32-layer pool holds 32× the resident bytes for the same block count",
    );
}

#[test]
fn filled_tokens_is_the_context_len_source_not_blocks_times_block_size() {
    let mut pool = KvBlockPool::new(geom(16, 4));
    let s = pool.open();
    assert_eq!(
        pool.filled_tokens(s),
        Some(0),
        "fresh session: 0 filled tokens"
    );
    pool.append(s, 6).unwrap(); // 6 tokens → 2 blocks, last block half full
    assert_eq!(pool.session_blocks(s), Some(2));
    assert_eq!(
        pool.filled_tokens(s),
        Some(6),
        "context_len is the token count (6), NOT blocks×block_size (8)",
    );
    assert_eq!(
        pool.filled_tokens(SessionHandle(999)),
        None,
        "unknown session → None"
    );
}

#[test]
fn session_block_table_returns_resident_physical_ids_in_logical_order() {
    let mut pool = KvBlockPool::new(geom(16, 4));
    let s = pool.open();
    pool.append(s, 9).unwrap(); // 3 blocks
    let bt = pool.session_block_table(s).unwrap();
    let expected: Vec<PhysBlockId> = (0..3).map(|i| pool.resident_block(s, i).unwrap()).collect();
    assert_eq!(
        bt, expected,
        "block table = per-slot resident physical id, in order"
    );
    assert_eq!(
        pool.session_block_table(SessionHandle(999)),
        Err(KvAllocError::UnknownSession),
        "unknown session → typed error, never a panic",
    );
}

#[test]
fn session_block_table_errors_on_an_externalized_slot_never_routes_a_reclaimed_block() {
    // A fully-evicted session keeps its slots as `Externalized` (bytes live in
    // the handle). Materializing a block table over it must be a typed error —
    // routing attention through a physical block already handed back to the
    // pool would be silent cross-session corruption.
    let mut pool = KvBlockPool::new(geom(16, 4));
    let s = pool.open();
    pool.append(s, 9).unwrap(); // 3 exclusive blocks
    let _rep = pool.evict(s).unwrap(); // all 3 externalized (none shared)
    assert_eq!(
        pool.session_block_table(s),
        Err(KvAllocError::SessionNotResident { slot: 0 }),
        "an externalized slot is a mis-sequenced materialize → typed error",
    );
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
    assert_eq!(
        rep.freed,
        vec![2],
        "only the exclusive block (index 2) frees"
    );
    assert_eq!(
        rep.still_shared,
        vec![0, 1],
        "the two shared blocks reported by index"
    );
    assert_eq!(
        pool.free_blocks(),
        free_before + 1,
        "exactly one block returned"
    );

    // The sharer B is intact: its blocks still resolve to the SAME physical
    // blocks, which are still allocated (refcount dropped 2→1, not freed).
    assert_eq!(
        pool.resident_block(b, 0),
        Some(p0),
        "B's shared prefix intact"
    );
    assert_eq!(pool.resident_block(b, 1), Some(p1));
    // A retains its refs on the shared blocks: evict is PARTIAL for shared
    // blocks — they aren't A's alone to reclaim, and the Q9 self-contained-
    // restore rule forbids copying a shared block's bytes into A's handle. So
    // p0/p1 stay allocated at refcount 2 (A + B). A naive detach-all evict
    // would drop them to 0 and free them out from under B — the hazard.
    assert_eq!(
        pool.block_refcount(p0),
        2,
        "still shared by A and B, not freed"
    );
    assert_eq!(pool.block_refcount(p1), 2);
    // The freed block p2 is genuinely reusable and not referenced by B.
    assert_ne!(pool.resident_block(b, 0), Some(p2));
    assert_ne!(pool.resident_block(b, 1), Some(p2));
}

/// Shared setup for the pair of tests below: a donor `a` with 2 filled,
/// exclusive blocks (the prefix) and a NON-EMPTY target `b` (the
/// refusal trigger for `splice_prefix`). Returns
/// `(pool, a, b, p0, p1)`.
fn donor_and_nonempty_target() -> (
    KvBlockPool,
    SessionHandle,
    SessionHandle,
    PhysBlockId,
    PhysBlockId,
) {
    let mut pool = KvBlockPool::new(geom(16, 4));
    let a = pool.open();
    pool.append(a, 8).unwrap();
    let (p0, p1) = (
        pool.resident_block(a, 0).unwrap(),
        pool.resident_block(a, 1).unwrap(),
    );
    let b = pool.open();
    pool.append(b, 4).unwrap();
    (pool, a, b, p0, p1)
}

/// TRANSACTIONAL GUARD (Lightbulb-flagged scar): `splice_prefix` must validate
/// BEFORE it mutates, so a REFUSED splice leaves the pool byte-identical — no
/// half-spliced target with bumped refcounts and no unsplice. The teeth are in
/// asserting the POOL STATE after the refusal, NOT the returned `Err`: the
/// Err-only assertion passes even on a broken validate-AFTER-splice impl,
/// because the guard still fires — just too late, after the damage. The
/// meaningful refusal here is "target not empty": a shared prefix must be the
/// target's FIRST blocks, but the underlying `splice` blindly APPENDS, so a
/// non-empty target is exactly the case a late check would corrupt.
#[test]
fn refused_splice_prefix_leaves_the_pool_completely_untouched() {
    let (mut pool, a, b, p0, p1) = donor_and_nonempty_target();
    let q0 = pool.resident_block(b, 0).unwrap();

    // Snapshot everything a corruption could move.
    let free_before = pool.free_blocks();
    let (rc_p0, rc_p1) = (pool.block_refcount(p0), pool.block_refcount(p1));
    let b_blocks_before = pool.session_blocks(b);
    let b_filled_before = pool.filled_tokens(b);

    // Refuse: cannot splice a prefix into a non-empty target.
    let res = pool.splice_prefix(a, b, 2);
    assert!(
        res.is_err(),
        "splice_prefix into a non-empty target must refuse"
    );

    // THE GUARD — assert on the POOL, not the Err. A validate-after-splice impl
    // would have appended A's 2 blocks to B (B → 3 blocks) and bumped p0/p1 to
    // refcount 2 before erroring; every assertion below then fails.
    assert_eq!(
        pool.session_blocks(b),
        b_blocks_before,
        "B's block count unchanged (no prefix appended)"
    );
    assert_eq!(pool.filled_tokens(b), b_filled_before, "B's fill unchanged");
    assert_eq!(
        pool.resident_block(b, 0),
        Some(q0),
        "B's own block untouched"
    );
    assert_eq!(pool.resident_block(b, 1), None, "B gained no second block");
    assert_eq!(
        pool.block_refcount(p0),
        rc_p0,
        "donor refcount not bumped by a refused splice"
    );
    assert_eq!(pool.block_refcount(p1), rc_p1);
    assert_eq!(pool.free_blocks(), free_before, "free list unmoved");
}

#[test]
fn refused_splice_prefix_does_not_poison_the_donor_for_a_later_splice() {
    let (mut pool, a, b, p0, p1) = donor_and_nonempty_target();
    let _ = pool.splice_prefix(a, b, 2); // the refusal from the test above

    // The refusal did not poison the donor for a later legitimate caller:
    // splicing the same prefix into a FRESH empty session still succeeds.
    let c = pool.open();
    let shared = pool
        .splice_prefix(a, c, 2)
        .expect("legit prefix splice into empty target");
    assert_eq!(
        shared, 8,
        "2 blocks × block_size 4 = 8 shared tokens (donor fully filled)"
    );
    assert_eq!(
        pool.filled_tokens(c),
        Some(8),
        "C inherits the prefix's fill"
    );
    assert_eq!(pool.block_refcount(p0), 2, "now genuinely shared A+C");
    assert_eq!(pool.block_refcount(p1), 2);
    assert_eq!(
        pool.resident_block(c, 0),
        Some(p0),
        "C reads A's exact prefix blocks (zero-copy)"
    );
    assert_eq!(pool.resident_block(c, 1), Some(p1));
}

/// ALIGNMENT INVARIANT (Lightbulb-flagged): only FULLY-filled whole blocks may
/// be shared, so the sharer's fill stays a block multiple and its first suffix
/// write lands on a fresh (unshared) block — never mid a shared block. A
/// partial last block is refused. (Block-granular COUNT is not block-aligned
/// FILL: 6 tokens at bs=4 is 2 blocks but `filled==6`.)
#[test]
fn splice_prefix_refuses_a_partial_last_block() {
    let mut pool = KvBlockPool::new(geom(16, 4));
    let a = pool.open();
    pool.append(a, 6).unwrap(); // 2 blocks, block 1 half-full (filled 6)
    let free_before = pool.free_blocks();

    // Sharing 2 blocks would give a misaligned fill of 6 → refuse, untouched.
    let c = pool.open();
    assert_eq!(
        pool.splice_prefix(a, c, 2),
        Err(KvAllocError::PrefixNotFullyFilled {
            prefix_blocks: 2,
            donor_filled: 6
        }),
        "a partial last block cannot be shared (would misalign the sharer's fill)",
    );
    assert_eq!(
        pool.session_blocks(c),
        Some(0),
        "refused share leaves C empty"
    );
    assert_eq!(
        pool.free_blocks(),
        free_before,
        "free list unmoved by the refusal"
    );

    // Sharing the ONE fully-filled block is aligned and succeeds.
    assert_eq!(
        pool.splice_prefix(a, c, 1).unwrap(),
        4,
        "one full block = 4 shared tokens (block-aligned)",
    );
    assert_eq!(
        pool.filled_tokens(c),
        Some(4),
        "sharer fill is block-aligned"
    );
}

#[test]
fn alloc_shifted_prefix_slots_validates_and_allocates() {
    // rung-2 bookkeeping: a shifted-prefix splice needs the target's fill to be
    // block-aligned (so the prefix lands on fresh whole blocks), and allocates
    // fresh COPY-target blocks (not a refcount share).
    let mut pool = KvBlockPool::new(geom(64, 4));
    let donor = pool.open();
    pool.append(donor, 8).unwrap(); // 2 full blocks
    let pid = pool.register_prefix(donor, 2).unwrap();

    // NON-aligned target fill → refused before any mutation.
    let dst = pool.open();
    pool.append(dst, 5).unwrap();
    let free0 = pool.free_blocks();
    assert_eq!(
        pool.alloc_shifted_prefix_slots(pid, dst),
        Err(KvAllocError::OffsetNotBlockAligned {
            filled: 5,
            block_size: 4
        }),
    );
    assert_eq!(pool.free_blocks(), free0, "refusal allocates nothing");
    assert_eq!(
        pool.filled_tokens(dst),
        Some(5),
        "refusal does not bump fill"
    );
    assert_eq!(
        pool.session_blocks(dst),
        Some(2),
        "refusal does not extend the table"
    );

    // Block-aligned target (M=8) → allocates 2 fresh copy-target blocks.
    let dst2 = pool.open();
    pool.append(dst2, 8).unwrap();
    let (m, pairs) = pool.alloc_shifted_prefix_slots(pid, dst2).unwrap();
    assert_eq!(m, 8, "offset is the target's block-aligned fill");
    assert_eq!(pairs.len(), 2, "one copy pair per prefix block");
    assert_eq!(
        pool.filled_tokens(dst2),
        Some(16),
        "fill bumped by 2*block_size"
    );
    for (src, fresh) in &pairs {
        assert_eq!(
            pool.block_refcount(*fresh),
            1,
            "fresh dst block is exclusive (a COPY target)"
        );
        assert_ne!(
            src, fresh,
            "dst block is a copy target, not the shared original"
        );
    }
}

#[test]
fn capacity_is_geometry_keyed_and_tracks_the_free_list() {
    let mut pool = KvBlockPool::new(geom(10, 4));
    let cap = pool.capacity();
    assert_eq!(
        cap.geometry,
        geom(10, 4),
        "geometry-keyed for cross-pool admission"
    );
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
    assert_eq!(
        rep.freed,
        vec![0, 1, 2],
        "all exclusive → all freed, by index"
    );
    assert!(rep.still_shared.is_empty());
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
fn evict_range_sheds_a_span_and_leaves_the_live_session_decoding() {
    let mut pool = KvBlockPool::new(geom(16, 4));
    let s = pool.open();
    pool.append(s, 20).unwrap(); // 5 blocks, all exclusive
    let free_after = pool.free_blocks();
    // Shed the cold middle [1, 4): blocks 1, 2, 3 — the point of tiering.
    let rep = pool.evict_range(s, 1, 4).unwrap();
    assert_eq!(rep.freed, vec![1, 2, 3]);
    assert!(rep.still_shared.is_empty());
    assert_eq!(pool.free_blocks(), free_after + 3);
    // Head + tail stay resident; the session is still live and can grow.
    assert!(pool.resident_block(s, 0).is_some(), "head resident");
    assert!(pool.resident_block(s, 4).is_some(), "tail resident");
    assert_eq!(pool.resident_block(s, 2), None, "middle externalized");
    pool.append(s, 4).unwrap(); // still decoding after a partial evict
    assert_eq!(pool.session_blocks(s), Some(6));
    // Restore the shed span at its original logical positions (→ RoPE ranges
    // reconstruct); the rest is untouched.
    pool.restore(s, rep.handle).unwrap();
    assert!(pool.resident_block(s, 1).is_some());
    assert!(pool.resident_block(s, 3).is_some());
}

#[test]
fn evict_blocks_partially_overlapping_a_shared_prefix_reports_both() {
    // The exact shape a conversation sharing a system prompt generates: a
    // requested set straddling a spliced (shared) prefix and an exclusive
    // tail. A count-based report would hide WHICH blocks stayed shared, and a
    // consumer marking the span demoted on the count would diverge from the
    // pool. Per-block `freed`/`still_shared` keeps it honest.
    let mut pool = KvBlockPool::new(geom(16, 4));
    let a = pool.open();
    pool.append(a, 20).unwrap(); // 5 blocks
    let b = pool.open();
    pool.splice(a, b, 0, 2).unwrap(); // A's blocks 0,1 shared with B
    let rep = pool.evict_blocks(a, &[1, 2, 3]).unwrap(); // straddles shared + exclusive
    assert_eq!(rep.freed, vec![2, 3], "exclusive blocks freed, by index");
    assert_eq!(
        rep.still_shared,
        vec![1],
        "the shared block reported, not freed"
    );
    // Block 1 untouched: A still holds it, B still resolves to it, refcount 2.
    assert!(pool.resident_block(a, 1).is_some());
    assert_eq!(pool.block_refcount(pool.resident_block(b, 1).unwrap()), 2);
}

#[test]
fn evict_blocks_rejects_out_of_range_index_atomically() {
    let mut pool = KvBlockPool::new(geom(16, 4));
    let s = pool.open();
    pool.append(s, 12).unwrap(); // 3 blocks
    let free_before = pool.free_blocks();
    let err = pool.evict_blocks(s, &[0, 5]).unwrap_err(); // 5 is out of range
    assert!(matches!(
        err,
        KvAllocError::BadBlockIndex {
            index: 5,
            session_blocks: 3
        }
    ));
    // Atomic: the valid block 0 was NOT evicted despite appearing in the set.
    assert_eq!(
        pool.free_blocks(),
        free_before,
        "nothing evicted on a bad set"
    );
    assert!(pool.resident_block(s, 0).is_some());
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
    let (p0, p1) = (
        pool.resident_block(a, 0).unwrap(),
        pool.resident_block(a, 1).unwrap(),
    );
    let b = pool.open();
    pool.splice(a, b, 0, 2).unwrap();
    let free_before = pool.free_blocks();
    pool.discard(a); // A gone, but B still references p0,p1
    assert_eq!(
        pool.free_blocks(),
        free_before,
        "shared blocks NOT freed — B holds them"
    );
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
    assert_eq!(
        pool.resident_block(a, 0),
        Some(p0),
        "A (the sharer) unchanged"
    );
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
    assert_eq!(
        pool.kv_bytes_resident(),
        2 * per,
        "sharing adds no resident bytes"
    );
    pool.append(b, 4).unwrap(); // B grows by 1 exclusive block
    assert_eq!(pool.kv_bytes_resident(), 3 * per);
}

/// Shared setup for the pair of tests below: registers a 2-block prefix
/// on donor `a`, then discards `a` -- leaving the prefix's OWNER as the
/// blocks' sole reference (refcount 1 each). Returns
/// `(pool, id, p0, p1, free_before)` where `free_before` is the free
/// count right after the discard.
fn registered_prefix_after_donor_discarded()
-> (KvBlockPool, PrefixId, PhysBlockId, PhysBlockId, usize) {
    let mut pool = KvBlockPool::new(geom(16, 4));
    let a = pool.open();
    pool.append(a, 8).unwrap(); // 2 FULL blocks (block_size 4)
    let (p0, p1) = (
        pool.resident_block(a, 0).unwrap(),
        pool.resident_block(a, 1).unwrap(),
    );
    let id = pool.register_prefix(a, 2).expect("register 2 full blocks");
    pool.discard(a);
    let free_before = pool.free_blocks();
    (pool, id, p0, p1, free_before)
}

#[test]
fn prefix_owner_keeps_blocks_alive_after_donor_discarded() {
    let (pool, id, p0, p1, free_before) = registered_prefix_after_donor_discarded();
    assert_eq!(pool.prefix_blocks(id).unwrap(), 2);
    // The owner keeps the blocks alive (refcount → 1) and NONE are freed
    // by discarding the donor — the prefix outlives the session that
    // computed it, so a consumer's prefix reference no longer races the
    // donor's teardown.
    assert_eq!(
        pool.block_refcount(p0),
        1,
        "owner alone still references p0"
    );
    assert_eq!(pool.block_refcount(p1), 1);
    assert_eq!(
        pool.free_blocks(),
        free_before,
        "no block freed — the owner holds them"
    );
}

#[test]
fn owner_only_prefix_blocks_report_still_shared_and_release_frees_them() {
    let (mut pool, id, p0, p1, free_before) = registered_prefix_after_donor_discarded();

    // THE OWNER-ONLY still_shared PIN (silent-corruption trap): a block held
    // ONLY by a prefix owner (refcount 1) must report `still_shared`, NEVER
    // `freed`, from an evict — a registered prefix's blocks are eviction-immune
    // (only `release_prefix` frees them). An owner-only block reporting `freed`
    // would let a consumer's report-reconciliation treat live shared-prefix
    // tokens as reclaimed and re-prefill over a live prefix. Evict-query the
    // owner's own blocks at their sharpest (sole reference).
    let owner_h = pool.prefixes[&id].owner; // tests reach the internal owner
    let rep = pool
        .evict_blocks(owner_h, &[0, 1])
        .expect("evict-query owner blocks");
    assert_eq!(
        rep.still_shared,
        vec![0, 1],
        "owner-only prefix blocks must report still_shared (eviction-immune)",
    );
    assert!(
        rep.freed.is_empty(),
        "a registered prefix's blocks are NEVER freed by evict"
    );
    assert_eq!(
        pool.block_refcount(p0),
        1,
        "still resident — evict did not detach it"
    );
    assert_eq!(pool.block_refcount(p1), 1);

    // Only release_prefix frees them: owner-only → refcount 0 → back to the pool.
    pool.release_prefix(id).unwrap();
    assert_eq!(
        pool.block_refcount(p0),
        0,
        "release_prefix frees the owner-only block"
    );
    assert_eq!(pool.block_refcount(p1), 0);
    assert_eq!(
        pool.free_blocks(),
        free_before + 2,
        "both prefix blocks back in the pool"
    );

    // A released id is a typed error on every path, never a panic.
    assert_eq!(pool.release_prefix(id), Err(KvAllocError::UnknownPrefix));
    assert_eq!(pool.prefix_blocks(id), Err(KvAllocError::UnknownPrefix));
}

#[test]
fn splice_prefix_from_shares_a_registered_prefix_after_donor_gone() {
    let mut pool = KvBlockPool::new(geom(16, 4));
    let a = pool.open();
    pool.append(a, 8).unwrap(); // 2 full blocks
    let (p0, p1) = (
        pool.resident_block(a, 0).unwrap(),
        pool.resident_block(a, 1).unwrap(),
    );
    let id = pool.register_prefix(a, 2).unwrap();
    pool.discard(a); // donor gone; the owner keeps the prefix alive (refcount 1)
    assert_eq!(pool.block_refcount(p0), 1);

    // A fresh consumer splices the registered prefix — no donor needed.
    let c = pool.open();
    let shared = pool
        .splice_prefix_from(id, c)
        .expect("splice registered prefix");
    assert_eq!(shared, 8, "2 blocks × block_size 4 = 8 shared tokens");
    assert_eq!(
        pool.filled_tokens(c),
        Some(8),
        "consumer fill = shared prefix length"
    );
    assert_eq!(pool.session_blocks(c), Some(2));
    assert_eq!(
        pool.block_refcount(p0),
        2,
        "owner + consumer reference the prefix block"
    );
    assert_eq!(pool.block_refcount(p1), 2);
    // Zero-copy: the consumer's slots point at the SAME physical blocks.
    assert_eq!(pool.resident_block(c, 0).unwrap(), p0);
    assert_eq!(pool.resident_block(c, 1).unwrap(), p1);

    // Same transactional guard as splice_prefix: refuses a non-empty target.
    let d = pool.open();
    pool.append(d, 4).unwrap();
    assert_eq!(
        pool.splice_prefix_from(id, d),
        Err(KvAllocError::PrefixTargetNotEmpty)
    );

    // After release, the id is unknown → typed error, never a panic.
    pool.release_prefix(id).unwrap();
    let e = pool.open();
    assert_eq!(
        pool.splice_prefix_from(id, e),
        Err(KvAllocError::UnknownPrefix)
    );
}
