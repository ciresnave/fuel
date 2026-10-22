// SPDX-License-Identifier: MIT OR Apache-2.0
//! A content-addressed index over FULL blocks (ROADMAP item 17, row H20).
//!
//! Prefix reuse by whole-prefix hashing only hits at lengths the caller chose
//! to store; a request that shares a system prompt but diverges after it misses
//! unless the system prompt alone was stored. A block index fixes that at the
//! mechanism level: every full block is registered under a **chained hash**
//! (`hash(parent_hash, block_tokens, extra)`), so a new request finds the
//! longest run of already-computed blocks by walking its own hash chain, at any
//! block boundary, in O(blocks).
//!
//! **Mechanism, not policy** ([15-consumer-contract](../../docs/architecture/15-consumer-contract.md)):
//! the pool stores opaque 128-bit keys the CALLER computed. It decides nothing
//! about what goes into a key (tokens, a LoRA id, a tenant salt are the
//! caller's `extra`), when to commit, or what to evict. It supplies the
//! building blocks: [`commit_block`](KvBlockPool::commit_block),
//! [`lookup_chain`](KvBlockPool::lookup_chain),
//! [`splice_cached_chain`](KvBlockPool::splice_cached_chain), and the eviction
//! candidates in LRU order ([`evictable_cached`](KvBlockPool::evictable_cached)),
//! so a policy can score them however it likes.
//!
//! Ownership: the index holds ONE reference on each cached block, so a block
//! outlives the session that computed it. A cached block with a session also
//! referencing it has refcount > 1 and is protected by the existing
//! copy-on-write rules (writers must [`cow_break`](KvBlockPool::cow_break)
//! first), so the content behind a hash is immutable for as long as it is
//! cached.

use crate::{KvAllocError, KvBlockPool, PhysBlockId, SessionHandle, Slot};

/// An opaque 128-bit content key for one full block, computed by the caller.
pub type BlockHash = u128;

/// One cached block: where it lives and when it was last used.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CachedBlock {
    pub(crate) block: PhysBlockId,
    pub(crate) last_used: u64,
}

/// Cumulative counters over the block index since the pool was built.
///
/// **Mechanism, not policy:** the pool only counts. Whether a hit rate is good,
/// or what to evict because of it, is the caller's call. A lookup is one
/// [`lookup_chain`](KvBlockPool::lookup_chain) call, including the one inside
/// [`splice_cached_chain`](KvBlockPool::splice_cached_chain), counted once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// `lookup_chain` calls.
    pub lookups: u64,
    /// Blocks matched across all lookups.
    pub block_hits: u64,
    /// Requested blocks not matched: for each lookup, the hashes at and after
    /// the first miss.
    pub block_misses: u64,
    /// Blocks newly registered by `commit_block` (a refused duplicate is not
    /// counted).
    pub commits: u64,
    /// Entries dropped by `release_cached`: the evictions a policy chose.
    pub evictions: u64,
}

/// A convenience chained hash: `hash(parent, tokens, extra)` over two
/// independent 64-bit lanes. **Not cryptographic.** Accidental collisions are
/// negligible at 128 bits, but a party who controls the tokens can craft one, so
/// a multi-tenant consumer should fold a secret into `extra` or supply its own
/// keyed hash. The pool itself never calls this.
pub fn chain_block_hash(parent: Option<BlockHash>, tokens: &[u32], extra: &[u8]) -> BlockHash {
    fn mix(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    let (mut a, mut b) = match parent {
        Some(p) => ((p >> 64) as u64, p as u64),
        None => (0x6A09_E667_F3BC_C908, 0xBB67_AE85_84CA_A73B),
    };
    let mut feed = |x: u64| {
        a = mix(a ^ x).wrapping_add(0x9E37_79B9_7F4A_7C15);
        b = mix(b.rotate_left(23) ^ x.wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    };
    feed(tokens.len() as u64);
    for &t in tokens {
        feed(u64::from(t));
    }
    feed(0xFFFF_FFFF_0000_0000 | extra.len() as u64);
    for chunk in extra.chunks(8) {
        let mut w = [0u8; 8];
        w[..chunk.len()].copy_from_slice(chunk);
        feed(u64::from_le_bytes(w));
    }
    (u128::from(a) << 64) | u128::from(b)
}

impl KvBlockPool {
    /// Register block `i` of session `s` under `hash`. The block must be
    /// resident and FULLY filled (a partial block's content is still changing).
    ///
    /// Returns `Ok(true)` if the index took a reference, `Ok(false)` if the hash
    /// (or this physical block) is already cached: the existing entry is
    /// kept and the session's block is left alone. A cached block survives the
    /// session; release it with [`release_cached`](Self::release_cached).
    pub fn commit_block(
        &mut self,
        s: SessionHandle,
        i: usize,
        hash: BlockHash,
    ) -> Result<bool, KvAllocError> {
        let bs = self.geom.block_size;
        let table = self.tables.get(&s).ok_or(KvAllocError::UnknownSession)?;
        let slot = table
            .slots
            .get(i)
            .copied()
            .ok_or(KvAllocError::BadBlockIndex {
                index: i,
                session_blocks: table.slots.len(),
            })?;
        let Slot::Resident(p) = slot else {
            return Err(KvAllocError::SessionNotResident { slot: i });
        };
        if table.filled_tokens < (i + 1) * bs {
            return Err(KvAllocError::BlockNotFull { index: i });
        }
        if self.cached.contains_key(&hash) || self.cached_by_block.contains_key(&p) {
            return Ok(false);
        }
        self.refcount[p as usize] += 1;
        self.tick += 1;
        self.cached.insert(
            hash,
            CachedBlock {
                block: p,
                last_used: self.tick,
            },
        );
        self.cached_by_block.insert(p, hash);
        self.cache_stats.commits += 1;
        Ok(true)
    }

    /// The longest run of cached blocks matching `hashes` from the start: the
    /// physical block for `hashes[0]`, then `hashes[1]`, … stopping at the first
    /// miss. Each hit is marked recently used. Pure lookup otherwise: no
    /// refcount changes, so the result is only valid until the next mutation
    /// (use [`splice_cached_chain`](Self::splice_cached_chain) to claim it).
    pub fn lookup_chain(&mut self, hashes: &[BlockHash]) -> Vec<PhysBlockId> {
        let mut out = Vec::new();
        for h in hashes {
            let Some(entry) = self.cached.get_mut(h) else {
                break;
            };
            self.tick += 1;
            entry.last_used = self.tick;
            out.push(entry.block);
        }
        self.cache_stats.lookups += 1;
        self.cache_stats.block_hits += out.len() as u64;
        self.cache_stats.block_misses += (hashes.len() - out.len()) as u64;
        out
    }

    /// Attach the longest cached run matching `hashes` to the EMPTY session
    /// `dst` (shared copy-on-write: each block's refcount rises by one). Returns
    /// the number of tokens now covered, `blocks * block_size`, so the caller
    /// prefills only `prompt[covered..]`. A target that already holds blocks is
    /// refused before any mutation, like [`splice_prefix`](Self::splice_prefix).
    pub fn splice_cached_chain(
        &mut self,
        dst: SessionHandle,
        hashes: &[BlockHash],
    ) -> Result<usize, KvAllocError> {
        let dst_table = self.tables.get(&dst).ok_or(KvAllocError::UnknownSession)?;
        if !dst_table.slots.is_empty() {
            return Err(KvAllocError::PrefixTargetNotEmpty);
        }
        let blocks = self.lookup_chain(hashes);
        for &p in &blocks {
            self.refcount[p as usize] += 1;
            self.table_mut(dst)?.slots.push(Slot::Resident(p));
        }
        let tokens = blocks.len() * self.geom.block_size;
        self.table_mut(dst)?.filled_tokens = tokens;
        Ok(tokens)
    }

    /// Cached blocks that nothing but the index references (refcount 1), least
    /// recently used first: the candidates a policy may release to make room.
    /// A block a live session still reads is not listed (releasing it would free
    /// nothing).
    pub fn evictable_cached(&self) -> Vec<BlockHash> {
        let mut v: Vec<(u64, BlockHash)> = self
            .cached
            .iter()
            .filter(|(_, c)| self.refcount[c.block as usize] == 1)
            .map(|(h, c)| (c.last_used, *h))
            .collect();
        v.sort_unstable();
        v.into_iter().map(|(_, h)| h).collect()
    }

    /// Drop the index's reference on `hash`. The block returns to the free list
    /// if no session still references it.
    pub fn release_cached(&mut self, hash: BlockHash) -> Result<(), KvAllocError> {
        let entry = self
            .cached
            .remove(&hash)
            .ok_or(KvAllocError::UnknownCachedBlock)?;
        self.cached_by_block.remove(&entry.block);
        self.cache_stats.evictions += 1;
        let rc = &mut self.refcount[entry.block as usize];
        *rc -= 1;
        if *rc == 0 {
            self.free.push(entry.block);
        }
        Ok(())
    }

    /// The index counters since the pool was built.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache_stats
    }

    /// How many blocks the index currently holds.
    pub fn cached_blocks(&self) -> usize {
        self.cached.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvGeometry;

    fn pool(blocks: usize) -> KvBlockPool {
        KvBlockPool::new(KvGeometry {
            n_layers: 1,
            num_blocks: blocks,
            block_size: 4,
            n_kv_heads: 1,
            head_dim: 2,
            elem_size: 4,
        })
    }

    /// A session filled with `tokens` tokens, its full blocks committed under a
    /// chained hash of `ids`. Returns the session and the hash chain.
    fn session_with(
        p: &mut KvBlockPool,
        ids: &[u32],
        extra: &[u8],
    ) -> (SessionHandle, Vec<BlockHash>) {
        let s = p.open();
        p.append(s, ids.len()).unwrap();
        let mut chain = Vec::new();
        let mut parent = None;
        for (i, blk) in ids.chunks(4).enumerate() {
            if blk.len() < 4 {
                break;
            }
            let h = chain_block_hash(parent, blk, extra);
            p.commit_block(s, i, h).unwrap();
            chain.push(h);
            parent = Some(h);
        }
        (s, chain)
    }

    fn chain_of(ids: &[u32], extra: &[u8]) -> Vec<BlockHash> {
        let mut parent = None;
        let mut out = Vec::new();
        for blk in ids.chunks(4).filter(|b| b.len() == 4) {
            let h = chain_block_hash(parent, blk, extra);
            out.push(h);
            parent = Some(h);
        }
        out
    }

    #[test]
    fn chain_hash_depends_on_parent_tokens_extra_and_order() {
        let h0 = chain_block_hash(None, &[1, 2, 3, 4], b"");
        assert_eq!(h0, chain_block_hash(None, &[1, 2, 3, 4], b""));
        assert_ne!(h0, chain_block_hash(None, &[1, 2, 3, 5], b""), "tokens");
        assert_ne!(h0, chain_block_hash(None, &[4, 3, 2, 1], b""), "order");
        assert_ne!(
            h0,
            chain_block_hash(None, &[1, 2, 3, 4], b"tenant-a"),
            "extra"
        );
        assert_ne!(
            chain_block_hash(Some(h0), &[9, 9, 9, 9], b""),
            chain_block_hash(None, &[9, 9, 9, 9], b""),
            "parent"
        );
        assert_ne!(
            chain_block_hash(Some(h0), &[9, 9, 9, 9], b""),
            chain_block_hash(Some(h0 ^ 1), &[9, 9, 9, 9], b""),
            "parent identity"
        );
    }

    #[test]
    fn a_new_request_finds_the_shared_blocks_at_any_boundary() {
        let mut p = pool(16);
        // Request 1: system prompt (8 tokens = 2 blocks) + tail A (8) + a partial block.
        let sys: Vec<u32> = (100..108).collect();
        let tail_a: Vec<u32> = (200..208).collect();
        let r1: Vec<u32> = [&sys[..], &tail_a[..], &[7, 7]].concat();
        let (_s1, _c1) = session_with(&mut p, &r1, b"");
        assert_eq!(
            p.cached_blocks(),
            4,
            "4 full blocks committed, the partial one not"
        );

        // Request 2: same system prompt, DIFFERENT tail. Whole-prefix hashing
        // would miss; the chain matches the 2 shared blocks and stops.
        let tail_b: Vec<u32> = (300..308).collect();
        let r2: Vec<u32> = [&sys[..], &tail_b[..]].concat();
        let want = chain_of(&r2, b"");
        let s2 = p.open();
        assert_eq!(p.splice_cached_chain(s2, &want).unwrap(), 8);
        assert_eq!(p.session_blocks(s2), Some(2));
        assert_eq!(p.filled_tokens(s2), Some(8));
        // Shared with request 1's session AND the index.
        let t2 = p.session_block_table(s2).unwrap();
        assert!(t2.iter().all(|b| p.block_refcount(*b) == 3));
    }

    #[test]
    fn the_extra_key_isolates_tenants() {
        let mut p = pool(8);
        let ids: Vec<u32> = (0..8).collect();
        let (_s, _) = session_with(&mut p, &ids, b"tenant-a");
        assert_eq!(p.lookup_chain(&chain_of(&ids, b"tenant-a")).len(), 2);
        assert!(p.lookup_chain(&chain_of(&ids, b"tenant-b")).is_empty());
    }

    #[test]
    fn a_cached_block_outlives_its_session_and_stays_out_of_the_free_list() {
        let mut p = pool(4);
        let ids: Vec<u32> = (0..8).collect();
        let (s, chain) = session_with(&mut p, &ids, b"");
        assert_eq!(p.free_blocks(), 2);
        p.discard(s);
        assert_eq!(p.free_blocks(), 2, "the index still holds both blocks");
        // A later session claims them.
        let s2 = p.open();
        assert_eq!(p.splice_cached_chain(s2, &chain).unwrap(), 8);
        assert_eq!(p.free_blocks(), 2);
    }

    #[test]
    fn partial_blocks_are_refused_and_commits_are_idempotent() {
        let mut p = pool(4);
        let s = p.open();
        p.append(s, 6).unwrap(); // block 0 full, block 1 has 2 of 4
        assert_eq!(p.commit_block(s, 0, 11), Ok(true));
        assert_eq!(
            p.commit_block(s, 1, 12),
            Err(KvAllocError::BlockNotFull { index: 1 })
        );
        // Same hash again, or the same physical block under another hash: no-op.
        assert_eq!(p.commit_block(s, 0, 11), Ok(false));
        assert_eq!(p.commit_block(s, 0, 99), Ok(false));
        assert_eq!(p.cached_blocks(), 1);
        assert_eq!(
            p.commit_block(s, 5, 13),
            Err(KvAllocError::BadBlockIndex {
                index: 5,
                session_blocks: 2
            })
        );
    }

    #[test]
    fn eviction_candidates_are_lru_first_and_exclude_blocks_a_session_reads() {
        let mut p = pool(8);
        let a: Vec<u32> = (0..4).collect();
        let b: Vec<u32> = (10..14).collect();
        let c: Vec<u32> = (20..24).collect();
        let (sa, ca) = session_with(&mut p, &a, b"");
        let (sb, cb) = session_with(&mut p, &b, b"");
        let (sc, cc) = session_with(&mut p, &c, b"");
        // All three are still read by their sessions: nothing is evictable.
        assert!(p.evictable_cached().is_empty());
        p.discard(sa);
        p.discard(sb);
        p.discard(sc);
        assert_eq!(
            p.evictable_cached(),
            vec![ca[0], cb[0], cc[0]],
            "commit order = LRU"
        );
        // Touch `a`: it becomes the most recently used.
        assert_eq!(p.lookup_chain(&ca).len(), 1);
        assert_eq!(p.evictable_cached(), vec![cb[0], cc[0], ca[0]]);
        // Releasing frees the block.
        let before = p.free_blocks();
        p.release_cached(cb[0]).unwrap();
        assert_eq!(p.free_blocks(), before + 1);
        assert_eq!(
            p.release_cached(cb[0]),
            Err(KvAllocError::UnknownCachedBlock)
        );
    }

    #[test]
    fn releasing_a_block_a_session_still_reads_does_not_free_it() {
        let mut p = pool(4);
        let ids: Vec<u32> = (0..4).collect();
        let (s, chain) = session_with(&mut p, &ids, b"");
        let before = p.free_blocks();
        p.release_cached(chain[0]).unwrap();
        assert_eq!(p.free_blocks(), before, "the session's reference remains");
        p.discard(s);
        assert_eq!(p.free_blocks(), before + 1);
    }

    #[test]
    fn splicing_into_a_non_empty_session_is_refused_untouched() {
        let mut p = pool(8);
        let ids: Vec<u32> = (0..4).collect();
        let (_s, chain) = session_with(&mut p, &ids, b"");
        let busy = p.open();
        p.append(busy, 1).unwrap();
        let rc: Vec<u32> = (0..8).map(|b| p.block_refcount(b)).collect();
        assert_eq!(
            p.splice_cached_chain(busy, &chain),
            Err(KvAllocError::PrefixTargetNotEmpty)
        );
        let after: Vec<u32> = (0..8).map(|b| p.block_refcount(b)).collect();
        assert_eq!(rc, after);
        let gone = p.open();
        p.discard(gone);
        assert_eq!(
            p.splice_cached_chain(gone, &chain),
            Err(KvAllocError::UnknownSession)
        );
    }

    #[test]
    fn stats_count_hits_misses_commits_and_evictions() {
        let mut p = pool(16);
        assert_eq!(p.cache_stats(), CacheStats::default());
        let ids: Vec<u32> = (0..12).collect();
        let (_s, chain) = session_with(&mut p, &ids, b"");
        assert_eq!(p.cache_stats().commits, 3);
        // A duplicate commit is refused and must not count.
        p.commit_block(_s, 0, chain[0]).unwrap();
        assert_eq!(p.cache_stats().commits, 3);
        // Two shared blocks then a diverging third: 2 hits, 1 miss.
        let mut other: Vec<u32> = (0..8).collect();
        other.extend([99, 98, 97, 96]);
        assert_eq!(p.lookup_chain(&chain_of(&other, b"")).len(), 2);
        let st = p.cache_stats();
        assert_eq!((st.lookups, st.block_hits, st.block_misses), (1, 2, 1));
        // Splice counts its single inner lookup once.
        let dst = p.open();
        p.splice_cached_chain(dst, &chain).unwrap();
        let st = p.cache_stats();
        assert_eq!((st.lookups, st.block_hits, st.block_misses), (2, 5, 1));
        p.release_cached(chain[2]).unwrap();
        assert_eq!(p.cache_stats().evictions, 1);
        assert!(p.release_cached(chain[2]).is_err());
        assert_eq!(
            p.cache_stats().evictions,
            1,
            "a failed release is not an eviction"
        );
    }

    #[test]
    fn a_broken_chain_stops_at_the_first_miss() {
        let mut p = pool(8);
        let ids: Vec<u32> = (0..12).collect();
        let (_s, chain) = session_with(&mut p, &ids, b"");
        // Ask for blocks 0, 1 correct, then a wrong third hash, then the right 3rd.
        let asked = vec![chain[0], chain[1], 0xDEAD, chain[2]];
        assert_eq!(p.lookup_chain(&asked).len(), 2);
    }
}
