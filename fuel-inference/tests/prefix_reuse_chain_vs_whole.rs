// SPDX-License-Identifier: MIT OR Apache-2.0
//! Measured comparison for ROADMAP item 17, row H20: how many prompt tokens does
//! block-chain matching (`fuel-kv-pool`'s content-addressed block index) serve
//! from cache, against `PrefixCache`'s whole-prefix keys, on a synthetic
//! multi-turn trace?
//!
//! Trace: one shared system prompt, a few conversation "tails" (the history
//! different users build on top of it), and a unique final user turn per
//! request. Everything is deterministic (seeded), counts only, no timing.
//!
//! What `PrefixCache` can reuse is what the caller stored. Three policies:
//!
//! * `whole/end`: store each prompt when its request finishes. A later request
//!   that shares only a part never matches.
//! * `whole/bounds`: the caller also stores the logical boundaries (system
//!   prompt, system + tail) after each request: the best case for whole-prefix
//!   keys, and it needs the caller to know them.
//! * `chain`: commit every full block under a chained hash; a request splices
//!   the longest matching run, at any block boundary, with no boundary
//!   annotations.
//!
//! The model assumption: reuse is byte-exact for the matched tokens (that is
//! all this counts); quality and speed are not measured here.

use fuel::lazy::Tensor;
use fuel::{DType, Device};
use fuel_inference::prefix_cache::PrefixCache;
use fuel_kv_pool::{KvAllocError, KvBlockPool, KvGeometry, chain_block_hash};

const BLOCK: usize = 16;
const SYS: usize = 512;
const TAIL: usize = 256;
const VARIANTS: usize = 4;
const REQUESTS: usize = 80;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// One request: the full prompt and the logical boundary lengths inside it.
struct Request {
    prompt: Vec<u32>,
    sys_len: usize,
    sys_tail_len: usize,
}

fn trace() -> Vec<Request> {
    let sys: Vec<u32> = (1_000..1_000 + SYS as u32).collect();
    let tails: Vec<Vec<u32>> = (0..VARIANTS as u32)
        .map(|v| (0..TAIL as u32).map(|i| 10_000 + v * 1_000 + i).collect())
        .collect();
    let mut rng = Rng(42);
    (0..REQUESTS)
        .map(|i| {
            let v = (rng.next() as usize) % VARIANTS;
            let user_len = 50 + (rng.next() as usize) % 100;
            let user: Vec<u32> = (0..user_len as u32)
                .map(|j| 100_000 + i as u32 * 1_000 + j)
                .collect();
            let prompt = [&sys[..], &tails[v][..], &user[..]].concat();
            Request {
                prompt,
                sys_len: SYS,
                sys_tail_len: SYS + TAIL,
            }
        })
        .collect()
}

fn tiny_kv() -> Vec<(Tensor, Tensor)> {
    let z = || Tensor::zeros((1, 1, 1, 1), DType::F32, &Device::cpu()).unwrap();
    vec![(z(), z())]
}

/// Tokens served from `PrefixCache` over the trace under the given policy.
fn whole_prefix_hits(reqs: &[Request], store_bounds: bool) -> usize {
    let mut cache = PrefixCache::new(1_000);
    let mut hits = 0;
    for r in reqs {
        if let Some((len, _)) = cache.longest_prefix_match(&r.prompt) {
            hits += len;
        }
        cache.insert(&r.prompt, tiny_kv());
        if store_bounds {
            for len in [r.sys_len, r.sys_tail_len] {
                if !cache.contains(&r.prompt[..len]) {
                    cache.insert(&r.prompt[..len], tiny_kv());
                }
            }
        }
    }
    hits
}

fn chain_of(prompt: &[u32]) -> Vec<u128> {
    let mut parent = None;
    prompt
        .chunks(BLOCK)
        .filter(|b| b.len() == BLOCK)
        .map(|b| {
            let h = chain_block_hash(parent, b, b"");
            parent = Some(h);
            h
        })
        .collect()
}

/// Tokens served from the block index over the trace, with a pool of
/// `num_blocks` blocks. When the pool runs short the least-recently-used
/// unreferenced cached blocks are released first. Returns (hits, evictions).
fn chain_hits(reqs: &[Request], num_blocks: usize) -> (usize, usize) {
    let mut pool = KvBlockPool::new(KvGeometry {
        n_layers: 1,
        num_blocks,
        block_size: BLOCK,
        n_kv_heads: 1,
        head_dim: 2,
        elem_size: 4,
    });
    let (mut hits, mut evictions) = (0, 0);
    for r in reqs {
        let chain = chain_of(&r.prompt);
        let s = pool.open();
        let covered = pool.splice_cached_chain(s, &chain).unwrap();
        hits += covered;
        let rest = r.prompt.len() - covered;
        let need = pool.blocks_required(covered, rest);
        while pool.free_blocks() < need {
            let victim = *pool
                .evictable_cached()
                .first()
                .expect("an evictable block exists when the pool is merely full of cache");
            pool.release_cached(victim).unwrap();
            evictions += 1;
        }
        match pool.append(s, rest) {
            Ok(()) => {}
            Err(KvAllocError::OutOfBlocks { .. }) => panic!("sized above"),
            Err(e) => panic!("{e:?}"),
        }
        for (i, h) in chain.iter().enumerate().skip(covered / BLOCK) {
            pool.commit_block(s, i, *h).unwrap();
        }
        pool.discard(s);
    }
    (hits, evictions)
}

#[test]
fn block_chain_matching_vs_whole_prefix_keys_on_a_multi_turn_trace() {
    let reqs = trace();
    let total: usize = reqs.iter().map(|r| r.prompt.len()).sum();
    let pct = |h: usize| 100.0 * h as f64 / total as f64;

    let w_end = whole_prefix_hits(&reqs, false);
    let w_bounds = whole_prefix_hits(&reqs, true);
    let (c_big, ev_big) = chain_hits(&reqs, 4_096);
    // 160 blocks = 2,560 tokens: holds the system prompt (32 blocks), all four
    // tails (4 x 16) and about half of the transient user turns: forces eviction.
    let (c_small, ev_small) = chain_hits(&reqs, 160);

    println!("trace: {REQUESTS} requests, {total} prompt tokens, block {BLOCK}");
    println!(
        "whole-prefix, store at request end : {w_end:>6} tokens ({:.1}%)",
        pct(w_end)
    );
    println!(
        "whole-prefix, store logical bounds : {w_bounds:>6} tokens ({:.1}%)",
        pct(w_bounds)
    );
    println!(
        "block chain, roomy pool (4096)     : {c_big:>6} tokens ({:.1}%), {ev_big} evictions",
        pct(c_big)
    );
    println!(
        "block chain, tight pool (160)      : {c_small:>6} tokens ({:.1}%), {ev_small} evictions",
        pct(c_small)
    );

    // Storing only whole prompts never matches: every prompt ends in a unique turn.
    assert_eq!(w_end, 0, "whole-prefix keys at request end must not match");
    // The chain finds shared blocks with no boundary annotations...
    assert!(pct(c_big) > 60.0, "chain hit rate {:.1}%", pct(c_big));
    // ...and is within block rounding of the annotated whole-prefix best case.
    assert!(
        c_big as f64 >= 0.95 * w_bounds as f64,
        "chain {c_big} vs annotated whole-prefix {w_bounds}"
    );
    // Eviction is exercised and the hot shared blocks survive it.
    assert!(ev_small > 0, "the tight pool must evict");
    assert!(
        c_small as f64 >= 0.80 * c_big as f64,
        "tight pool kept {c_small} of {c_big}"
    );
}
