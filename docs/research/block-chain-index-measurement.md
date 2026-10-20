# Block-chain prefix matching vs whole-prefix keys: a measured comparison

ROADMAP item 17, row **H20**. Measured 2026-10-20 by `fuel-inference/tests/prefix_reuse_chain_vs_whole.rs` (deterministic, counts only, debug build, no timing). It compares the new content-addressed block index in `fuel-kv-pool` (`commit_block`, `lookup_chain`, `splice_cached_chain`, `evictable_cached`, `release_cached`, with the convenience `chain_block_hash`) against `fuel_inference::prefix_cache::PrefixCache`, whose keys are a hash of the whole token prefix.

## The trace

80 requests of one shared system prompt (512 tokens), one of 4 conversation tails (256 tokens each) chosen at random, and a unique final user turn (50-149 tokens). 69,506 prompt tokens in total. Block size 16. Everything is synthetic (distinct token ids), so a "hit" is exact: this counts how many prompt tokens a policy can serve from cache, not what serving them is worth.

## Results

| policy | tokens served from cache | share of prompt tokens |
|---|---:|---:|
| whole-prefix keys, stored when each request finishes (`longest_prefix_match` + `insert(prompt)`) | 0 | 0.0% |
| whole-prefix keys, caller also stores the system-prompt and system+tail boundaries | 59,904 | 86.2% |
| block chain, roomy pool (4,096 blocks), no boundary knowledge | 59,904 | 86.2% |
| block chain, tight pool (160 blocks = 2,560 tokens), LRU release of unreferenced cached blocks | 59,392 | 85.4% (431 evictions) |

## How to read it

- **The gain is not needing the boundaries.** Storing whole prompts never matches, because every prompt ends in a turn nobody else has: 0%. The chain finds the shared run at any block boundary with no annotations: 86.2%. A caller that already knows where the shared parts end and stores exactly those prefixes gets the same number. Here the two tie exactly because the logical boundaries (512 and 768 tokens) are multiples of the block size; a boundary inside a block would cost the chain up to 15 tokens per request (the partial block) and cost the annotated whole-prefix policy nothing.
- **Eviction works on the hot set.** With a pool that holds only 2,560 tokens, the index released 431 least-recently-used blocks and still served 85.4%: the system prompt and the four tails stay hot, the unique user turns are what get evicted. The release order is the pool's `evictable_cached()` (LRU, only blocks no session still reads); which block to release is the caller's choice, the pool only lists the candidates.
- **Cost (analysis, not measurement).** `PrefixCache::longest_prefix_match` as written hashes every prefix length from the longest down, each hash over that many tokens: O(L^2) token hashes per lookup. The chain computes one hash per block and does one map lookup per block: O(L). No timings were taken for either.
- **What is not shown:** speed, output quality, behaviour under concurrent sessions, adversarial hash inputs (`chain_block_hash` is a 128-bit non-cryptographic mix; a multi-tenant consumer should fold a secret into `extra` or supply its own hash, and the `extra` key is what isolates tenants, tested), or any real model's KV bytes (the trace uses the pool's metadata only; `PrefixCache` stores 1-element tensors for the comparison).

## Reproduce

```text
cargo test -p fuel-inference --test prefix_reuse_chain_vs_whole -- --nocapture
```
