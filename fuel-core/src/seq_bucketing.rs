//! Sequence-length bucketing for dynamic shapes (Phase 6a).
//!
//! # Why this exists
//!
//! The lazy graph is shape-polymorphic at build time but every node
//! carries a concrete [`Shape`]. For autoregressive decode the context
//! length `N` grows by one each step, so naively the forward graph has
//! to be rebuilt every token. Graph construction is cheap per-node but
//! non-trivial at transformer scale (thousands of nodes × tens of
//! layers); rebuilding it once per token shows up in CPU decode traces
//! as real overhead.
//!
//! Bucketing fixes this by rounding `N` up to the next entry in a
//! fixed ladder of **bucket sizes** — e.g. `[64, 128, 256, 512, 1024,
//! 2048, 4096]`. A user whose actual context is 73 tokens runs the
//! 128-bucket graph, pads KV cache + attention mask to length 128,
//! and the mask zeros out the 55 slots that don't carry real data.
//! Adjacent decode steps share the same bucket (and therefore the
//! same graph) until `N` crosses the next ladder step — one graph
//! build amortized across many tokens.
//!
//! # Phase 6d supersession
//!
//! This module's existence is explicitly time-boxed. Phase 6d replaces
//! bucketing with **paged attention**, which consults a page table to
//! fetch only populated cache blocks — collapsing every decode shape
//! into a single execution path. When paged attention lands the
//! bucketing utility becomes dead code; callers migrate off it and
//! this module is removed.
//!
//! # What's here today
//!
//! Just the primitives: a [`pick_bucket`] function and a
//! [`BucketedLen`] helper that wraps an actual length + a ladder.
//! Integration into the [`generate()`](crate::lazy::LlamaModel::generate)
//! loop is a follow-up; it requires graph-memoization plumbing in
//! the executor that's a bigger change than this utility, and the
//! payoff only matters for long prompts where graph-rebuild overhead
//! is visible. The utility stands alone so downstream code can start
//! bucketing incrementally.

/// Round `seq_len` up to the smallest bucket ≥ `seq_len`. Panics if
/// `seq_len` exceeds the largest bucket in the table.
///
/// The `buckets` slice must be sorted ascending. Duplicates are
/// tolerated but wasteful. An empty `buckets` slice panics — callers
/// must supply at least one bucket.
///
/// # Example
///
/// ```
/// use fuel_core::seq_bucketing::pick_bucket;
/// assert_eq!(pick_bucket(50,  &[64, 128, 256]), 64);
/// assert_eq!(pick_bucket(64,  &[64, 128, 256]), 64);
/// assert_eq!(pick_bucket(73,  &[64, 128, 256]), 128);
/// assert_eq!(pick_bucket(256, &[64, 128, 256]), 256);
/// ```
pub fn pick_bucket(seq_len: usize, buckets: &[usize]) -> usize {
    assert!(!buckets.is_empty(), "pick_bucket: buckets slice is empty");
    for &b in buckets {
        if b >= seq_len {
            return b;
        }
    }
    panic!(
        "pick_bucket: seq_len {seq_len} exceeds the largest bucket {} — \
         either grow the bucket ladder or truncate the sequence",
        buckets.last().unwrap()
    );
}

/// The standard power-of-two bucket ladder used across Fuel's lazy
/// inference paths. Matches common serving-side conventions (vLLM /
/// TGI defaults) and covers typical LLM context sizes from short
/// chat turns up to 4K extended contexts.
///
/// This is just a convenience constant — callers are free to supply
/// their own ladder tuned to their workload (shorter buckets for
/// tight memory, wider buckets for low-latency decode).
pub const DEFAULT_BUCKETS: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096];

/// A wrapped (actual_len, bucket_len) pair. Simplifies downstream code
/// that needs both the true length (for attention mask generation) and
/// the padded length (for KV cache allocation / graph shapes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketedLen {
    /// The real sequence length — how many tokens actually carry data.
    /// Attention masks use this to zero out the padding slots.
    pub actual: usize,
    /// The padded length — `pick_bucket(actual, ladder)`. KV caches
    /// and attention input tensors are sized to this.
    pub bucket: usize,
}

impl BucketedLen {
    /// Build a `BucketedLen` from an actual length and a bucket ladder.
    pub fn new(actual: usize, buckets: &[usize]) -> Self {
        Self { actual, bucket: pick_bucket(actual, buckets) }
    }

    /// Number of padding positions between `actual` and `bucket`.
    pub fn padding(&self) -> usize {
        self.bucket - self.actual
    }

    /// Whether the sequence exactly fills its bucket — i.e. adding
    /// one more token will cross into the next bucket and require a
    /// graph rebuild.
    pub fn at_bucket_edge(&self) -> bool {
        self.padding() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_bucket_rounds_up() {
        let buckets = &[64, 128, 256, 512];
        assert_eq!(pick_bucket(1, buckets), 64);
        assert_eq!(pick_bucket(63, buckets), 64);
        assert_eq!(pick_bucket(64, buckets), 64);
        assert_eq!(pick_bucket(65, buckets), 128);
        assert_eq!(pick_bucket(127, buckets), 128);
        assert_eq!(pick_bucket(128, buckets), 128);
        assert_eq!(pick_bucket(129, buckets), 256);
        assert_eq!(pick_bucket(512, buckets), 512);
    }

    #[test]
    #[should_panic(expected = "seq_len 1000 exceeds the largest bucket")]
    fn pick_bucket_overflow_panics() {
        let _ = pick_bucket(1000, &[64, 128, 256]);
    }

    #[test]
    #[should_panic(expected = "buckets slice is empty")]
    fn pick_bucket_empty_panics() {
        let _ = pick_bucket(10, &[]);
    }

    #[test]
    fn bucketed_len_fields_and_padding() {
        let b = BucketedLen::new(73, &[64, 128, 256]);
        assert_eq!(b.actual, 73);
        assert_eq!(b.bucket, 128);
        assert_eq!(b.padding(), 55);
        assert!(!b.at_bucket_edge());

        let exact = BucketedLen::new(128, &[64, 128, 256]);
        assert_eq!(exact.padding(), 0);
        assert!(exact.at_bucket_edge());
    }

    #[test]
    fn default_buckets_is_powers_of_two() {
        assert_eq!(DEFAULT_BUCKETS, &[64, 128, 256, 512, 1024, 2048, 4096]);
        // The "next bucket is 2x the previous" invariant — anything
        // non-monotonic or non-geometric would surprise callers.
        for pair in DEFAULT_BUCKETS.windows(2) {
            assert_eq!(pair[1], pair[0] * 2);
        }
    }

    #[test]
    fn rebucket_simulation() {
        // Simulate a decode loop: prompt of 50, generate 200 tokens.
        // Count how many bucket transitions we cross (= graph rebuilds).
        let mut rebuilds = 0;
        let mut last_bucket = 0;
        for n in 50..=250 {
            let b = BucketedLen::new(n, DEFAULT_BUCKETS).bucket;
            if b != last_bucket {
                rebuilds += 1;
                last_bucket = b;
            }
        }
        // Lengths 50..=64 → bucket 64 (1 build)
        // Lengths 65..=128 → bucket 128 (1 build)
        // Lengths 129..=250 → bucket 256 (1 build)
        // Total: 3 graph builds for 201 tokens.
        assert_eq!(rebuilds, 3, "expected 3 bucket transitions in 50..250");
    }
}
