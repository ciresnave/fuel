//! StreamingLLM: Sink-token + recent-window KV cache management.
//!
//! When generating very long sequences that exceed the model's training context window,
//! standard KV caches either truncate (losing early context) or OOM. StreamingLLM
//! (Xiao et al., 2023) observes that **attention sinks** — the first few tokens in the
//! sequence — accumulate disproportionate attention mass regardless of their semantic
//! content. Keeping these "sink tokens" plus a sliding window of recent tokens allows
//! stable generation well beyond the training context, trading perfect recall of middle
//! context for unbounded sequence support.
//!
//! # Architecture
//!
//! [`StreamingPolicy`] manages a logical token budget split into two regions:
//!
//! ```text
//! ┌──────────────┬─────────────────────────────────────────────┐
//! │  Sink tokens  │            Recent window                    │
//! │  (fixed)      │            (sliding)                        │
//! └──────────────┴─────────────────────────────────────────────┘
//!  0..num_sinks   budget - num_sinks tokens (most recent)
//! ```
//!
//! - **Sink tokens** (default: 4) are always retained. These are the first tokens in the
//!   sequence and serve as stable attention anchors.
//! - The **recent window** holds the most recent tokens up to the remaining budget.
//! - Tokens between the sinks and the recent window are evicted.
//!
//! # Example
//!
//! ```rust
//! use candle_inference::streaming::StreamingPolicy;
//!
//! // Keep 4 sink tokens + a window of 60 recent tokens = 64 total budget
//! let policy = StreamingPolicy::new(4, 64);
//! assert_eq!(policy.num_sinks(), 4);
//! assert_eq!(policy.budget(), 64);
//! assert_eq!(policy.window_size(), 60);
//!
//! // With 100 tokens in the cache, which indices should we keep?
//! let keep = policy.select_keep(100);
//! // Keeps: [0, 1, 2, 3] (sinks) + [40, 41, ..., 99] (recent 60)
//! assert_eq!(keep.len(), 64);
//! assert_eq!(&keep[..4], &[0, 1, 2, 3]);
//! assert_eq!(keep[4], 40);
//! assert_eq!(*keep.last().unwrap(), 99);
//! ```
//!
//! # Reference
//!
//! Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024).

/// Streaming KV cache policy with fixed attention sinks and a sliding recent window.
///
/// This policy does not own KV tensors — it computes which token indices to keep,
/// and the caller performs the actual tensor slicing.
#[derive(Debug, Clone)]
pub struct StreamingPolicy {
    /// Number of initial tokens to always retain (attention sinks).
    num_sinks: usize,
    /// Total token budget (sinks + recent window).
    budget: usize,
}

impl StreamingPolicy {
    /// Creates a new streaming policy.
    ///
    /// * `num_sinks` — Number of initial tokens to always keep (typically 1–4).
    /// * `budget` — Total token budget. Must be greater than `num_sinks`.
    ///
    /// # Panics
    ///
    /// Panics if `budget <= num_sinks` (there must be room for at least one recent token).
    pub fn new(num_sinks: usize, budget: usize) -> Self {
        assert!(
            budget > num_sinks,
            "budget ({budget}) must be > num_sinks ({num_sinks})"
        );
        Self { num_sinks, budget }
    }

    /// Returns the number of sink tokens retained.
    pub fn num_sinks(&self) -> usize {
        self.num_sinks
    }

    /// Returns the total token budget (sinks + window).
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Returns the size of the recent-token window (`budget - num_sinks`).
    pub fn window_size(&self) -> usize {
        self.budget - self.num_sinks
    }

    /// Returns `true` if the cache needs eviction at the given sequence length.
    pub fn needs_eviction(&self, current_seq_len: usize) -> bool {
        current_seq_len > self.budget
    }

    /// Computes the token indices to keep for a cache of `current_seq_len` tokens.
    ///
    /// Returns a sorted vector of indices in `0..current_seq_len`. If the current
    /// length is within budget, all indices are returned (no eviction needed).
    pub fn select_keep(&self, current_seq_len: usize) -> Vec<usize> {
        if current_seq_len <= self.budget {
            // Everything fits, keep all
            return (0..current_seq_len).collect();
        }

        let window = self.window_size();
        let recent_start = current_seq_len - window;

        let mut keep = Vec::with_capacity(self.budget);

        // Sink tokens (first num_sinks positions)
        let sinks_end = self.num_sinks.min(current_seq_len);
        keep.extend(0..sinks_end);

        // Recent window (last `window` positions)
        // Avoid overlap if recent_start < sinks_end
        let actual_start = recent_start.max(sinks_end);
        keep.extend(actual_start..current_seq_len);

        keep
    }

    /// Computes the token indices to evict for a cache of `current_seq_len` tokens.
    ///
    /// Returns indices of tokens that should be removed, sorted ascending. Empty
    /// if no eviction is needed.
    pub fn select_evict(&self, current_seq_len: usize) -> Vec<usize> {
        if current_seq_len <= self.budget {
            return Vec::new();
        }

        let keep = self.select_keep(current_seq_len);
        let keep_set: std::collections::HashSet<usize> = keep.into_iter().collect();

        (0..current_seq_len)
            .filter(|i| !keep_set.contains(i))
            .collect()
    }

    /// Computes the corrected position IDs for the retained tokens.
    ///
    /// After eviction, the model's RoPE (Rotary Position Embedding) must use the
    /// original absolute positions, not the compressed sequential indices. This method
    /// returns the original position for each retained token.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle_inference::streaming::StreamingPolicy;
    ///
    /// let policy = StreamingPolicy::new(2, 6);
    /// // 10 tokens → keep [0, 1] (sinks) + [6, 7, 8, 9] (recent 4)
    /// let positions = policy.position_ids(10);
    /// assert_eq!(positions, vec![0, 1, 6, 7, 8, 9]);
    /// ```
    pub fn position_ids(&self, current_seq_len: usize) -> Vec<usize> {
        // The kept indices ARE the original positions
        self.select_keep(current_seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_streaming_policy() {
        let policy = StreamingPolicy::new(4, 64);
        assert_eq!(policy.num_sinks(), 4);
        assert_eq!(policy.budget(), 64);
        assert_eq!(policy.window_size(), 60);
    }

    #[test]
    fn no_eviction_when_within_budget() {
        let policy = StreamingPolicy::new(4, 64);
        let keep = policy.select_keep(32);
        assert_eq!(keep.len(), 32);
        assert_eq!(keep, (0..32).collect::<Vec<_>>());
        assert!(!policy.needs_eviction(32));
    }

    #[test]
    fn no_eviction_at_exact_budget() {
        let policy = StreamingPolicy::new(4, 64);
        let keep = policy.select_keep(64);
        assert_eq!(keep.len(), 64);
        assert!(!policy.needs_eviction(64));
    }

    #[test]
    fn eviction_beyond_budget() {
        let policy = StreamingPolicy::new(4, 64);
        assert!(policy.needs_eviction(100));

        let keep = policy.select_keep(100);
        assert_eq!(keep.len(), 64);

        // First 4 are sinks
        assert_eq!(&keep[..4], &[0, 1, 2, 3]);

        // Last 60 are recent window
        assert_eq!(keep[4], 40); // 100 - 60 = 40
        assert_eq!(*keep.last().unwrap(), 99);
    }

    #[test]
    fn evict_indices() {
        let policy = StreamingPolicy::new(2, 6);
        // 10 tokens: keep [0, 1] + [6, 7, 8, 9] → evict [2, 3, 4, 5]
        let evict = policy.select_evict(10);
        assert_eq!(evict, vec![2, 3, 4, 5]);
    }

    #[test]
    fn position_ids_preserve_original() {
        let policy = StreamingPolicy::new(2, 6);
        let positions = policy.position_ids(10);
        // Keep: [0, 1] sinks + [6, 7, 8, 9] recent
        assert_eq!(positions, vec![0, 1, 6, 7, 8, 9]);
    }

    #[test]
    fn single_sink_token() {
        let policy = StreamingPolicy::new(1, 5);
        let keep = policy.select_keep(20);
        assert_eq!(keep.len(), 5);
        assert_eq!(keep[0], 0); // sink
        assert_eq!(keep[1], 16); // recent window starts at 20 - 4 = 16
        assert_eq!(*keep.last().unwrap(), 19);
    }

    #[test]
    fn overlap_sinks_and_window() {
        // When sequence is short enough that sinks overlap with recent window
        let policy = StreamingPolicy::new(4, 8);
        let keep = policy.select_keep(6);
        // Budget = 8, but only 6 tokens exist → keep all
        assert_eq!(keep.len(), 6);
        assert_eq!(keep, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn just_over_budget() {
        let policy = StreamingPolicy::new(2, 6);
        // 7 tokens: keep [0, 1] + [3, 4, 5, 6] → evict [2]
        let keep = policy.select_keep(7);
        assert_eq!(keep.len(), 6);
        assert_eq!(&keep[..2], &[0, 1]);
        assert_eq!(&keep[2..], &[3, 4, 5, 6]);

        let evict = policy.select_evict(7);
        assert_eq!(evict, vec![2]);
    }

    #[test]
    #[should_panic(expected = "budget (4) must be > num_sinks (4)")]
    fn budget_must_exceed_sinks() {
        StreamingPolicy::new(4, 4);
    }

    #[test]
    fn large_sequence() {
        let policy = StreamingPolicy::new(4, 2048);
        let keep = policy.select_keep(100_000);
        assert_eq!(keep.len(), 2048);
        // Sinks
        assert_eq!(&keep[..4], &[0, 1, 2, 3]);
        // Recent window starts at 100000 - 2044 = 97956
        assert_eq!(keep[4], 97956);
        assert_eq!(*keep.last().unwrap(), 99999);
    }

    #[test]
    fn empty_evict_when_under_budget() {
        let policy = StreamingPolicy::new(2, 10);
        let evict = policy.select_evict(5);
        assert!(evict.is_empty());
    }
}
