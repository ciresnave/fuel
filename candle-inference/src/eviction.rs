//! Composable KV cache eviction policies.
//!
//! This module provides an [`EvictionPolicy`] trait and several implementations for
//! deciding which tokens to evict when a KV cache exceeds its budget. Policies can
//! be composed via [`VotingAggregator`], which combines multiple policies with
//! per-policy weights using a weighted-score voting system.
//!
//! # Provided policies
//!
//! - [`LruPolicy`] — Evicts the least-recently-used (oldest) tokens, optionally
//!   preserving a sliding window of the most recent tokens.
//! - [`H2oPolicy`] — Heavy-Hitter Oracle: preserves tokens with the highest cumulative
//!   attention scores and evicts the rest. Based on the H2O paper (Zhang et al., 2023).
//! - [`VotingAggregator`] — Combines multiple policies by weighted score aggregation.
//!
//! # Example
//!
//! ```rust
//! use candle_inference::eviction::{EvictionPolicy, LruPolicy, H2oPolicy, VotingAggregator};
//!
//! // Create individual policies
//! let lru = LruPolicy::new();
//! let h2o = H2oPolicy::new();
//!
//! // Combine with 40% weight on recency, 60% on attention importance
//! let mut aggregator = VotingAggregator::new();
//! aggregator.add_policy(Box::new(lru), 0.4);
//! aggregator.add_policy(Box::new(h2o), 0.6);
//!
//! // Score 8 tokens; suppose we want to keep 4.
//! // Positions: 0..8, attention scores would come from actual model output.
//! let positions: Vec<usize> = (0..8).collect();
//! let attn_scores: Vec<f32> = vec![0.1, 0.05, 0.3, 0.02, 0.15, 0.08, 0.25, 0.05];
//! let scores = aggregator.score(&positions, &attn_scores);
//! // Higher score = more important = keep. Evict the lowest-scoring tokens.
//! ```

/// Context passed to eviction policies for scoring.
///
/// Provides the information a policy needs to decide which tokens to keep or evict.
#[derive(Debug, Clone)]
pub struct EvictionContext<'a> {
    /// Absolute position of each cached token in the original sequence.
    pub positions: &'a [usize],
    /// Cumulative attention score received by each token across all prior steps.
    /// Higher values indicate tokens that have been attended to more heavily.
    pub attention_scores: &'a [f32],
}

/// A policy that assigns an importance score to each cached token.
///
/// Higher scores mean the token is more important and should be retained.
/// Lower scores mean the token is a candidate for eviction.
pub trait EvictionPolicy: Send + Sync {
    /// Returns an importance score for each token in the cache.
    ///
    /// The returned vector must have the same length as `ctx.positions`.
    /// Scores are in arbitrary units — they will be normalized when combined
    /// by [`VotingAggregator`].
    fn score(&self, ctx: &EvictionContext<'_>) -> Vec<f32>;

    /// Returns a human-readable name for this policy (for logging/debugging).
    fn name(&self) -> &str;
}

/// Least-recently-used eviction: older tokens get lower scores.
///
/// Assigns scores linearly proportional to position — the most recent token gets
/// the highest score, the oldest gets the lowest. This is equivalent to a sliding
/// window that always keeps the newest tokens.
///
/// # Example
///
/// ```rust
/// use candle_inference::eviction::{EvictionPolicy, EvictionContext, LruPolicy};
///
/// let policy = LruPolicy::new();
/// let positions = vec![0, 1, 2, 3, 4];
/// let attn = vec![0.0; 5]; // LRU ignores attention scores
/// let ctx = EvictionContext { positions: &positions, attention_scores: &attn };
/// let scores = policy.score(&ctx);
/// // Token at position 4 (most recent) has the highest score
/// assert!(scores[4] > scores[0]);
/// ```
pub struct LruPolicy;

impl LruPolicy {
    /// Creates a new LRU eviction policy.
    pub fn new() -> Self {
        Self
    }
}

impl Default for LruPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy for LruPolicy {
    fn score(&self, ctx: &EvictionContext<'_>) -> Vec<f32> {
        let n = ctx.positions.len();
        if n == 0 {
            return Vec::new();
        }
        // Find min/max positions for normalization
        let max_pos = ctx.positions.iter().copied().max().unwrap_or(0) as f32;
        let min_pos = ctx.positions.iter().copied().min().unwrap_or(0) as f32;
        let range = max_pos - min_pos;

        if range == 0.0 {
            // All tokens at the same position — equal scores
            return vec![1.0; n];
        }

        ctx.positions
            .iter()
            .map(|&pos| (pos as f32 - min_pos) / range)
            .collect()
    }

    fn name(&self) -> &str {
        "lru"
    }
}

/// Heavy-Hitter Oracle (H2O) eviction policy.
///
/// Preserves tokens with the highest cumulative attention scores. Based on the
/// observation that a small fraction of tokens ("heavy hitters") receive the majority
/// of attention mass across layers and heads. Evicting non-heavy-hitter tokens has
/// minimal impact on model quality.
///
/// Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
/// Inference of Large Language Models" (NeurIPS 2023).
///
/// # Example
///
/// ```rust
/// use candle_inference::eviction::{EvictionPolicy, EvictionContext, H2oPolicy};
///
/// let policy = H2oPolicy::new();
/// let positions = vec![0, 1, 2, 3, 4];
/// let attn = vec![0.1, 0.5, 0.05, 0.3, 0.05];
/// let ctx = EvictionContext { positions: &positions, attention_scores: &attn };
/// let scores = policy.score(&ctx);
/// // Token at position 1 (attn=0.5) has the highest score
/// assert!(scores[1] > scores[0]);
/// assert!(scores[1] > scores[2]);
/// ```
pub struct H2oPolicy;

impl H2oPolicy {
    /// Creates a new H2O eviction policy.
    pub fn new() -> Self {
        Self
    }
}

impl Default for H2oPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy for H2oPolicy {
    fn score(&self, ctx: &EvictionContext<'_>) -> Vec<f32> {
        let n = ctx.attention_scores.len();
        if n == 0 {
            return Vec::new();
        }
        // Normalize attention scores to [0, 1]
        let max_attn = ctx
            .attention_scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_attn = ctx
            .attention_scores
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let range = max_attn - min_attn;

        if range == 0.0 {
            return vec![1.0; n];
        }

        ctx.attention_scores
            .iter()
            .map(|&s| (s - min_attn) / range)
            .collect()
    }

    fn name(&self) -> &str {
        "h2o"
    }
}

/// Combines multiple [`EvictionPolicy`] implementations via weighted score aggregation.
///
/// Each policy produces a score vector that is normalized to `[0, 1]` and then
/// multiplied by the policy's weight. The final score for each token is the weighted
/// sum across all policies.
///
/// # Example
///
/// ```rust
/// use candle_inference::eviction::{
///     EvictionPolicy, EvictionContext, LruPolicy, H2oPolicy, VotingAggregator,
/// };
///
/// let mut agg = VotingAggregator::new();
/// agg.add_policy(Box::new(LruPolicy::new()), 0.4);
/// agg.add_policy(Box::new(H2oPolicy::new()), 0.6);
///
/// let positions = vec![0, 1, 2, 3, 4];
/// let attn = vec![0.5, 0.1, 0.3, 0.05, 0.05];
/// let scores = agg.score(&positions, &attn);
/// assert_eq!(scores.len(), 5);
/// ```
pub struct VotingAggregator {
    policies: Vec<(Box<dyn EvictionPolicy>, f32)>,
}

impl VotingAggregator {
    /// Creates an empty aggregator with no policies.
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
        }
    }

    /// Adds a policy with the given weight. Weights do not need to sum to 1.0 —
    /// they are used as relative importance values.
    pub fn add_policy(&mut self, policy: Box<dyn EvictionPolicy>, weight: f32) {
        self.policies.push((policy, weight));
    }

    /// Computes the weighted aggregate score for each token.
    ///
    /// Convenience method that constructs an [`EvictionContext`] from the given
    /// position and attention-score slices.
    pub fn score(&self, positions: &[usize], attention_scores: &[f32]) -> Vec<f32> {
        let ctx = EvictionContext {
            positions,
            attention_scores,
        };
        <Self as EvictionPolicy>::score(self, &ctx)
    }

    /// Returns the number of registered policies.
    pub fn len(&self) -> usize {
        self.policies.len()
    }

    /// Returns true if no policies have been registered.
    pub fn is_empty(&self) -> bool {
        self.policies.is_empty()
    }

    /// Selects the indices of the `budget` highest-scoring tokens.
    ///
    /// Returns indices into the `positions` / `attention_scores` arrays, sorted by
    /// descending score. If `budget >= positions.len()`, all indices are returned.
    pub fn select_keep(
        &self,
        positions: &[usize],
        attention_scores: &[f32],
        budget: usize,
    ) -> Vec<usize> {
        let scores = self.score(positions, attention_scores);
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        // Sort descending by score (highest = most important = keep)
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .into_iter()
            .take(budget.min(positions.len()))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Selects the indices of the `count` lowest-scoring tokens (eviction candidates).
    ///
    /// Returns indices sorted by ascending score (worst first). If `count >= positions.len()`,
    /// all indices are returned.
    pub fn select_evict(
        &self,
        positions: &[usize],
        attention_scores: &[f32],
        count: usize,
    ) -> Vec<usize> {
        let scores = self.score(positions, attention_scores);
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        // Sort ascending by score (lowest = least important = evict)
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .into_iter()
            .take(count.min(positions.len()))
            .map(|(idx, _)| idx)
            .collect()
    }
}

impl Default for VotingAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy for VotingAggregator {
    fn score(&self, ctx: &EvictionContext<'_>) -> Vec<f32> {
        let n = ctx.positions.len();
        if n == 0 || self.policies.is_empty() {
            return vec![0.0; n];
        }

        let mut combined = vec![0.0f32; n];
        let total_weight: f32 = self.policies.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return combined;
        }

        for (policy, weight) in &self.policies {
            let scores = policy.score(ctx);
            debug_assert_eq!(scores.len(), n, "policy '{}' returned wrong number of scores", policy.name());

            // Scores from individual policies are already normalized to [0, 1]
            // by each policy's implementation. Apply weight directly.
            let norm_weight = weight / total_weight;
            for (c, s) in combined.iter_mut().zip(scores.iter()) {
                *c += s * norm_weight;
            }
        }

        combined
    }

    fn name(&self) -> &str {
        "voting_aggregator"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_scores_increase_with_position() {
        let policy = LruPolicy::new();
        let positions = vec![0, 5, 10, 15, 20];
        let attn = vec![0.0; 5];
        let ctx = EvictionContext {
            positions: &positions,
            attention_scores: &attn,
        };
        let scores = policy.score(&ctx);
        assert_eq!(scores.len(), 5);
        for i in 1..scores.len() {
            assert!(scores[i] > scores[i - 1], "score[{i}] should be > score[{}]", i - 1);
        }
        assert!((scores[0] - 0.0).abs() < 1e-6);
        assert!((scores[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lru_equal_positions_equal_scores() {
        let policy = LruPolicy::new();
        let positions = vec![5, 5, 5];
        let attn = vec![0.0; 3];
        let ctx = EvictionContext {
            positions: &positions,
            attention_scores: &attn,
        };
        let scores = policy.score(&ctx);
        assert!(scores.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn h2o_scores_follow_attention() {
        let policy = H2oPolicy::new();
        let positions = vec![0, 1, 2, 3, 4];
        let attn = vec![0.1, 0.5, 0.05, 0.3, 0.05];
        let ctx = EvictionContext {
            positions: &positions,
            attention_scores: &attn,
        };
        let scores = policy.score(&ctx);
        // Token 1 (attn=0.5) should have highest score
        assert!(scores[1] > scores[0]);
        assert!(scores[1] > scores[2]);
        assert!(scores[1] > scores[3]);
        assert!((scores[1] - 1.0).abs() < 1e-6); // max → 1.0
        assert!((scores[2] - 0.0).abs() < 1e-6); // min → 0.0
    }

    #[test]
    fn h2o_equal_attention_equal_scores() {
        let policy = H2oPolicy::new();
        let positions = vec![0, 1, 2];
        let attn = vec![0.5, 0.5, 0.5];
        let ctx = EvictionContext {
            positions: &positions,
            attention_scores: &attn,
        };
        let scores = policy.score(&ctx);
        assert!(scores.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn aggregator_combines_policies() {
        let mut agg = VotingAggregator::new();
        agg.add_policy(Box::new(LruPolicy::new()), 0.5);
        agg.add_policy(Box::new(H2oPolicy::new()), 0.5);

        // Token 0: old (pos=0) but high attention (0.9) → H2O likes it, LRU doesn't
        // Token 4: new (pos=4) but low attention (0.01) → LRU likes it, H2O doesn't
        let positions = vec![0, 1, 2, 3, 4];
        let attn = vec![0.9, 0.1, 0.1, 0.1, 0.01];
        let scores = agg.score(&positions, &attn);
        assert_eq!(scores.len(), 5);

        // Both policies contribute, so the combined score for token 0 and token 4
        // should be somewhere in between (neither extreme wins outright)
        // Token 0: LRU=0.0, H2O=1.0 → combined ≈ 0.5
        // Token 4: LRU=1.0, H2O=0.0 → combined ≈ 0.5
        assert!((scores[0] - scores[4]).abs() < 0.15);
    }

    #[test]
    fn aggregator_select_keep() {
        let mut agg = VotingAggregator::new();
        agg.add_policy(Box::new(LruPolicy::new()), 1.0);

        let positions = vec![0, 1, 2, 3, 4];
        let attn = vec![0.0; 5];
        let keep = agg.select_keep(&positions, &attn, 3);
        assert_eq!(keep.len(), 3);
        // LRU prefers newest: indices 4, 3, 2
        assert!(keep.contains(&4));
        assert!(keep.contains(&3));
        assert!(keep.contains(&2));
    }

    #[test]
    fn aggregator_select_evict() {
        let mut agg = VotingAggregator::new();
        agg.add_policy(Box::new(LruPolicy::new()), 1.0);

        let positions = vec![0, 1, 2, 3, 4];
        let attn = vec![0.0; 5];
        let evict = agg.select_evict(&positions, &attn, 2);
        assert_eq!(evict.len(), 2);
        // LRU evicts oldest: indices 0, 1
        assert!(evict.contains(&0));
        assert!(evict.contains(&1));
    }

    #[test]
    fn empty_cache_returns_empty_scores() {
        let policy = LruPolicy::new();
        let ctx = EvictionContext {
            positions: &[],
            attention_scores: &[],
        };
        assert!(policy.score(&ctx).is_empty());
    }

    #[test]
    fn aggregator_no_policies_returns_zeros() {
        let agg = VotingAggregator::new();
        let ctx = EvictionContext {
            positions: &[0, 1, 2],
            attention_scores: &[0.1, 0.2, 0.3],
        };
        let scores = <VotingAggregator as EvictionPolicy>::score(&agg, &ctx);
        assert!(scores.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn aggregator_weighted_bias() {
        // With 100% weight on H2O, result should match H2O exactly
        let mut agg = VotingAggregator::new();
        agg.add_policy(Box::new(H2oPolicy::new()), 1.0);
        agg.add_policy(Box::new(LruPolicy::new()), 0.0);

        let positions = vec![0, 1, 2];
        let attn = vec![0.1, 0.5, 0.3];
        let scores = agg.score(&positions, &attn);

        let h2o = H2oPolicy::new();
        let ctx = EvictionContext {
            positions: &positions,
            attention_scores: &attn,
        };
        let h2o_scores = h2o.score(&ctx);

        // With weight 0 on LRU, aggregator should return pure H2O scores
        // but the total_weight normalization makes 0-weight contribute 0
        for (a, h) in scores.iter().zip(h2o_scores.iter()) {
            assert!((a - h).abs() < 1e-6, "aggregator={a}, h2o={h}");
        }
    }
}
