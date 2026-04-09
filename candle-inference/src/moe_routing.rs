//! Mixture-of-Experts (MoE) token routing.
//!
//! MoE architectures (Mixtral, Qwen-MoE, DeepSeek-MoE, etc.) activate only a
//! subset of experts per token, trading parameter count for compute efficiency.
//! This module provides the **routing and capacity management** logic:
//!
//! - **Top-K gating**: Selects the top-K experts per token based on router logits.
//! - **Capacity control**: Enforces per-expert token limits via configurable overflow
//!   policies (Token Drop, Expanded Drop, no-drop).
//! - **Batch construction**: Groups tokens by assigned expert for parallel execution.
//!
//! # Architecture
//!
//! The [`MoeRouter`] takes router logits `[batch_size, num_experts]` and produces
//! a [`RoutingResult`] containing per-token expert assignments and weights.
//! The caller then runs the appropriate expert for each group of tokens.
//!
//! ```text
//! Router logits [B, E]
//!   │
//!   ├─► top-K selection    ─► expert_indices [B, K]
//!   ├─► softmax weights    ─► expert_weights [B, K]
//!   └─► capacity control   ─► overflow policy applied
//!                               │
//!                               └─► ExpertBatch { expert_id, token_indices, weights }
//! ```
//!
//! # Example
//!
//! ```rust
//! use candle_inference::moe_routing::{MoeRouter, MoeConfig, OverflowPolicy};
//!
//! let config = MoeConfig::new(8, 2)             // 8 experts, top-2
//!     .with_capacity_factor(1.25)                 // 25% headroom
//!     .with_overflow_policy(OverflowPolicy::TokenDrop);
//!
//! let router = MoeRouter::new(config);
//!
//! // Simulated router logits: 4 tokens, 8 experts
//! let logits = vec![
//!     vec![2.0, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0], // token 0
//!     vec![0.0, 0.0, 0.5, 3.0, 0.1, 0.0, 0.0, 0.0], // token 1
//!     vec![0.1, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.1], // token 2
//!     vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0], // token 3
//! ];
//!
//! let result = router.route(&logits);
//! assert_eq!(result.num_tokens(), 4);
//! assert_eq!(result.top_k(), 2);
//! ```

use std::collections::HashMap;

/// Overflow policy when an expert exceeds its token capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    /// Drop excess tokens (they get zero weight for this expert).
    TokenDrop,
    /// Allow expanded capacity (no dropping — used for inference where batch
    /// sizes are small and dropping would hurt quality).
    NoDrop,
}

/// Configuration for MoE routing.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Total number of experts.
    pub num_experts: usize,
    /// Number of experts activated per token.
    pub top_k: usize,
    /// Capacity factor: max tokens per expert = capacity_factor * (B * K / E).
    /// Only used with `TokenDrop` policy.
    pub capacity_factor: f32,
    /// Overflow policy.
    pub overflow_policy: OverflowPolicy,
}

impl MoeConfig {
    /// Create a new MoE routing config.
    ///
    /// # Panics
    ///
    /// Panics if `top_k > num_experts` or either is 0.
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        assert!(num_experts > 0, "num_experts must be > 0");
        assert!(top_k > 0, "top_k must be > 0");
        assert!(top_k <= num_experts, "top_k must be <= num_experts");

        Self {
            num_experts,
            top_k,
            capacity_factor: 1.0,
            overflow_policy: OverflowPolicy::NoDrop,
        }
    }

    /// Set the capacity factor.
    pub fn with_capacity_factor(mut self, factor: f32) -> Self {
        assert!(factor > 0.0, "capacity_factor must be > 0");
        self.capacity_factor = factor;
        self
    }

    /// Set the overflow policy.
    pub fn with_overflow_policy(mut self, policy: OverflowPolicy) -> Self {
        self.overflow_policy = policy;
        self
    }
}

/// Per-token routing assignment.
#[derive(Debug, Clone)]
pub struct TokenAssignment {
    /// The expert indices assigned to this token (length = top_k).
    pub expert_indices: Vec<usize>,
    /// The corresponding softmax weights (length = top_k, sums to ~1.0).
    pub expert_weights: Vec<f32>,
    /// Whether any assignment was dropped due to capacity overflow.
    pub dropped: bool,
}

/// A batch of tokens assigned to a single expert.
#[derive(Debug, Clone)]
pub struct ExpertBatch {
    /// Expert index.
    pub expert_id: usize,
    /// Indices of tokens routed to this expert.
    pub token_indices: Vec<usize>,
    /// Corresponding routing weights for each token.
    pub weights: Vec<f32>,
}

/// Result of MoE routing.
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Per-token assignments.
    assignments: Vec<TokenAssignment>,
    /// Per-expert batches (only experts with at least one token).
    expert_batches: Vec<ExpertBatch>,
    /// Number of tokens dropped due to capacity overflow.
    num_dropped: usize,
    /// Top-K value used.
    top_k: usize,
}

impl RoutingResult {
    /// Number of tokens routed.
    pub fn num_tokens(&self) -> usize {
        self.assignments.len()
    }

    /// Top-K value used.
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Per-token routing assignments.
    pub fn assignments(&self) -> &[TokenAssignment] {
        &self.assignments
    }

    /// Per-expert token batches. Only includes experts with ≥ 1 assigned token.
    pub fn expert_batches(&self) -> &[ExpertBatch] {
        &self.expert_batches
    }

    /// Number of (token, expert) assignments dropped due to capacity.
    pub fn num_dropped(&self) -> usize {
        self.num_dropped
    }

    /// Returns a map from expert_id to the number of tokens assigned.
    pub fn expert_load(&self) -> HashMap<usize, usize> {
        let mut load = HashMap::new();
        for batch in &self.expert_batches {
            load.insert(batch.expert_id, batch.token_indices.len());
        }
        load
    }
}

/// MoE token router.
///
/// See the [module-level documentation](self) for details.
#[derive(Debug, Clone)]
pub struct MoeRouter {
    config: MoeConfig,
}

impl MoeRouter {
    pub fn new(config: MoeConfig) -> Self {
        Self { config }
    }

    /// Route tokens to experts.
    ///
    /// * `logits` — Router logits, shape `[batch_size][num_experts]`.
    ///   Each inner Vec must have length `num_experts`.
    ///
    /// Returns a [`RoutingResult`] with per-token assignments and per-expert batches.
    pub fn route(&self, logits: &[Vec<f32>]) -> RoutingResult {
        let batch_size = logits.len();
        let num_experts = self.config.num_experts;
        let top_k = self.config.top_k;

        if batch_size == 0 {
            return RoutingResult {
                assignments: Vec::new(),
                expert_batches: Vec::new(),
                num_dropped: 0,
                top_k,
            };
        }

        // Step 1: Softmax over experts per token, then top-K selection.
        let mut assignments = Vec::with_capacity(batch_size);
        for row in logits {
            assert_eq!(
                row.len(),
                num_experts,
                "logits row length must equal num_experts"
            );

            let (indices, weights) = self.top_k_softmax(row, top_k);
            assignments.push(TokenAssignment {
                expert_indices: indices,
                expert_weights: weights,
                dropped: false,
            });
        }

        // Step 2: Capacity control
        let num_dropped = match self.config.overflow_policy {
            OverflowPolicy::TokenDrop => {
                self.apply_capacity_control(&mut assignments, batch_size)
            }
            OverflowPolicy::NoDrop => 0,
        };

        // Step 3: Build per-expert batches
        let expert_batches = self.build_expert_batches(&assignments);

        RoutingResult {
            assignments,
            expert_batches,
            num_dropped,
            top_k,
        }
    }

    /// Softmax + top-K selection for a single token's logits.
    fn top_k_softmax(&self, logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        // Find top-K indices by logit value
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        let indices: Vec<usize> = indexed.iter().map(|&(i, _)| i).collect();
        let top_logits: Vec<f32> = indexed.iter().map(|&(_, v)| v).collect();

        // Softmax over the selected top-K logits only
        let max_logit = top_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top_logits.iter().map(|&v| (v - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = if sum > 0.0 {
            exps.iter().map(|&e| e / sum).collect()
        } else {
            vec![1.0 / k as f32; k]
        };

        (indices, weights)
    }

    /// Apply Token Drop capacity control. Returns the number of dropped assignments.
    fn apply_capacity_control(
        &self,
        assignments: &mut [TokenAssignment],
        batch_size: usize,
    ) -> usize {
        let num_experts = self.config.num_experts;
        let top_k = self.config.top_k;

        // Per-expert capacity: capacity_factor * ceil(batch_size * top_k / num_experts)
        let avg_tokens_per_expert =
            ((batch_size * top_k) as f32 / num_experts as f32).ceil() as usize;
        let expert_capacity =
            (self.config.capacity_factor * avg_tokens_per_expert as f32).ceil() as usize;
        let expert_capacity = expert_capacity.max(1);

        // Count how many tokens each expert has
        let mut expert_counts = vec![0usize; num_experts];
        let mut dropped = 0;

        for assign in assignments.iter_mut() {
            let mut new_indices = Vec::with_capacity(top_k);
            let mut new_weights = Vec::with_capacity(top_k);

            for (&eidx, &w) in assign
                .expert_indices
                .iter()
                .zip(&assign.expert_weights)
            {
                if expert_counts[eidx] < expert_capacity {
                    expert_counts[eidx] += 1;
                    new_indices.push(eidx);
                    new_weights.push(w);
                } else {
                    dropped += 1;
                    assign.dropped = true;
                }
            }

            // Re-normalize weights if some experts were dropped
            if new_indices.len() < assign.expert_indices.len() {
                let wsum: f32 = new_weights.iter().sum();
                if wsum > 0.0 {
                    for w in &mut new_weights {
                        *w /= wsum;
                    }
                }
            }

            assign.expert_indices = new_indices;
            assign.expert_weights = new_weights;
        }

        dropped
    }

    /// Build per-expert token batches from assignments.
    fn build_expert_batches(&self, assignments: &[TokenAssignment]) -> Vec<ExpertBatch> {
        let mut expert_map: HashMap<usize, (Vec<usize>, Vec<f32>)> = HashMap::new();

        for (token_idx, assign) in assignments.iter().enumerate() {
            for (&expert_id, &weight) in assign
                .expert_indices
                .iter()
                .zip(&assign.expert_weights)
            {
                let entry = expert_map.entry(expert_id).or_default();
                entry.0.push(token_idx);
                entry.1.push(weight);
            }
        }

        let mut batches: Vec<ExpertBatch> = expert_map
            .into_iter()
            .map(|(expert_id, (token_indices, weights))| ExpertBatch {
                expert_id,
                token_indices,
                weights,
            })
            .collect();

        batches.sort_by_key(|b| b.expert_id);
        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logits(rows: &[&[f32]]) -> Vec<Vec<f32>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    #[test]
    fn basic_top2_routing() {
        let router = MoeRouter::new(MoeConfig::new(4, 2));
        let logits = make_logits(&[
            &[3.0, 1.0, 0.0, 0.0], // expert 0 and 1
            &[0.0, 0.0, 2.0, 4.0], // expert 3 and 2
        ]);

        let result = router.route(&logits);
        assert_eq!(result.num_tokens(), 2);
        assert_eq!(result.top_k(), 2);

        // Token 0 should pick experts 0 and 1
        let a0 = &result.assignments()[0];
        assert!(a0.expert_indices.contains(&0));
        assert!(a0.expert_indices.contains(&1));

        // Token 1 should pick experts 3 and 2
        let a1 = &result.assignments()[1];
        assert!(a1.expert_indices.contains(&3));
        assert!(a1.expert_indices.contains(&2));
    }

    #[test]
    fn weights_sum_to_one() {
        let router = MoeRouter::new(MoeConfig::new(8, 2));
        let logits = make_logits(&[
            &[1.0, 2.0, 3.0, 0.5, 0.1, 0.0, 0.0, 0.0],
        ]);

        let result = router.route(&logits);
        let wsum: f32 = result.assignments()[0].expert_weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-5, "weights sum = {wsum}");
    }

    #[test]
    fn token_drop_capacity() {
        // 4 tokens, 2 experts, top-1 → avg 2 tokens/expert
        // capacity_factor=1.0 → capacity=2
        let config = MoeConfig::new(2, 1)
            .with_capacity_factor(1.0)
            .with_overflow_policy(OverflowPolicy::TokenDrop);
        let router = MoeRouter::new(config);

        // All 4 tokens prefer expert 0
        let logits = make_logits(&[
            &[10.0, 0.0],
            &[10.0, 0.0],
            &[10.0, 0.0],
            &[10.0, 0.0],
        ]);

        let result = router.route(&logits);
        // Expert 0 capacity = ceil(1.0 * ceil(4*1/2)) = 2
        // So 2 tokens should be dropped from expert 0
        assert!(result.num_dropped() > 0);
    }

    #[test]
    fn no_drop_policy() {
        let config = MoeConfig::new(2, 1).with_overflow_policy(OverflowPolicy::NoDrop);
        let router = MoeRouter::new(config);

        let logits = make_logits(&[
            &[10.0, 0.0],
            &[10.0, 0.0],
            &[10.0, 0.0],
        ]);

        let result = router.route(&logits);
        assert_eq!(result.num_dropped(), 0);
        // All 3 tokens should be assigned to expert 0
        let load = result.expert_load();
        assert_eq!(load[&0], 3);
    }

    #[test]
    fn expert_batches_built() {
        let router = MoeRouter::new(MoeConfig::new(4, 1));
        let logits = make_logits(&[
            &[5.0, 0.0, 0.0, 0.0], // → expert 0
            &[0.0, 5.0, 0.0, 0.0], // → expert 1
            &[0.0, 0.0, 0.0, 5.0], // → expert 3
            &[5.0, 0.0, 0.0, 0.0], // → expert 0
        ]);

        let result = router.route(&logits);
        let batches = result.expert_batches();

        // 3 experts active (0, 1, 3)
        assert_eq!(batches.len(), 3);

        // Expert 0 has 2 tokens
        let e0 = batches.iter().find(|b| b.expert_id == 0).unwrap();
        assert_eq!(e0.token_indices, vec![0, 3]);
    }

    #[test]
    fn empty_batch() {
        let router = MoeRouter::new(MoeConfig::new(4, 2));
        let result = router.route(&[]);
        assert_eq!(result.num_tokens(), 0);
        assert!(result.expert_batches().is_empty());
    }

    #[test]
    fn top_k_equals_num_experts() {
        let router = MoeRouter::new(MoeConfig::new(3, 3));
        let logits = make_logits(&[&[1.0, 2.0, 3.0]]);

        let result = router.route(&logits);
        // All 3 experts should be selected
        assert_eq!(result.assignments()[0].expert_indices.len(), 3);
    }

    #[test]
    fn expert_load_distribution() {
        let router = MoeRouter::new(MoeConfig::new(4, 2));
        let logits = make_logits(&[
            &[5.0, 4.0, 0.0, 0.0],
            &[0.0, 4.0, 5.0, 0.0],
            &[0.0, 0.0, 5.0, 4.0],
            &[4.0, 0.0, 0.0, 5.0],
        ]);

        let result = router.route(&logits);
        let load = result.expert_load();

        // Each expert should get 2 tokens (balanced routing)
        for expert_id in 0..4 {
            assert_eq!(load[&expert_id], 2, "expert {expert_id} load mismatch");
        }
    }

    #[test]
    #[should_panic(expected = "top_k must be <= num_experts")]
    fn top_k_exceeds_experts() {
        MoeConfig::new(4, 5);
    }

    #[test]
    fn dropped_tokens_flag() {
        let config = MoeConfig::new(2, 1)
            .with_capacity_factor(0.5) // Very restrictive
            .with_overflow_policy(OverflowPolicy::TokenDrop);
        let router = MoeRouter::new(config);

        let logits = make_logits(&[
            &[10.0, 0.0],
            &[10.0, 0.0],
            &[10.0, 0.0],
            &[10.0, 0.0],
        ]);

        let result = router.route(&logits);
        // Some tokens should have the dropped flag
        let any_dropped = result.assignments().iter().any(|a| a.dropped);
        assert!(any_dropped);
    }
}
