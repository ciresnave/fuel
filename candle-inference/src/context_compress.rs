//! Context compression for long conversations.
//!
//! When a conversation grows beyond the model's context window the oldest
//! turns must be *compressed* — reduced to a shorter representation — so that
//! the effective history fits while preserving coherence.
//!
//! This module provides the policy and bookkeeping; actual summarisation
//! (calling a model) is left to the caller.
//!
//! # Design
//!
//! [`ContextCompressor`] maintains a list of *turns* ([`Turn`]) and enforces
//! a token budget.  When the total exceeds the budget, the compressor selects
//! the lowest-scored turns for compression and replaces their token counts
//! with the compressed sizes reported by the caller.
//!
//! Scoring uses a combination of *recency* (newer turns score higher) and
//! *importance* (user-settable weight per turn, e.g., system prompts always
//! important).
//!
//! # Example
//!
//! ```rust
//! use candle_inference::context_compress::{ContextCompressor, Turn, Role};
//!
//! let mut cc = ContextCompressor::new(1024); // 1024-token budget
//!
//! cc.push(Turn::new(Role::System, 50).with_importance(1.0));
//! cc.push(Turn::new(Role::User, 200));
//! cc.push(Turn::new(Role::Assistant, 600));
//! cc.push(Turn::new(Role::User, 300));
//!
//! assert!(cc.total_tokens() > cc.budget());
//!
//! // Ask which turns to compress
//! let plan = cc.plan_compression();
//! assert!(!plan.is_empty());
//!
//! // After external summarisation, report compressed sizes
//! for entry in &plan {
//!     cc.mark_compressed(entry.turn_index, entry.tokens / 4);
//! }
//!
//! assert!(cc.total_tokens() <= cc.budget());
//! ```

/// Speaker role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One conversational turn.
#[derive(Debug, Clone)]
pub struct Turn {
    /// Speaker.
    pub role: Role,
    /// Current token count (decreases after compression).
    pub tokens: usize,
    /// Original (uncompressed) token count — never changes after creation.
    pub original_tokens: usize,
    /// User-settable importance weight in `[0, 1]`.  Higher = more
    /// resistant to compression.
    pub importance: f64,
    /// Whether this turn has already been compressed.
    pub compressed: bool,
    /// If `true`, never compress this turn (e.g., system prompt).
    pub pinned: bool,
}

impl Turn {
    /// Create a new turn with default importance (0.0).
    pub fn new(role: Role, tokens: usize) -> Self {
        Self {
            role,
            tokens,
            original_tokens: tokens,
            importance: 0.0,
            compressed: false,
            pinned: false,
        }
    }

    /// Builder: set importance weight.
    pub fn with_importance(mut self, w: f64) -> Self {
        self.importance = w.clamp(0.0, 1.0);
        self
    }

    /// Builder: pin turn (never compress).
    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }

    /// Compression ratio: current / original tokens.
    pub fn compression_ratio(&self) -> f64 {
        if self.original_tokens == 0 {
            1.0
        } else {
            self.tokens as f64 / self.original_tokens as f64
        }
    }
}

/// Entry in a compression plan.
#[derive(Debug, Clone)]
pub struct CompressionEntry {
    /// Index of the turn in the compressor's turn list.
    pub turn_index: usize,
    /// Current token count of this turn.
    pub tokens: usize,
    /// The score that led to selection (lower = more compressible).
    pub score: f64,
}

/// Context compressor managing turn-level token budgeting.
#[derive(Debug)]
pub struct ContextCompressor {
    /// Maximum total tokens allowed.
    budget: usize,
    /// Turns in conversation order.
    turns: Vec<Turn>,
}

impl ContextCompressor {
    /// Create a compressor with the given token budget.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            turns: Vec::new(),
        }
    }

    /// Append a turn.
    pub fn push(&mut self, turn: Turn) {
        self.turns.push(turn);
    }

    /// Total tokens across all turns.
    pub fn total_tokens(&self) -> usize {
        self.turns.iter().map(|t| t.tokens).sum()
    }

    /// Token budget.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// How many tokens over budget (0 if within budget).
    pub fn overflow(&self) -> usize {
        self.total_tokens().saturating_sub(self.budget)
    }

    /// Number of turns.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// True if no turns.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Access a turn by index.
    pub fn get(&self, index: usize) -> Option<&Turn> {
        self.turns.get(index)
    }

    /// Produce a compression plan: select turns to compress in order to
    /// bring total tokens ≤ budget.
    ///
    /// Turns are scored by `recency * (1 + importance)`.  Lower-scored
    /// turns are selected first.  Pinned and already-compressed turns are
    /// skipped.
    ///
    /// Returns an empty vec when already within budget.
    pub fn plan_compression(&self) -> Vec<CompressionEntry> {
        if self.total_tokens() <= self.budget {
            return Vec::new();
        }

        let n = self.turns.len();
        let mut scored: Vec<(usize, f64)> = self
            .turns
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.pinned && !t.compressed)
            .map(|(i, t)| {
                // Recency: newer turns (higher index) get higher score
                let recency = if n <= 1 { 1.0 } else { i as f64 / (n - 1) as f64 };
                let score = recency * (1.0 + t.importance);
                (i, score)
            })
            .collect();

        // Sort ascending score — least important first
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut plan = Vec::new();
        let mut hypothetical_freed: usize = 0;
        let overflow = self.overflow();

        for (idx, score) in scored {
            if hypothetical_freed >= overflow {
                break;
            }
            let t = &self.turns[idx];
            // Assume compression yields ~25% of original tokens (conservative)
            let expected_savings = t.tokens.saturating_sub(t.tokens / 4);
            hypothetical_freed += expected_savings;
            plan.push(CompressionEntry {
                turn_index: idx,
                tokens: t.tokens,
                score,
            });
        }

        plan
    }

    /// Mark a turn as compressed with a new (smaller) token count.
    ///
    /// Returns `false` if the index is out of bounds.
    pub fn mark_compressed(&mut self, turn_index: usize, new_tokens: usize) -> bool {
        if let Some(turn) = self.turns.get_mut(turn_index) {
            turn.tokens = new_tokens;
            turn.compressed = true;
            true
        } else {
            false
        }
    }

    /// Update the token budget.
    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
    }

    /// Iterate turns.
    pub fn iter(&self) -> impl Iterator<Item = &Turn> {
        self.turns.iter()
    }

    /// Fraction of total tokens that are compressed.
    pub fn compressed_fraction(&self) -> f64 {
        let total: usize = self.turns.iter().map(|t| t.original_tokens).sum();
        if total == 0 {
            return 0.0;
        }
        let compressed_original: usize = self
            .turns
            .iter()
            .filter(|t| t.compressed)
            .map(|t| t.original_tokens)
            .sum();
        compressed_original as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn within_budget_no_plan() {
        let mut cc = ContextCompressor::new(1000);
        cc.push(Turn::new(Role::User, 200));
        cc.push(Turn::new(Role::Assistant, 300));

        assert_eq!(cc.total_tokens(), 500);
        assert!(cc.plan_compression().is_empty());
    }

    #[test]
    fn overflow_produces_plan() {
        let mut cc = ContextCompressor::new(500);
        cc.push(Turn::new(Role::User, 200));
        cc.push(Turn::new(Role::Assistant, 200));
        cc.push(Turn::new(Role::User, 200));

        assert_eq!(cc.overflow(), 100);
        let plan = cc.plan_compression();
        assert!(!plan.is_empty());
        // First turn (oldest, lowest recency) should be picked first
        assert_eq!(plan[0].turn_index, 0);
    }

    #[test]
    fn mark_compressed_reduces_tokens() {
        let mut cc = ContextCompressor::new(500);
        cc.push(Turn::new(Role::User, 400));
        cc.push(Turn::new(Role::Assistant, 300));

        assert_eq!(cc.overflow(), 200);

        cc.mark_compressed(0, 50);
        assert_eq!(cc.total_tokens(), 350);
        assert_eq!(cc.overflow(), 0);
        assert!(cc.get(0).unwrap().compressed);
    }

    #[test]
    fn pinned_turns_excluded() {
        let mut cc = ContextCompressor::new(500);
        cc.push(Turn::new(Role::System, 400).pinned());
        cc.push(Turn::new(Role::User, 200));

        let plan = cc.plan_compression();
        // Only the user turn (index 1) should be in the plan
        for entry in &plan {
            assert_ne!(entry.turn_index, 0);
        }
    }

    #[test]
    fn already_compressed_excluded() {
        let mut cc = ContextCompressor::new(500);
        cc.push(Turn::new(Role::User, 300));
        cc.push(Turn::new(Role::Assistant, 300));

        cc.mark_compressed(0, 50);
        // Now total = 50 + 300 = 350, which is under budget
        assert!(cc.plan_compression().is_empty());
    }

    #[test]
    fn importance_affects_selection_order() {
        let mut cc = ContextCompressor::new(300);
        // Two turns at same position; important one should be picked later
        cc.push(Turn::new(Role::User, 200));
        cc.push(Turn::new(Role::User, 200).with_importance(0.9));

        let plan = cc.plan_compression();
        assert!(!plan.is_empty());
        // The non-important turn (index 0) should have lower score
        assert_eq!(plan[0].turn_index, 0);
    }

    #[test]
    fn compression_ratio() {
        let mut turn = Turn::new(Role::User, 400);
        assert_eq!(turn.compression_ratio(), 1.0);

        turn.tokens = 100;
        assert!((turn.compression_ratio() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn compressed_fraction() {
        let mut cc = ContextCompressor::new(1000);
        cc.push(Turn::new(Role::User, 200));
        cc.push(Turn::new(Role::Assistant, 300));

        assert_eq!(cc.compressed_fraction(), 0.0);

        cc.mark_compressed(0, 50);
        // 200 / 500 = 0.4
        assert!((cc.compressed_fraction() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn empty_compressor() {
        let cc = ContextCompressor::new(100);
        assert!(cc.is_empty());
        assert_eq!(cc.len(), 0);
        assert_eq!(cc.total_tokens(), 0);
        assert!(cc.plan_compression().is_empty());
    }

    #[test]
    fn set_budget() {
        let mut cc = ContextCompressor::new(1000);
        cc.push(Turn::new(Role::User, 500));

        assert_eq!(cc.overflow(), 0);
        cc.set_budget(400);
        assert_eq!(cc.overflow(), 100);
    }

    #[test]
    fn role_variants() {
        // Ensure all roles are usable
        let roles = [Role::System, Role::User, Role::Assistant, Role::Tool];
        let mut cc = ContextCompressor::new(10000);
        for role in roles {
            cc.push(Turn::new(role, 100));
        }
        assert_eq!(cc.len(), 4);
    }

    #[test]
    fn recency_favours_newer_turns() {
        let mut cc = ContextCompressor::new(100);
        // 5 turns of equal importance, total 500 vs budget 100
        for _ in 0..5 {
            cc.push(Turn::new(Role::User, 100));
        }

        let plan = cc.plan_compression();
        // Should be sorted oldest-first (index 0, 1, 2, ...)
        for window in plan.windows(2) {
            assert!(window[0].turn_index < window[1].turn_index);
        }
    }
}
