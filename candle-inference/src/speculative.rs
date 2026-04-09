//! Speculative decoding for accelerated autoregressive generation.
//!
//! Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) uses a small, fast
//! **draft model** to propose `K` candidate tokens, then verifies them in a single
//! forward pass of the larger **target model**. Accepted tokens are emitted immediately;
//! rejected tokens trigger a resample from the target distribution. This can yield
//! 1.5–3× latency improvement because the target model processes `K` tokens in parallel
//! instead of autoregressively.
//!
//! # Architecture
//!
//! This module provides the **verification logic and statistics tracking**, not the
//! models themselves. The caller supplies closures (or trait implementations) that
//! produce logits from token sequences — the module handles the accept/reject math.
//!
//! ```text
//! ┌─────────────┐     K draft tokens      ┌─────────────────┐
//! │ Draft model  ├───────────────────────►  │  Speculative    │
//! │ (fast, small)│                          │  Verifier       │
//! └─────────────┘                          │                 │
//! ┌─────────────┐     K+1 target logits    │  accept/reject  │──► accepted tokens
//! │ Target model ├───────────────────────►  │  + resample     │
//! │ (large)      │                          └─────────────────┘
//! └─────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use candle_inference::speculative::{SpeculativeConfig, SpeculativeStats, verify_draft};
//! use candle::{DType, Device, Tensor};
//!
//! # fn main() -> candle::Result<()> {
//! let device = Device::Cpu;
//! let vocab_size = 100;
//!
//! // Draft model proposed 3 tokens: [10, 20, 30]
//! let draft_tokens = vec![10u32, 20, 30];
//!
//! // Draft model's log-probabilities at each step (shape: [K, vocab_size])
//! let draft_logprobs = Tensor::zeros((3, vocab_size), DType::F32, &device)?;
//!
//! // Target model's log-probabilities for the prefix + draft (shape: [K+1, vocab_size])
//! // (K+1 because the target also produces logits for the next token after all drafts)
//! let target_logprobs = Tensor::zeros((4, vocab_size), DType::F32, &device)?;
//!
//! let config = SpeculativeConfig::new(3);
//! let mut stats = SpeculativeStats::new();
//!
//! let result = verify_draft(
//!     &draft_tokens,
//!     &draft_logprobs,
//!     &target_logprobs,
//!     &config,
//!     &mut stats,
//! )?;
//!
//! // result.accepted_tokens: tokens accepted by the target model
//! // result.next_token: the resampled/bonus token from the target
//! // result.num_accepted: how many of the K draft tokens were accepted
//! assert!(result.accepted_tokens.len() <= 3);
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! - Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)
//! - Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)

use candle::{DType, Result, Tensor};

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens the draft model proposes per verification round.
    pub draft_len: usize,
    /// Minimum acceptance rate before auto-fallback disables speculation.
    /// Set to `0.0` to never fall back. Default: `0.0`.
    pub min_acceptance_rate: f32,
    /// Number of rounds to track for the rolling acceptance rate.
    /// Default: `100`.
    pub stats_window: usize,
}

impl SpeculativeConfig {
    /// Creates a new config with the given draft length.
    pub fn new(draft_len: usize) -> Self {
        Self {
            draft_len: draft_len.max(1),
            min_acceptance_rate: 0.0,
            stats_window: 100,
        }
    }

    /// Sets the minimum acceptance rate for auto-fallback.
    pub fn with_min_acceptance_rate(mut self, rate: f32) -> Self {
        self.min_acceptance_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Sets the rolling window size for statistics tracking.
    pub fn with_stats_window(mut self, window: usize) -> Self {
        self.stats_window = window.max(1);
        self
    }
}

/// Tracks acceptance statistics across verification rounds.
#[derive(Debug, Clone)]
pub struct SpeculativeStats {
    /// Total number of verification rounds.
    pub total_rounds: u64,
    /// Total draft tokens proposed across all rounds.
    pub total_drafted: u64,
    /// Total draft tokens accepted across all rounds.
    pub total_accepted: u64,
    /// Rolling window of per-round acceptance rates.
    recent_rates: Vec<f32>,
    /// Write index into the rolling window.
    window_idx: usize,
}

impl SpeculativeStats {
    /// Creates a new empty statistics tracker.
    pub fn new() -> Self {
        Self {
            total_rounds: 0,
            total_drafted: 0,
            total_accepted: 0,
            recent_rates: Vec::new(),
            window_idx: 0,
        }
    }

    /// Records one verification round.
    pub fn record(&mut self, drafted: usize, accepted: usize, window_size: usize) {
        self.total_rounds += 1;
        self.total_drafted += drafted as u64;
        self.total_accepted += accepted as u64;

        let rate = if drafted > 0 {
            accepted as f32 / drafted as f32
        } else {
            1.0
        };

        if self.recent_rates.len() < window_size {
            self.recent_rates.push(rate);
        } else {
            let idx = self.window_idx % window_size;
            self.recent_rates[idx] = rate;
        }
        self.window_idx += 1;
    }

    /// Returns the overall acceptance rate across all rounds.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_drafted == 0 {
            1.0
        } else {
            self.total_accepted as f32 / self.total_drafted as f32
        }
    }

    /// Returns the rolling-window acceptance rate.
    pub fn recent_acceptance_rate(&self) -> f32 {
        if self.recent_rates.is_empty() {
            1.0
        } else {
            self.recent_rates.iter().sum::<f32>() / self.recent_rates.len() as f32
        }
    }

    /// Returns `true` if speculation should be disabled based on the config threshold.
    pub fn should_fallback(&self, config: &SpeculativeConfig) -> bool {
        config.min_acceptance_rate > 0.0
            && self.total_rounds >= config.stats_window as u64
            && self.recent_acceptance_rate() < config.min_acceptance_rate
    }
}

impl Default for SpeculativeStats {
    fn default() -> Self {
        Self::new()
    }
}

/// The result of verifying a batch of draft tokens against the target model.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Draft tokens that were accepted (may be empty if all rejected).
    pub accepted_tokens: Vec<u32>,
    /// The next token sampled from the target model's distribution.
    /// This is either a resampled token (at the first rejection point) or
    /// a "bonus" token (if all drafts were accepted, sampled from position K+1).
    pub next_token: u32,
    /// Number of draft tokens accepted (0..=K).
    pub num_accepted: usize,
}

/// Verifies draft tokens against the target model using speculative sampling.
///
/// # Algorithm
///
/// For each draft position `i` (0..K):
/// 1. Compute acceptance probability: `min(1, target_prob[token_i] / draft_prob[token_i])`
/// 2. Draw uniform random `r ~ U(0, 1)`
/// 3. If `r < acceptance_prob`: **accept** token_i, continue to i+1
/// 4. If `r >= acceptance_prob`: **reject** — resample from the adjusted distribution
///    `max(0, target_prob - draft_prob)` (normalized), emit that token, stop
///
/// If all K tokens are accepted, the bonus token at position K+1 is sampled from the
/// target distribution directly.
///
/// # Arguments
///
/// * `draft_tokens` — The K tokens proposed by the draft model.
/// * `draft_logprobs` — Log-probabilities from the draft model, shape `[K, vocab_size]`.
/// * `target_logprobs` — Log-probabilities from the target model, shape `[K+1, vocab_size]`.
/// * `config` — Speculative decoding configuration.
/// * `stats` — Statistics tracker (updated in place).
///
/// # Returns
///
/// A [`VerifyResult`] containing the accepted tokens and the next token.
pub fn verify_draft(
    draft_tokens: &[u32],
    draft_logprobs: &Tensor,
    target_logprobs: &Tensor,
    config: &SpeculativeConfig,
    stats: &mut SpeculativeStats,
) -> Result<VerifyResult> {
    let k = draft_tokens.len().min(config.draft_len);

    if k == 0 {
        // No draft tokens — just sample from the target
        let target_probs = softmax_1d(&target_logprobs.get(0)?)?;
        let next_token = sample_from_probs(&target_probs)?;
        stats.record(0, 0, config.stats_window);
        return Ok(VerifyResult {
            accepted_tokens: Vec::new(),
            next_token,
            num_accepted: 0,
        });
    }

    let mut accepted = Vec::with_capacity(k);

    for i in 0..k {
        let draft_probs = softmax_1d(&draft_logprobs.get(i)?)?;
        let target_probs = softmax_1d(&target_logprobs.get(i)?)?;

        let token = draft_tokens[i];
        let draft_p = prob_of_token(&draft_probs, token)?;
        let target_p = prob_of_token(&target_probs, token)?;

        // Acceptance probability: min(1, target_p / draft_p)
        let accept_prob = if draft_p > 0.0 {
            (target_p / draft_p).min(1.0)
        } else if target_p > 0.0 {
            1.0 // draft assigned 0 prob but target likes it — always accept
        } else {
            0.0 // both assign 0 — reject
        };

        // Deterministic comparison using the token value as a simple hash.
        // In production, you'd use a proper RNG here — but for a pure-function
        // verification API that doesn't own an RNG, we use a deterministic
        // threshold based on the acceptance probability itself.
        // Tokens with accept_prob >= 1.0 are always accepted.
        // For accept_prob < 1.0, we use a seeded per-position pseudo-random test.
        let r = pseudo_uniform(i as u64, token as u64);

        if r < accept_prob {
            accepted.push(token);
        } else {
            // Rejection: resample from adjusted distribution max(0, target - draft)
            let next_token = resample_adjusted(&target_probs, &draft_probs)?;
            stats.record(k, accepted.len(), config.stats_window);
            return Ok(VerifyResult {
                accepted_tokens: accepted,
                next_token,
                num_accepted: i,
            });
        }
    }

    // All K tokens accepted — sample bonus token from target at position K
    let bonus_probs = softmax_1d(&target_logprobs.get(k)?)?;
    let next_token = sample_from_probs(&bonus_probs)?;
    stats.record(k, k, config.stats_window);

    Ok(VerifyResult {
        accepted_tokens: accepted,
        next_token,
        num_accepted: k,
    })
}

/// Softmax a 1-D logits tensor to probabilities.
fn softmax_1d(logits: &Tensor) -> Result<Vec<f32>> {
    let logits = logits.to_dtype(DType::F32)?;
    let max_val = logits.max(0)?.to_scalar::<f32>()?;
    let shifted = (logits - max_val as f64)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_all()?.to_scalar::<f32>()?;
    let probs = (exp / sum as f64)?;
    probs.to_vec1::<f32>()
}

/// Extract the probability assigned to a specific token.
fn prob_of_token(probs: &[f32], token: u32) -> Result<f32> {
    Ok(probs.get(token as usize).copied().unwrap_or(0.0))
}

/// Resample from the adjusted distribution max(0, target_p - draft_p), normalized.
fn resample_adjusted(target_probs: &[f32], draft_probs: &[f32]) -> Result<u32> {
    let adjusted: Vec<f32> = target_probs
        .iter()
        .zip(draft_probs.iter())
        .map(|(&t, &d)| (t - d).max(0.0))
        .collect();

    let sum: f32 = adjusted.iter().sum();
    if sum <= 0.0 {
        // Fallback: sample from target distribution directly
        return sample_from_probs(target_probs);
    }

    // Sample from normalized adjusted distribution
    let normalized: Vec<f32> = adjusted.iter().map(|&p| p / sum).collect();
    sample_from_probs(&normalized)
}

/// Argmax sampling from a probability vector.
fn sample_from_probs(probs: &[f32]) -> Result<u32> {
    let mut best_idx = 0u32;
    let mut best_p = f32::NEG_INFINITY;
    for (i, &p) in probs.iter().enumerate() {
        if p > best_p {
            best_p = p;
            best_idx = i as u32;
        }
    }
    Ok(best_idx)
}

/// Simple deterministic pseudo-random uniform in [0, 1).
/// Used for accept/reject decisions so that `verify_draft` is a pure function
/// (no RNG state required). In production the caller would supply an RNG.
fn pseudo_uniform(position: u64, token: u64) -> f32 {
    // Mixing function (splitmix64-inspired)
    let mut x = position
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(token);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    // Map to [0, 1)
    (x >> 40) as f32 / (1u64 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn uniform_logprobs(k: usize, vocab: usize, device: &Device) -> Tensor {
        // Uniform distribution — all logits = 0 → softmax = 1/vocab
        Tensor::zeros((k, vocab), DType::F32, device).unwrap()
    }

    #[test]
    fn all_accepted_when_distributions_match() -> Result<()> {
        let device = Device::Cpu;
        let vocab = 50;
        let k = 3;
        let draft_tokens = vec![5u32, 10, 15];

        // When draft and target have identical distributions, acceptance prob = 1.0
        let logprobs = uniform_logprobs(k, vocab, &device);
        let target_logprobs = uniform_logprobs(k + 1, vocab, &device);

        let config = SpeculativeConfig::new(k);
        let mut stats = SpeculativeStats::new();

        let result = verify_draft(
            &draft_tokens,
            &logprobs,
            &target_logprobs,
            &config,
            &mut stats,
        )?;

        // With uniform distributions, target_p/draft_p = 1.0 → always accept
        assert_eq!(result.num_accepted, 3);
        assert_eq!(result.accepted_tokens, vec![5, 10, 15]);
        assert_eq!(stats.total_drafted, 3);
        assert_eq!(stats.total_accepted, 3);
        Ok(())
    }

    #[test]
    fn stats_tracking() {
        let mut stats = SpeculativeStats::new();
        stats.record(5, 3, 10);
        stats.record(5, 5, 10);

        assert_eq!(stats.total_rounds, 2);
        assert_eq!(stats.total_drafted, 10);
        assert_eq!(stats.total_accepted, 8);
        assert!((stats.acceptance_rate() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn stats_rolling_window() {
        let mut stats = SpeculativeStats::new();
        // Fill window of size 3
        stats.record(10, 10, 3); // rate 1.0
        stats.record(10, 5, 3); // rate 0.5
        stats.record(10, 0, 3); // rate 0.0

        assert_eq!(stats.recent_rates.len(), 3);
        let recent = stats.recent_acceptance_rate();
        assert!((recent - 0.5).abs() < 1e-6);

        // Window wraps: new entry overwrites oldest
        stats.record(10, 8, 3); // rate 0.8, overwrites 1.0
        // Window: [0.8, 0.5, 0.0]
        let recent = stats.recent_acceptance_rate();
        let expected = (0.8 + 0.5 + 0.0) / 3.0;
        assert!((recent - expected).abs() < 1e-6);
    }

    #[test]
    fn fallback_detection() {
        let config = SpeculativeConfig::new(5).with_min_acceptance_rate(0.5);
        let mut stats = SpeculativeStats::new();

        // Not enough rounds yet
        assert!(!stats.should_fallback(&config));

        // Fill 100 rounds with poor acceptance
        for _ in 0..100 {
            stats.record(5, 1, config.stats_window);
        }

        // 1/5 = 0.2 < 0.5 → should fallback
        assert!(stats.should_fallback(&config));
    }

    #[test]
    fn empty_draft_returns_bonus_token() -> Result<()> {
        let device = Device::Cpu;
        let vocab = 50;

        // No draft tokens
        let draft_tokens: Vec<u32> = vec![];
        let draft_logprobs = Tensor::zeros((0, vocab), DType::F32, &device)?;
        let target_logprobs = Tensor::zeros((1, vocab), DType::F32, &device)?;

        let config = SpeculativeConfig::new(3);
        let mut stats = SpeculativeStats::new();

        let result = verify_draft(
            &draft_tokens,
            &draft_logprobs,
            &target_logprobs,
            &config,
            &mut stats,
        )?;

        assert_eq!(result.num_accepted, 0);
        assert!(result.accepted_tokens.is_empty());
        Ok(())
    }

    #[test]
    fn rejection_with_divergent_distributions() -> Result<()> {
        let device = Device::Cpu;
        let vocab = 10;
        let k = 3;

        // Draft strongly prefers token 0
        let mut draft_data = vec![0.0f32; k * vocab];
        for i in 0..k {
            draft_data[i * vocab] = 10.0; // token 0 has high logit
        }
        let draft_logprobs = Tensor::from_vec(draft_data, (k, vocab), &device)?;

        // Target strongly prefers token 5
        let mut target_data = vec![0.0f32; (k + 1) * vocab];
        for i in 0..=k {
            target_data[i * vocab + 5] = 10.0; // token 5 has high logit
        }
        let target_logprobs = Tensor::from_vec(target_data, (k + 1, vocab), &device)?;

        // Draft proposes token 0 (which target dislikes)
        let draft_tokens = vec![0u32, 0, 0];

        let config = SpeculativeConfig::new(k);
        let mut stats = SpeculativeStats::new();

        let result = verify_draft(
            &draft_tokens,
            &draft_logprobs,
            &target_logprobs,
            &config,
            &mut stats,
        )?;

        // The target assigns near-zero probability to token 0,
        // so acceptance probability ≈ 0 and most/all drafts should be rejected.
        // The resampled token should be 5 (target's preference).
        assert!(result.num_accepted < k);
        // The next_token should be from the target's favored distribution
        assert_eq!(result.next_token, 5);
        Ok(())
    }

    #[test]
    fn verify_result_invariants() -> Result<()> {
        let device = Device::Cpu;
        let vocab = 20;
        let k = 5;

        let draft_tokens: Vec<u32> = (0..k as u32).collect();
        let draft_logprobs = uniform_logprobs(k, vocab, &device);
        let target_logprobs = uniform_logprobs(k + 1, vocab, &device);

        let config = SpeculativeConfig::new(k);
        let mut stats = SpeculativeStats::new();

        let result = verify_draft(
            &draft_tokens,
            &draft_logprobs,
            &target_logprobs,
            &config,
            &mut stats,
        )?;

        // Invariant: accepted_tokens.len() == num_accepted
        assert_eq!(result.accepted_tokens.len(), result.num_accepted);
        // Invariant: num_accepted <= k
        assert!(result.num_accepted <= k);
        // Invariant: accepted tokens are a prefix of draft tokens
        for (i, &tok) in result.accepted_tokens.iter().enumerate() {
            assert_eq!(tok, draft_tokens[i]);
        }
        Ok(())
    }

    #[test]
    fn pseudo_uniform_in_range() {
        for pos in 0..1000 {
            for tok in [0u64, 1, 42, 1000, u64::MAX] {
                let v = pseudo_uniform(pos, tok);
                assert!((0.0..1.0).contains(&v), "pos={pos}, tok={tok}, v={v}");
            }
        }
    }

    #[test]
    fn config_builder() {
        let config = SpeculativeConfig::new(5)
            .with_min_acceptance_rate(0.3)
            .with_stats_window(50);
        assert_eq!(config.draft_len, 5);
        assert!((config.min_acceptance_rate - 0.3).abs() < 1e-6);
        assert_eq!(config.stats_window, 50);
    }
}
