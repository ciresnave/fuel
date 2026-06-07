//! Logit Processing and Sampling
//!
//! Functionality for modeling sampling strategies and logits processing in text generation
//! with support for temperature-based sampling, top-k filtering, nucleus sampling (top-p),
//! and combinations thereof.
//!
//! As of the eager-retirement Phase H follow-up, `LogitsProcessor` operates entirely on
//! host-side `&[f32]` slices — there is no `Tensor` in this module any more. Callers
//! are expected to realize their lazy logits to a `Vec<f32>` (or borrow an existing
//! slice) and pass them in directly.
//!
//! ```rust,no_run
//! use fuel_transformers::generation::{LogitsProcessor, Sampling};
//!
//! let mut lp = LogitsProcessor::from_sampling(42, Sampling::All { temperature: 0.7 });
//! let logits: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
//! let next = lp.sample(&logits).unwrap();
//! # let _ = next;
//! ```
use fuel::{Error, Result};
use rand::{distr::Distribution, SeedableRng};

/// Token-sampling strategy used during autoregressive text generation.
///
/// Each variant selects a different decode-time algorithm that controls the
/// trade-off between output diversity and quality.
#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    /// Deterministic greedy decoding – always pick the highest-logit token.
    ArgMax,
    /// Full-vocabulary sampling re-scaled by `temperature`.
    ///
    /// A temperature near `0.0` approaches greedy behaviour; `1.0` samples
    /// from the raw model distribution; values above `1.0` increase entropy.
    All { temperature: f64 },
    /// Restrict sampling to the `k` highest-probability tokens before applying
    /// the temperature re-scaling.
    TopK { k: usize, temperature: f64 },
    /// Nucleus (top-*p*) sampling: keep only the smallest set of tokens whose
    /// cumulative probability reaches `p`, then sample from that subset.
    TopP { p: f64, temperature: f64 },
    /// Apply top-*k* truncation first, then nucleus sampling within the survivors.
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
    /// Gumbel-Softmax sampling: add Gumbel noise before argmax, equivalent to
    /// sampling proportionally from the softmax distribution.
    // Note that the rng is not used for the Gumbel-Softmax sampling.
    GumbelSoftmax { temperature: f64 },
}

/// Applies a [`Sampling`] strategy to raw model logits to produce the next token id.
///
/// An internal seeded RNG is used for all stochastic strategies.  For
/// reproducible outputs, always construct the processor with the same seed.
pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}

impl LogitsProcessor {
    /// Creates a `LogitsProcessor` with an explicit [`Sampling`] strategy.
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling }
    }

    /// Creates a `LogitsProcessor` using the legacy `temperature` / `top_p` API.
    ///
    /// | `temperature` | `top_p`  | Resulting strategy            |
    /// |--------------|----------|-------------------------------|
    /// | `None` or ≈0  | any      | [`Sampling::ArgMax`]          |
    /// | `Some(t)`     | `None`   | [`Sampling::All`]             |
    /// | `Some(t)`     | `Some(p)`| [`Sampling::TopP`]            |
    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    fn sample_argmax(&mut self, logits: &[f32]) -> Result<u32> {
        // Argmax with `total_cmp` to handle NaN deterministically and match the
        // previous fuel-core `argmax` behaviour on float dtypes.
        let (idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .ok_or_else(|| Error::Msg("empty logits slice passed to sample_argmax".into()))?;
        Ok(idx as u32)
    }

    /// Host-side Gumbel-Softmax sampling.
    ///
    /// Implements `argmax_i ( (logits_i / temperature) + g_i )` where `g_i` is
    /// drawn from a standard Gumbel(0, 1) distribution. For temperature → 0 this
    /// degenerates to greedy argmax; for temperature == 1 it is equivalent in
    /// distribution to drawing a sample from `softmax(logits)`.
    ///
    /// The RNG is intentionally the same `StdRng` used by the multinomial samplers
    /// so that callers get reproducible sequences from a single seed.
    fn sample_gumbel_softmax(&mut self, logits: &[f32], temperature: f64) -> Result<u32> {
        if logits.is_empty() {
            return Err(Error::Msg(
                "empty logits slice passed to sample_gumbel_softmax".into(),
            ));
        }
        use rand::Rng;
        let t = temperature.max(1e-7) as f32;
        let mut best_idx: usize = 0;
        let mut best_val: f32 = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            // u in (0, 1) to keep -ln(-ln(u)) finite.
            let u: f32 = self.rng.random_range(f32::EPSILON..1.0);
            let g = -(-u.ln()).ln();
            let score = (l / t) + g;
            if score > best_val {
                best_val = score;
                best_idx = i;
            }
        }
        Ok(best_idx as u32)
    }

    fn sample_multinomial(&mut self, prs: &[f32]) -> Result<u32> {
        let distr = rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that have very low probabilities and are
    /// less likely to go "off the rails".
    fn sample_topp(&mut self, prs: &mut [f32], top_p: f32) -> Result<u32> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(prs)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&mut self, prs: &mut [f32], top_k: usize) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&prs)?;
            Ok(indices[index as usize] as u32)
        }
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(&mut self, prs: &mut [f32], top_k: usize, top_p: f32) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs)?
            } else {
                self.sample_topp(&mut prs, top_p)?
            };
            Ok(indices[index as usize] as u32)
        }
    }

    /// Samples the next token id from `logits` using the configured strategy.
    ///
    /// `logits` is the raw vocab-sized logit vector — typically the last row of
    /// the model's `(batch, seq, vocab)` output, already realized to `f32` on the
    /// host. No dtype conversion or device transfer is performed here.
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        self.sample_f(logits, |_| {})
    }

    /// Samples the next token id, allowing a closure `f` to mutate the
    /// probability vector before the final sampling step.
    ///
    /// The closure receives a mutable slice of `f32` probabilities (after
    /// temperature scaling and softmax) and can be used to implement custom
    /// logit processors such as repetition penalties.
    pub fn sample_f(&mut self, logits: &[f32], f: impl FnOnce(&mut [f32])) -> Result<u32> {
        // Numerically-stable softmax computed directly on the host-side logits.
        // `temperature` is applied as `logits / temperature` before the max-shift,
        // which is the standard formulation used by HuggingFace / fuel-core's
        // tensor-based path that this replaces.
        let prs = |temperature: f64| -> Result<Vec<f32>> {
            let t = temperature as f32;
            // Pre-divide by temperature so the subsequent max/exp/normalize is the
            // softmax over the scaled logits.
            let mut scaled: Vec<f32> = logits.iter().map(|&x| x / t).collect();
            let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in scaled.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for v in scaled.iter_mut() {
                    *v *= inv;
                }
            }
            f(&mut scaled);
            Ok(scaled)
        };

        let next_token = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(logits)?,
            Sampling::GumbelSoftmax { temperature } => {
                self.sample_gumbel_softmax(logits, *temperature)?
            }
            Sampling::All { temperature } => {
                let prs = prs(*temperature)?;
                self.sample_multinomial(&prs)?
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = prs(*temperature)?;
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk(&mut prs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk_topp(&mut prs, *k, *p as f32)?
            }
        };
        Ok(next_token)
    }
}
