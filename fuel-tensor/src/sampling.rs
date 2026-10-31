// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sampling controls for decode loops (ROADMAP row 133, slice 1: the pure host sampler).
//!
//! [`sample_with_params`] picks the next token from one logits row under a [`SamplingParams`] and
//! the tokens seen so far. It is host-side and deterministic: the same logits, params, history and
//! RNG state give the same token. With only a positive temperature set it reduces exactly to
//! [`crate::lazy::sample_logits`] (same RNG stream, same probabilities), which a test pins.
//!
//! **Order of operations**, so a caller can predict the result:
//! 1. penalties on the raw logits, from `history`: repetition (HF form: a seen token's positive
//!    logit is divided by the penalty, a negative one multiplied), then
//!    `logit -= frequency * count + presence * (count > 0)`;
//! 2. `temperature <= 0` or NaN: the argmax of the penalised logits (greedy), lowest index on a tie;
//! 3. otherwise scale by `1 / temperature`, softmax, then `top_k`, then `top_p` (nucleus: the
//!    smallest prefix of descending probabilities reaching `top_p`, at least one token), then
//!    `min_p` (drop tokens below `min_p` times the largest remaining probability), renormalise,
//!    and draw with [`crate::lazy::sample_multinomial`].
//!
//! **Logprobs are of the model's own distribution**: `log_softmax` of the INPUT logits, before
//! penalties, temperature and filters (vLLM's "raw logprobs"). They describe what the model said,
//! not what was sampled from, and they do not depend on the sampling controls.
//!
//! `history` is whatever the caller wants penalised. OpenAI penalises generated tokens only;
//! HF's repetition penalty covers the prompt too. The scheduler slice decides which it passes.

use crate::lazy::sample_multinomial;
use fuel_ir::{Error, Result};

/// Every control for one sampling step. `Default` is greedy with no penalties and no logprobs.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature; `<= 0` or NaN means greedy.
    pub temperature: f32,
    /// Keep only the `k` most probable tokens (`k >= 1`).
    pub top_k: Option<usize>,
    /// Nucleus mass in `(0, 1]`.
    pub top_p: Option<f32>,
    /// Minimum probability relative to the most probable token, in `[0, 1]`.
    pub min_p: Option<f32>,
    /// Subtracted once from a token's logit if it appears in the history.
    pub presence_penalty: f32,
    /// Subtracted per occurrence of a token in the history.
    pub frequency_penalty: f32,
    /// Multiplicative penalty on seen tokens (`> 0`; `1.0` is off).
    pub repetition_penalty: f32,
    /// Seeds the caller's RNG state for a new sequence.
    pub seed: u64,
    /// Report this many top logprobs (plus the chosen token's); `None` reports none.
    pub logprobs: Option<usize>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_p: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
            logprobs: None,
        }
    }
}

/// One sampled step.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampled {
    /// The chosen token.
    pub token: u32,
    /// Its logprob under the model's own distribution; `None` unless `logprobs` was requested.
    pub logprob: Option<f32>,
    /// The `logprobs` most probable tokens, most probable first (lowest index on a tie).
    pub top_logprobs: Vec<(u32, f32)>,
}

fn err(msg: String) -> Error {
    Error::Msg(msg).bt()
}

impl SamplingParams {
    /// Check every control against the vocabulary size; the first problem is named.
    pub fn validate(&self, vocab: usize) -> Result<()> {
        if let Some(0) = self.top_k {
            return Err(err("SamplingParams: top_k must be >= 1".into()));
        }
        if let Some(p) = self.top_p
            && !(p > 0.0 && p <= 1.0)
        {
            return Err(err(format!("SamplingParams: top_p {p} is not in (0, 1]")));
        }
        if let Some(p) = self.min_p
            && !(0.0..=1.0).contains(&p)
        {
            return Err(err(format!("SamplingParams: min_p {p} is not in [0, 1]")));
        }
        if !(self.repetition_penalty > 0.0 && self.repetition_penalty.is_finite()) {
            return Err(err(format!(
                "SamplingParams: repetition_penalty {} must be finite and > 0",
                self.repetition_penalty
            )));
        }
        if !self.presence_penalty.is_finite() || !self.frequency_penalty.is_finite() {
            return Err(err("SamplingParams: penalties must be finite".into()));
        }
        if let Some(n) = self.logprobs
            && n > vocab
        {
            return Err(err(format!(
                "SamplingParams: logprobs {n} exceeds the vocabulary size {vocab}"
            )));
        }
        Ok(())
    }
}

/// Pick the next token. Errors on empty or non-finite logits, a history token outside the
/// vocabulary, or invalid params; never panics.
pub fn sample_with_params(
    logits: &[f32],
    params: &SamplingParams,
    history: &[u32],
    rng_state: &mut u64,
) -> Result<Sampled> {
    let vocab = logits.len();
    if vocab == 0 {
        return Err(err("sample_with_params: empty logits".into()));
    }
    if logits.iter().any(|x| !x.is_finite()) {
        return Err(err("sample_with_params: non-finite logit".into()));
    }
    params.validate(vocab)?;

    let adjusted = apply_penalties(logits, params, history)?;
    let token = if params.temperature.is_nan() || params.temperature <= 0.0 {
        argmax(&adjusted)
    } else {
        let probs = filtered_probs(&adjusted, params);
        sample_multinomial(&probs, rng_state)
    };

    let (logprob, top_logprobs) = match params.logprobs {
        None => (None, Vec::new()),
        Some(n) => {
            let lp = log_softmax(logits);
            (Some(lp[token as usize]), top_n(&lp, n))
        }
    };
    Ok(Sampled {
        token,
        logprob,
        top_logprobs,
    })
}

fn apply_penalties(logits: &[f32], p: &SamplingParams, history: &[u32]) -> Result<Vec<f32>> {
    let mut counts = vec![0_u32; logits.len()];
    for t in history {
        let slot = counts.get_mut(*t as usize).ok_or_else(|| {
            err(format!(
                "sample_with_params: history token {t} is outside the vocabulary of {}",
                logits.len()
            ))
        })?;
        *slot += 1;
    }
    Ok(logits
        .iter()
        .zip(&counts)
        .map(|(&l, &c)| {
            if c == 0 {
                return l;
            }
            let rep = if l > 0.0 {
                l / p.repetition_penalty
            } else {
                l * p.repetition_penalty
            };
            rep - p.frequency_penalty * c as f32 - p.presence_penalty
        })
        .collect())
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in x.iter().enumerate().skip(1) {
        if *v > x[best] {
            best = i;
        }
    }
    best as u32
}

/// Temperature softmax, then the top_k, top_p and min_p filters, renormalised.
fn filtered_probs(adjusted: &[f32], p: &SamplingParams) -> Vec<f32> {
    // Same arithmetic as `sample_logits` so the unfiltered case matches it bit for bit.
    let inv_temp = 1.0 / p.temperature;
    let scaled: Vec<f32> = adjusted.iter().map(|&x| x * inv_temp).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let mut probs: Vec<f32> = exp.iter().map(|&x| x / sum).collect();
    if p.top_k.is_none() && p.top_p.is_none() && p.min_p.is_none() {
        return probs;
    }

    // Token order by descending probability, lowest index first on a tie.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    let mut keep = order.len();
    if let Some(k) = p.top_k {
        keep = keep.min(k);
    }
    if let Some(top_p) = p.top_p {
        let mut cum = 0.0_f32;
        let mut n = 0;
        for &i in order.iter().take(keep) {
            cum += probs[i];
            n += 1;
            if cum >= top_p {
                break;
            }
        }
        keep = keep.min(n);
    }
    if let Some(min_p) = p.min_p {
        let floor = min_p * probs[order[0]];
        keep = order
            .iter()
            .take(keep)
            .take_while(|&&i| probs[i] >= floor)
            .count()
            .max(1);
    }
    let kept: std::collections::HashSet<usize> = order.iter().take(keep).copied().collect();
    let mut total = 0.0_f32;
    for (i, pr) in probs.iter_mut().enumerate() {
        if kept.contains(&i) {
            total += *pr;
        } else {
            *pr = 0.0;
        }
    }
    probs.iter_mut().for_each(|pr| *pr /= total);
    probs
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln() + max;
    logits.iter().map(|&x| x - lse).collect()
}

fn top_n(logprobs: &[f32], n: usize) -> Vec<(u32, f32)> {
    let mut order: Vec<usize> = (0..logprobs.len()).collect();
    order.sort_by(|&a, &b| logprobs[b].total_cmp(&logprobs[a]).then(a.cmp(&b)));
    order
        .into_iter()
        .take(n)
        .map(|i| (i as u32, logprobs[i]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::{SamplingStrategy, sample_logits};

    const L: [f32; 5] = [0.1, 0.2, 0.0, 0.3, 0.15];

    fn p() -> SamplingParams {
        SamplingParams {
            temperature: 1.0,
            ..Default::default()
        }
    }

    fn draws(params: &SamplingParams, history: &[u32], n: usize) -> Vec<u32> {
        let mut rng = 7_u64;
        (0..n)
            .map(|_| {
                sample_with_params(&L, params, history, &mut rng)
                    .unwrap()
                    .token
            })
            .collect()
    }

    #[test]
    fn default_is_greedy_and_reports_no_logprobs() {
        let s = sample_with_params(&L, &SamplingParams::default(), &[], &mut 0).unwrap();
        assert_eq!(s.token, 3);
        assert_eq!((s.logprob, s.top_logprobs.len()), (None, 0));
    }

    #[test]
    fn plain_temperature_reduces_to_sample_logits_on_the_same_rng_stream() {
        let (mut a, mut b) = (42_u64, 42_u64);
        for _ in 0..50 {
            let new = sample_with_params(&L, &p(), &[], &mut a).unwrap().token;
            let old = sample_logits(
                &L,
                SamplingStrategy::Temperature {
                    temp: 1.0,
                    seed: 42,
                },
                &mut b,
            );
            assert_eq!(new, old);
        }
        assert_eq!(a, b, "both must consume the RNG identically");
    }

    #[test]
    fn top_k_confines_the_draws_and_the_unfiltered_control_escapes() {
        let top2 = SamplingParams {
            top_k: Some(2),
            ..p()
        };
        assert!(draws(&top2, &[], 200).iter().all(|t| *t == 3 || *t == 1));
        assert!(draws(&p(), &[], 200).iter().any(|t| *t != 3 && *t != 1));
        let one = SamplingParams {
            top_k: Some(1),
            temperature: 5.0,
            ..p()
        };
        assert!(draws(&one, &[], 50).iter().all(|t| *t == 3));
    }

    #[test]
    fn top_p_keeps_the_smallest_prefix_and_always_at_least_one_token() {
        let tiny = SamplingParams {
            top_p: Some(0.0001),
            ..p()
        };
        assert!(draws(&tiny, &[], 50).iter().all(|t| *t == 3));
        // top_p = 1.0 filters nothing: identical draws to the unfiltered sampler.
        let all = SamplingParams {
            top_p: Some(1.0),
            ..p()
        };
        assert_eq!(draws(&all, &[], 100), draws(&p(), &[], 100));
    }

    #[test]
    fn min_p_one_keeps_only_the_most_probable_and_zero_keeps_all() {
        let one = SamplingParams {
            min_p: Some(1.0),
            ..p()
        };
        assert!(draws(&one, &[], 50).iter().all(|t| *t == 3));
        let zero = SamplingParams {
            min_p: Some(0.0),
            ..p()
        };
        assert_eq!(draws(&zero, &[], 100), draws(&p(), &[], 100));
    }

    #[test]
    fn repetition_penalty_divides_positive_and_multiplies_negative_logits() {
        let g = |l: &[f32], h: &[u32], rp: f32| {
            let params = SamplingParams {
                repetition_penalty: rp,
                ..Default::default()
            };
            sample_with_params(l, &params, h, &mut 0).unwrap().token
        };
        assert_eq!(g(&[2.0, 1.9], &[], 2.0), 0);
        assert_eq!(g(&[2.0, 1.9], &[0], 2.0), 1, "2.0 / 2 = 1.0 < 1.9");
        assert_eq!(g(&[-1.0, -1.5], &[0], 2.0), 1, "-1.0 * 2 = -2.0 < -1.5");
    }

    #[test]
    fn frequency_counts_every_occurrence_and_presence_counts_once() {
        let run = |f: f32, pr: f32, h: &[u32]| {
            let params = SamplingParams {
                frequency_penalty: f,
                presence_penalty: pr,
                ..Default::default()
            };
            sample_with_params(&[1.0, 0.7], &params, h, &mut 0)
                .unwrap()
                .token
        };
        assert_eq!(run(0.1, 0.0, &[0]), 0, "1.0 - 0.1 = 0.9 still beats 0.7");
        assert_eq!(run(0.1, 0.0, &[0, 0, 0, 0]), 1, "1.0 - 0.4 = 0.6 loses");
        assert_eq!(
            run(0.0, 0.5, &[0, 0, 0, 0]),
            1,
            "presence 0.5 once: 0.5 loses"
        );
        assert_eq!(
            run(0.0, 0.2, &[0, 0, 0, 0]),
            0,
            "presence is not multiplied by count"
        );
    }

    #[test]
    fn logprobs_are_the_models_own_distribution_and_ignore_the_controls() {
        let all = SamplingParams {
            logprobs: Some(5),
            top_k: Some(1),
            repetition_penalty: 3.0,
            ..p()
        };
        let s = sample_with_params(&L, &all, &[3], &mut 1).unwrap();
        let total: f32 = s.top_logprobs.iter().map(|(_, lp)| lp.exp()).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "full-vocab logprobs sum to {total}"
        );
        let order: Vec<u32> = s.top_logprobs.iter().map(|(t, _)| *t).collect();
        assert_eq!(order, vec![3, 1, 4, 0, 2]);
        let chosen = s
            .top_logprobs
            .iter()
            .find(|(t, _)| *t == s.token)
            .unwrap()
            .1;
        assert_eq!(s.logprob, Some(chosen));
        let none = sample_with_params(
            &L,
            &SamplingParams {
                logprobs: Some(5),
                ..Default::default()
            },
            &[],
            &mut 1,
        )
        .unwrap();
        assert_eq!(
            none.top_logprobs, s.top_logprobs,
            "controls must not move logprobs"
        );
    }

    #[test]
    fn same_seed_same_tokens_and_a_different_seed_diverges() {
        let run = |seed: u64| {
            let mut rng = seed;
            (0..64)
                .map(|_| sample_with_params(&L, &p(), &[], &mut rng).unwrap().token)
                .collect::<Vec<_>>()
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    #[test]
    fn bad_input_is_a_typed_error_not_a_panic() {
        let bad = |params: SamplingParams, logits: &[f32], h: &[u32]| {
            sample_with_params(logits, &params, h, &mut 0).is_err()
        };
        let d = SamplingParams::default;
        assert!(bad(d(), &[], &[]));
        assert!(bad(d(), &[1.0, f32::NAN], &[]));
        assert!(bad(d(), &[1.0, 2.0], &[9]));
        assert!(bad(
            SamplingParams {
                top_k: Some(0),
                ..d()
            },
            &L,
            &[]
        ));
        assert!(bad(
            SamplingParams {
                top_p: Some(0.0),
                ..d()
            },
            &L,
            &[]
        ));
        assert!(bad(
            SamplingParams {
                top_p: Some(1.5),
                ..d()
            },
            &L,
            &[]
        ));
        assert!(bad(
            SamplingParams {
                min_p: Some(-0.1),
                ..d()
            },
            &L,
            &[]
        ));
        assert!(bad(
            SamplingParams {
                repetition_penalty: 0.0,
                ..d()
            },
            &L,
            &[]
        ));
        assert!(bad(
            SamplingParams {
                logprobs: Some(6),
                ..d()
            },
            &L,
            &[]
        ));
        assert!(
            !bad(
                SamplingParams {
                    logprobs: Some(5),
                    ..d()
                },
                &L,
                &[]
            ),
            "control: valid input passes"
        );
    }
}
