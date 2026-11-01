// SPDX-License-Identifier: MIT OR Apache-2.0
//! Multi-session decode for hybrid models (ROADMAP row 117, design note
//! `docs/design-notes/2026-10-30-hybrid-decode-state-row-117.md`).
//!
//! [`crate::multi_session::DecodeModel`] hands every step a `&mut KvCache`, which cannot hold a
//! Gated DeltaNet layer's fixed-size recurrent state. This module is the additive sibling: a
//! model names its own per-sequence cache type ([`HybridDecodeModel::Cache`]) and the scheduler
//! owns one per session without knowing what is in it. Existing `DecodeModel` impls and
//! [`crate::multi_session::SessionScheduler`] are untouched.
//!
//! **Why the scheduler's `KvCache` is not in the session.** `Qwen3_5Model::decode` already owns one
//! host cache holding both the growing attention K/V and the fixed GDN state, so a scheduler-owned
//! `KvCache` beside it would be allocated and never read. The container for the later plan-once
//! path, where the fixed state is bound as graph input, is [`crate::hybrid_state::FixedState`].
//!
//! **All-or-nothing.** [`HybridDecodeModel::step`] must leave the cache unchanged when it returns
//! `Err`. The scheduler relies on it: a failed step finishes that session and does not touch the
//! others, and the cache it leaves behind is the last good one.

use crate::multi_session::{SessionId, SessionPhase, TokenLogprob, admit_params, sample_next};
use fuel::lazy::SamplingStrategy;
use fuel::sampling::SamplingParams;
use fuel_ir::{Error, Result};

/// A model that decodes with its own per-sequence cache.
pub trait HybridDecodeModel {
    /// Everything one sequence keeps between steps (attention K/V and fixed recurrent state).
    type Cache;

    /// Logit row width.
    fn vocab_size(&self) -> usize;

    /// An empty cache for a new sequence.
    fn new_cache(&self) -> Self::Cache;

    /// Host bytes `cache` holds, for capacity accounting.
    fn cache_bytes(&self, cache: &Self::Cache) -> usize;

    /// Feed `tokens` after those already consumed and return the logits of the LAST token
    /// (`vocab_size()` values). **On `Err` the cache is unchanged.**
    fn step(&self, cache: &mut Self::Cache, tokens: &[u32]) -> Result<Vec<f32>>;
}

/// One hybrid session: its cache plus sampler, RNG and token bookkeeping. Shares nothing.
pub struct HybridSession<C> {
    cache: C,
    id: SessionId,
    tokens: Vec<u32>,
    new_tokens: Vec<u32>,
    rng_state: u64,
    strategy: SamplingStrategy,
    /// Full sampling controls, when admitted with them; `None` = `strategy` alone.
    params: Option<SamplingParams>,
    /// Per-token logprobs, one per generated token, when `params.logprobs` is set.
    logprobs: Vec<TokenLogprob>,
    eos_ids: Option<Vec<u32>>,
    remaining: usize,
    phase: SessionPhase,
}

impl<C> HybridSession<C> {
    /// This session's id.
    pub fn id(&self) -> SessionId {
        self.id
    }

    /// Prompt plus generated tokens.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Generated tokens only.
    pub fn new_tokens(&self) -> &[u32] {
        &self.new_tokens
    }

    /// Logprobs of the generated tokens, one per entry of [`Self::new_tokens`]; empty unless the
    /// session was admitted with `SamplingParams::logprobs` set.
    pub fn logprobs(&self) -> &[TokenLogprob] {
        &self.logprobs
    }

    /// The session's cache, for inspection.
    pub fn cache(&self) -> &C {
        &self.cache
    }

    /// Can this session still advance?
    pub fn is_ready(&self) -> bool {
        self.phase != SessionPhase::Finished
    }

    fn accept(&mut self, logits: &[f32]) -> Result<u32> {
        let (next, lp) = sample_next(
            self.strategy,
            self.params.as_ref(),
            logits,
            &self.tokens,
            &mut self.rng_state,
        )?;
        self.logprobs.extend(lp);
        self.tokens.push(next);
        self.new_tokens.push(next);
        self.remaining = self.remaining.saturating_sub(1);
        let eos = self.eos_ids.as_ref().is_some_and(|ids| ids.contains(&next));
        self.phase = if eos || self.remaining == 0 {
            SessionPhase::Finished
        } else {
            SessionPhase::Decode
        };
        Ok(next)
    }
}

/// Round-robin scheduler over hybrid sessions sharing one read-only model.
pub struct HybridSessionScheduler<'m, M: HybridDecodeModel> {
    model: &'m M,
    sessions: Vec<HybridSession<M::Cache>>,
    next_id: u64,
}

impl<'m, M: HybridDecodeModel> HybridSessionScheduler<'m, M> {
    /// An empty scheduler.
    pub fn new(model: &'m M) -> Self {
        Self {
            model,
            sessions: Vec::new(),
            next_id: 0,
        }
    }

    /// Admit a session in the prefill phase. Refuses an empty prompt or a zero budget.
    pub fn add_session(
        &mut self,
        prompt: &[u32],
        strategy: SamplingStrategy,
        eos_ids: Option<Vec<u32>>,
        max_new: usize,
    ) -> Result<SessionId> {
        if prompt.is_empty() {
            return Err(Error::Msg("HybridSessionScheduler: prompt is empty".into()).bt());
        }
        if max_new == 0 {
            return Err(Error::Msg("HybridSessionScheduler: max_new must be > 0".into()).bt());
        }
        let id = SessionId(self.next_id);
        self.next_id += 1;
        let rng_state = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };
        self.sessions.push(HybridSession {
            cache: self.model.new_cache(),
            id,
            tokens: prompt.to_vec(),
            new_tokens: Vec::new(),
            rng_state,
            strategy,
            params: None,
            logprobs: Vec::new(),
            eos_ids: eos_ids.filter(|ids| !ids.is_empty()),
            remaining: max_new,
            phase: SessionPhase::Prefill,
        });
        Ok(id)
    }

    /// [`add_session`](Self::add_session) with the full sampling controls (ROADMAP row 133): top_k,
    /// top_p, min_p and penalties, the penalties reading the session's own token history. See
    /// `SessionScheduler::add_session_with_params`.
    pub fn add_session_with_params(
        &mut self,
        prompt: &[u32],
        params: SamplingParams,
        eos_ids: Option<Vec<u32>>,
        max_new: usize,
    ) -> Result<SessionId> {
        let strategy = admit_params(&params)?;
        let id = self.add_session(prompt, strategy, eos_ids, max_new)?;
        if let Some(s) = self.sessions.last_mut() {
            s.rng_state = params.seed;
            s.params = Some(params);
        }
        Ok(id)
    }

    /// Live sessions (finished ones stay until [`Self::reap_finished`]).
    pub fn sessions(&self) -> &[HybridSession<M::Cache>] {
        &self.sessions
    }

    /// Host bytes held by every live session's cache.
    pub fn resident_bytes(&self) -> usize {
        self.sessions
            .iter()
            .map(|s| self.model.cache_bytes(&s.cache))
            .sum()
    }

    /// Advance every ready session by one token. A session whose step fails is finished with its
    /// cache unchanged and its error returned in the result; the others still advance.
    pub fn step(&mut self) -> Vec<(SessionId, Result<u32>)> {
        let model = self.model;
        self.sessions
            .iter_mut()
            .filter(|s| s.is_ready())
            .map(|s| (s.id, advance(model, s)))
            .collect()
    }

    /// Step until no session is ready; returns every session's generated tokens by id.
    pub fn run_to_completion(&mut self) -> Vec<(SessionId, Vec<u32>)> {
        while self.sessions.iter().any(HybridSession::is_ready) {
            self.step();
        }
        self.sessions
            .iter()
            .map(|s| (s.id, s.new_tokens.clone()))
            .collect()
    }

    /// Drop finished sessions (freeing their caches); returns how many.
    pub fn reap_finished(&mut self) -> usize {
        let before = self.sessions.len();
        self.sessions.retain(HybridSession::is_ready);
        before - self.sessions.len()
    }
}

fn advance<M: HybridDecodeModel>(model: &M, s: &mut HybridSession<M::Cache>) -> Result<u32> {
    // Prefill feeds the whole prompt; decode feeds only the token sampled last step.
    let fed: &[u32] = match s.phase {
        SessionPhase::Prefill => &s.tokens,
        _ => &s.tokens[s.tokens.len() - 1..],
    };
    match model.step(&mut s.cache, fed) {
        Ok(logits) => s
            .accept(&logits)
            .inspect_err(|_| s.phase = SessionPhase::Finished),
        Err(e) => {
            s.phase = SessionPhase::Finished;
            Err(e)
        }
    }
}

impl HybridDecodeModel for fuel_transformers::models::lazy_qwen3_5::Qwen3_5Model {
    type Cache = fuel_transformers::models::lazy_qwen3_5_decode::Qwen3_5Cache;

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn new_cache(&self) -> Self::Cache {
        Self::new_cache(self)
    }

    fn cache_bytes(&self, cache: &Self::Cache) -> usize {
        cache.bytes()
    }

    fn step(&self, cache: &mut Self::Cache, tokens: &[u32]) -> Result<Vec<f32>> {
        let all = self.decode(cache, tokens)?;
        let v = self.config.vocab_size;
        let last = all.len().checked_sub(v).ok_or_else(|| {
            Error::Msg("Qwen3_5Model::decode returned fewer than vocab_size logits".into()).bt()
        })?;
        Ok(all[last..].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: usize = 7;

    /// A toy hybrid model: the cache is the whole token history, the logits peak at a function of
    /// the full history, so a session that lost or reordered state samples a different token.
    /// `fail_on` makes a step fail (before or after mutating, per `mutate_first`).
    struct Toy {
        fail_on: Option<u32>,
        mutate_first: bool,
    }

    fn peak(history: &[u32]) -> usize {
        history
            .iter()
            .enumerate()
            .map(|(i, t)| (i + 1) * (*t as usize + 1))
            .sum::<usize>()
            % V
    }

    impl HybridDecodeModel for Toy {
        type Cache = Vec<u32>;
        fn vocab_size(&self) -> usize {
            V
        }
        fn new_cache(&self) -> Vec<u32> {
            Vec::new()
        }
        fn cache_bytes(&self, c: &Vec<u32>) -> usize {
            c.len() * 4
        }
        fn step(&self, cache: &mut Vec<u32>, tokens: &[u32]) -> Result<Vec<f32>> {
            let fails = self.fail_on.is_some_and(|f| tokens.contains(&f));
            if fails && self.mutate_first {
                cache.extend_from_slice(tokens);
            }
            if fails {
                return Err(Error::Msg("toy failure".into()).bt());
            }
            cache.extend_from_slice(tokens);
            let mut row = vec![0.0; V];
            row[peak(cache)] = 1.0;
            Ok(row)
        }
    }

    fn ok_toy() -> Toy {
        Toy {
            fail_on: None,
            mutate_first: false,
        }
    }

    /// The oracle: the same recurrence run by hand over the whole history each step.
    fn reference(prompt: &[u32], n: usize) -> Vec<u32> {
        let mut h = prompt.to_vec();
        let mut out = Vec::new();
        for _ in 0..n {
            let t = peak(&h) as u32;
            h.push(t);
            out.push(t);
        }
        out
    }

    #[test]
    fn sessions_match_a_whole_history_reference_and_do_not_interfere() {
        let m = ok_toy();
        let mut s = HybridSessionScheduler::new(&m);
        let a = s
            .add_session(&[0, 0], SamplingStrategy::Greedy, None, 6)
            .unwrap();
        let b = s
            .add_session(&[0, 1], SamplingStrategy::Greedy, None, 4)
            .unwrap();
        let out = s.run_to_completion();
        let get = |id| out.iter().find(|(i, _)| *i == id).unwrap().1.clone();
        assert_eq!(get(a), reference(&[0, 0], 6));
        assert_eq!(get(b), reference(&[0, 1], 4));
        // Control: the two references differ and neither is constant, so lost or crossed state
        // would show.
        assert_ne!(
            get(a),
            get(b)[..]
                .iter()
                .chain(&[0, 0])
                .copied()
                .collect::<Vec<_>>()
        );
        assert!(get(a).windows(2).any(|w| w[0] != w[1]));
    }

    #[test]
    fn a_failed_step_finishes_only_that_session_and_leaves_its_cache_unchanged() {
        let m = Toy {
            fail_on: Some(6),
            mutate_first: false,
        };
        let mut s = HybridSessionScheduler::new(&m);
        let bad = s
            .add_session(&[1, 6], SamplingStrategy::Greedy, None, 3)
            .unwrap();
        let good = s
            .add_session(&[2], SamplingStrategy::Greedy, None, 3)
            .unwrap();
        let r = s.step();
        assert!(r.iter().find(|(i, _)| *i == bad).unwrap().1.is_err());
        assert!(r.iter().find(|(i, _)| *i == good).unwrap().1.is_ok());
        let bad_s = s.sessions().iter().find(|x| x.id() == bad).unwrap();
        assert!(!bad_s.is_ready());
        assert!(bad_s.cache().is_empty(), "cache must be the last good one");
        let good_s = s.sessions().iter().find(|x| x.id() == good).unwrap();
        assert!(good_s.is_ready());
        assert_eq!(s.reap_finished(), 1);
    }

    /// Negative control for the test above: a model that breaks the all-or-nothing rule IS seen
    /// to leave a dirty cache, so the property test can fail.
    #[test]
    fn a_model_that_mutates_before_failing_is_detected() {
        let m = Toy {
            fail_on: Some(6),
            mutate_first: true,
        };
        let mut s = HybridSessionScheduler::new(&m);
        s.add_session(&[1, 6], SamplingStrategy::Greedy, None, 3)
            .unwrap();
        s.step();
        assert!(!s.sessions()[0].cache().is_empty());
    }

    #[test]
    fn eos_and_budget_finish_a_session_and_resident_bytes_track_the_caches() {
        let m = ok_toy();
        let first = reference(&[1], 1)[0];
        let mut s = HybridSessionScheduler::new(&m);
        s.add_session(&[1], SamplingStrategy::Greedy, Some(vec![first]), 9)
            .unwrap();
        s.step();
        assert!(!s.sessions()[0].is_ready(), "eos on the first token");
        assert_eq!(s.resident_bytes(), 4);
        assert!(
            s.add_session(&[], SamplingStrategy::Greedy, None, 1)
                .is_err()
        );
        assert!(
            s.add_session(&[1], SamplingStrategy::Greedy, None, 0)
                .is_err()
        );
    }

    // ---------------- ROADMAP row 133, slice 2 ----------------

    fn greedy_params() -> SamplingParams {
        SamplingParams::default()
    }

    /// The hybrid arm reduces to the strategy path: a temperature-only `SamplingParams` samples the
    /// same tokens as `Temperature` with the same seed.
    #[test]
    fn params_reduce_to_the_strategy_path() {
        let model = ok_toy();
        let run = |with_params: bool| {
            let mut s = HybridSessionScheduler::new(&model);
            let id = if with_params {
                s.add_session_with_params(
                    &[1, 2],
                    SamplingParams {
                        temperature: 2.0,
                        seed: 5,
                        ..greedy_params()
                    },
                    None,
                    8,
                )
                .unwrap()
            } else {
                s.add_session(
                    &[1, 2],
                    SamplingStrategy::Temperature { temp: 2.0, seed: 5 },
                    None,
                    8,
                )
                .unwrap()
            };
            s.run_to_completion()
                .into_iter()
                .find(|(i, _)| *i == id)
                .unwrap()
                .1
        };
        assert_eq!(run(true), run(false));
    }

    /// The penalties see the session's own history: with a large presence penalty the greedy walk
    /// never repeats a token, where the unpenalised walk does (guarded below, so the test cannot
    /// pass on a trajectory that never repeats anyway).
    #[test]
    fn presence_penalty_uses_the_session_history() {
        let model = ok_toy();
        let prompt = [0u32];
        let n = 6;
        let plain = reference(&prompt, n);
        let mut seen = prompt.to_vec();
        seen.extend(&plain);
        let distinct: std::collections::BTreeSet<_> = seen.iter().collect();
        assert!(
            distinct.len() < seen.len(),
            "control: the unpenalised walk repeats a token"
        );

        let mut s = HybridSessionScheduler::new(&model);
        let id = s
            .add_session_with_params(
                &prompt,
                SamplingParams {
                    presence_penalty: 5.0,
                    ..greedy_params()
                },
                None,
                n,
            )
            .unwrap();
        let out = s
            .run_to_completion()
            .into_iter()
            .find(|(i, _)| *i == id)
            .unwrap()
            .1;
        let mut all = prompt.to_vec();
        all.extend(&out);
        let distinct: std::collections::BTreeSet<_> = all.iter().collect();
        assert_eq!(
            distinct.len(),
            all.len(),
            "no token repeats under the penalty: {all:?}"
        );
    }

    /// Bad params are refused at admission, admitting nothing.
    #[test]
    fn bad_params_are_refused_and_admit_nothing() {
        let model = ok_toy();
        let mut s = HybridSessionScheduler::new(&model);
        let bad = SamplingParams {
            min_p: Some(2.0),
            ..greedy_params()
        };
        assert!(s.add_session_with_params(&[1], bad, None, 3).is_err());
        assert!(s.sessions().is_empty());
        // And the id counter did not move: the next admission is session 0.
        let id = s
            .add_session_with_params(&[1], greedy_params(), None, 3)
            .unwrap();
        assert_eq!(id, SessionId(0));
    }

    /// A history token outside the vocabulary is a typed sampling error on the params path: that
    /// session is finished with the error returned, and the other session still advances.
    #[test]
    fn a_sampling_error_finishes_only_that_session() {
        let model = ok_toy();
        let mut s = HybridSessionScheduler::new(&model);
        let bad = s
            .add_session_with_params(&[V as u32 + 2], greedy_params(), None, 3)
            .unwrap();
        let good = s
            .add_session_with_params(&[1, 2], greedy_params(), None, 3)
            .unwrap();
        let results = s.step();
        let bad_res = &results.iter().find(|(i, _)| *i == bad).unwrap().1;
        let good_res = &results.iter().find(|(i, _)| *i == good).unwrap().1;
        assert!(
            bad_res.is_err(),
            "the out-of-vocabulary history token is an error"
        );
        assert!(good_res.is_ok());
        assert!(
            !s.sessions()
                .iter()
                .find(|x| x.id() == bad)
                .unwrap()
                .is_ready()
        );
        assert!(
            s.sessions()
                .iter()
                .find(|x| x.id() == good)
                .unwrap()
                .is_ready()
        );
    }

    /// Logprobs are recorded per generated token when asked for and not otherwise; greedy params
    /// make the chosen token the top-1 entry.
    #[test]
    fn logprobs_are_recorded_per_token_only_when_asked() {
        let model = ok_toy();
        let mut s = HybridSessionScheduler::new(&model);
        let with = s
            .add_session_with_params(
                &[1, 2],
                SamplingParams {
                    logprobs: Some(2),
                    ..greedy_params()
                },
                None,
                4,
            )
            .unwrap();
        let without = s
            .add_session_with_params(&[1, 2], greedy_params(), None, 4)
            .unwrap();
        for _ in 0..8 {
            s.step();
        }
        let find = |id| s.sessions().iter().find(|x| x.id() == id).unwrap();
        let a = find(with);
        assert_eq!(a.new_tokens().len(), 4);
        assert_eq!(a.logprobs().len(), 4);
        for (t, lp) in a.new_tokens().iter().zip(a.logprobs()) {
            assert_eq!(lp.top.len(), 2);
            assert_eq!((lp.top[0].0, lp.top[0].1), (*t, lp.logprob));
            assert!(lp.logprob <= 0.0 && lp.top[0].1 >= lp.top[1].1);
        }
        assert!(find(without).logprobs().is_empty());
    }
}
