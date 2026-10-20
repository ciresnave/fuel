// SPDX-License-Identifier: MIT OR Apache-2.0
//! Model-free speculative drafting by n-gram lookup in the token history
//! (ROADMAP item 17, row H24).
//!
//! "Prompt lookup" drafting needs no second model: when the text being
//! generated repeats text already in the context (a code edit that re-emits most
//! of the file, a summary that quotes, a structured answer that repeats keys),
//! the most likely continuation of the last few tokens is whatever followed
//! the same few tokens earlier. [`NgramDrafter::draft`] finds the longest
//! recent n-gram match and proposes the tokens that followed it. The target
//! model still verifies every proposed token ([`crate::speculative::verify_draft`]),
//! so the output distribution is the target's; a bad draft costs only the
//! (cheap) lookup and a wasted verification slot.
//!
//! This is **policy** (it lives above the Foundation seam, next to the
//! verifier). The draft is a point mass, so it plugs into the existing
//! verifier through [`one_hot_logprobs`], which gives it the one-hot
//! distribution the verifier's acceptance test (`min(1, p_target / p_draft)`)
//! needs to be exact for a deterministic proposal.

/// A longest-match n-gram drafter over a caller-owned token history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramDrafter {
    /// Longest suffix n-gram tried first (at least 1).
    pub max_ngram: usize,
    /// Shortest suffix n-gram accepted as evidence (at least 1). Short n-grams
    /// match everywhere; 2 is the usual floor.
    pub min_ngram: usize,
    /// Most tokens proposed per call.
    pub max_draft: usize,
}

impl NgramDrafter {
    /// A drafter trying suffixes of `max_ngram` down to `min_ngram` tokens and
    /// proposing up to `max_draft` tokens. Values below 1 are raised to 1 and
    /// `min_ngram` is capped at `max_ngram`.
    pub fn new(max_ngram: usize, min_ngram: usize, max_draft: usize) -> Self {
        let max_ngram = max_ngram.max(1);
        Self {
            max_ngram,
            min_ngram: min_ngram.clamp(1, max_ngram),
            max_draft: max_draft.max(1),
        }
    }

    /// Propose a continuation of `history`.
    ///
    /// For each suffix length from `max_ngram` down to `min_ngram`, look for
    /// the MOST RECENT earlier occurrence of the last `n` tokens (the suffix
    /// itself is excluded) and, on the first length that matches, return up to
    /// `max_draft` tokens that followed it. Returns an empty draft when
    /// nothing matches or the history is too short. Never panics.
    ///
    /// Cost is O(history x max_ngram) per call; a rolling index is the obvious
    /// next step if drafting ever shows up in a profile next to a model step.
    pub fn draft(&self, history: &[u32]) -> Vec<u32> {
        let len = history.len();
        for n in (self.min_ngram..=self.max_ngram).rev() {
            // Need the suffix plus at least one earlier token to match against.
            if len < n + 1 {
                continue;
            }
            let suffix = &history[len - n..];
            // Candidate start positions p of an earlier occurrence: the
            // occurrence must end before the suffix starts, so p + n <= len - n
            // would be strict non-overlap; overlapping repeats ("a a a") are
            // legitimate evidence, so only the suffix's own position is
            // excluded: p < len - n.
            for p in (0..len - n).rev() {
                if &history[p..p + n] == suffix {
                    let start = p + n;
                    let end = (start + self.max_draft).min(len);
                    if start < end {
                        return history[start..end].to_vec();
                    }
                }
            }
        }
        Vec::new()
    }
}

/// One-hot log-probabilities (`0.0` at the drafted token, `-inf` elsewhere), one
/// row per drafted token: the draft distribution of a deterministic proposal,
/// in the shape [`crate::speculative::verify_draft`] takes.
pub fn one_hot_logprobs(draft: &[u32], vocab_size: usize) -> Vec<Vec<f32>> {
    draft
        .iter()
        .map(|&t| {
            let mut row = vec![f32::NEG_INFINITY; vocab_size];
            if let Some(slot) = row.get_mut(t as usize) {
                *slot = 0.0;
            }
            row
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposes_what_followed_the_most_recent_match() {
        // ... 5 6 7 8 9 ... 5 6   -> after "5 6" came 7 8 9.
        let h = [1, 5, 6, 7, 8, 9, 2, 5, 6];
        assert_eq!(NgramDrafter::new(2, 2, 3).draft(&h), vec![7, 8, 9]);
    }

    #[test]
    fn prefers_the_most_recent_of_several_matches() {
        // "5 6" occurs twice before the end; the later one was followed by 70.
        let h = [5, 6, 7, 0, 5, 6, 70, 0, 5, 6];
        assert_eq!(NgramDrafter::new(2, 2, 1).draft(&h), vec![70]);
    }

    #[test]
    fn prefers_the_longer_ngram_over_a_more_recent_shorter_one() {
        // The 3-gram "1 5 6" was followed by 7; the more recent bare "5 6"
        // was followed by 99. The longer evidence wins.
        let h = [1, 5, 6, 7, 0, 5, 6, 99, 0, 1, 5, 6];
        assert_eq!(NgramDrafter::new(3, 2, 1).draft(&h), vec![7]);
    }

    #[test]
    fn falls_back_to_a_shorter_ngram_when_the_long_one_is_absent() {
        let h = [9, 5, 6, 7, 0, 3, 5, 6];
        assert_eq!(NgramDrafter::new(3, 2, 1).draft(&h), vec![7]);
    }

    #[test]
    fn no_match_and_short_histories_give_an_empty_draft() {
        let d = NgramDrafter::new(3, 2, 4);
        assert!(d.draft(&[]).is_empty());
        assert!(d.draft(&[1]).is_empty());
        assert!(d.draft(&[1, 2]).is_empty());
        assert!(d.draft(&[1, 2, 3, 4, 5, 6]).is_empty());
    }

    #[test]
    fn the_draft_is_clamped_to_the_end_of_history_and_to_max_draft() {
        // "5 6" at the start is followed by only 7 before the suffix begins to
        // overlap the history's end.
        let h = [5, 6, 7, 5, 6];
        assert_eq!(NgramDrafter::new(2, 2, 10).draft(&h), vec![7, 5, 6]);
        assert_eq!(NgramDrafter::new(2, 2, 2).draft(&h), vec![7, 5]);
    }

    #[test]
    fn an_overlapping_repeat_is_still_evidence() {
        // "a a a a": the suffix "a a" also occurs one step earlier.
        assert_eq!(NgramDrafter::new(2, 2, 3).draft(&[4, 4, 4, 4]), vec![4]);
    }

    #[test]
    fn constructor_clamps_degenerate_arguments() {
        let d = NgramDrafter::new(0, 9, 0);
        assert_eq!((d.max_ngram, d.min_ngram, d.max_draft), (1, 1, 1));
        // And such a drafter is still total.
        assert_eq!(d.draft(&[3, 3]), vec![3]);
    }

    #[test]
    fn one_hot_rows_have_a_single_zero_and_ignore_out_of_vocab_tokens() {
        let rows = one_hot_logprobs(&[2, 9], 4);
        assert_eq!(
            rows[0],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]
        );
        assert!(rows[1].iter().all(|v| *v == f32::NEG_INFINITY));
    }

    /// The point-mass draft plugs into the real verifier: a target that puts
    /// its mass on the drafted tokens accepts all of them and adds the bonus
    /// token; a target that disagrees at the second token accepts one.
    #[test]
    fn verifies_through_verify_draft() {
        use crate::speculative::{SpeculativeConfig, SpeculativeStats, verify_draft};
        let vocab = 6;
        let draft = [3u32, 4];
        let dlp = one_hot_logprobs(&draft, vocab);
        let peak = |t: usize| {
            let mut r = vec![-30.0f32; vocab];
            r[t] = 0.0;
            r
        };
        let cfg = SpeculativeConfig::new(2);
        let mut stats = SpeculativeStats::new();
        let agree = vec![peak(3), peak(4), peak(5)];
        let r = verify_draft(&draft, &dlp, &agree, &cfg, &mut stats).unwrap();
        assert_eq!((r.num_accepted, r.accepted_tokens.clone()), (2, vec![3, 4]));
        assert_eq!(r.next_token, 5);
        let disagree = vec![peak(3), peak(1), peak(5)];
        let r = verify_draft(&draft, &dlp, &disagree, &cfg, &mut stats).unwrap();
        assert_eq!(r.num_accepted, 1);
        assert_eq!(r.accepted_tokens, vec![3]);
    }
}
