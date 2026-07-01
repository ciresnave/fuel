//! Host-side sampling helpers over `LazyTensor` logits.
//!
//! Realizes the input logits with [`LazyTensor::realize_f32`] and runs
//! the actual token selection on the host. The lazy-graph side is just
//! a one-shot D2H — the sampling math itself is too branchy to express
//! cleanly as a graph and the cost is dominated by the realize already.
//!
//! Determinism: every stochastic helper takes a `&mut StdRng` so the
//! caller controls the RNG seed end-to-end.
//!
//! Logits convention: the input is treated as a single 1-D distribution
//! over the trailing dim. For rank-N inputs, the helper picks the
//! argmax / sampled index over the full flattened tensor — callers
//! with a `[batch, vocab]` shape should narrow / squeeze to `[vocab]`
//! first.

use crate::Result;
use crate::lazy::LazyTensor;
use rand::distr::Distribution;
use rand::rngs::StdRng;

/// Realize the logits into a host `Vec<f32>` and return it.
fn realize_logits(logits: &LazyTensor) -> Result<Vec<f32>> {
    let v = logits.realize_f32();
    if v.is_empty() {
        return Err(crate::Error::Msg(
            "sampling: realized logits vector is empty".to_string(),
        )
        .bt());
    }
    Ok(v)
}

/// Compute the argmax index over a 1-D `f32` slice. Uses `total_cmp`
/// so NaNs sort to the bottom rather than poisoning the pick.
fn argmax(values: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = values[0];
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v.total_cmp(&best_v).is_gt() {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

/// Numerically stable softmax in place on a `Vec<f32>` after dividing
/// by `temp`. Returns the same vector, normalized to sum to 1.
fn softmax_with_temp(mut logits: Vec<f32>, temp: f32) -> Result<Vec<f32>> {
    if !(temp.is_finite()) || temp <= 0.0 {
        return Err(crate::Error::Msg(format!(
            "sampling: temperature must be > 0 and finite, got {temp}",
        ))
        .bt());
    }
    for v in logits.iter_mut() {
        *v /= temp;
    }
    let mut max_v = logits[0];
    for &v in logits.iter().skip(1) {
        if v.total_cmp(&max_v).is_gt() {
            max_v = v;
        }
    }
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(crate::Error::Msg(format!(
            "sampling: softmax sum is non-positive or non-finite ({sum})",
        ))
        .bt());
    }
    for v in logits.iter_mut() {
        *v /= sum;
    }
    Ok(logits)
}

/// Multinomial sample from a probability vector. Returns the sampled
/// index.
fn sample_multinomial(prs: &[f32], rng: &mut StdRng) -> Result<usize> {
    let distr = rand::distr::weighted::WeightedIndex::new(prs)
        .map_err(crate::Error::wrap)?;
    Ok(distr.sample(rng))
}

/// Greedy / argmax selection. Returns the index of the largest logit
/// — equivalent to deterministic decoding.
pub fn greedy(logits: &LazyTensor) -> Result<u32> {
    let values = realize_logits(logits)?;
    let idx = argmax(&values);
    Ok(idx as u32)
}

/// Temperature-only multinomial sample. Divides the logits by `temp`,
/// applies a numerically stable softmax, and draws one index from the
/// resulting categorical distribution.
pub fn temperature_sample(
    logits: &LazyTensor,
    temp: f32,
    rng: &mut StdRng,
) -> Result<u32> {
    let values = realize_logits(logits)?;
    let prs = softmax_with_temp(values, temp)?;
    let idx = sample_multinomial(&prs, rng)?;
    Ok(idx as u32)
}

/// Top-K restricted multinomial sample. Selects the `k` highest-logit
/// indices, renormalizes their softmax mass to sum to 1, and samples
/// from the restricted set. `k` is clamped to `[1, vocab]`.
pub fn top_k_sample(
    logits: &LazyTensor,
    k: usize,
    temp: f32,
    rng: &mut StdRng,
) -> Result<u32> {
    let values = realize_logits(logits)?;
    let prs = softmax_with_temp(values, temp)?;
    if k == 0 {
        return Err(crate::Error::Msg(
            "sampling: top_k_sample requires k >= 1".to_string(),
        )
        .bt());
    }
    let k = k.min(prs.len());
    let mut idxs: Vec<usize> = (0..prs.len()).collect();
    // `select_nth_unstable_by` partitions in place and returns
    // `(left, pivot, right)` where `left` holds the first `k-1`
    // entries (strictly above the pivot under the comparator) and
    // `pivot` is the k-th entry itself. The top-k index set is
    // therefore `left ++ [pivot_index]`.
    let (left, &mut pivot_idx, _) =
        idxs.select_nth_unstable_by(k - 1, |&i, &j| prs[j].total_cmp(&prs[i]));
    let mut top: Vec<usize> = left.to_vec();
    top.push(pivot_idx);
    let restricted: Vec<f32> = top.iter().map(|&i| prs[i]).collect();
    let pick_local = sample_multinomial(&restricted, rng)?;
    Ok(top[pick_local] as u32)
}

/// Top-P (nucleus) sample. Sorts the softmax probabilities in
/// descending order, keeps the smallest prefix whose cumulative mass
/// reaches `p`, zeros the rest, and samples. With `p >= 1.0` this is
/// equivalent to a plain temperature sample; with `p <= 0.0` it falls
/// back to greedy.
pub fn top_p_sample(
    logits: &LazyTensor,
    p: f32,
    temp: f32,
    rng: &mut StdRng,
) -> Result<u32> {
    let values = realize_logits(logits)?;
    let mut prs = softmax_with_temp(values, temp)?;
    if !p.is_finite() {
        return Err(crate::Error::Msg(format!(
            "sampling: top_p must be finite, got {p}",
        ))
        .bt());
    }
    if p <= 0.0 {
        let idx = argmax(&prs);
        return Ok(idx as u32);
    }
    if p >= 1.0 {
        let idx = sample_multinomial(&prs, rng)?;
        return Ok(idx as u32);
    }
    let mut order: Vec<usize> = (0..prs.len()).collect();
    order.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));
    let mut cumsum = 0.0_f32;
    for &index in &order {
        if cumsum >= p {
            prs[index] = 0.0;
        } else {
            cumsum += prs[index];
        }
    }
    let idx = sample_multinomial(&prs, rng)?;
    Ok(idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use fuel_ir::Shape;
    use rand::SeedableRng;

    fn lazy_logits(values: Vec<f32>) -> LazyTensor {
        let n = values.len();
        LazyTensor::from_f32(values, Shape::from_dims(&[n]), &Device::cpu())
    }

    #[test]
    fn greedy_picks_max_index() {
        let logits = lazy_logits(vec![0.1, 5.0, 0.2, 0.3, -1.0]);
        let idx = greedy(&logits).unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn temperature_sample_with_very_low_temp_picks_argmax() {
        // With a tiny temperature the softmax collapses to a one-hot
        // on the argmax, so the multinomial sample must hit that
        // index with probability ~1.
        let logits = lazy_logits(vec![0.0, 0.0, 4.0, 0.0]);
        let mut rng = StdRng::seed_from_u64(0xdeadbeef);
        let idx = temperature_sample(&logits, 1e-4, &mut rng).unwrap();
        assert_eq!(idx, 2);
    }

    #[test]
    fn top_k_sample_only_picks_from_top_k_indices() {
        // Logits sized so the top-3 indices are deterministically
        // {1, 3, 5}. Run many samples and confirm zero outliers.
        let logits = lazy_logits(vec![
            -2.0, 3.0, -1.5, 2.5, -1.0, 2.0, -0.5, -3.0,
        ]);
        let mut rng = StdRng::seed_from_u64(7);
        let allowed: [u32; 3] = [1, 3, 5];
        for _ in 0..200 {
            let idx = top_k_sample(&logits, 3, 1.0, &mut rng).unwrap();
            assert!(
                allowed.contains(&idx),
                "top_k_sample returned {idx}, not in {allowed:?}",
            );
        }
    }

    #[test]
    fn top_p_sample_with_p_1_equals_full_distribution() {
        // p == 1.0 must reduce to a plain temperature_sample with the
        // same seed. Compare per-draw equality.
        let logits = lazy_logits(vec![0.5, 1.5, -0.5, 2.0, 0.25]);
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        for _ in 0..32 {
            let a = top_p_sample(&logits, 1.0, 1.0, &mut rng_a).unwrap();
            let b = temperature_sample(&logits, 1.0, &mut rng_b).unwrap();
            assert_eq!(a, b);
        }
    }
}
