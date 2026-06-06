//! Host-side weight initializers.
//!
//! Pure functions returning `Vec<f32>` so the caller can wrap the
//! result in any `WeightStorage` flavor (F32, BF16-down-cast, LoRA
//! seed, …). The lazy-graph layer never touches the RNG itself —
//! determinism is the caller's responsibility via a single
//! `rand::rngs::StdRng`.
//!
//! Formulas match PyTorch's `torch.nn.init`:
//!   - `xavier_uniform`: `U(-a, a)` with
//!     `a = sqrt(6 / (fan_in + fan_out))`. Glorot & Bengio (2010).
//!   - `kaiming_uniform`: `U(-bound, bound)` with
//!     `bound = gain * sqrt(3 / fan_in)`. He et al. (2015). The gain
//!     convention follows `torch.nn.init.calculate_gain` — pass
//!     `sqrt(2)` for ReLU, `1.0` for linear, etc.
//!   - `normal`: i.i.d. samples from `N(mean, std^2)`.
//!   - `uniform`: i.i.d. samples from `U(lo, hi)`.

use crate::Result;
use rand::distr::Distribution;
use rand::rngs::StdRng;

/// Xavier (Glorot) uniform initializer. Returns a `Vec<f32>` of length
/// `fan_in * fan_out` sampled from `U(-a, a)` where
/// `a = sqrt(6 / (fan_in + fan_out))`.
pub fn xavier_uniform(
    fan_in: usize,
    fan_out: usize,
    rng: &mut StdRng,
) -> Result<Vec<f32>> {
    if fan_in == 0 || fan_out == 0 {
        return Err(crate::Error::Msg(format!(
            "xavier_uniform: fan_in and fan_out must be >= 1, got \
             fan_in={fan_in} fan_out={fan_out}",
        ))
        .bt());
    }
    let denom = (fan_in + fan_out) as f32;
    let a = (6.0_f32 / denom).sqrt();
    uniform(-a, a, fan_in * fan_out, rng)
}

/// Kaiming (He) uniform initializer. Returns a `Vec<f32>` of length
/// `fan_in` sampled from `U(-bound, bound)` where
/// `bound = gain * sqrt(3 / fan_in)`.
///
/// `gain` follows the PyTorch `calculate_gain` convention — pass
/// `sqrt(2)` for ReLU, `1.0` for linear, etc.
///
/// Note: the eager fuel-nn module stores a 2-D weight and chooses
/// `fan_in` from its shape. Here the helper takes the already-computed
/// `fan_in` and returns a flat buffer the caller reshapes itself.
pub fn kaiming_uniform(
    fan_in: usize,
    gain: f32,
    rng: &mut StdRng,
) -> Result<Vec<f32>> {
    if fan_in == 0 {
        return Err(crate::Error::Msg(
            "kaiming_uniform: fan_in must be >= 1".to_string(),
        )
        .bt());
    }
    if !gain.is_finite() {
        return Err(crate::Error::Msg(format!(
            "kaiming_uniform: gain must be finite, got {gain}",
        ))
        .bt());
    }
    let std = gain / (fan_in as f32).sqrt();
    let bound = (3.0_f32).sqrt() * std;
    uniform(-bound, bound, fan_in, rng)
}

/// Normal-distribution initializer. Returns `n` i.i.d. samples from
/// `N(mean, std^2)`.
pub fn normal(
    mean: f32,
    std: f32,
    n: usize,
    rng: &mut StdRng,
) -> Result<Vec<f32>> {
    if std < 0.0 || !std.is_finite() || !mean.is_finite() {
        return Err(crate::Error::Msg(format!(
            "normal: invalid parameters mean={mean} std={std}",
        ))
        .bt());
    }
    let dist = rand_distr::Normal::new(mean, std).map_err(crate::Error::wrap)?;
    Ok((0..n).map(|_| dist.sample(rng)).collect())
}

/// Uniform-distribution initializer. Returns `n` i.i.d. samples from
/// `U(lo, hi)`. Requires `lo < hi`.
pub fn uniform(
    lo: f32,
    hi: f32,
    n: usize,
    rng: &mut StdRng,
) -> Result<Vec<f32>> {
    if !lo.is_finite() || !hi.is_finite() || !(lo < hi) {
        return Err(crate::Error::Msg(format!(
            "uniform: require lo < hi and both finite, got lo={lo} hi={hi}",
        ))
        .bt());
    }
    let dist = rand::distr::Uniform::new(lo, hi).map_err(crate::Error::wrap)?;
    Ok((0..n).map(|_| dist.sample(rng)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn mean_and_var(xs: &[f32]) -> (f32, f32) {
        let n = xs.len() as f32;
        let mean = xs.iter().sum::<f32>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        (mean, var)
    }

    #[test]
    fn xavier_uniform_mean_zero_variance_2_over_fan_in_plus_fan_out() {
        let fan_in = 64;
        let fan_out = 64;
        let mut rng = StdRng::seed_from_u64(0xabad1dea);
        // Generate a large sample to keep the statistical tolerance
        // tight without flakiness.
        let n_trials = 64;
        let mut all = Vec::with_capacity(n_trials * fan_in * fan_out);
        for _ in 0..n_trials {
            all.extend(xavier_uniform(fan_in, fan_out, &mut rng).unwrap());
        }
        let (mean, var) = mean_and_var(&all);
        // U(-a, a) has variance a^2 / 3 = (6 / (fan_in+fan_out)) / 3
        //                              = 2 / (fan_in + fan_out)
        let expected_var = 2.0_f32 / (fan_in + fan_out) as f32;
        assert!(
            mean.abs() < 5e-4,
            "xavier mean {mean} not near 0 (n={})",
            all.len(),
        );
        let rel = (var - expected_var).abs() / expected_var;
        assert!(
            rel < 0.02,
            "xavier var {var} expected {expected_var} (rel diff {rel})",
        );
        for v in &all {
            assert!(v.is_finite(), "xavier produced non-finite value");
        }
    }

    #[test]
    fn kaiming_uniform_distribution_check() {
        let fan_in = 512;
        let gain = (2.0_f32).sqrt(); // ReLU gain
        let mut rng = StdRng::seed_from_u64(123);
        let n_trials = 256;
        let mut all = Vec::with_capacity(n_trials * fan_in);
        for _ in 0..n_trials {
            all.extend(kaiming_uniform(fan_in, gain, &mut rng).unwrap());
        }
        // Variance of U(-bound, bound) = bound^2 / 3 = gain^2 / fan_in
        // For ReLU gain (sqrt 2): = 2 / fan_in.
        let (mean, var) = mean_and_var(&all);
        let expected_var = gain * gain / fan_in as f32;
        let std = expected_var.sqrt();
        let bound = (3.0_f32).sqrt() * std;
        assert!(
            mean.abs() < 5e-4,
            "kaiming mean {mean} not near 0 (n={})",
            all.len(),
        );
        let rel = (var - expected_var).abs() / expected_var;
        assert!(
            rel < 0.02,
            "kaiming var {var} expected {expected_var} (rel diff {rel})",
        );
        for v in &all {
            assert!(
                v.is_finite() && v.abs() <= bound + 1e-5,
                "kaiming value {v} out of [-{bound}, {bound}]",
            );
        }
    }

    #[test]
    fn normal_initializer_mean_within_tolerance() {
        let mean_in = 0.7_f32;
        let std_in = 0.5_f32;
        let n = 1 << 16; // 65536
        let mut rng = StdRng::seed_from_u64(0xcafe_f00d);
        let samples = normal(mean_in, std_in, n, &mut rng).unwrap();
        assert_eq!(samples.len(), n);
        let (m, v) = mean_and_var(&samples);
        // SE of the mean for N(mean, std^2) with n=65536 is
        // std / sqrt(n) = 0.5 / 256 ≈ 0.00195. A 4-sigma tolerance is
        // ~0.008.
        assert!(
            (m - mean_in).abs() < 0.01,
            "normal mean {m} not near {mean_in}",
        );
        let expected_var = std_in * std_in;
        let rel = (v - expected_var).abs() / expected_var;
        assert!(
            rel < 0.04,
            "normal var {v} expected {expected_var} (rel diff {rel})",
        );
        for x in &samples {
            assert!(x.is_finite(), "normal produced non-finite sample");
        }
    }

    #[test]
    fn uniform_in_range() {
        let mut rng = StdRng::seed_from_u64(99);
        let lo = -1.5_f32;
        let hi = 2.5_f32;
        let samples = uniform(lo, hi, 4096, &mut rng).unwrap();
        for &x in &samples {
            assert!(x.is_finite());
            assert!(x >= lo && x < hi, "uniform sample {x} out of [{lo}, {hi})");
        }
    }
}
