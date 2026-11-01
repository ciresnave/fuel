// SPDX-License-Identifier: MIT OR Apache-2.0
//! `SamplingStrategy::Temperature { temp: 0.0 }` is documented as "effectively greedy". It used to
//! set the inverse temperature to 1.0 instead, i.e. sample UNSCALED from the softmax, so a caller
//! asking for greedy through the temperature knob got random tokens.

use fuel_tensor::lazy::{SamplingStrategy, sample_logits};

/// Near-uniform logits with the maximum at index 3: unscaled sampling picks index 3 only about a
/// quarter of the time, so 64 draws all landing on it is not luck.
const LOGITS: [f32; 5] = [0.1, 0.2, 0.0, 0.3, 0.15];

fn draws(strategy: SamplingStrategy, n: usize) -> Vec<u32> {
    let mut rng = 12345_u64;
    (0..n)
        .map(|_| sample_logits(&LOGITS, strategy, &mut rng))
        .collect()
}

#[test]
fn temperature_zero_is_greedy_on_every_draw() {
    let got = draws(
        SamplingStrategy::Temperature {
            temp: 0.0,
            seed: 12345,
        },
        64,
    );
    assert!(
        got.iter().all(|t| *t == 3),
        "temp 0.0 must be greedy: {got:?}"
    );
}

/// Negative control: the same harness sees a non-greedy sampler as non-greedy, so the test above
/// can fail. A positive temperature on these logits visits more than one token in 64 draws.
#[test]
fn a_positive_temperature_is_not_greedy_so_the_test_can_fail() {
    let got = draws(
        SamplingStrategy::Temperature {
            temp: 1.0,
            seed: 12345,
        },
        64,
    );
    assert!(got.iter().any(|t| *t != 3), "temp 1.0 should vary: {got:?}");
}

#[test]
fn a_negative_or_nan_temperature_is_greedy_not_an_inverted_distribution() {
    for temp in [-1.0_f32, f32::NAN] {
        let got = draws(SamplingStrategy::Temperature { temp, seed: 1 }, 32);
        assert!(got.iter().all(|t| *t == 3), "temp {temp}: {got:?}");
    }
}
