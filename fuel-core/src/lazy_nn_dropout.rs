//! Lazy port of `fuel-nn`'s [`Dropout`](crate::lazy_nn_dropout::Dropout)
//! module.
//!
//! During training each element of the input is independently
//! zeroed with probability `drop_p` and the survivors are scaled by
//! `1 / (1 - drop_p)` so the expected value of every activation is
//! preserved. During evaluation the module is the identity.
//!
//! Lazy-graph semantics
//! --------------------
//!
//! Unlike the eager [`fuel_nn::ops::dropout`] op (which calls
//! `Tensor::rand` at execution time on the storage backend), the
//! lazy bridge has no graph-level random-number primitive yet. The
//! v1 implementation therefore samples the Bernoulli mask
//! **host-side** at graph-build time from a caller-supplied seed,
//! wraps it as a `const_f32_like` const node, and multiplies. This
//! means:
//!
//!   * The mask is **baked into the graph** for the lifetime of the
//!     resulting `LazyTensor`. Re-executing the same lazy graph
//!     re-applies the same mask — there is no fresh sampling per
//!     `realize_*` call. Callers that want a fresh mask per step
//!     (the normal training loop case) must rebuild the dropout
//!     node each step with a fresh seed.
//!   * Determinism is the seed's responsibility. Pass the same
//!     seed twice and you get the same mask twice.
//!   * `drop_p == 0.0` short-circuits to the identity (matching the
//!     fuel-nn eager op's behavior in the no-op case).
//!
//! v1 scope
//! --------
//!
//!   * F32 inputs only. BF16/F16/F64 inputs are rejected at
//!     graph-build time; once a graph-level `Op::BernoulliMask`
//!     lands the dtype gate goes away.
//!   * Forward only. The mask is a const node, so autograd through
//!     it is the identity-with-scale and "just works" through the
//!     existing `Mul` backward — no per-element scatter is needed.
//!   * No external rng state: the seed is consumed once at
//!     construction time. Threading a fresh seed in per step is the
//!     caller's responsibility (see [`Dropout::forward_with_seed`]).
//!
//! Limitations to revisit when graph-level RNG lands
//! -------------------------------------------------
//!
//! The proper end state is `Op::BernoulliMask { p, seed }` (or a
//! `Tensor::rand` lazy primitive) so the mask is sampled on the
//! backend at execution time, the graph stays small, and re-execution
//! of a training-step graph naturally produces a fresh mask. This
//! file is the host-side stop-gap that unblocks ports of models with
//! dropout layers in the training path until that primitive ships.

use crate::lazy::LazyTensor;
use crate::Result;
use fuel_core_types::{DType, Shape};
use std::sync::Arc;

/// A dropout layer that randomly zeroes input elements during
/// training and is the identity at inference.
///
/// Mirrors the shape of [`fuel_nn::ops::Dropout`]. The drop
/// probability is captured at construction; `forward` takes the
/// `train` flag (and an explicit seed if you want deterministic
/// behavior across step boundaries).
///
/// ```rust,no_run
/// # use fuel_core::{Device, lazy::LazyTensor};
/// # use fuel_core::lazy_nn_dropout::Dropout;
/// # use fuel_core_types::Shape;
/// let device = Device::cpu();
/// let x = LazyTensor::from_f32(
///     vec![1.0_f32, 2.0, 3.0, 4.0],
///     Shape::from_dims(&[4]),
///     &device,
/// );
/// let drop = Dropout::new(0.5);
/// // Eval mode: identity.
/// let y_eval = drop.forward(&x, /* train = */ false).unwrap();
/// // Train mode: requires a seed.
/// let y_train = drop.forward_with_seed(&x, 0xC0FFEE).unwrap();
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Dropout {
    drop_p: f64,
}

impl Dropout {
    /// Build a dropout layer with the given drop probability. `drop_p`
    /// is the *fraction of elements zeroed* (PyTorch / fuel-nn
    /// convention), not the keep probability. Must be in `[0, 1)`.
    pub fn new(drop_p: f64) -> Self {
        Self { drop_p }
    }

    /// The drop probability the layer was constructed with.
    pub fn drop_p(&self) -> f64 {
        self.drop_p
    }

    /// Apply dropout when `train` is true, otherwise return `x`
    /// unchanged. Uses the address of `x`'s underlying graph node as
    /// the seed — adequate for a *single* forward pass but **not**
    /// suitable for a training loop that wants a fresh mask per
    /// step. Use [`Self::forward_with_seed`] for that.
    pub fn forward(&self, x: &LazyTensor, train: bool) -> Result<LazyTensor> {
        if !train {
            return Ok(x.clone());
        }
        // Address-of-graph-node fallback: stable within a single graph
        // build, varies between LazyTensor instances. Suitable for
        // ad-hoc one-shot use; the training-loop path should pass a
        // real seed through `forward_with_seed`.
        let seed = (x.graph_tensor() as *const _ as usize) as u64;
        self.forward_with_seed(x, seed)
    }

    /// Apply dropout unconditionally with the given seed. The mask
    /// is sampled host-side and baked into the graph as a const
    /// node, so two calls with the same `seed` and the same input
    /// shape produce the same mask. Useful for tests and for
    /// deterministic training loops that thread their own rng state.
    pub fn forward_with_seed(
        &self,
        x: &LazyTensor,
        seed: u64,
    ) -> Result<LazyTensor> {
        if !(0.0..1.0).contains(&self.drop_p) {
            return Err(crate::Error::Msg(format!(
                "dropout: drop_p must be in [0, 1), got {}",
                self.drop_p,
            ))
            .bt());
        }
        if self.drop_p == 0.0 {
            // No-op short-circuit. Matches fuel-nn eager when
            // drop_p is exactly zero.
            return Ok(x.clone());
        }
        if x.dtype() != DType::F32 {
            return Err(crate::Error::Msg(format!(
                "dropout: v1 only supports F32 inputs, got {:?} (graph-level \
                 BernoulliMask primitive lands later)",
                x.dtype(),
            ))
            .bt());
        }

        let shape = x.shape();
        let n = shape.elem_count();
        let mask_shape = Shape::from_dims(shape.dims());

        let mask = build_bernoulli_mask(n, self.drop_p, seed);
        let mask_t = x.const_f32_like(Arc::<[f32]>::from(mask), mask_shape);
        Ok(x.mul(&mask_t)?)
    }
}

/// Sample `n` Bernoulli draws with keep probability `1 - drop_p`,
/// scaled by `1 / (1 - drop_p)` so the expected value of each
/// element matches the unmasked input.
///
/// Uses a deterministic SplitMix64 stream derived from `seed`. This
/// is sufficient for masking — we are not generating cryptographic
/// or high-dimensional MCMC samples here — and avoids pulling a
/// fresh `rand` dep into fuel-core just for dropout.
fn build_bernoulli_mask(n: usize, drop_p: f64, seed: u64) -> Vec<f32> {
    let keep_p = 1.0 - drop_p;
    let scale = (1.0 / keep_p) as f32;
    let threshold = drop_p; // draw < threshold ⇒ drop; otherwise keep
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // SplitMix64 step.
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        // Map to [0, 1) by taking the top 53 bits and dividing by 2^53.
        let u = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        let m = if u < threshold { 0.0_f32 } else { scale };
        out.push(m);
    }
    out
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn forward_eval_is_identity() {
        let device = Device::cpu();
        let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let x = LazyTensor::from_f32(
            data.clone(),
            Shape::from_dims(&[6]),
            &device,
        );
        let drop = Dropout::new(0.5);
        let y = drop.forward(&x, /* train = */ false).unwrap();
        let out = y.realize_f32();
        assert_eq!(out, data, "eval-mode dropout must be the identity");
    }

    #[test]
    fn forward_train_zeros_or_scales_each_element() {
        // With drop_p = 0.5 the scale factor is 1 / (1 - 0.5) = 2.0,
        // so every output element is either 0.0 or 2.0 * input.
        let device = Device::cpu();
        let data: Vec<f32> = (1..=128).map(|i| i as f32).collect();
        let x = LazyTensor::from_f32(
            data.clone(),
            Shape::from_dims(&[128]),
            &device,
        );
        let drop = Dropout::new(0.5);
        let y = drop.forward_with_seed(&x, 0xDEADBEEF).unwrap();
        let out = y.realize_f32();
        assert_eq!(out.len(), data.len());
        let mut zeros = 0usize;
        let mut keeps = 0usize;
        for (i, (&o, &d)) in out.iter().zip(data.iter()).enumerate() {
            if o == 0.0 {
                zeros += 1;
            } else {
                assert!(
                    (o - 2.0 * d).abs() < 1e-6,
                    "elem {i}: expected 0.0 or {} got {o}",
                    2.0 * d,
                );
                keeps += 1;
            }
        }
        assert!(
            zeros > 0 && keeps > 0,
            "drop_p=0.5 over 128 elems should produce a mix \
             (zeros={zeros}, keeps={keeps})",
        );
    }

    #[test]
    fn forward_train_preserves_expected_value_in_aggregate() {
        // With a large sample, sum(output) / sum(input) should be
        // close to 1.0 because survivors are scaled by 1/(1 - p).
        let device = Device::cpu();
        let n = 4096;
        let data: Vec<f32> = vec![1.0; n];
        let x = LazyTensor::from_f32(
            data.clone(),
            Shape::from_dims(&[n]),
            &device,
        );
        let drop = Dropout::new(0.3);
        let y = drop.forward_with_seed(&x, 0x5EED_5EED).unwrap();
        let out = y.realize_f32();
        let in_sum: f32 = data.iter().copied().sum();
        let out_sum: f32 = out.iter().copied().sum();
        let ratio = out_sum / in_sum;
        // Coarse Monte-Carlo check; tolerance is generous on purpose
        // (4096 Bernoulli draws @ keep_p=0.7).
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "expected E[out]/E[in] ≈ 1.0, got ratio={ratio}",
        );
    }

    #[test]
    fn same_seed_gives_same_mask() {
        let device = Device::cpu();
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5).collect();
        let x = LazyTensor::from_f32(
            data.clone(),
            Shape::from_dims(&[64]),
            &device,
        );
        let drop = Dropout::new(0.4);
        let a = drop.forward_with_seed(&x, 12345).unwrap().realize_f32();
        let b = drop.forward_with_seed(&x, 12345).unwrap().realize_f32();
        assert_eq!(a, b, "identical seed must yield identical mask");
    }

    #[test]
    fn drop_p_zero_short_circuits() {
        let device = Device::cpu();
        let data: Vec<f32> = vec![7.0, -3.5, 0.25, 100.0];
        let x = LazyTensor::from_f32(
            data.clone(),
            Shape::from_dims(&[4]),
            &device,
        );
        let drop = Dropout::new(0.0);
        let y = drop.forward_with_seed(&x, 1).unwrap();
        assert_eq!(y.realize_f32(), data);
    }

    #[test]
    fn drop_p_out_of_range_errors() {
        let device = Device::cpu();
        let x = LazyTensor::from_f32(
            vec![1.0_f32, 2.0],
            Shape::from_dims(&[2]),
            &device,
        );
        assert!(Dropout::new(1.0).forward_with_seed(&x, 0).is_err());
        assert!(Dropout::new(-0.1).forward_with_seed(&x, 0).is_err());
    }
}
