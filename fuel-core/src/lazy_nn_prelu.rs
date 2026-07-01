//! Lazy port of `fuel-nn`'s `PReLU` activation.
//!
//! `PReLU(x) = max(0, x) + alpha * min(0, x)` — a leaky ReLU
//! where the negative-side slope `alpha` is itself a learned
//! parameter rather than a fixed hyperparameter.
//!
//! Two parameterizations, mirroring PyTorch's
//! `torch.nn.PReLU(num_parameters=...)`:
//!   * `num_parameters == 1` — a single scalar `alpha` shared
//!     across every element.
//!   * `num_parameters == C` — a per-channel `alpha` vector. The
//!     channel axis is dim 1 of the input (channels-first NCHW /
//!     NCL / NC layouts).
//!
//! The forward pass is built entirely from existing [`LazyTensor`]
//! primitives at graph-build time. Shape and channel-count checks
//! run when [`PReLU::forward`] is invoked — none of the work is
//! deferred to realize-time.
//!
//! v1 scope:
//!   * F32 weights only. Activations may be any dtype the
//!     downstream `broadcast_mul` accepts after a cast.
//!   * Forward only. Backward for the `alpha` parameter is handled
//!     by autograd through the primitive ops we emit (relu / min /
//!     broadcast_mul).
//!   * Per-channel form requires `rank(x) >= 2` and `x.dim(1) ==
//!     num_parameters`, matching the eager `fuel_nn::PReLU` check.

use crate::Result;
use crate::lazy::LazyTensor;
use fuel_ir::Shape;
use std::sync::Arc;

/// PReLU activation with a learned per-channel (or scalar) slope.
///
/// Stored weight layout matches PyTorch's checkpoint convention: a
/// flat length-`num_parameters` vector. The reshape that prepares
/// it for broadcasting happens inside [`Self::forward`].
#[derive(Debug, Clone)]
pub struct PReLU {
    /// Learned negative-side slope(s). Length must equal
    /// `num_parameters`.
    pub weight: Arc<[f32]>,
    /// `1` for the shared-scalar form, `C` for the per-channel
    /// form.
    pub num_parameters: usize,
}

impl PReLU {
    /// Construct a `PReLU` from an explicit weight buffer.
    ///
    /// Errors if `weight.len() != num_parameters` — caught here
    /// rather than at the first forward call.
    pub fn new(weight: Arc<[f32]>, num_parameters: usize) -> Result<Self> {
        if weight.len() != num_parameters {
            return Err(crate::Error::Msg(format!(
                "PReLU::new: weight length {} != num_parameters {}",
                weight.len(),
                num_parameters,
            ))
            .bt());
        }
        if num_parameters == 0 {
            return Err(crate::Error::Msg(
                "PReLU::new: num_parameters must be >= 1".to_string(),
            )
            .bt());
        }
        Ok(Self { weight, num_parameters })
    }

    /// Construct a shared-scalar PReLU. PyTorch's default
    /// initializer is `0.25`.
    pub fn scalar(alpha: f32) -> Self {
        let w: Arc<[f32]> = Arc::<[f32]>::from(vec![alpha]);
        Self { weight: w, num_parameters: 1 }
    }

    /// Construct a per-channel PReLU with `c` channels initialized
    /// to PyTorch's default `0.25`.
    pub fn per_channel_default(c: usize) -> Result<Self> {
        if c == 0 {
            return Err(crate::Error::Msg(
                "PReLU::per_channel_default: c must be >= 1".to_string(),
            )
            .bt());
        }
        let w: Arc<[f32]> = Arc::<[f32]>::from(vec![0.25_f32; c]);
        Ok(Self { weight: w, num_parameters: c })
    }

    /// Apply the activation. Returns a `LazyTensor` of the same
    /// shape and dtype as `x`.
    ///
    /// Per-channel form: `weight` is reshaped to `[1, C, 1, ...]`
    /// — `1` on every axis except the channel axis — so a single
    /// `broadcast_mul` against the negative half of `x` covers any
    /// rank `>= 2`.
    ///
    /// Scalar form: `weight` becomes a rank-0 tensor; the same
    /// `broadcast_mul` then fans it out across the full input.
    pub fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
        // Materialize the slope tensor on x's graph in the broadcast
        // shape we need.
        let weight = if self.num_parameters == 1 {
            // Rank-0 scalar; broadcasts against anything.
            let w = x.const_f32_like(
                Arc::clone(&self.weight),
                Shape::from_dims(&[]),
            );
            w
        } else {
            // Per-channel — require rank >= 2 and channel-axis match.
            let dims = x.shape();
            let dims = dims.dims();
            if dims.len() < 2 {
                return Err(crate::Error::Msg(format!(
                    "PReLU::forward: per-channel form requires rank >= 2, \
                     got shape {dims:?}",
                ))
                .bt());
            }
            let c = dims[1];
            if c != self.num_parameters {
                return Err(crate::Error::Msg(format!(
                    "PReLU::forward: channel-axis size {c} != \
                     num_parameters {} (input shape {dims:?})",
                    self.num_parameters,
                ))
                .bt());
            }
            // Shape `[1, C, 1, 1, ...]` with `1` on every non-channel
            // axis so broadcast_mul picks the right alpha per channel.
            let mut bshape: Vec<usize> = vec![1; dims.len()];
            bshape[1] = c;
            x.const_f32_like(
                Arc::clone(&self.weight),
                Shape::from_dims(&bshape),
            )
        };

        // pos = relu(x); neg = min(x, 0)
        let pos = x.relu();
        let zero = x.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32]),
            Shape::from_dims(&[]),
        );
        let neg = x.minimum(&broadcast_zero_like(&zero, x)?)?;

        // weighted_neg = alpha * neg (broadcast).
        let weighted_neg = neg.broadcast_mul(&weight)?;
        pos.add(&weighted_neg)
    }

    /// Load a PReLU weight from a memory-mapped safetensors file.
    ///
    /// Reads `"{prefix}.weight"` and interprets it as F32.
    /// `num_parameters` is `Some(c)` for the per-channel form or
    /// `None` for the shared-scalar form (matching the eager
    /// `prelu(num_channels, vb)` factory).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        num_parameters: Option<usize>,
    ) -> Result<Self> {
        use crate::lazy::load_tensor_as_f32;
        let n = num_parameters.unwrap_or(1);
        let w = load_tensor_as_f32(st, &format!("{prefix}.weight"))?;
        if w.len() != n {
            crate::bail!(
                "{prefix}.weight: {} elements, expected {n}",
                w.len(),
            );
        }
        let weight: Arc<[f32]> = Arc::from(w);
        Ok(PReLU { weight, num_parameters: n })
    }
}

/// Broadcast a rank-0 zero against `like`'s shape so that
/// `x.minimum(&zero)` lands in the strict-shape path. Promotes the
/// scalar via `broadcast_to`.
fn broadcast_zero_like(
    zero_scalar: &LazyTensor,
    like: &LazyTensor,
) -> Result<LazyTensor> {
    Ok(zero_scalar.broadcast_to(like.shape())?)
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn prelu_scalar_matches_textbook_formula() {
        // alpha = 0.1; for x in [-2, -1, 0, 1, 2] we expect
        //   y = [-0.2, -0.1, 0, 1, 2].
        let device = Device::cpu();
        let act = PReLU::scalar(0.1);
        let x = LazyTensor::from_f32(
            vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0],
            Shape::from_dims(&[5]),
            &device,
        );
        let y = act.forward(&x).unwrap().realize_f32();
        assert_eq!(y.len(), 5);
        let expected = [-0.2_f32, -0.1, 0.0, 1.0, 2.0];
        for (got, want) in y.iter().zip(expected.iter()) {
            assert!(
                approx_eq(*got, *want, 1e-6),
                "got {got} expected {want}",
            );
        }
    }

    #[test]
    fn prelu_per_channel_broadcasts_along_dim1() {
        // Shape [N=1, C=2, L=3]; per-channel alpha = [0.25, 0.5].
        // Input:
        //   ch0 = [-4, -2, 0]; ch1 = [-3, 1, -1]
        // Expected:
        //   ch0 = [-1.0, -0.5, 0]
        //   ch1 = [-1.5, 1.0, -0.5]
        let device = Device::cpu();
        let weights: Arc<[f32]> =
            Arc::<[f32]>::from(vec![0.25_f32, 0.5]);
        let act = PReLU::new(weights, 2).unwrap();
        let x = LazyTensor::from_f32(
            vec![-4.0_f32, -2.0, 0.0, -3.0, 1.0, -1.0],
            Shape::from_dims(&[1, 2, 3]),
            &device,
        );
        let y = act.forward(&x).unwrap().realize_f32();
        let expected = [
            -1.0_f32, -0.5, 0.0,
            -1.5, 1.0, -0.5,
        ];
        assert_eq!(y.len(), expected.len());
        for (got, want) in y.iter().zip(expected.iter()) {
            assert!(
                approx_eq(*got, *want, 1e-6),
                "got {got} expected {want}",
            );
        }
    }

    #[test]
    fn prelu_scalar_identity_on_nonnegative_inputs() {
        // PReLU(x) = x for x >= 0 regardless of alpha.
        let device = Device::cpu();
        let act = PReLU::scalar(0.7);
        let x = LazyTensor::from_f32(
            vec![0.0_f32, 0.5, 1.0, 2.5, 100.0],
            Shape::from_dims(&[5]),
            &device,
        );
        let y = act.forward(&x).unwrap().realize_f32();
        let expected = [0.0_f32, 0.5, 1.0, 2.5, 100.0];
        for (got, want) in y.iter().zip(expected.iter()) {
            assert!(
                approx_eq(*got, *want, 1e-6),
                "got {got} expected {want}",
            );
        }
    }

    #[test]
    fn prelu_new_rejects_length_mismatch() {
        let bad: Arc<[f32]> = Arc::<[f32]>::from(vec![0.25_f32, 0.5]);
        let err = PReLU::new(bad, 3).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("weight length") && msg.contains("num_parameters"),
            "unexpected error: {msg}",
        );
    }

    #[test]
    fn prelu_per_channel_rejects_rank1_input() {
        let act = PReLU::per_channel_default(4).unwrap();
        let device = Device::cpu();
        let x = LazyTensor::from_f32(
            vec![1.0_f32, 2.0, 3.0, 4.0],
            Shape::from_dims(&[4]),
            &device,
        );
        let err = act.forward(&x).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("rank >= 2"),
            "expected rank-error, got {msg}",
        );
    }
}
