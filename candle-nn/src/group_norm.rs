//! Group Normalization.
//!
//! This layer applies [Group Normalization](https://arxiv.org/abs/1803.08494) over a
//! mini-batch of inputs. The input tensor is expected to have shape `(N, C, ...)` where
//! `C` is divisible by `num_groups`. Each group of `C / num_groups` channels is normalized
//! independently using the mean and variance computed over that group and the spatial
//! dimensions.
//!
//! Learnable per-channel `weight` (scale) and `bias` (shift) parameters are always applied
//! after normalization.
//!
//! Use [`group_norm`] to construct a `GroupNorm` from a [`VarBuilder`](crate::VarBuilder),
//! or use [`GroupNorm::new`] to construct one directly from tensors.
use candle::{Context, DType, Result, Tensor};

/// Group Normalization layer.
///
/// Divides the channels into `num_groups` groups and normalizes each group independently.
/// Implements [`Module`](crate::Module).
///
/// # Example
///
/// ```rust
/// use candle::{Tensor, Device, Module};
/// use candle_nn::GroupNorm;
///
/// let num_groups = 2;
/// let num_channels = 4;
/// let w = Tensor::ones(num_channels, candle::DType::F32, &Device::Cpu)?;
/// let b = Tensor::zeros(num_channels, candle::DType::F32, &Device::Cpu)?;
/// let gn = GroupNorm::new(w, b, num_channels, num_groups, 1e-5)?;
/// let x = Tensor::zeros((1, num_channels, 8), candle::DType::F32, &Device::Cpu)?;
/// let y = gn.forward(&x)?;
/// assert_eq!(y.dims(), &[1, num_channels, 8]);
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    num_channels: usize,
    num_groups: usize,
}

impl GroupNorm {
    /// Creates a new `GroupNorm` layer.
    ///
    /// `weight` and `bias` must have shape `[num_channels]`, and `num_channels` must be
    /// divisible by `num_groups`.
    pub fn new(
        weight: Tensor,
        bias: Tensor,
        num_channels: usize,
        num_groups: usize,
        eps: f64,
    ) -> Result<Self> {
        if !num_channels.is_multiple_of(num_groups) {
            candle::bail!(
                "GroupNorm: num_groups ({num_groups}) must divide num_channels ({num_channels})"
            )
        }
        Ok(Self {
            weight,
            bias,
            eps,
            num_channels,
            num_groups,
        })
    }
}

impl crate::Module for GroupNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_shape = x.dims();
        if x_shape.len() <= 2 {
            candle::bail!("input rank for GroupNorm should be at least 3");
        }
        let (b_sz, n_channels) = (x_shape[0], x_shape[1]);
        let hidden_size = x_shape[2..].iter().product::<usize>() * n_channels / self.num_groups;
        if n_channels != self.num_channels {
            candle::bail!(
                "unexpected num-channels in GroupNorm ({n_channels} <> {}",
                self.num_channels
            )
        }
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let x = x.reshape((b_sz, self.num_groups, hidden_size))?;
        let x = x.to_dtype(internal_dtype)?;

        // Compute mean and variance in a single set of reductions over x, avoiding
        // the intermediate full-size centered tensor.  var = E[x^2] - E[x]^2
        let mean_x = (x.sum_keepdim(2)? / hidden_size as f64)?;
        let var = ((x.sqr()?.sum_keepdim(2)? / hidden_size as f64)? - mean_x.sqr()?)?;

        // Pre-compute a fused scale and offset that combine normalization with the
        // per-channel weight and bias, so we only need 2 passes over the full tensor
        // instead of 4 (subtract mean, divide by std, multiply weight, add bias).
        //   y = weight / std * x + (bias - mean * weight / std)
        //     = scale * x + offset
        let std = (var + self.eps)?.sqrt()?;
        let channels_per_group = self.num_channels / self.num_groups;
        // Reshape weight/bias to (1, num_groups, channels_per_group) so they
        // broadcast against the per-group std of shape (b_sz, num_groups, 1).
        let w = self
            .weight
            .reshape((1, self.num_groups, channels_per_group))?
            .to_dtype(internal_dtype)?;
        let b = self
            .bias
            .reshape((1, self.num_groups, channels_per_group))?
            .to_dtype(internal_dtype)?;
        let scale = w.broadcast_div(&std)?;
        let offset = b.broadcast_sub(&mean_x.broadcast_mul(&scale)?)?;

        // Reshape x so channels_per_group is its own axis, then fuse the
        // normalisation + affine into two passes (mul + add).
        let spatial_size = hidden_size / channels_per_group;
        let x = x.reshape((b_sz, self.num_groups, channels_per_group, spatial_size))?;
        let scale = scale.unsqueeze(3)?;
        let offset = offset.unsqueeze(3)?;

        x.broadcast_mul(&scale)?
            .broadcast_add(&offset)?
            .to_dtype(x_dtype)?
            .reshape(x_shape)
            .with_context(|| {
                format!(
                    "GroupNorm(groups={}, channels={}): input shape {x_shape:?}",
                    self.num_groups, self.num_channels
                )
            })
    }
}

/// Creates a [`GroupNorm`] layer by loading `weight` and `bias` from a
/// [`VarBuilder`](crate::VarBuilder).
///
/// # Example
///
/// ```no_run
/// use candle_nn::group_norm;
///
/// // let gn = group_norm(4, 16, 1e-5, vb.pp("gn"))?;
/// ```
pub fn group_norm(
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    vb: crate::VarBuilder,
) -> Result<GroupNorm> {
    let weight = vb.get_with_hints(num_channels, "weight", crate::Init::Const(1.))?;
    let bias = vb.get_with_hints(num_channels, "bias", crate::Init::Const(0.))?;
    GroupNorm::new(weight, bias, num_channels, num_groups, eps)
}
