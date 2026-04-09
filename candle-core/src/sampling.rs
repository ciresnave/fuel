use crate::{DType, Result, Tensor};

/// Sample according to the Gumbel-Softmax distribution.
///
/// With `temperature > 0`, adds Gumbel noise to `logits` then returns
/// `argmax((logits + noise) / temperature)` — a differentiable approximation to sampling.
/// With `temperature == 0`, returns a plain `argmax`.
///
/// # Example
///
/// ```rust
/// use candle_core::{Tensor, Device};
/// // With temperature=0, acts as argmax (deterministic).
/// let logits = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
/// let idx = candle_core::sampling::gumbel_softmax(&logits, 0.0, 0)?;;
/// assert_eq!(idx.to_scalar::<u32>()?, 2);
/// # Ok::<(), candle_core::Error>(())
/// ```
pub fn gumbel_softmax<D: crate::shape::Dim>(
    logits: &Tensor,
    temperature: f64,
    dim: D,
) -> Result<Tensor> {
    if temperature <= 0.0 {
        logits.argmax(dim)
    } else {
        // Cast to f32, doing the Gumbel softmax in bf16 is a bit unstable.
        let logits = logits.to_dtype(DType::F32)?;
        let minus_g = logits.rand_like(1e-7, 0.999)?.log()?.neg()?.log()?;
        if temperature == 1.0 {
            let sampled = (logits - minus_g)?.argmax(dim)?;
            Ok(sampled)
        } else {
            let sampled = (logits + minus_g * (-temperature))?.argmax(dim)?;
            Ok(sampled)
        }
    }
}
