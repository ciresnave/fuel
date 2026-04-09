//! Gradient clipping utilities.
//!
//! Gradient clipping prevents exploding gradients during training by capping
//! the gradient magnitude before the optimizer step.
//!
//! Two strategies are provided:
//!
//! - [`clip_grad_norm`] — Rescale all gradients so the **global** L2 norm does
//!   not exceed `max_norm`. This is the most widely used strategy (equivalent
//!   to `torch.nn.utils.clip_grad_norm_`).
//! - [`clip_grad_value`] — Clamp each element individually to `[-clip_value,
//!   clip_value]`.
//!
//! # Example
//!
//! ```rust
//! use candle::{Tensor, Var, Device, DType};
//! use candle_training::grad_clip::{clip_grad_norm, clip_grad_value};
//!
//! let x = Var::new(&[10.0f32, 20.0, 30.0][..], &Device::Cpu)?;
//! let loss = x.as_tensor().sqr()?.sum_all()?;
//! let mut grads = loss.backward()?;
//!
//! // Before clipping, gradient norm is large
//! let norm = clip_grad_norm(&[&x], &mut grads, 1.0)?;
//! assert!(norm > 1.0); // Returns the *original* norm
//!
//! # Ok::<(), candle::Error>(())
//! ```

use candle::{Result, Var};

/// Clip the **global L2 norm** of the gradients in `grads` for the given `vars`.
///
/// If the total gradient norm exceeds `max_norm`, all gradients are rescaled
/// proportionally so that the resulting norm equals `max_norm`. Gradients
/// below the threshold are left unchanged.
///
/// Returns the **original** (unclipped) total norm as `f64`.
///
/// # Arguments
///
/// - `vars` — The variables whose gradients should be clipped.
/// - `grads` — Mutable reference to the gradient store. Gradients are replaced
///   in-place.
/// - `max_norm` — The maximum allowed L2 norm across all gradients.
///
/// # Example
///
/// ```rust
/// use candle::{Var, Device};
/// use candle_training::grad_clip::clip_grad_norm;
///
/// let x = Var::new(&[3.0f32, 4.0][..], &Device::Cpu)?;
/// let loss = x.as_tensor().sqr()?.sum_all()?;
/// let mut grads = loss.backward()?;
/// let original_norm = clip_grad_norm(&[&x], &mut grads, 1.0)?;
/// // original grad was [6, 8], norm = 10
/// assert!((original_norm - 10.0).abs() < 1e-4);
/// // After clipping, the gradient tensor has norm ≤ 1.0
/// let g = grads.get(x.as_tensor()).unwrap();
/// let clipped_norm: f64 = g.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
/// assert!(clipped_norm <= 1.0 + 1e-6);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn clip_grad_norm(
    vars: &[&Var],
    grads: &mut candle::backprop::GradStore,
    max_norm: f64,
) -> Result<f64> {
    // Compute global L2 norm across all gradients.
    let mut total_norm_sq = 0f64;
    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let norm_sq: f64 = grad
                .to_dtype(candle::DType::F64)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f64>()?;
            total_norm_sq += norm_sq;
        }
    }
    let total_norm = total_norm_sq.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for var in vars {
            if let Some(grad) = grads.remove(var.as_tensor()) {
                let clipped = (grad * scale)?;
                grads.insert(var.as_tensor(), clipped);
            }
        }
    }

    Ok(total_norm)
}

/// Clamp every gradient element to `[-clip_value, clip_value]`.
///
/// This is a simpler (but less commonly used) alternative to norm-based
/// clipping. It modifies gradients element-wise rather than by global scale.
///
/// # Example
///
/// ```rust
/// use candle::{Var, Device};
/// use candle_training::grad_clip::clip_grad_value;
///
/// let x = Var::new(&[10.0f32, -20.0][..], &Device::Cpu)?;
/// let loss = x.as_tensor().sqr()?.sum_all()?;
/// let mut grads = loss.backward()?;
/// clip_grad_value(&[&x], &mut grads, 5.0)?;
/// let g = grads.get(x.as_tensor()).unwrap();
/// let vals = g.to_vec1::<f32>()?;
/// assert!(vals[0] <= 5.0);   // was 20.0, now clamped
/// assert!(vals[1] >= -5.0);  // was -40.0, now clamped
/// # Ok::<(), candle::Error>(())
/// ```
pub fn clip_grad_value(
    vars: &[&Var],
    grads: &mut candle::backprop::GradStore,
    clip_value: f64,
) -> Result<()> {
    for var in vars {
        if let Some(grad) = grads.remove(var.as_tensor()) {
            let clipped = grad.clamp(-clip_value, clip_value)?;
            grads.insert(var.as_tensor(), clipped);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Var};

    #[test]
    fn clip_norm_reduces_large_gradients() -> Result<()> {
        let x = Var::new(&[3.0f32, 4.0][..], &Device::Cpu)?;
        let loss = x.as_tensor().sqr()?.sum_all()?;
        let mut grads = loss.backward()?;
        // grad = [6, 8], norm = 10
        let orig_norm = clip_grad_norm(&[&x], &mut grads, 1.0)?;
        assert!((orig_norm - 10.0).abs() < 1e-4);

        let g = grads.get(x.as_tensor()).unwrap();
        let clipped_norm: f64 = g.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
        assert!(clipped_norm <= 1.0 + 1e-5);
        Ok(())
    }

    #[test]
    fn clip_norm_noop_for_small_gradients() -> Result<()> {
        let x = Var::new(&[0.1f32, 0.1][..], &Device::Cpu)?;
        let loss = x.as_tensor().sqr()?.sum_all()?;
        let mut grads = loss.backward()?;
        // grad = [0.2, 0.2], norm ≈ 0.283 — below max_norm=1.0
        let g_before = grads.get(x.as_tensor()).unwrap().to_vec1::<f32>()?;
        let _ = clip_grad_norm(&[&x], &mut grads, 1.0)?;
        let g_after = grads.get(x.as_tensor()).unwrap().to_vec1::<f32>()?;
        assert!((g_before[0] - g_after[0]).abs() < 1e-7);
        assert!((g_before[1] - g_after[1]).abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn clip_value_clamps() -> Result<()> {
        let x = Var::new(&[100.0f32, -100.0][..], &Device::Cpu)?;
        let loss = x.as_tensor().sqr()?.sum_all()?;
        let mut grads = loss.backward()?;
        // grad = [200, -200]
        clip_grad_value(&[&x], &mut grads, 5.0)?;
        let vals = grads.get(x.as_tensor()).unwrap().to_vec1::<f32>()?;
        assert!((vals[0] - 5.0).abs() < 1e-6);
        assert!((vals[1] - (-5.0)).abs() < 1e-6);
        Ok(())
    }
}
