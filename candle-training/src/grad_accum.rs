//! Gradient accumulation for simulating larger batch sizes.
//!
//! When GPU memory is limited, you can accumulate gradients over multiple
//! mini-batches before applying a single optimizer step, effectively
//! increasing the effective batch size.
//!
//! [`GradAccumulator`] manages this process: call [`accumulate`] after each
//! forward/backward pass, and it returns `true` every `accum_steps` calls to
//! signal that the optimizer should step.
//!
//! # Example
//!
//! ```rust
//! use candle::{Var, Device, DType, Tensor};
//! use candle_nn::{SGD, Optimizer};
//! use candle_training::grad_accum::GradAccumulator;
//!
//! let x = Var::new(&[1.0f32, 2.0, 3.0][..], &Device::Cpu)?;
//! let mut opt = SGD::new(vec![x.clone()], 0.01)?;
//! let mut accum = GradAccumulator::new(4); // accumulate over 4 mini-batches
//!
//! for i in 0..8 {
//!     let loss = x.as_tensor().sqr()?.sum_all()?;
//!     if accum.accumulate(&loss, &[&x])? {
//!         // Every 4th iteration: apply accumulated gradients
//!         opt.step(accum.gradients().unwrap())?;
//!         accum.zero_grad();
//!     }
//! }
//! # Ok::<(), candle::Error>(())
//! ```

use candle::backprop::GradStore;
use candle::{Result, Tensor, Var};

/// Accumulates gradients across multiple mini-batches.
///
/// Gradients are averaged (divided by `accum_steps`) so the effective
/// learning rate matches what you would get with a single large batch.
pub struct GradAccumulator {
    accum_steps: usize,
    current_count: usize,
    accumulated: Option<GradStore>,
}

impl GradAccumulator {
    /// Create a new accumulator that triggers every `accum_steps` calls.
    ///
    /// # Panics
    ///
    /// Panics if `accum_steps` is zero.
    pub fn new(accum_steps: usize) -> Self {
        assert!(accum_steps > 0, "accum_steps must be >= 1");
        Self {
            accum_steps,
            current_count: 0,
            accumulated: None,
        }
    }

    /// Compute gradients for `loss` and add them to the accumulator.
    ///
    /// Returns `true` when `accum_steps` mini-batches have been accumulated
    /// and the optimizer should step. The accumulated gradients are available
    /// via [`gradients()`](GradAccumulator::gradients).
    ///
    /// Gradients are scaled by `1 / accum_steps` at accumulation time so the
    /// optimizer does not need to adjust its learning rate.
    pub fn accumulate(&mut self, loss: &Tensor, vars: &[&Var]) -> Result<bool> {
        let grads = loss.backward()?;
        let scale = 1.0 / self.accum_steps as f64;

        match &mut self.accumulated {
            None => {
                let mut store = GradStore::new();
                for var in vars {
                    if let Some(g) = grads.get(var.as_tensor()) {
                        let scaled = (g * scale)?;
                        store.insert(var.as_tensor(), scaled);
                    }
                }
                self.accumulated = Some(store);
            }
            Some(store) => {
                for var in vars {
                    if let Some(g) = grads.get(var.as_tensor()) {
                        let scaled = (g * scale)?;
                        match store.remove(var.as_tensor()) {
                            Some(existing) => {
                                let summed = (existing + scaled)?;
                                store.insert(var.as_tensor(), summed);
                            }
                            None => {
                                store.insert(var.as_tensor(), scaled);
                            }
                        }
                    }
                }
            }
        }

        self.current_count += 1;
        Ok(self.current_count >= self.accum_steps)
    }

    /// Return a reference to the accumulated gradients, if any.
    pub fn gradients(&self) -> Option<&GradStore> {
        self.accumulated.as_ref()
    }

    /// Reset the accumulator for the next cycle.
    pub fn zero_grad(&mut self) {
        self.accumulated = None;
        self.current_count = 0;
    }

    /// Return the number of mini-batches accumulated so far in this cycle.
    pub fn current_count(&self) -> usize {
        self.current_count
    }

    /// Return the configured accumulation step count.
    pub fn accum_steps(&self) -> usize {
        self.accum_steps
    }
}
