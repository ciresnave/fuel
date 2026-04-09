//! Convenience wrapper for common training setup patterns.
//!
//! Using candle for training requires understanding four concepts before
//! writing any model code: [`candle::DType`], [`candle::Device`], [`VarMap`],
//! and [`VarBuilder`]. [`TrainingContext`] bundles all four so that newcomers
//! and quick scripts can skip the boilerplate.
//!
//! # When to use this
//!
//! Use [`TrainingContext`] when:
//!
//! - You are writing a training script and don't need the full flexibility of
//!   managing VarMap and VarBuilder independently.
//! - You want a single object to pass around that carries both the
//!   parameter store and the device/dtype context.
//!
//! Use [`VarMap`] + [`VarBuilder`] directly when:
//!
//! - You need fine-grained control over where parameters are stored.
//! - You are loading a pre-trained checkpoint (use
//!   [`VarBuilder::from_mmaped_safetensors`] directly).
//! - You are sharing a parameter store across multiple model components.
//!
//! # Example
//!
//! ```rust
//! use candle::{DType, Tensor};
//! use candle_nn::{TrainingContext, Module};
//!
//! # fn main() -> candle::Result<()> {
//! // All parameters created via `ctx.vb()` live in the same VarMap.
//! let ctx = TrainingContext::cpu_f32();
//!
//! let linear = candle_nn::linear(128, 64, ctx.vb())?;
//!
//! let x = Tensor::randn(0f32, 1., (4, 128), ctx.device())?;
//! let y = linear.forward(&x)?;     // (4, 64)
//!
//! // Hand all variables to an optimizer:
//! let _params = ctx.vars();    // Vec<candle::Var>
//! # Ok(())
//! # }
//! ```

use candle::{DType, Device, Var};

use crate::{VarBuilder, VarMap};

/// A convenience handle that bundles a [`VarMap`], a dtype, and a device.
///
/// All [`VarBuilder`]s issued by the same `TrainingContext` share the **same
/// underlying [`VarMap`]**, so every parameter created through `ctx.vb()` is
/// automatically visible in `ctx.vars()`.
///
/// # Cloning
///
/// `TrainingContext` is cheap to clone. The clone shares the same
/// `VarMap` (same `Arc`), so parameters created through either the original or
/// the clone are returned by `vars()` on both.
#[derive(Clone, Debug)]
pub struct TrainingContext {
    varmap: VarMap,
    dtype: DType,
    device: Device,
}

impl TrainingContext {
    /// Create a `TrainingContext` with the given dtype and device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use candle::{DType, Device};
    /// use candle_nn::TrainingContext;
    ///
    /// let ctx = TrainingContext::new(DType::F32, Device::Cpu);
    /// ```
    pub fn new(dtype: DType, device: Device) -> Self {
        Self {
            varmap: VarMap::new(),
            dtype,
            device,
        }
    }

    /// Shorthand for `TrainingContext::new(DType::F32, Device::Cpu)`.
    pub fn cpu_f32() -> Self {
        Self::new(DType::F32, Device::Cpu)
    }

    /// Shorthand for `TrainingContext::new(DType::BF16, Device::Cpu)`.
    pub fn cpu_bf16() -> Self {
        Self::new(DType::BF16, Device::Cpu)
    }

    // ── Issuing VarBuilders ──────────────────────────────────────────────

    /// Return a root [`VarBuilder`] backed by this context's `VarMap`.
    ///
    /// All parameters created through the returned builder (or any
    /// `push_prefix`/`pp` child of it) are stored in this context's `VarMap`.
    pub fn vb(&self) -> VarBuilder<'static> {
        VarBuilder::from_varmap(&self.varmap, self.dtype, &self.device)
    }

    /// Return a [`VarBuilder`] rooted at `prefix`.
    ///
    /// Equivalent to `ctx.vb().pp(prefix)`.
    pub fn vb_pp(&self, prefix: &str) -> VarBuilder<'static> {
        self.vb().pp(prefix)
    }

    // ── Inspecting the parameter store ──────────────────────────────────

    /// Return all trainable variables currently registered in this context.
    ///
    /// Pass the result directly to optimizer constructors:
    ///
    /// ```rust
    /// # use candle::{DType, Device};
    /// # use candle_nn::TrainingContext;
    /// # let ctx = TrainingContext::new(DType::F32, Device::Cpu);
    /// let _params = ctx.vars();       // Vec<candle::Var>
    /// ```
    pub fn vars(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }

    /// Return a reference to the underlying [`VarMap`].
    ///
    /// Useful for saving and loading checkpoints:
    ///
    /// ```rust,no_run
    /// # use candle::{DType, Device};
    /// # use candle_nn::TrainingContext;
    /// # let ctx = TrainingContext::new(DType::F32, Device::Cpu);
    /// ctx.varmap().save("checkpoint.safetensors")?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    // ── Device / dtype accessors ─────────────────────────────────────────

    /// Return the device this context is configured for.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Return the dtype this context is configured for.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
