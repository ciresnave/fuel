// SPDX-License-Identifier: MIT OR Apache-2.0
//! `fuel-tensor` — Fuel's user-facing `Tensor` handle and `Device`
//! abstraction, moved here from `fuel-core` (board #109; CireSnave:
//! "Move Tensor into fuel-tensor. We can worry about a semantic split
//! later."). This is a MECHANICAL move -- every item below kept its own
//! content; only the crate boundary changed. `fuel-core`'s own
//! `lib.rs` re-exports these same paths so existing
//! `fuel_core::lazy::Tensor` / `fuel::Tensor` call sites keep compiling.
//!
//! The moved cluster (`lazy`, `device`, the 3 GPU backend bridges,
//! `dtype`'s `WithDType`/`IntDType`/`FloatDType` trio, `pipelined_bridge`,
//! `judge`, `factories`, `planner`, `decode_shape`, `lazy_latent_cache`,
//! `test_utils`, `scheduling`, `inference_context`,
//! `kv_block_pool_device`, `persistent_decode`, `nf4`) moved together in
//! one PR because it is one mutually-referencing graph, not several
//! independent files that happen to touch `Tensor` -- measured via a
//! full crate::-reference census before any file moved, not assumed
//! from file names. See `docs/release-0.13.0-wave.md`.
//!
//! ```rust
//! use fuel_tensor::lazy::{Tensor, realize_many_f32};
//! use fuel_tensor::Device;
//! # use fuel_tensor::Error;
//! # fn main() -> Result<(), Error> {
//! let dev = Device::cpu();
//! let a = Tensor::from_f32((0..6).map(|x| x as f32).collect::<Vec<_>>(), (2, 3), &dev)?;
//! let b = Tensor::from_f32_on(a.graph(), (0..12).map(|x| x as f32).collect::<Vec<_>>(), (3, 4), &dev)?;
//! let c = a.matmul(&b)?;
//! assert_eq!(c.shape().dims(), &[2, 4]);
//! let out = realize_many_f32(&[&c]);
//! assert_eq!(out[0].len(), 8);
//! # Ok(())}
//! ```

pub mod cuda_backend;
pub mod decode_shape;
pub mod device;
pub mod dtype;
pub mod factories;
pub mod inference_context;
pub mod judge;
pub mod kv_block_pool_device;
pub mod lazy;
pub mod lazy_latent_cache;
pub mod metal_backend;
pub mod nf4;
pub mod persistent_decode;
pub mod pipelined_bridge;
pub mod planner;
pub mod scheduling;
pub mod test_utils;
#[cfg(feature = "vulkan")]
pub mod vulkan_backend;

pub use device::{Device, DeviceLocation, NdArray};
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use fuel_ir::error::{Context, Error, Result};
// `device.rs` uses `crate::Layout`/`crate::Shape` (root-qualified) internally;
// these are the same fuel_ir re-exports fuel-core's own root carries, needed
// here because fuel-tensor is now the crate device.rs's `crate::` resolves
// within.
pub use fuel_ir::layout::Layout;
pub use fuel_ir::shape::{D, Shape};

#[doc(hidden)]
#[cfg(feature = "cuda")]
pub use cuda_backend as cuda;

#[cfg(feature = "cuda")]
pub use cuda_backend::{CudaDevice, CudaStorage};

#[cfg(feature = "cuda")]
pub use fuel_cuda_backend::builder_arg;

#[cfg(feature = "cudnn")]
pub use cuda_backend::cudnn;

#[cfg(feature = "metal")]
pub use metal_backend::{MetalDevice, MetalError, MetalStorage};

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

/// Returns early from a function with a formatted error message.
///
/// A LOCAL copy of `fuel_core::bail!` (fuel-core/src/error.rs): the
/// macro's `$crate::Error` resolves to whichever crate it is invoked
/// from, and `fuel-tensor` needing `fuel_core::bail!` directly would
/// create a `fuel-tensor` <-> `fuel-core` cargo cycle (`fuel-core`
/// still needs `fuel-tensor` for `train.rs`'s references into this
/// crate). Duplicating this 12-line macro verbatim avoids the cycle
/// without inventing a new shared-macro crate for one move. Tracked as
/// a follow-up (single home, most likely alongside `Error`/`Result` in
/// `fuel_ir::error`) in `docs/release-0.13.0-wave.md` so the duplicate
/// can't drift silently.
#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()).bt())
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()).bt())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()).bt())
    };
}

#[cfg(test)]
mod bail_tests {
    use fuel_ir::error::Result;

    fn always_bails() -> Result<()> {
        bail!("deliberate failure for the macro test");
        #[allow(unreachable_code)]
        Ok(())
    }

    #[test]
    fn bail_macro_returns_a_typed_err() {
        let err = always_bails().expect_err("bail! must produce an Err, never panic");
        assert!(err.to_string().contains("deliberate failure"));
    }
}
