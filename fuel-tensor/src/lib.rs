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

// GAP-229: `clippy::identity_op` fires across fuel-core+fuel-dispatch (and now
// fuel-tensor, since the flagged sites in judge/mod.rs and lazy.rs moved here)
// and is a defect in 0 of them — it measures a house idiom, not debt, so it is
// allowed at the crate root. Carried forward verbatim from fuel-core/src/lib.rs
// (board #109 move) rather than re-derived, since the measurement and the two
// intentional classes it documents (DOC-INDEX, DOC-SHAPE) are about the CODE
// that moved, not about which crate currently compiles it. See fuel-core's own
// copy of this comment (unaffected by this move) for the full rationale.
#![allow(clippy::identity_op)]

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
// device.rs's `use crate::{DType, HostBuffer, Result, Shape, Storage,
// WithDType};` -- a GROUPED import, which an earlier `crate::X` text census
// missed (it doesn't contain the substring `crate::HostBuffer`, only
// `HostBuffer` inside `crate::{...}`'s braces). Caught by the first real
// `cargo check`, not the census.
pub use fuel_backend_contract::Storage;
pub use fuel_ir::HostBuffer;

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

// `bail!` consolidation (fuel-core dissolution, GAP-347 PR 4): this used to
// be a LOCAL duplicate of `fuel_core::bail!`, kept separate only because
// depending on `fuel_core::bail!` directly would have created a
// `fuel-tensor` <-> `fuel-core` cargo cycle (`fuel-core` needs
// `fuel-tensor` for `train.rs`). That reasoning never applied to
// `fuel_ir`: this crate already depends on it directly (`Error`/`Result`
// above are a bare re-export of `fuel_ir::error::{Error, Result}`, not a
// distinct type), and `fuel_ir` is tier 10 — below everything. `$crate`
// inside a `macro_rules!` body resolves to the crate that WROTE the
// macro, not the one that re-exports or invokes it, so re-exporting here
// keeps `fuel_tensor::bail!` producing byte-identical `fuel_ir::Error::Msg`
// values — proved by the pre-existing `bail_tests::bail_macro_returns_a_typed_err`
// below, which is unchanged and still exercises this re-export.
pub use fuel_ir::bail;

#[cfg(test)]
mod bail_tests {
    use crate::bail;
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
