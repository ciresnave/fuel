//! In-place custom operations.
//!
//! Traits in this module describe operations that mutate a tensor's storage
//! directly rather than producing a new one. Because they don't produce a
//! new value, they cannot participate in autograd; use the
//! `CustomOp1`/`CustomOp2`/`CustomOp3` family in `fuel-core` if you need
//! a backward pass.
//!
//! These traits live here (and not in `fuel-core`) so that backend crates
//! such as `fuel-cuda-backend` and `fuel-metal-backend` can implement them on
//! backend-specific bridge types (e.g. `CudaUgIOp1`) without depending on
//! `fuel-core` — that would form a cycle, since `fuel-core` already depends
//! on the backend crates.

use crate::dyn_backend::DynBackendStorage;
use crate::{Layout, Result};

/// A custom in-place unary operation that modifies tensor storage directly.
///
/// Apply with `Tensor::inplace_op1` (in `fuel-core`).
///
/// # Example
///
/// ```no_run
/// use fuel_core_types::{Layout, Result};
/// use fuel_core_types::dyn_backend::DynBackendStorage;
/// use fuel_core_types::inplace_op::InplaceOp1;
/// struct ZeroOut;
/// impl InplaceOp1 for ZeroOut {
///     fn name(&self) -> &'static str { "zero_out" }
///     fn fwd(&self, s: &mut dyn DynBackendStorage, l: &Layout) -> Result<()> { todo!() }
/// }
/// ```
pub trait InplaceOp1 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass. The storage is a mutable trait object; use
    /// `storage.as_any_mut().downcast_mut()` to access backend-specific types.
    fn fwd(&self, storage: &mut dyn DynBackendStorage, layout: &Layout) -> Result<()>;
}

/// A custom in-place binary operation that modifies the first tensor using data from a second.
///
/// The first tensor's storage is mutated; the second tensor is read-only.
///
/// Apply with `Tensor::inplace_op2` (in `fuel-core`).
pub trait InplaceOp2 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass. The first storage is mutable; the second is read-only.
    fn fwd(
        &self,
        s1: &mut dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
    ) -> Result<()>;
}

/// A custom in-place ternary operation that modifies the first tensor using data from two others.
///
/// The first tensor's storage is mutated; the other two are read-only.
///
/// Apply with `Tensor::inplace_op3` (in `fuel-core`).
pub trait InplaceOp3 {
    /// Returns a human-readable name for this operation, used in error messages.
    fn name(&self) -> &'static str;

    /// The forward pass. The first storage is mutable; the second and third are read-only.
    fn fwd(
        &self,
        s1: &mut dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
        s3: &dyn DynBackendStorage,
        l3: &Layout,
    ) -> Result<()>;
}

// Blanket impls for boxed trait objects, so callers can hold one of several
// concrete impls in a `Box<dyn InplaceOpN>` and still pass it to the
// generic `Tensor::inplace_opN` API.

impl InplaceOp1 for Box<dyn InplaceOp1 + '_> {
    fn name(&self) -> &'static str {
        (**self).name()
    }
    fn fwd(&self, s: &mut dyn DynBackendStorage, l: &Layout) -> Result<()> {
        (**self).fwd(s, l)
    }
}

impl InplaceOp2 for Box<dyn InplaceOp2 + '_> {
    fn name(&self) -> &'static str {
        (**self).name()
    }
    fn fwd(
        &self,
        s1: &mut dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
    ) -> Result<()> {
        (**self).fwd(s1, l1, s2, l2)
    }
}

impl InplaceOp3 for Box<dyn InplaceOp3 + '_> {
    fn name(&self) -> &'static str {
        (**self).name()
    }
    fn fwd(
        &self,
        s1: &mut dyn DynBackendStorage,
        l1: &Layout,
        s2: &dyn DynBackendStorage,
        l2: &Layout,
        s3: &dyn DynBackendStorage,
        l3: &Layout,
    ) -> Result<()> {
        (**self).fwd(s1, l1, s2, l2, s3, l3)
    }
}
