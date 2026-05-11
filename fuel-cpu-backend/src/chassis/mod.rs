//! Trait-chassis surface for CPU kernels.
//!
//! Each submodule centralizes the shape / stride / loop logic for one
//! kernel family. Op-specific math lives in tiny trait impls; per-dtype
//! impls only carry what actually changes per dtype (accumulator type
//! for low-precision floats, etc.). The result is that adding a new
//! dtype or op to a family is a few trait-method lines instead of a
//! copy-paste of the entire kernel.
//!
//! See `docs/session-prompts/cpu-kernel-trait-chassis-refactor.md`
//! for the design rationale.

pub mod binary;
pub mod reduction;
pub mod unary;
