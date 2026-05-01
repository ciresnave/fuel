//! Thin re-export layer for Metal backend types.
//!
//! After step 8 of the backend-agnostic refactor, all Metal logic lives in
//! `fuel-metal`; this module just re-exports its surface so existing crate
//! paths (`fuel_core::metal_backend::*`) keep working.

pub use fuel_metal::*;
