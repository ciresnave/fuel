//! Thin re-export layer for CUDA backend types.
//!
//! After step 8 of the backend-agnostic refactor, all CUDA logic lives in
//! `fuel-graph-cuda`; this module just re-exports its surface so existing
//! crate paths (`fuel_core::cuda_backend::*`) keep working.

pub use fuel_graph_cuda::*;
