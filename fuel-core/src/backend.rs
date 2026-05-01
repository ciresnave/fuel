//! Backend trait re-exports (legacy module path).
//!
//! After step 8 of the backend-agnostic refactor (2026-04-30), the static
//! `BackendStorage` and `BackendDevice` traits were deleted: every backend
//! now implements `DynBackendStorage` / `DynBackendDevice` directly. The
//! only trait still surfaced through this module is [`HostStorage`], the
//! capability marker for storage living in host-addressable memory.
//!
//! Prefer importing directly from `fuel_core::dyn_backend` or
//! `fuel_core_types`. This module remains as a re-export so existing call
//! sites such as `use fuel_core::backend::HostStorage` keep compiling.

pub use fuel_core_types::backend::HostStorage;
