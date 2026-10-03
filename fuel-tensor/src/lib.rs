// SPDX-License-Identifier: MIT OR Apache-2.0
//! `fuel-tensor` — Fuel's user-facing `Tensor` handle and `Device`
//! abstraction, the destination for board #109's fuel-core dissolution
//! (CireSnave: "Move Tensor into fuel-tensor. We can worry about a
//! semantic split later.").
//!
//! This crate is currently a scaffold: it exists in the workspace and
//! has a crate_dependency_tiers.txt entry (tier 100, "tensor /
//! user-facing handle"), but nothing has moved into it yet. See
//! `docs/release-0.13.0-wave.md` for the move's own tracking entry and
//! the PR sequence it's landing under.
