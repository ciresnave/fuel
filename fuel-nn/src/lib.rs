// SPDX-License-Identifier: MIT OR Apache-2.0
//! Neural-network building blocks for Fuel's lazy graph.
//!
//! Extracted from `fuel-core` (2026-08-19). This crate sits **above**
//! `fuel-core`: the whole surface is built on `Tensor`, which lives in
//! `fuel_core::lazy`, so `fuel-nn` is a *consumer* of `fuel-core` and
//! `fuel-core` deliberately does **not** re-export it — that would be a
//! dependency cycle.
//!
//! # Why this could be extracted now
//!
//! The `fuel-core` dissolution was recorded as "downstream of the eager-
//! dispatch retirement (B6)". **B6 is complete** — verify with
//! `git grep "pub struct Tensor" -- fuel-core/src/` (expect 0).
//!
//! ⚠️ **The control here was CORRUPTED by my own `Lazy`-prefix rename and is
//! repaired.** It used to read *"control: `pub struct LazyTensor` returns 1"* —
//! the sweep rewrote the control string to match the claim string, leaving one
//! sentence asserting the same query returns both 0 and 1. A control must anchor
//! on something a rename cannot reach, so it is now STRUCTURAL:
//! `git ls-files 'fuel-core/src/*.rs' | wc -l` → **190** (proves the path is
//! live; if it returns 0 the claim above is vacuous rather than true).
//!
//! And this surface never depended on the EAGER tensor type in the first place:
//! measured across all 22 files, every tensor mention is the lazy
//! `fuel_core::lazy::Tensor` or a doc comment, with one exception
//! (`optim.rs`, `fuel_graph::NodeHandle::from_existing`) which is the graph
//! handle from a crate already *below* `fuel-core`.
//!
//! So this crate inherited a blocker it never had, from a program that has
//! since finished.

pub mod conv_transpose;
pub mod dropout;
pub mod gru;
pub mod loss;
pub mod modules;
pub mod one_hot;
pub mod optim;
pub mod prelu;
pub mod varbuilder;
pub mod varmap;
