// SPDX-License-Identifier: MIT OR Apache-2.0
//! # fuel-parallel
//!
//! **Layer**: Use-Case Orchestration  |  **Stability**: experimental
//!
//! Multi-GPU parallelism primitives for the Fuel ML framework. This crate
//! provides the building blocks for distributing model computation across
//! multiple devices.
//!
//! ## What is here
//!
//! - **Device topology** — [`topology`] models device interconnects and
//!   bandwidth for cost-aware placement decisions.
//! - **Device groups** — [`device_group`] provides real single-process,
//!   multi-device collectives over the lazy graph, staging through the host
//!   when a hop crosses vendors.
//! - **Tensor parallelism** — [`tensor_parallel`] provides column-parallel and
//!   row-parallel sharding strategies for linear layers with all-reduce
//!   communication.
//! - **Pipeline parallelism** — [`pipeline_parallel`] provides stage assignment
//!   and micro-batch scheduling (1F1B, GPipe) for models too large for a single
//!   device.
//! - **Distributed cache** — [`distributed_cache`] coordinates KV cache state
//!   across devices for paged and prefix caches.
//!
//! ## Lazy-only
//!
//! Every tensor-touching surface here takes [`Tensor`](fuel::lazy::Tensor).
//! Fuel retired eager entirely in B6, and a collective written against the old
//! eager `Tensor` could not have reduced lazy shards — the most likely reason
//! this crate sat unwired. Only [`comm`] and [`tensor_parallel`] ever touched
//! tensors; [`topology`], [`pipeline_parallel`] and [`distributed_cache`] are
//! pure policy and metadata, so they were already dtype- and tensor-free.
//!
//! ## Design principles
//!
//! This crate is a **leaf crate** — nothing in the Fuel ecosystem depends on
//! it. It provides policy, metadata, and strategies.
//!
//! Two collective surfaces live here and serve different deployments:
//! [`device_group::DeviceGroup`] is the one-process/N-devices form and works
//! today; [`comm::Communicator`] is the SPMD seam an out-of-process transport
//! (NCCL, Gloo, MPI) plugs into, which Fuel does not implement because rank
//! assignment and rendezvous are consumer policy.
//!
//! ## What is NOT here
//!
//! - **An out-of-process collective transport** — see [`comm::Communicator`],
//!   the seam one would plug into.
//! - **Model definitions** — those stay in `fuel-transformers`.
//! - **Inference orchestration** — that's `fuel-inference`.
//! - **Training loops** — that's `fuel-training`.
//!
//! ## Layer placement
//!
//! ```text
//! fuel-parallel      ← you are here (multi-GPU orchestration)
//! fuel-transformers    (model definitions)
//! fuel-core            (Tensor, Device)
//! fuel-graph           (the DAG + optimizer)
//! fuel-ir              (Shape, DType, Op, errors)
//! ```
//!
//! *(This block previously listed `fuel-nn (layers, optimisers, VarBuilder)`
//! between `fuel-transformers` and `fuel-core`, and the "NOT here" list pointed
//! at `baracuda_nccl`. **No `fuel-nn` crate exists** in the workspace — the
//! `Linear` in [`tensor_parallel`] says outright that it was inlined so this
//! crate need not depend on it, so the diagram contradicted the code beside it.
//! `fuel-core-types` is likewise gone: it is now `fuel-ir`.)*

pub mod comm;
pub mod device_group;
pub mod distributed_cache;
pub mod pipeline_parallel;
pub mod tensor_parallel;
pub mod topology;
