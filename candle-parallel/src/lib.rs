//! # candle-parallel
//!
//! **Layer**: Use-Case Orchestration  |  **Stability**: experimental
//!
//! Multi-GPU parallelism primitives for the Candle ML framework. This crate
//! provides the building blocks for distributing model computation across
//! multiple devices.
//!
//! ## What is here
//!
//! - **Device topology** — [`topology`] models device interconnects and
//!   bandwidth for cost-aware placement decisions.
//! - **Tensor parallelism** — [`tensor_parallel`] provides column-parallel and
//!   row-parallel sharding strategies for linear layers with all-reduce
//!   communication.
//! - **Pipeline parallelism** — [`pipeline_parallel`] provides stage assignment
//!   and micro-batch scheduling (1F1B, GPipe) for models too large for a single
//!   device.
//! - **Distributed cache** — [`distributed_cache`] coordinates KV cache state
//!   across devices for paged and prefix caches.
//!
//! ## Design principles
//!
//! This crate is a **leaf crate** — nothing in the Candle ecosystem depends on
//! it. It provides policy, metadata, and strategies. Actual GPU communication
//! (NCCL AllReduce, etc.) is injected through a [`Communicator`](comm::Communicator)
//! trait that callers implement for their specific backend.
//!
//! ## What is NOT here
//!
//! - **NCCL bindings** — use `cudarc::nccl` directly and wrap with the
//!   `Communicator` trait.
//! - **Model definitions** — those stay in `candle-transformers`.
//! - **Inference orchestration** — that's `candle-inference`.
//! - **Training loops** — that's `candle-training`.
//!
//! ## Layer placement
//!
//! ```text
//! candle-parallel  ← you are here (multi-GPU orchestration)
//! candle-transformers (model definitions)
//! candle-nn          (layers, optimisers, VarBuilder)
//! candle-core        (tensors, devices, autograd)
//! ```

pub mod comm;
pub mod distributed_cache;
pub mod pipeline_parallel;
pub mod tensor_parallel;
pub mod topology;
