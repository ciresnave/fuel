// SPDX-License-Identifier: MIT OR Apache-2.0
//! Communication abstraction for collective operations — **lazy-only**.
//!
//! Defines the [`Communicator`] trait that backends (NCCL, Gloo, etc.) implement
//! to provide collective primitives. Code in [`tensor_parallel`](crate::tensor_parallel)
//! and [`pipeline_parallel`](crate::pipeline_parallel) is generic over this trait.
//!
//! # Which shape do I want?
//!
//! This crate carries **two** collective surfaces, and they are not competitors:
//!
//! | | [`Communicator`] | [`DeviceGroup`](crate::device_group::DeviceGroup) |
//! |---|---|---|
//! | Model | SPMD — one process per rank | one process, N devices |
//! | You pass | *your* shard | *every* shard |
//! | Transport | external (NCCL/Gloo/MPI) | in-graph `Op::Copy` |
//! | Works today | only at `world_size == 1` | yes |
//!
//! A single process driving four GPUs holds all four shards at once and has no
//! peer to rendezvous with, so the per-rank signature does not fit it — use
//! `DeviceGroup`. `Communicator` is the seam an out-of-process transport plugs
//! into, and Fuel deliberately does not implement one: *which* process is rank 3,
//! and how ranks meet, is consumer policy (see `docs/architecture/15-consumer-contract.md`).
//!
//! # Lazy-only
//!
//! These signatures take [`Tensor`]. The eager `fuel::Tensor` surface is
//! being retired, and a collective written against it could not reduce lazy
//! shards — which is the likeliest reason this crate never acquired a consumer.
//!
//! # Example
//!
//! ```rust
//! use fuel_parallel::comm::{Communicator, ReduceOp, CommInfo};
//!
//! // A mock communicator for testing (passes data through unchanged).
//! let info = CommInfo { rank: 0, world_size: 2 };
//! assert_eq!(info.is_root(), true);
//! assert_eq!(info.world_size, 2);
//! ```

use fuel::Result;
use fuel::lazy::Tensor;

/// Reduce operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Element-wise sum.
    Sum,
    /// Element-wise product.
    Product,
    /// Element-wise minimum.
    Min,
    /// Element-wise maximum.
    Max,
}

/// Basic communicator metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommInfo {
    /// This process's rank (0-indexed).
    pub rank: usize,
    /// Total number of participating processes.
    pub world_size: usize,
}

impl CommInfo {
    /// Whether this is rank 0.
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }
}

/// Abstraction over collective communication backends.
///
/// Implementations wrap NCCL, Gloo, MPI, or mock backends. All operations
/// are synchronous — they block until the collective is complete.
///
/// The trait is object-safe so it can be stored as `Box<dyn Communicator>`.
///
/// **Every shard handed to an implementation must belong to the same graph as
/// the value it is combined with.** `Tensor` ops are graph-affine: two
/// tensors can be combined iff their `graph_id`s match. An out-of-process
/// implementation that receives bytes over the wire must therefore materialize
/// them onto the *caller's* graph (`const_f32_like`, `from_f32_on`), not mint a
/// fresh one.
pub trait Communicator: Send {
    /// Communicator info (rank, world size).
    fn info(&self) -> CommInfo;

    /// All-reduce: compute element-wise reduction across all ranks,
    /// distributing the result to every rank.
    fn all_reduce(&self, tensor: &Tensor, op: ReduceOp) -> Result<Tensor>;

    /// All-gather: concatenate tensors from all ranks along `dim`.
    fn all_gather(&self, tensor: &Tensor, dim: usize) -> Result<Tensor>;

    /// Reduce-scatter: reduce across ranks then scatter equal chunks to each rank.
    fn reduce_scatter(&self, tensor: &Tensor, op: ReduceOp, dim: usize) -> Result<Tensor>;

    /// Broadcast tensor from `root` to all ranks.
    fn broadcast(&self, tensor: &Tensor, root: usize) -> Result<Tensor>;

    /// Barrier: block until all ranks reach this point.
    fn barrier(&self) -> Result<()>;
}

/// A single-process "communicator" that passes tensors through unchanged.
///
/// Useful for testing parallel code on a single device.
///
/// **This reduces nothing, and that is correct.** With `world_size == 1` there
/// is no peer to combine with, so the identity *is* the reduction. It is not a
/// working multi-device collective and must not be read as one — for that, use
/// [`DeviceGroup`](crate::device_group::DeviceGroup), which holds every shard
/// and folds them in-graph.
pub struct IdentityComm;

impl Communicator for IdentityComm {
    fn info(&self) -> CommInfo {
        CommInfo {
            rank: 0,
            world_size: 1,
        }
    }

    fn all_reduce(&self, tensor: &Tensor, _op: ReduceOp) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn all_gather(&self, tensor: &Tensor, _dim: usize) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn reduce_scatter(&self, tensor: &Tensor, _op: ReduceOp, _dim: usize) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn broadcast(&self, tensor: &Tensor, _root: usize) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn barrier(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel::{Device, Shape};

    /// A 1-D f32 lazy tensor on CPU, on a fresh graph.
    fn t(data: &[f32]) -> Tensor {
        Tensor::from_f32(
            data.to_vec(),
            Shape::from_dims(&[data.len()]),
            &Device::cpu(),
        )
    }

    #[test]
    fn identity_comm_info() {
        let comm = IdentityComm;
        let info = comm.info();
        assert_eq!(info.rank, 0);
        assert_eq!(info.world_size, 1);
        assert!(info.is_root());
    }

    #[test]
    fn identity_all_reduce() {
        let comm = IdentityComm;
        let result = comm
            .all_reduce(&t(&[1.0, 2.0, 3.0]), ReduceOp::Sum)
            .unwrap();
        assert_eq!(result.realize_f32(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn identity_all_gather() {
        let comm = IdentityComm;
        let result = comm.all_gather(&t(&[1.0, 2.0]), 0).unwrap();
        assert_eq!(result.realize_f32(), vec![1.0, 2.0]);
    }

    // `Communicator` is the SPMD shape: each rank calls with its own tensor.
    // A single process driving N devices holds ALL shards at once, so it wants
    // a slice-shaped interface instead — see [`crate::device_group`].
    // `IdentityComm` stays the correct degenerate impl here.

    #[test]
    fn identity_comm_reduces_nothing_and_that_is_correct() {
        // Characterization guard. IdentityComm is world_size 1 returning its
        // input — right for a single rank, and NOT a working collective. This
        // test exists so the distinction stays visible if someone reads
        // `all_reduce` and assumes it reduces.
        let comm = IdentityComm;
        let out = comm.all_reduce(&t(&[1.0, 2.0]), ReduceOp::Sum).unwrap();
        assert_eq!(comm.info().world_size, 1);
        assert_eq!(out.realize_f32(), vec![1.0, 2.0]);
    }

    #[test]
    fn identity_preserves_graph_affinity() {
        // The ported signature must not silently re-root a shard onto a new
        // graph: a returned tensor that no longer shares the caller's graph
        // could not be combined with anything, and the failure would surface
        // far from here as an operand-graph assert.
        let comm = IdentityComm;
        let input = t(&[1.0, 2.0]);
        let out = comm.all_reduce(&input, ReduceOp::Sum).unwrap();
        assert_eq!(out.graph_id(), input.graph_id());
        // ...and the proof that matters: it still composes.
        assert_eq!(out.add(&input).unwrap().realize_f32(), vec![2.0, 4.0]);
    }

    #[test]
    fn identity_broadcast() {
        let comm = IdentityComm;
        let result = comm.broadcast(&t(&[5.0]), 0).unwrap();
        assert_eq!(result.realize_f32(), vec![5.0]);
    }

    #[test]
    fn identity_reduce_scatter() {
        let comm = IdentityComm;
        let result = comm
            .reduce_scatter(&t(&[1.0, 2.0, 3.0]), ReduceOp::Sum, 0)
            .unwrap();
        assert_eq!(result.realize_f32(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn identity_barrier() {
        let comm = IdentityComm;
        assert!(comm.barrier().is_ok());
    }

    #[test]
    fn comm_info_non_root() {
        let info = CommInfo {
            rank: 3,
            world_size: 4,
        };
        assert!(!info.is_root());
    }

    #[test]
    fn reduce_op_variants() {
        // Ensure all variants exist and are distinct
        let ops = [
            ReduceOp::Sum,
            ReduceOp::Product,
            ReduceOp::Min,
            ReduceOp::Max,
        ];
        for (i, a) in ops.iter().enumerate() {
            for (j, b) in ops.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }
}
