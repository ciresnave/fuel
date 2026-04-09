//! Tensor parallelism: column-parallel and row-parallel linear layers.
//!
//! Tensor parallelism (TP) splits weight matrices across devices so each device
//! computes a partial result, then collective communication merges the outputs.
//!
//! ## Sharding strategies
//!
//! | Strategy | Weight split | Communication |
//! |----------|-------------|---------------|
//! | [`ColumnParallel`] | Split output dim | None (outputs are disjoint shards) |
//! | [`RowParallel`] | Split input dim | AllReduce after matmul |
//!
//! In a standard MLP (`Y = XA`):
//! - **Column-parallel**: `A` is split column-wise → each rank gets `A_i` (columns
//!   `[i*cols/N..(i+1)*cols/N]`) → `Y_i = X @ A_i` → results are concatenated.
//! - **Row-parallel**: `A` is split row-wise → each rank gets `A_i` (rows
//!   `[i*rows/N..(i+1)*rows/N]`) → `Y_i = X_i @ A_i` → results are *summed*
//!   via all-reduce.
//!
//! ## Usage pattern
//!
//! ```rust
//! use candle_parallel::tensor_parallel::{TensorParallelConfig, ShardDim};
//!
//! let config = TensorParallelConfig::new(4); // 4-way TP
//! assert_eq!(config.world_size(), 4);
//!
//! // Compute shard boundaries for a [4096, 4096] weight on rank 1
//! let (start, end) = config.shard_range(1, 4096, ShardDim::Column);
//! assert_eq!(start, 1024);
//! assert_eq!(end, 2048);
//! ```

use crate::comm::{Communicator, ReduceOp};
use candle::{Module, Result, Tensor};
use candle_nn::Linear;
use serde::{Deserialize, Serialize};

/// Which dimension to shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShardDim {
    /// Shard along columns (output features / dim 1 of weight).
    Column,
    /// Shard along rows (input features / dim 0 of weight).
    Row,
}

/// Metadata for one weight shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorShard {
    /// Rank that owns this shard.
    pub rank: usize,
    /// Total number of ranks.
    pub world_size: usize,
    /// Shard dimension.
    pub dim: ShardDim,
    /// Original (unsharded) shape as `[rows, cols]`.
    pub original_shape: [usize; 2],
}

impl TensorShard {
    /// Start index of this shard along the sharded dimension.
    pub fn start(&self) -> usize {
        let full = match self.dim {
            ShardDim::Column => self.original_shape[1],
            ShardDim::Row => self.original_shape[0],
        };
        let per_rank = full / self.world_size;
        self.rank * per_rank
    }

    /// End index (exclusive) of this shard.
    pub fn end(&self) -> usize {
        let full = match self.dim {
            ShardDim::Column => self.original_shape[1],
            ShardDim::Row => self.original_shape[0],
        };
        let per_rank = full / self.world_size;
        if self.rank == self.world_size - 1 {
            full // last rank gets remainder
        } else {
            (self.rank + 1) * per_rank
        }
    }

    /// Number of elements along the sharded dimension for this rank.
    pub fn shard_size(&self) -> usize {
        self.end() - self.start()
    }
}

/// Configuration for tensor parallelism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorParallelConfig {
    world_size: usize,
}

impl TensorParallelConfig {
    /// Create a TP config with the given number of devices.
    pub fn new(world_size: usize) -> Self {
        assert!(world_size > 0, "world_size must be > 0");
        Self { world_size }
    }

    /// Number of devices in the TP group.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Compute the `[start, end)` range for a given rank along a dimension of
    /// size `full_size`.
    pub fn shard_range(&self, rank: usize, full_size: usize, dim: ShardDim) -> (usize, usize) {
        let _ = dim; // used for API consistency; range is the same regardless
        let per_rank = full_size / self.world_size;
        let start = rank * per_rank;
        let end = if rank == self.world_size - 1 {
            full_size
        } else {
            (rank + 1) * per_rank
        };
        (start, end)
    }

    /// Create a [`TensorShard`] descriptor.
    pub fn make_shard(&self, rank: usize, dim: ShardDim, original_shape: [usize; 2]) -> TensorShard {
        TensorShard {
            rank,
            world_size: self.world_size,
            dim,
            original_shape,
        }
    }
}

/// Column-parallel linear layer.
///
/// Each rank holds columns `[rank * out/N .. (rank+1) * out/N]` of the weight.
/// Forward pass: `Y_local = X @ W_local` — no communication needed.
/// The caller is responsible for gathering outputs if needed.
pub struct ColumnParallel {
    linear: Linear,
    shard: TensorShard,
}

impl ColumnParallel {
    /// Wrap a pre-sharded linear layer.
    pub fn new(linear: Linear, shard: TensorShard) -> Self {
        Self { linear, shard }
    }

    /// Shard metadata.
    pub fn shard(&self) -> &TensorShard {
        &self.shard
    }

    /// Forward: local matmul only, no communication.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

/// Row-parallel linear layer.
///
/// Each rank holds rows `[rank * in/N .. (rank+1) * in/N]` of the weight.
/// Forward pass: `Y_local = X_local @ W_local`, then AllReduce(Sum) to combine.
pub struct RowParallel<C: Communicator> {
    linear: Linear,
    shard: TensorShard,
    comm: C,
}

impl<C: Communicator> RowParallel<C> {
    /// Wrap a pre-sharded linear layer with a communicator for all-reduce.
    pub fn new(linear: Linear, shard: TensorShard, comm: C) -> Self {
        Self { linear, shard, comm }
    }

    /// Shard metadata.
    pub fn shard(&self) -> &TensorShard {
        &self.shard
    }

    /// Forward: local matmul then all-reduce sum.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let local = self.linear.forward(x)?;
        self.comm.all_reduce(&local, ReduceOp::Sum)
    }
}

/// Describes how a model layer should be parallelized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerParallelPlan {
    /// Layer name (for debugging / weight loading).
    pub name: String,
    /// Sharding strategy for this layer.
    pub strategy: ShardDim,
    /// Which TP group this layer belongs to (for multi-group configs).
    pub group_id: usize,
}

impl LayerParallelPlan {
    /// Create a plan entry.
    pub fn new(name: impl Into<String>, strategy: ShardDim) -> Self {
        Self {
            name: name.into(),
            strategy,
            group_id: 0,
        }
    }

    /// Builder: set group ID.
    pub fn with_group(mut self, group_id: usize) -> Self {
        self.group_id = group_id;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::IdentityComm;
    use candle::{DType, Device};

    #[test]
    fn config_shard_range() {
        let config = TensorParallelConfig::new(4);
        let (s, e) = config.shard_range(0, 4096, ShardDim::Column);
        assert_eq!((s, e), (0, 1024));

        let (s, e) = config.shard_range(3, 4096, ShardDim::Column);
        assert_eq!((s, e), (3072, 4096));
    }

    #[test]
    fn shard_range_with_remainder() {
        let config = TensorParallelConfig::new(3);
        // 10 / 3 = 3 per rank, last rank gets 10 - 6 = 4
        let (s, e) = config.shard_range(2, 10, ShardDim::Row);
        assert_eq!((s, e), (6, 10));
    }

    #[test]
    fn tensor_shard_metadata() {
        let config = TensorParallelConfig::new(2);
        let shard = config.make_shard(1, ShardDim::Column, [4096, 4096]);

        assert_eq!(shard.start(), 2048);
        assert_eq!(shard.end(), 4096);
        assert_eq!(shard.shard_size(), 2048);
    }

    #[test]
    fn column_parallel_forward() {
        let device = Device::Cpu;
        // 3×2 input, 2×4 weight → 3×4 output (column-sharded to 2×2 on this rank)
        let w = Tensor::ones((2, 2), DType::F32, &device).unwrap();
        let linear = Linear::new(w, None);
        let shard = TensorShard {
            rank: 0, world_size: 2, dim: ShardDim::Column,
            original_shape: [2, 4],
        };

        let col = ColumnParallel::new(linear, shard);
        let x = Tensor::ones((3, 2), DType::F32, &device).unwrap();
        let y = col.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 2]);
    }

    #[test]
    fn row_parallel_forward() {
        let device = Device::Cpu;
        let w = Tensor::ones((4, 2), DType::F32, &device).unwrap();
        let linear = Linear::new(w, None);
        let shard = TensorShard {
            rank: 0, world_size: 2, dim: ShardDim::Row,
            original_shape: [4, 4],
        };

        let row = RowParallel::new(linear, shard, IdentityComm);
        let x = Tensor::ones((3, 2), DType::F32, &device).unwrap();
        let y = row.forward(&x).unwrap();
        // With identity comm, all_reduce is a no-op → same as local matmul
        assert_eq!(y.dims(), &[3, 4]);
    }

    #[test]
    fn layer_parallel_plan() {
        let plan = LayerParallelPlan::new("mlp.gate_proj", ShardDim::Column)
            .with_group(1);
        assert_eq!(plan.name, "mlp.gate_proj");
        assert_eq!(plan.strategy, ShardDim::Column);
        assert_eq!(plan.group_id, 1);
    }

    #[test]
    #[should_panic]
    fn zero_world_size_panics() {
        TensorParallelConfig::new(0);
    }

    #[test]
    fn single_device_shard_covers_all() {
        let config = TensorParallelConfig::new(1);
        let (s, e) = config.shard_range(0, 4096, ShardDim::Column);
        assert_eq!((s, e), (0, 4096));
    }

    #[test]
    fn shard_dims_are_distinct() {
        assert_ne!(ShardDim::Column, ShardDim::Row);
    }
}
