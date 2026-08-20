// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tensor parallelism: column-parallel and row-parallel linear layers — **lazy-only**.
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
//! use fuel_parallel::tensor_parallel::{TensorParallelConfig, ShardDim};
//!
//! let config = TensorParallelConfig::new(4)?; // 4-way TP
//! assert_eq!(config.world_size(), 4);
//!
//! // Compute shard boundaries for a [4096, 4096] weight on rank 1
//! let (start, end) = config.shard_range(1, 4096, ShardDim::Column);
//! assert_eq!(start, 1024);
//! assert_eq!(end, 2048);
//! # Ok::<(), fuel::Error>(())
//! ```

use crate::comm::{Communicator, ReduceOp};
use fuel::lazy::{Tensor, WeightStorage};
use fuel::{Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Minimal inlined Linear layer, **lazy**.
///
/// Computes `y = x @ W + b`, delegating to
/// [`WeightStorage::apply_linear`](fuel::lazy::WeightStorage::apply_linear) —
/// the same builder every lazy model in `fuel-core` uses. That buys F32, BF16,
/// Q4_0 and LoRA weights for free; a hand-rolled `const_f32_like` + `matmul`
/// here would have been f32-only.
///
/// # Weight layout — CHANGED from the eager version
///
/// **`[in_features, out_features]`, applied as `x @ W`.**
///
/// The eager `Linear` this replaces stored `[out_features, in_features]` and
/// computed `x @ W.t()` (PyTorch's `nn.Linear` convention). `WeightStorage`
/// uses the transposed layout, and adopting it is what makes the quantized and
/// LoRA paths reachable. The difference is silent — feeding
/// `[out, in]` data to a square layer produces a wrong answer rather than an
/// error — so it is called out here rather than left to be discovered.
///
/// `fuel-parallel` has no consumers today, so nothing in-tree breaks.
///
/// # Why host storage rather than a `Tensor` weight
///
/// `Tensor` is **graph-affine**: two tensors combine iff they share a
/// graph. A `Linear` holding a `Tensor` weight could only ever be applied
/// to inputs on that one graph — useless for a model that builds a fresh graph
/// per step, which is exactly what Fuel's lazy decode path does.
/// `WeightStorage` materializes onto **the input's** graph at `forward` time,
/// so one `Linear` serves any number of graphs.
#[derive(Clone, Debug)]
pub struct Linear {
    weight: WeightStorage,
    in_features: usize,
    out_features: usize,
    /// Length `out_features`.
    bias: Option<Arc<[f32]>>,
}

impl Linear {
    /// Build from a [`WeightStorage`] laid out `[in_features, out_features]`.
    ///
    /// Returns `Err` on a size mismatch rather than deferring it: the
    /// `apply_linear` path this delegates to reports mismatches via
    /// `assert_eq!`/`unwrap()`, which would be a panic on a production path.
    /// Validating here keeps Fuel's never-panic rule intact.
    ///
    /// The element-count check applies to F32/BF16 only — a Q4_0 weight's
    /// `elem_count` counts packed `u32` words, not logical elements, so its
    /// dimensions are checked by `WeightStorage` itself against the values
    /// stored in the variant.
    pub fn new(
        weight: WeightStorage,
        in_features: usize,
        out_features: usize,
        bias: Option<Arc<[f32]>>,
    ) -> Result<Self> {
        let dtype = weight.dtype();
        if matches!(dtype, fuel::DType::F32 | fuel::DType::BF16) {
            let expected = in_features * out_features;
            if weight.elem_count() != expected {
                return Err(Error::Msg(format!(
                    "Linear::new: weight has {} elements but [in={in_features}, out={out_features}] \
                     needs {expected}",
                    weight.elem_count()
                )));
            }
        }
        if let Some(b) = &bias
            && b.len() != out_features
        {
            return Err(Error::Msg(format!(
                "Linear::new: bias has {} elements but out_features is {out_features}",
                b.len()
            )));
        }
        Ok(Self {
            weight,
            in_features,
            out_features,
            bias,
        })
    }

    /// Convenience constructor for a plain f32 weight, `[in_features, out_features]`.
    pub fn from_f32(
        weight: impl Into<Arc<[f32]>>,
        in_features: usize,
        out_features: usize,
        bias: Option<Arc<[f32]>>,
    ) -> Result<Self> {
        Self::new(
            WeightStorage::F32(weight.into()),
            in_features,
            out_features,
            bias,
        )
    }

    /// `(in_features, out_features)`.
    pub fn features(&self) -> (usize, usize) {
        (self.in_features, self.out_features)
    }

    /// Forward: `y = x @ W + b`, built on **`x`'s** graph.
    ///
    /// The eager original branched on `x.is_contiguous()` and hand-picked
    /// between a reshape-then-matmul and a broadcast matmul, "to avoid a
    /// broadcasted matmul — it's much slower". That is a dispatch-time
    /// optimisation, and in a lazy DAG choosing it here would be the model layer
    /// making a strategy decision that belongs to the optimizer. Lazy `matmul`
    /// is N-D batched with rank-2 broadcasting, so the whole dance collapses to
    /// one delegated call and the optimizer keeps its job.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (in_features, out_features) = (self.in_features, self.out_features);

        // The trailing-dim and stored-shape checks that were duplicated here
        // now live inside `WeightStorage::apply_linear`, which returns
        // `Result` rather than panicking. Guarding at this one caller only
        // ever protected this one caller; validating inside the builder
        // protects all ~620 of them. What is left here is naming the layer in
        // the error chain.
        let labelled = |e: Error| Error::Msg(format!("Linear({in_features}->{out_features}): {e}"));
        match &self.bias {
            None => self
                .weight
                .apply_linear(x, in_features, out_features)
                .map_err(labelled),
            Some(bias) => self
                .weight
                .apply_linear_with_bias(x, in_features, out_features, Arc::clone(bias))
                .map_err(labelled),
        }
    }
}

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
        let per_rank = full / self.world_size.max(1);
        self.rank * per_rank
    }

    /// End index (exclusive) of this shard.
    pub fn end(&self) -> usize {
        let full = match self.dim {
            ShardDim::Column => self.original_shape[1],
            ShardDim::Row => self.original_shape[0],
        };
        let per_rank = full / self.world_size.max(1);
        if self.rank + 1 >= self.world_size {
            full // last rank gets remainder
        } else {
            (self.rank + 1) * per_rank
        }
    }

    /// Number of elements along the sharded dimension for this rank.
    pub fn shard_size(&self) -> usize {
        self.end().saturating_sub(self.start())
    }
}

/// Configuration for tensor parallelism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorParallelConfig {
    world_size: usize,
}

impl TensorParallelConfig {
    /// Create a TP config with the given number of devices.
    ///
    /// Returns `Err` for `world_size == 0` rather than panicking: a zero-rank TP
    /// group has no valid shard range, and Fuel's never-panic rule applies to
    /// config construction as much as to a kernel.
    pub fn new(world_size: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(Error::Msg(
                "TensorParallelConfig::new: world_size must be > 0".into(),
            ));
        }
        Ok(Self { world_size })
    }

    /// Number of devices in the TP group.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Compute the `[start, end)` range for a given rank along a dimension of
    /// size `full_size`.
    ///
    /// Ranges are contiguous and cover `0..full_size` exactly; the last rank
    /// absorbs the remainder when `full_size` does not divide evenly. When
    /// `world_size > full_size` the leading ranks get **empty** ranges and the
    /// last rank takes everything — degenerate but still a partition, never
    /// overlapping.
    pub fn shard_range(&self, rank: usize, full_size: usize, dim: ShardDim) -> (usize, usize) {
        let _ = dim; // used for API consistency; range is the same regardless
        let per_rank = full_size / self.world_size;
        let start = rank * per_rank;
        let end = if rank + 1 >= self.world_size {
            full_size
        } else {
            (rank + 1) * per_rank
        };
        (start, end)
    }

    /// Create a [`TensorShard`] descriptor.
    pub fn make_shard(
        &self,
        rank: usize,
        dim: ShardDim,
        original_shape: [usize; 2],
    ) -> TensorShard {
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
/// The caller is responsible for gathering outputs if needed
/// ([`DeviceGroup::all_gather`](crate::device_group::DeviceGroup::all_gather)).
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
///
/// # This is the SPMD form
///
/// It reduces via a [`Communicator`], so it describes *one rank's* half of the
/// work. Fuel ships no multi-process `Communicator` (see [`crate::comm`]), so
/// with the only in-tree impl — `IdentityComm`, `world_size == 1` — the
/// all-reduce is an identity and the result is the single rank's own product.
/// That is correct for one rank and is **not** a multi-device reduction.
///
/// For one process driving N devices, hold all N partials and fold them with
/// [`DeviceGroup::all_reduce`](crate::device_group::DeviceGroup::all_reduce),
/// which is a real cross-device collective.
pub struct RowParallel<C: Communicator> {
    linear: Linear,
    shard: TensorShard,
    comm: C,
}

impl<C: Communicator> RowParallel<C> {
    /// Wrap a pre-sharded linear layer with a communicator for all-reduce.
    pub fn new(linear: Linear, shard: TensorShard, comm: C) -> Self {
        Self {
            linear,
            shard,
            comm,
        }
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
    use fuel::{Device, Shape};

    /// All-ones weight, `[in_features, out_features]`. Note the argument order
    /// follows the `WeightStorage` layout, not the eager `[out, in]` one.
    fn ones_linear(in_features: usize, out_features: usize) -> Linear {
        Linear::from_f32(
            vec![1.0f32; in_features * out_features],
            in_features,
            out_features,
            None,
        )
        .unwrap()
    }

    fn ones_input(dims: &[usize]) -> Tensor {
        let n: usize = dims.iter().product();
        Tensor::from_f32(vec![1.0f32; n], Shape::from_dims(dims), &Device::cpu())
    }

    #[test]
    fn config_shard_range() {
        let config = TensorParallelConfig::new(4).unwrap();
        let (s, e) = config.shard_range(0, 4096, ShardDim::Column);
        assert_eq!((s, e), (0, 1024));

        let (s, e) = config.shard_range(3, 4096, ShardDim::Column);
        assert_eq!((s, e), (3072, 4096));
    }

    #[test]
    fn shard_range_with_remainder() {
        let config = TensorParallelConfig::new(3).unwrap();
        // 10 / 3 = 3 per rank, last rank gets 10 - 6 = 4
        let (s, e) = config.shard_range(2, 10, ShardDim::Row);
        assert_eq!((s, e), (6, 10));
    }

    #[test]
    fn shard_ranges_partition_exactly() {
        // The property that matters and that per-rank spot checks miss: the
        // ranges must tile [0, full) with no gap and no overlap, for sizes that
        // divide evenly and sizes that do not.
        for world in 1..=5usize {
            let config = TensorParallelConfig::new(world).unwrap();
            for full in [0usize, 1, 7, 10, 4096] {
                let mut cursor = 0usize;
                for rank in 0..world {
                    let (s, e) = config.shard_range(rank, full, ShardDim::Row);
                    assert_eq!(
                        s, cursor,
                        "gap/overlap at rank {rank} (world={world}, full={full})"
                    );
                    assert!(
                        e >= s,
                        "inverted range at rank {rank} (world={world}, full={full})"
                    );
                    cursor = e;
                }
                assert_eq!(
                    cursor, full,
                    "ranges do not cover full={full} (world={world})"
                );
            }
        }
    }

    #[test]
    fn tensor_shard_metadata() {
        let config = TensorParallelConfig::new(2).unwrap();
        let shard = config.make_shard(1, ShardDim::Column, [4096, 4096]);

        assert_eq!(shard.start(), 2048);
        assert_eq!(shard.end(), 4096);
        assert_eq!(shard.shard_size(), 2048);
    }

    #[test]
    fn column_parallel_forward() {
        // 3×2 input, 2×2 weight shard → 3×2 output.
        let col = ColumnParallel::new(
            ones_linear(2, 2),
            TensorShard {
                rank: 0,
                world_size: 2,
                dim: ShardDim::Column,
                original_shape: [2, 4],
            },
        );
        let y = col.forward(&ones_input(&[3, 2])).unwrap();
        assert_eq!(y.shape().dims(), &[3, 2]);
        // ones(3,2) @ ones(2,2).t() => every element is the contracted dim, 2.
        assert_eq!(y.realize_f32(), vec![2.0f32; 6]);
    }

    #[test]
    fn row_parallel_forward() {
        let row = RowParallel::new(
            ones_linear(2, 4),
            TensorShard {
                rank: 0,
                world_size: 2,
                dim: ShardDim::Row,
                original_shape: [4, 4],
            },
            IdentityComm,
        );
        let y = row.forward(&ones_input(&[3, 2])).unwrap();
        // With identity comm, all_reduce is a no-op → same as local matmul
        assert_eq!(y.shape().dims(), &[3, 4]);
        assert_eq!(y.realize_f32(), vec![2.0f32; 12]);
    }

    #[test]
    fn linear_applies_to_inputs_on_different_graphs() {
        // The point of holding host data instead of a `Tensor` weight: one
        // `Linear` must work against inputs rooted on unrelated graphs. Holding
        // a `Tensor` would make the second call fail an operand-graph check.
        let lin = ones_linear(3, 2);
        let a = ones_input(&[1, 3]);
        let b = ones_input(&[2, 3]);
        assert_ne!(
            a.graph_id(),
            b.graph_id(),
            "inputs must be on separate graphs"
        );

        assert_eq!(lin.forward(&a).unwrap().realize_f32(), vec![3.0f32; 2]);
        assert_eq!(lin.forward(&b).unwrap().realize_f32(), vec![3.0f32; 4]);
    }

    #[test]
    fn linear_output_shares_the_input_graph() {
        let lin = ones_linear(3, 2);
        let x = ones_input(&[1, 3]);
        let y = lin.forward(&x).unwrap();
        assert_eq!(y.graph_id(), x.graph_id());
    }

    #[test]
    fn linear_adds_bias() {
        let lin =
            Linear::from_f32(vec![1.0f32; 3 * 2], 3, 2, Some(vec![10.0f32, 20.0].into())).unwrap();
        let y = lin.forward(&ones_input(&[1, 3])).unwrap();
        // row of ones · ones(3) = 3, then + per-output bias
        assert_eq!(y.realize_f32(), vec![13.0, 23.0]);
    }

    #[test]
    fn linear_forward_is_batched_over_leading_dims() {
        // Rank-3 input must work without the caller reshaping — lazy matmul is
        // N-D batched, which is why the eager contiguity branch could go.
        let lin = ones_linear(3, 2);
        let y = lin.forward(&ones_input(&[2, 4, 3])).unwrap();
        assert_eq!(y.shape().dims(), &[2, 4, 2]);
        assert_eq!(y.realize_f32(), vec![3.0f32; 16]);
    }

    #[test]
    fn linear_rejects_mismatched_input() {
        let lin = ones_linear(3, 2);
        let err = lin.forward(&ones_input(&[1, 5])).unwrap_err();
        let msg = format!("{err}");
        // The check now lives in `WeightStorage::apply_linear` rather than
        // being duplicated in `Linear::forward`, so the wording comes from
        // there; `Linear(..)` is this layer's label on the error chain. Assert
        // on the facts (offending shape + required dim), not the phrasing.
        assert!(
            msg.contains("[1, 5]") && msg.contains("trailing dim 3"),
            "error should name the offending shape and the required in_features, got: {err}"
        );
        assert!(
            msg.contains("Linear(3->2)"),
            "error should name the layer, got: {err}"
        );
    }

    #[test]
    fn linear_rejects_weight_shape_mismatch() {
        let err = Linear::from_f32(vec![1.0f32; 5], 3, 2, None).unwrap_err();
        assert!(format!("{err}").contains("needs 6"), "got: {err}");
    }

    #[test]
    fn linear_rejects_bias_length_mismatch() {
        let err =
            Linear::from_f32(vec![1.0f32; 6], 3, 2, Some(vec![0.0f32; 3].into())).unwrap_err();
        assert!(format!("{err}").contains("out_features is 2"), "got: {err}");
    }

    #[test]
    fn linear_rejects_rank_1_input_instead_of_panicking() {
        // `WeightStorage::apply_linear` reaches `x.matmul(&w).unwrap()`, which
        // panics on rank < 2. Guarding here is what keeps the never-panic rule
        // intact through the delegation.
        let lin = ones_linear(3, 2);
        let x = Tensor::from_f32(vec![1.0f32; 3], Shape::from_dims(&[3]), &Device::cpu());
        let err = lin.forward(&x).unwrap_err();
        assert!(format!("{err}").contains("rank >= 2"), "got: {err}");
    }

    #[test]
    fn linear_carries_a_bf16_weight() {
        // The point of delegating to `WeightStorage`: dtypes beyond f32 come
        // for free. A hand-rolled `const_f32_like` Linear could not do this.
        let w: Vec<half::bf16> = vec![half::bf16::from_f32(1.0); 3 * 2];
        let lin = Linear::new(WeightStorage::BF16(w.into()), 3, 2, None).unwrap();
        assert_eq!(lin.features(), (3, 2));
    }

    #[test]
    fn layer_parallel_plan() {
        let plan = LayerParallelPlan::new("mlp.gate_proj", ShardDim::Column).with_group(1);
        assert_eq!(plan.name, "mlp.gate_proj");
        assert_eq!(plan.strategy, ShardDim::Column);
        assert_eq!(plan.group_id, 1);
    }

    #[test]
    fn zero_world_size_is_an_error_not_a_panic() {
        // Was `#[should_panic]`. Fuel's never-panic rule applies to config
        // construction too, so this now surfaces as a typed error.
        let err = TensorParallelConfig::new(0).unwrap_err();
        assert!(
            format!("{err}").contains("world_size must be > 0"),
            "got: {err}"
        );
    }

    #[test]
    fn single_device_shard_covers_all() {
        let config = TensorParallelConfig::new(1).unwrap();
        let (s, e) = config.shard_range(0, 4096, ShardDim::Column);
        assert_eq!((s, e), (0, 4096));
    }

    #[test]
    fn shard_dims_are_distinct() {
        assert_ne!(ShardDim::Column, ShardDim::Row);
    }
}
