//! Communication abstraction for collective operations.
//!
//! Defines the [`Communicator`] trait that backends (NCCL, Gloo, etc.) implement
//! to provide collective primitives. Code in [`tensor_parallel`](crate::tensor_parallel)
//! and [`pipeline_parallel`](crate::pipeline_parallel) is generic over this trait.
//!
//! # Example
//!
//! ```rust
//! use candle_parallel::comm::{Communicator, ReduceOp, CommInfo};
//!
//! // A mock communicator for testing (passes data through unchanged).
//! let info = CommInfo { rank: 0, world_size: 2 };
//! assert_eq!(info.is_root(), true);
//! assert_eq!(info.world_size, 2);
//! ```

use candle::{Result, Tensor};

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
/// # Example
///
/// ```rust
/// use candle::{Device, Tensor};
/// use candle_parallel::comm::{Communicator, ReduceOp, IdentityComm};
///
/// let comm = IdentityComm;
/// assert_eq!(comm.info().rank, 0);
/// assert_eq!(comm.info().world_size, 1);
///
/// let t = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap();
/// let reduced = comm.all_reduce(&t, ReduceOp::Sum).unwrap();
/// // Identity: result equals input
/// assert_eq!(reduced.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
/// ```
pub struct IdentityComm;

impl Communicator for IdentityComm {
    fn info(&self) -> CommInfo {
        CommInfo { rank: 0, world_size: 1 }
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
    use candle::Device;

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
        let t = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap();
        let result = comm.all_reduce(&t, ReduceOp::Sum).unwrap();
        assert_eq!(result.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn identity_all_gather() {
        let comm = IdentityComm;
        let t = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();
        let result = comm.all_gather(&t, 0).unwrap();
        assert_eq!(result.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn identity_broadcast() {
        let comm = IdentityComm;
        let t = Tensor::new(&[5.0f32], &Device::Cpu).unwrap();
        let result = comm.broadcast(&t, 0).unwrap();
        assert_eq!(result.to_vec1::<f32>().unwrap(), vec![5.0]);
    }

    #[test]
    fn identity_reduce_scatter() {
        let comm = IdentityComm;
        let t = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap();
        let result = comm.reduce_scatter(&t, ReduceOp::Sum, 0).unwrap();
        assert_eq!(result.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn identity_barrier() {
        let comm = IdentityComm;
        assert!(comm.barrier().is_ok());
    }

    #[test]
    fn comm_info_non_root() {
        let info = CommInfo { rank: 3, world_size: 4 };
        assert!(!info.is_root());
    }

    #[test]
    fn reduce_op_variants() {
        // Ensure all variants exist and are distinct
        let ops = [ReduceOp::Sum, ReduceOp::Product, ReduceOp::Min, ReduceOp::Max];
        for (i, a) in ops.iter().enumerate() {
            for (j, b) in ops.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }
}
