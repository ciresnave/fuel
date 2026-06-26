//! `Storage` re-export + `StorageApplyOps` trait extension.
//!
//! Phase 7.5 work item G fix-up: the `Storage` struct and almost all of
//! its eager-dispatch methods moved to `fuel-core-types::storage` so
//! that `fuel_graph::Graph` can own a `HashMap<NodeId, Storage>` slot
//! map without the fuel-graph crate inverting its dependency on
//! fuel-core.
//!
//! What stays in fuel-core: the three `apply_op1/2/3` methods, because
//! they take `&dyn CustomOp1/2/3` trait objects whose `bwd` method
//! returns `Tensor` (autograd). They live as a trait extension on
//! `Storage` rather than inherent impls because Rust orphan rules
//! forbid inherent impls on a type defined in another crate.
//!
//! All three are scheduled for removal in Phase 7.5 work item B6
//! along with the rest of eager dispatch.

pub use fuel_ir::Storage;

use crate::custom_op::{CustomOp1, CustomOp2, CustomOp3};
use crate::{Layout, Result, Shape};

/// Trait extension that re-attaches the `CustomOp` apply methods to
/// the moved `Storage` type. `use crate::storage::StorageApplyOps;` to
/// bring them into scope.
pub trait StorageApplyOps {
    fn apply_op1(&self, l: &Layout, c: &dyn CustomOp1) -> Result<(Storage, Shape)>;

    #[allow(clippy::too_many_arguments)]
    fn apply_op2(
        &self,
        l1: &Layout,
        t2: &Storage,
        l2: &Layout,
        c: &dyn CustomOp2,
    ) -> Result<(Storage, Shape)>;

    #[allow(clippy::too_many_arguments)]
    fn apply_op3(
        &self,
        l1: &Layout,
        t2: &Storage,
        l2: &Layout,
        t3: &Storage,
        l3: &Layout,
        c: &dyn CustomOp3,
    ) -> Result<(Storage, Shape)>;
}

impl StorageApplyOps for Storage {
    fn apply_op1(&self, l: &Layout, c: &dyn CustomOp1) -> Result<(Storage, Shape)> {
        let (storage, shape) = c.fwd(self.as_dyn(), l)?;
        Ok((Storage::from_dyn(storage), shape))
    }

    fn apply_op2(
        &self,
        l1: &Layout,
        t2: &Storage,
        l2: &Layout,
        c: &dyn CustomOp2,
    ) -> Result<(Storage, Shape)> {
        self.same_device(t2, c.name())?;
        let (storage, shape) = c.fwd(self.as_dyn(), l1, t2.as_dyn(), l2)?;
        Ok((Storage::from_dyn(storage), shape))
    }

    fn apply_op3(
        &self,
        l1: &Layout,
        t2: &Storage,
        l2: &Layout,
        t3: &Storage,
        l3: &Layout,
        c: &dyn CustomOp3,
    ) -> Result<(Storage, Shape)> {
        self.same_device(t2, c.name())?;
        self.same_device(t3, c.name())?;
        let (storage, shape) = c.fwd(self.as_dyn(), l1, t2.as_dyn(), l2, t3.as_dyn(), l3)?;
        Ok((Storage::from_dyn(storage), shape))
    }
}
