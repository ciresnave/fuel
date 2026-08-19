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
//! `StorageApplyOps` was REMOVED in B6 along with eager dispatch; this module
//! is now just the `Storage` re-export.

pub use fuel_backend_contract::Storage;
