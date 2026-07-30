//! Single-process, multi-device collectives — **lazy-only**.
//!
//! # Why this exists alongside [`crate::comm::Communicator`]
//!
//! `Communicator` is the **SPMD** shape: each rank calls `all_reduce` with *its
//! own* tensor and the implementation rendezvouses with the other ranks. That is
//! the right shape for multi-process / multi-node.
//!
//! A **single process driving N devices holds every shard at once**. There is no
//! other rank to meet, so the per-rank signature does not fit — which is why
//! `IdentityComm` (the only `Communicator` impl today) returns its input: with
//! `world_size == 1` there is genuinely nothing to reduce. That is correct for a
//! single rank and is *not* a working collective.
//!
//! `DeviceGroup` is the slice-shaped counterpart: you hand it all the shards.
//! "One process, four GPUs" — the configuration most inference deployments
//! actually run — needs cross-device copies, not a multi-node transport.
//!
//! # Why this module is lazy-only
//!
//! The rest of this crate is written against the **legacy eager** `fuel::Tensor`
//! (`fuel-core/src/tensor.rs`, "eventually retires when B6 drops eager
//! dispatch"). New code targets [`LazyTensor`] — `fuel-core`'s own guidance is
//! *"new user code should use `lazy::LazyTensor`"*, and every model in Fuel is
//! built on the lazy graph. A collective written against eager tensors could not
//! reduce lazy shards, which is the likeliest reason this crate has no consumers.
//!
//! # Transport
//!
//! Cross-device movement is [`LazyTensor::copy_to_device`] → `Op::Copy`, resolved
//! by the executor's Copy arm. That path is **backend-agnostic** (a host
//! round-trip today; P2P later), so a group may span *different backends* — CUDA
//! and Vulkan, say — without special-casing.

use crate::comm::ReduceOp;
use fuel::lazy::LazyTensor;
use fuel::{Device, Error, Result};

/// A set of devices participating in single-process collectives.
///
/// The first device is the **leader**: reductions land there, because a reduced
/// value has to live somewhere and the caller needs to know where without asking.
#[derive(Debug, Clone)]
pub struct DeviceGroup {
    devices: Vec<Device>,
}

impl DeviceGroup {
    /// Build a group over `devices`. The first is the leader.
    ///
    /// Returns `Err` for an empty list rather than panicking — a group with no
    /// devices has no leader, so every collective on it would be undefined.
    pub fn new(devices: Vec<Device>) -> Result<Self> {
        if devices.is_empty() {
            return Err(Error::Msg(
                "DeviceGroup::new: at least one device is required (the first is the leader)".into(),
            ));
        }
        Ok(Self { devices })
    }

    /// Number of devices in the group.
    pub fn size(&self) -> usize {
        self.devices.len()
    }

    /// The leader device — where reductions land.
    pub fn leader(&self) -> &Device {
        &self.devices[0]
    }

    /// The devices in this group, in rank order.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Reduce `shards` element-wise onto the leader device.
    ///
    /// Each shard is copied to the leader (a no-op when already resident) and
    /// then folded with `op`. Shards must be the same shape and dtype, and must
    /// live on the same graph — see the error cases below.
    pub fn all_reduce(&self, shards: &[LazyTensor], op: ReduceOp) -> Result<LazyTensor> {
        let shards = self.check_shards(shards, "all_reduce")?;
        let leader = self.leader();

        let mut acc = shards[0].copy_to_device(leader);
        for shard in &shards[1..] {
            let here = shard.copy_to_device(leader);
            acc = match op {
                ReduceOp::Sum => acc.add(&here),
                ReduceOp::Product => acc.mul(&here),
                ReduceOp::Max => acc.maximum(&here),
                ReduceOp::Min => acc.minimum(&here),
            }
            .map_err(|e| Error::Msg(format!("DeviceGroup::all_reduce: {op:?} failed: {e}")))?;
        }
        Ok(acc)
    }

    /// Concatenate `shards` along `dim` onto the leader device.
    ///
    /// The gather counterpart of [`all_reduce`](Self::all_reduce): where reduce
    /// combines values element-wise, this preserves every shard's contribution.
    pub fn all_gather(&self, shards: &[LazyTensor], dim: usize) -> Result<LazyTensor> {
        let shards = self.check_shards(shards, "all_gather")?;
        let leader = self.leader();

        let mut acc = shards[0].copy_to_device(leader);
        for shard in &shards[1..] {
            let here = shard.copy_to_device(leader);
            acc = acc
                .concat(&here, dim)
                .map_err(|e| Error::Msg(format!("DeviceGroup::all_gather: concat on dim {dim} failed: {e}")))?;
        }
        Ok(acc)
    }

    /// Shared precondition check for the collectives.
    ///
    /// Both failures below would otherwise surface as a **panic** from deeper in
    /// the graph builder — empty slice as an index panic, mixed graphs as
    /// `Op`-level `Arc::ptr_eq` assertion. Catching them here keeps the
    /// never-panic contract at the collective boundary, and the graph case can
    /// name the fix because every tensor reports its
    /// [`graph_id`](LazyTensor::graph_id).
    fn check_shards<'a>(&self, shards: &'a [LazyTensor], who: &str) -> Result<&'a [LazyTensor]> {
        if shards.is_empty() {
            return Err(Error::Msg(format!(
                "DeviceGroup::{who}: no shards to combine (expected at least one)"
            )));
        }
        let g0 = shards[0].graph_id();
        if let Some(bad) = shards.iter().position(|s| s.graph_id() != g0) {
            return Err(Error::Msg(format!(
                "DeviceGroup::{who}: shards must live on the same graph — shard 0 is on graph \
                 #{g0}, shard {bad} on graph #{}; each `from_*` constructor mints a NEW graph, so \
                 build the shards with `from_*_on(first.graph(), ..)` or `const_*_like`",
                shards[bad].graph_id(),
            )));
        }
        Ok(shards)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel::Shape;

    fn shard(anchor: Option<&LazyTensor>, data: Vec<f32>, dev: &Device) -> LazyTensor {
        let shape = Shape::from_dims(&[data.len()]);
        match anchor {
            Some(a) => LazyTensor::from_f32_on(a.graph(), data, shape, dev),
            None => LazyTensor::from_f32(data, shape, dev),
        }
    }

    #[test]
    fn new_rejects_an_empty_device_list() {
        assert!(
            DeviceGroup::new(vec![]).is_err(),
            "a group with no devices has no leader, so collectives on it are undefined",
        );
    }

    #[test]
    fn leader_is_the_first_device() {
        let g = DeviceGroup::new(vec![Device::cpu()]).unwrap();
        assert_eq!(g.size(), 1);
        assert_eq!(g.leader().location(), Device::cpu().location());
    }

    #[test]
    fn all_reduce_sums_across_shards() {
        let dev = Device::cpu();
        let g = DeviceGroup::new(vec![dev.clone()]).unwrap();
        let a = shard(None, vec![1.0, 2.0], &dev);
        let b = shard(Some(&a), vec![10.0, 20.0], &dev);
        let out = g.all_reduce(&[a, b], ReduceOp::Sum).unwrap();
        assert_eq!(out.realize_f32(), vec![11.0, 22.0]);
    }

    #[test]
    fn all_reduce_takes_the_elementwise_max() {
        let dev = Device::cpu();
        let g = DeviceGroup::new(vec![dev.clone()]).unwrap();
        let a = shard(None, vec![1.0, 50.0], &dev);
        let b = shard(Some(&a), vec![10.0, 20.0], &dev);
        let out = g.all_reduce(&[a, b], ReduceOp::Max).unwrap();
        assert_eq!(out.realize_f32(), vec![10.0, 50.0]);
    }

    #[test]
    fn all_reduce_of_one_shard_is_that_shard() {
        let dev = Device::cpu();
        let g = DeviceGroup::new(vec![dev.clone()]).unwrap();
        let a = shard(None, vec![3.0, 4.0], &dev);
        let out = g.all_reduce(std::slice::from_ref(&a), ReduceOp::Sum).unwrap();
        assert_eq!(out.realize_f32(), vec![3.0, 4.0]);
    }

    #[test]
    fn all_reduce_rejects_empty_shards() {
        let g = DeviceGroup::new(vec![Device::cpu()]).unwrap();
        assert!(
            g.all_reduce(&[], ReduceOp::Sum).is_err(),
            "zero shards is an error, not a panic (never-panic contract)",
        );
    }

    #[test]
    fn all_reduce_rejects_cross_graph_shards_and_names_the_fix() {
        let dev = Device::cpu();
        let g = DeviceGroup::new(vec![dev.clone()]).unwrap();
        // Two INDEPENDENT constructors => two graphs. Combining them would trip
        // the Op-level affinity assert; the group must catch it first.
        let a = shard(None, vec![1.0, 2.0], &dev);
        let b = shard(None, vec![3.0, 4.0], &dev);
        let err = g.all_reduce(&[a, b], ReduceOp::Sum).unwrap_err().to_string();
        assert!(err.contains("same graph"), "got: {err}");
        assert!(err.contains("const_*_like") || err.contains("from_*_on"), "must name the fix; got: {err}");
    }

    #[test]
    fn all_gather_concatenates_shards() {
        let dev = Device::cpu();
        let g = DeviceGroup::new(vec![dev.clone()]).unwrap();
        let a = shard(None, vec![1.0, 2.0], &dev);
        let b = shard(Some(&a), vec![3.0, 4.0], &dev);
        let out = g.all_gather(&[a, b], 0).unwrap();
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
