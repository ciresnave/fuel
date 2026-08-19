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
//! Everything tensor-touching in this crate targets [`LazyTensor`], and every
//! model in Fuel is built on the lazy graph. A collective written against the
//! old eager `Tensor` could not have reduced lazy shards, which is the likeliest
//! reason this crate has no consumers. B6 deleted that eager `Tensor` outright
//! (`fuel-core/src/tensor.rs` is gone), so lazy-only is now the only option
//! rather than a preference.
//!
//! # Transport, and why cross-vendor hops are authored here
//!
//! Movement is [`LazyTensor::copy_to_device`] → `Op::Copy`, resolved by the
//! executor's Copy arm. **Every hop with CPU on one end is a single `Op::Copy`**
//! — CPU↔CUDA, CPU↔Vulkan, and same-device. A **direct cross-vendor GPU↔GPU copy
//! is not implemented**: both GPU wrappers reject a foreign-GPU output, and
//! `fuel-graph/src/opt.rs:2590` says so outright — "a cross-VENDOR GPU↔GPU edge
//! (e.g. CUDA↔Vulkan) can't be done as one `Op::Copy`".
//!
//! The optimizer *does* split such an edge into two hops, but
//! `insert_cross_device_copies` **deliberately skips any consumer that is itself
//! an `Op::Copy`/`Op::Move`** (`opt.rs:2571-2577`) — otherwise inserting a copy on
//! a copy's input would regress infinitely. That carve-out is correct, and it
//! means a **user-authored** copy across vendors is never split for you.
//!
//! So this module **stages through CPU itself** when source and leader are
//! different non-CPU backends: `src → CPU → leader`, two explicit `Op::Copy`
//! nodes. A group may therefore genuinely span CPU + CUDA + Vulkan.
//!
//! *(An earlier version of this comment claimed the path was "backend-agnostic …
//! without special-casing". That was inferred from `copy_to_device`'s doc comment
//! rather than read from the pass, and it was wrong.)*

use crate::comm::ReduceOp;
use fuel::lazy::LazyTensor;
use fuel::{Device, DeviceLocation, Error, Result};

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
                "DeviceGroup::new: at least one device is required (the first is the leader)"
                    .into(),
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
    /// **Shard `i` is taken to live on device `i`** — the group's device list is
    /// the residency map, so `shards.len()` must not exceed [`size`](Self::size).
    /// Each shard is brought to the leader (staging through CPU when it must
    /// cross vendors, see [`bring_to_leader`](Self::bring_to_leader)) and folded
    /// with `op`.
    pub fn all_reduce(&self, shards: &[LazyTensor], op: ReduceOp) -> Result<LazyTensor> {
        let shards = self.check_shards(shards, "all_reduce")?;

        let mut acc = self.bring_to_leader(&shards[0], 0);
        for (i, shard) in shards.iter().enumerate().skip(1) {
            let here = self.bring_to_leader(shard, i);
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

        let mut acc = self.bring_to_leader(&shards[0], 0);
        for (i, shard) in shards.iter().enumerate().skip(1) {
            let here = self.bring_to_leader(shard, i);
            acc = acc.concat(&here, dim).map_err(|e| {
                Error::Msg(format!(
                    "DeviceGroup::all_gather: concat on dim {dim} failed: {e}"
                ))
            })?;
        }
        Ok(acc)
    }

    /// Realize `t` as `f32`, seeding a device handle for **every** device in the
    /// group.
    ///
    /// A plain `LazyTensor::realize_f32` seeds only the *primary* device, so a
    /// graph whose nodes span several backends cannot be executed through it —
    /// the non-primary backends have no handle to allocate against. This routes
    /// through `pipelined_bridge::realize_one_as_multi_device` with the leader as
    /// primary and the remaining group devices as extras, which is the entry the
    /// existing live multi-vendor tests use.
    ///
    /// Callers should not have to know that: the group already owns the device
    /// list, so it is the natural place for the seeding to live.
    pub fn realize_f32(&self, t: &LazyTensor) -> Result<Vec<f32>> {
        let extras: Vec<&Device> = self.devices[1..].iter().collect();
        let graph = t.graph().clone();
        fuel::pipelined_bridge::realize_one_as_multi_device::<f32>(
            &graph,
            t.node_id(),
            self.leader(),
            &extras,
        )
    }

    /// Bring `shard`, which lives on `devices[rank]`, onto the leader device —
    /// **authoring the CPU hop ourselves when the move crosses vendors.**
    ///
    /// Three cases:
    /// - already on the leader → a same-device `Op::Copy`, which the backend can
    ///   elide;
    /// - one end is CPU → a single `Op::Copy` (the implemented direct hops);
    /// - **two different non-CPU backends** → `src → CPU → leader`, two explicit
    ///   `Op::Copy` nodes, because a direct cross-vendor GPU↔GPU copy does not
    ///   exist and the optimizer will not split a Copy's own input
    ///   (`opt.rs:2571-2577`). Emitting one hop here would fail at execute time
    ///   with a foreign-GPU-output rejection.
    ///
    /// A `rank` past the device list falls back to the leader unchanged: the
    /// count mismatch is already reported by [`check_shards`](Self::check_shards),
    /// and this must not panic on the way there.
    fn bring_to_leader(&self, shard: &LazyTensor, rank: usize) -> LazyTensor {
        let leader = self.leader();
        let Some(src) = self.devices.get(rank) else {
            return shard.copy_to_device(leader);
        };

        let (src_loc, leader_loc) = (src.location(), leader.location());
        if src_loc == leader_loc {
            return shard.copy_to_device(leader);
        }

        let cpu = DeviceLocation::Cpu;
        if src_loc == cpu || leader_loc == cpu {
            // One end is the host: a single implemented hop.
            return shard.copy_to_device(leader);
        }

        // Cross-vendor: author both hops.
        shard.copy_to_device(&Device::cpu()).copy_to_device(leader)
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
        if shards.len() > self.devices.len() {
            return Err(Error::Msg(format!(
                "DeviceGroup::{who}: {} shards for a group of {} devices — shard i is taken to \
                 live on device i, so there cannot be more shards than devices",
                shards.len(),
                self.devices.len(),
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

    /// A group of `n` CPU devices — the degenerate all-same-location case, which
    /// is what a host-only test can construct. `shard i` still corresponds to
    /// `device i`, so `n` shards need `n` devices.
    fn cpu_group(n: usize) -> (DeviceGroup, Device) {
        let dev = Device::cpu();
        (DeviceGroup::new(vec![dev.clone(); n]).unwrap(), dev)
    }

    #[test]
    fn all_reduce_sums_across_shards() {
        let (g, dev) = cpu_group(2);
        let a = shard(None, vec![1.0, 2.0], &dev);
        let b = shard(Some(&a), vec![10.0, 20.0], &dev);
        let out = g.all_reduce(&[a, b], ReduceOp::Sum).unwrap();
        assert_eq!(out.realize_f32(), vec![11.0, 22.0]);
    }

    #[test]
    fn all_reduce_takes_the_elementwise_max() {
        let (g, dev) = cpu_group(2);
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
        let out = g
            .all_reduce(std::slice::from_ref(&a), ReduceOp::Sum)
            .unwrap();
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
        let (g, dev) = cpu_group(2);
        // Two INDEPENDENT constructors => two graphs. Combining them would trip
        // the Op-level affinity assert; the group must catch it first.
        let a = shard(None, vec![1.0, 2.0], &dev);
        let b = shard(None, vec![3.0, 4.0], &dev);
        let err = g
            .all_reduce(&[a, b], ReduceOp::Sum)
            .unwrap_err()
            .to_string();
        assert!(err.contains("same graph"), "got: {err}");
        assert!(
            err.contains("const_*_like") || err.contains("from_*_on"),
            "must name the fix; got: {err}"
        );
    }

    #[test]
    fn all_reduce_rejects_more_shards_than_devices() {
        // shard i lives on device i, so more shards than devices is incoherent
        // — and must be an error rather than an out-of-range fallback.
        let dev = Device::cpu();
        let g = DeviceGroup::new(vec![dev.clone()]).unwrap();
        let a = shard(None, vec![1.0], &dev);
        let b = shard(Some(&a), vec![2.0], &dev);
        let err = g
            .all_reduce(&[a, b], ReduceOp::Sum)
            .unwrap_err()
            .to_string();
        assert!(err.contains("2 shards"), "got: {err}");
        assert!(err.contains("1 devices"), "got: {err}");
    }

    #[test]
    fn same_device_ranks_take_the_single_hop_path() {
        // Regression guard for the staging logic: when every device in the group
        // is the same location, no rank should be routed through a CPU stage.
        // Correctness is observable (values), hop count is not from here — this
        // pins the behaviour that the all-CPU group still reduces correctly
        // after `bring_to_leader` gained the cross-vendor branch.
        let dev = Device::cpu();
        let g = DeviceGroup::new(vec![dev.clone(), dev.clone(), dev.clone()]).unwrap();
        let a = shard(None, vec![1.0, 1.0], &dev);
        let b = shard(Some(&a), vec![2.0, 2.0], &dev);
        let c = shard(Some(&a), vec![3.0, 3.0], &dev);
        let out = g.all_reduce(&[a, b, c], ReduceOp::Sum).unwrap();
        assert_eq!(out.realize_f32(), vec![6.0, 6.0]);
    }

    #[test]
    fn all_gather_concatenates_shards() {
        let (g, dev) = cpu_group(2);
        let a = shard(None, vec![1.0, 2.0], &dev);
        let b = shard(Some(&a), vec![3.0, 4.0], &dev);
        let out = g.all_gather(&[a, b], 0).unwrap();
        assert_eq!(out.realize_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
