//! Device topology modelling.
//!
//! Describes the physical layout of compute devices and their interconnects so
//! that placement and scheduling decisions can take transfer costs into account.
//!
//! This module contains only descriptive types — no CUDA or Metal API calls.
//! Topology is constructed by the caller (or a platform-specific probe) and
//! passed to scheduling/placement code.
//!
//! # Example
//!
//! ```rust
//! use candle_parallel::topology::{DeviceTopology, DeviceInfo, DeviceKind, Link, Interconnect};
//!
//! let mut topo = DeviceTopology::new();
//! let gpu0 = topo.add_device(DeviceInfo::new(0, DeviceKind::Cuda, "RTX 4090")
//!     .with_memory_bytes(24 * 1024 * 1024 * 1024));
//! let gpu1 = topo.add_device(DeviceInfo::new(1, DeviceKind::Cuda, "RTX 4090")
//!     .with_memory_bytes(24 * 1024 * 1024 * 1024));
//! topo.add_link(gpu0, gpu1, Link::new(Interconnect::NvLink, 600_000));
//!
//! assert_eq!(topo.num_devices(), 2);
//! assert_eq!(topo.bandwidth_mb_s(gpu0, gpu1), Some(600_000));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Opaque device identifier within a topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId(pub usize);

/// Kind of compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceKind {
    /// CPU (host).
    Cpu,
    /// NVIDIA CUDA GPU.
    Cuda,
    /// AMD ROCm GPU.
    Rocm,
    /// Apple Metal GPU.
    Metal,
    /// Intel GPU (Level Zero / oneAPI).
    IntelGpu,
    /// Other / custom.
    Other,
}

/// Interconnect type between two devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Interconnect {
    /// NVIDIA NVLink (high bandwidth, low latency).
    NvLink,
    /// PCI Express (standard host↔device or peer-to-peer).
    Pcie,
    /// AMD Infinity Fabric.
    InfinityFabric,
    /// Shared memory (same-device or CPU↔CPU).
    SharedMemory,
    /// Network (e.g. InfiniBand, RoCE).
    Network,
    /// Unknown / not characterized.
    Unknown,
}

/// Descriptor for a single compute device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Hardware ordinal (e.g. CUDA device index).
    pub ordinal: usize,
    /// Device kind.
    pub kind: DeviceKind,
    /// Human-readable name (e.g. `"RTX 4090"`).
    pub name: String,
    /// Total device memory in bytes (0 = unknown).
    pub memory_bytes: u64,
    /// Compute throughput estimate in GFLOPS (0 = unknown).
    pub gflops: f64,
}

impl DeviceInfo {
    /// Create a minimal device descriptor.
    pub fn new(ordinal: usize, kind: DeviceKind, name: impl Into<String>) -> Self {
        Self {
            ordinal,
            kind,
            name: name.into(),
            memory_bytes: 0,
            gflops: 0.0,
        }
    }

    /// Builder: set memory capacity.
    pub fn with_memory_bytes(mut self, bytes: u64) -> Self {
        self.memory_bytes = bytes;
        self
    }

    /// Builder: set compute throughput.
    pub fn with_gflops(mut self, gflops: f64) -> Self {
        self.gflops = gflops;
        self
    }
}

/// A directed link between two devices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    /// Interconnect type.
    pub interconnect: Interconnect,
    /// Bandwidth in MB/s (0 = unknown).
    pub bandwidth_mb_s: u64,
    /// Latency in microseconds (0 = unknown).
    pub latency_us: u64,
}

impl Link {
    /// Create a link with the given interconnect and bandwidth.
    pub fn new(interconnect: Interconnect, bandwidth_mb_s: u64) -> Self {
        Self {
            interconnect,
            bandwidth_mb_s,
            latency_us: 0,
        }
    }

    /// Builder: set latency.
    pub fn with_latency_us(mut self, us: u64) -> Self {
        self.latency_us = us;
        self
    }

    /// Estimated transfer time in microseconds for `bytes` of data.
    ///
    /// Returns `None` if bandwidth is unknown (0).
    pub fn transfer_time_us(&self, bytes: u64) -> Option<f64> {
        if self.bandwidth_mb_s == 0 {
            return None;
        }
        let mb = bytes as f64 / (1024.0 * 1024.0);
        let seconds = mb / self.bandwidth_mb_s as f64;
        Some(self.latency_us as f64 + seconds * 1_000_000.0)
    }
}

/// Graph of devices and their interconnects.
#[derive(Debug, Clone, Default)]
pub struct DeviceTopology {
    devices: Vec<DeviceInfo>,
    /// Directed links keyed by (src, dst).
    links: HashMap<(DeviceId, DeviceId), Link>,
}

impl DeviceTopology {
    /// Create an empty topology.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a device. Returns its [`DeviceId`].
    pub fn add_device(&mut self, info: DeviceInfo) -> DeviceId {
        let id = DeviceId(self.devices.len());
        self.devices.push(info);
        id
    }

    /// Add a bidirectional link between two devices.
    pub fn add_link(&mut self, a: DeviceId, b: DeviceId, link: Link) {
        self.links.insert((a, b), link.clone());
        self.links.insert((b, a), link);
    }

    /// Add a directed (asymmetric) link.
    pub fn add_directed_link(&mut self, from: DeviceId, to: DeviceId, link: Link) {
        self.links.insert((from, to), link);
    }

    /// Number of devices.
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    /// Get device info.
    pub fn device(&self, id: DeviceId) -> Option<&DeviceInfo> {
        self.devices.get(id.0)
    }

    /// Get the link between two devices.
    pub fn link(&self, from: DeviceId, to: DeviceId) -> Option<&Link> {
        self.links.get(&(from, to))
    }

    /// Get bandwidth (MB/s) between two devices, if known.
    pub fn bandwidth_mb_s(&self, from: DeviceId, to: DeviceId) -> Option<u64> {
        self.links.get(&(from, to)).map(|l| l.bandwidth_mb_s)
    }

    /// Estimate transfer time (µs) for `bytes` between two devices.
    pub fn transfer_time_us(&self, from: DeviceId, to: DeviceId, bytes: u64) -> Option<f64> {
        self.links.get(&(from, to))?.transfer_time_us(bytes)
    }

    /// All device IDs.
    pub fn device_ids(&self) -> Vec<DeviceId> {
        (0..self.devices.len()).map(DeviceId).collect()
    }

    /// Devices of a specific kind.
    pub fn devices_of_kind(&self, kind: DeviceKind) -> Vec<DeviceId> {
        self.devices
            .iter()
            .enumerate()
            .filter(|(_, d)| d.kind == kind)
            .map(|(i, _)| DeviceId(i))
            .collect()
    }

    /// Total memory across all devices of a given kind.
    pub fn total_memory(&self, kind: DeviceKind) -> u64 {
        self.devices
            .iter()
            .filter(|d| d.kind == kind)
            .map(|d| d.memory_bytes)
            .sum()
    }

    /// Find the link with the highest bandwidth from a given device.
    pub fn fastest_peer(&self, from: DeviceId) -> Option<(DeviceId, &Link)> {
        self.links
            .iter()
            .filter(|((src, dst), _)| *src == from && *dst != from)
            .max_by_key(|(_, l)| l.bandwidth_mb_s)
            .map(|((_, dst), l)| (*dst, l))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_gpu_topology() -> DeviceTopology {
        let mut topo = DeviceTopology::new();
        topo.add_device(DeviceInfo::new(0, DeviceKind::Cuda, "GPU 0")
            .with_memory_bytes(24_000_000_000));
        topo.add_device(DeviceInfo::new(1, DeviceKind::Cuda, "GPU 1")
            .with_memory_bytes(24_000_000_000));
        topo.add_link(DeviceId(0), DeviceId(1), Link::new(Interconnect::NvLink, 600_000));
        topo
    }

    #[test]
    fn add_devices_and_links() {
        let topo = two_gpu_topology();
        assert_eq!(topo.num_devices(), 2);
        assert_eq!(topo.device(DeviceId(0)).unwrap().name, "GPU 0");
    }

    #[test]
    fn bidirectional_link() {
        let topo = two_gpu_topology();
        assert_eq!(topo.bandwidth_mb_s(DeviceId(0), DeviceId(1)), Some(600_000));
        assert_eq!(topo.bandwidth_mb_s(DeviceId(1), DeviceId(0)), Some(600_000));
    }

    #[test]
    fn no_link() {
        let topo = two_gpu_topology();
        assert!(topo.bandwidth_mb_s(DeviceId(0), DeviceId(0)).is_none());
    }

    #[test]
    fn transfer_time() {
        let topo = two_gpu_topology();
        // 600 GB/s ≈ 600_000 MB/s; 1 MB transfer ≈ 1/600_000 s ≈ 1.67 µs
        let time = topo.transfer_time_us(DeviceId(0), DeviceId(1), 1024 * 1024).unwrap();
        assert!(time > 0.0 && time < 100.0);
    }

    #[test]
    fn devices_of_kind() {
        let mut topo = two_gpu_topology();
        topo.add_device(DeviceInfo::new(0, DeviceKind::Cpu, "Host"));
        assert_eq!(topo.devices_of_kind(DeviceKind::Cuda).len(), 2);
        assert_eq!(topo.devices_of_kind(DeviceKind::Cpu).len(), 1);
    }

    #[test]
    fn total_memory() {
        let topo = two_gpu_topology();
        assert_eq!(topo.total_memory(DeviceKind::Cuda), 48_000_000_000);
    }

    #[test]
    fn fastest_peer() {
        let mut topo = DeviceTopology::new();
        topo.add_device(DeviceInfo::new(0, DeviceKind::Cuda, "GPU 0"));
        topo.add_device(DeviceInfo::new(1, DeviceKind::Cuda, "GPU 1"));
        topo.add_device(DeviceInfo::new(2, DeviceKind::Cuda, "GPU 2"));
        topo.add_link(DeviceId(0), DeviceId(1), Link::new(Interconnect::Pcie, 32_000));
        topo.add_link(DeviceId(0), DeviceId(2), Link::new(Interconnect::NvLink, 600_000));

        let (peer, link) = topo.fastest_peer(DeviceId(0)).unwrap();
        assert_eq!(peer, DeviceId(2));
        assert_eq!(link.interconnect, Interconnect::NvLink);
    }

    #[test]
    fn directed_link() {
        let mut topo = DeviceTopology::new();
        topo.add_device(DeviceInfo::new(0, DeviceKind::Cuda, "GPU 0"));
        topo.add_device(DeviceInfo::new(1, DeviceKind::Cuda, "GPU 1"));
        topo.add_directed_link(DeviceId(0), DeviceId(1), Link::new(Interconnect::Pcie, 16_000));

        assert!(topo.link(DeviceId(0), DeviceId(1)).is_some());
        assert!(topo.link(DeviceId(1), DeviceId(0)).is_none());
    }

    #[test]
    fn link_latency() {
        let link = Link::new(Interconnect::Pcie, 32_000).with_latency_us(5);
        // 1 MB at 32_000 MB/s ≈ 31.25 µs + 5 µs latency
        let time = link.transfer_time_us(1024 * 1024).unwrap();
        assert!(time > 30.0 && time < 40.0);
    }

    #[test]
    fn device_builder_pattern() {
        let info = DeviceInfo::new(0, DeviceKind::Cuda, "RTX 4090")
            .with_memory_bytes(24_000_000_000)
            .with_gflops(82_600.0);
        assert_eq!(info.memory_bytes, 24_000_000_000);
        assert!((info.gflops - 82_600.0).abs() < 1.0);
    }

    #[test]
    fn empty_topology() {
        let topo = DeviceTopology::new();
        assert_eq!(topo.num_devices(), 0);
        assert!(topo.device_ids().is_empty());
    }
}
