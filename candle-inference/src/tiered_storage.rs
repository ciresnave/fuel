//! Tiered KV cache storage: GPU → CPU → Disk.
//!
//! When KV cache memory on the GPU is scarce, cold segments can be *demoted*
//! to cheaper storage tiers and *promoted* back when needed.  This module
//! provides the metadata tracking and policy logic — it does **not** move
//! actual tensors (that responsibility belongs to the caller / runtime).
//!
//! # Tiers
//!
//! | Tier | Storage | Latency | Cost |
//! |------|---------|---------|------|
//! | `Gpu` | VRAM | ~µs | High |
//! | `Cpu` | System RAM | ~ms | Medium |
//! | `Disk` | Filesystem | ~10ms | Low |
//!
//! # Architecture
//!
//! [`TieredStore`] tracks which *segments* (identified by a string key) live
//! on which tier, along with their size and position metadata.  Callers ask
//! the store to [`demote`](TieredStore::demote) or
//! [`promote`](TieredStore::promote) segments; the store updates bookkeeping
//! and returns a [`TierTransfer`] describing the move the caller must execute.
//!
//! Position IDs are preserved across tier transitions so that RoPE / ALiBi
//! positional embeddings remain correct when a segment is promoted back.
//!
//! # Example
//!
//! ```rust
//! use candle_inference::tiered_storage::{TieredStore, Tier, SegmentMeta};
//!
//! let mut store = TieredStore::new(1_000_000, 8_000_000); // 1 MB GPU, 8 MB CPU
//!
//! // Register a segment on GPU
//! let meta = SegmentMeta::new("turn-1", 0..512, 200_000);
//! store.register(meta, Tier::Gpu);
//!
//! assert_eq!(store.gpu_used(), 200_000);
//!
//! // Demote to CPU
//! let transfer = store.demote("turn-1", Tier::Cpu).unwrap();
//! assert_eq!(transfer.from, Tier::Gpu);
//! assert_eq!(transfer.to, Tier::Cpu);
//! assert_eq!(store.gpu_used(), 0);
//! assert_eq!(store.cpu_used(), 200_000);
//!
//! // Promote back to GPU
//! let transfer = store.promote("turn-1", Tier::Gpu).unwrap();
//! assert_eq!(store.gpu_used(), 200_000);
//! ```

use std::collections::HashMap;
use std::ops::Range;

/// Storage tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    /// GPU VRAM (fastest, most expensive).
    Gpu = 0,
    /// System RAM (medium speed/cost).
    Cpu = 1,
    /// Disk / filesystem (slowest, cheapest).
    Disk = 2,
}

impl Tier {
    /// Returns `true` if `self` is a faster (higher) tier than `other`.
    pub fn is_faster_than(self, other: Self) -> bool {
        (self as u8) < (other as u8)
    }
}

/// Metadata for a stored KV segment.
#[derive(Debug, Clone)]
pub struct SegmentMeta {
    /// Unique segment key (e.g. `"turn-3"`, `"doc-chunk-7"`).
    pub key: String,
    /// Token position range `[start, end)` — preserved across tier moves
    /// for correct positional embedding re-injection.
    pub position_range: Range<usize>,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Current tier.
    pub tier: Tier,
    /// Access counter (incremented on promote / explicit touch).
    pub access_count: u64,
}

impl SegmentMeta {
    /// Create new segment metadata (tier is set when registered).
    pub fn new(key: impl Into<String>, position_range: Range<usize>, size_bytes: usize) -> Self {
        Self {
            key: key.into(),
            position_range,
            size_bytes,
            tier: Tier::Gpu,
            access_count: 0,
        }
    }

    /// Number of token positions in this segment.
    pub fn num_positions(&self) -> usize {
        self.position_range.len()
    }
}

/// Describes a tier transfer the caller must execute.
#[derive(Debug, Clone)]
pub struct TierTransfer {
    /// Segment key.
    pub key: String,
    /// Source tier.
    pub from: Tier,
    /// Destination tier.
    pub to: Tier,
    /// Bytes to move.
    pub size_bytes: usize,
    /// Position range (preserved — caller must store positions with the data).
    pub position_range: Range<usize>,
}

/// Tiered KV cache storage manager.
///
/// Tracks segment placement across GPU, CPU, and disk tiers with per-tier
/// byte budgets.
#[derive(Debug)]
pub struct TieredStore {
    /// GPU budget in bytes.
    gpu_budget: usize,
    /// CPU budget in bytes.
    cpu_budget: usize,
    /// Per-tier current usage.
    gpu_used: usize,
    cpu_used: usize,
    disk_used: usize,
    /// All segments keyed by name.
    segments: HashMap<String, SegmentMeta>,
}

impl TieredStore {
    /// Create a new tiered store with GPU and CPU byte budgets.
    /// Disk is unbounded.
    pub fn new(gpu_budget: usize, cpu_budget: usize) -> Self {
        Self {
            gpu_budget,
            cpu_budget,
            gpu_used: 0,
            cpu_used: 0,
            disk_used: 0,
            segments: HashMap::new(),
        }
    }

    /// Register a new segment at the given tier.
    ///
    /// Returns `false` if the segment key already exists or the tier has
    /// insufficient budget.
    pub fn register(&mut self, mut meta: SegmentMeta, tier: Tier) -> bool {
        if self.segments.contains_key(&meta.key) {
            return false;
        }
        if !self.has_budget(tier, meta.size_bytes) {
            return false;
        }
        meta.tier = tier;
        self.add_usage(tier, meta.size_bytes);
        self.segments.insert(meta.key.clone(), meta);
        true
    }

    /// Demote a segment to a lower (slower) tier.
    ///
    /// Returns the transfer descriptor, or `None` if the segment doesn't
    /// exist, is already at or below the target tier, or the target tier
    /// lacks budget.
    pub fn demote(&mut self, key: &str, to: Tier) -> Option<TierTransfer> {
        let seg = self.segments.get(key)?;
        let from = seg.tier;
        let size = seg.size_bytes;
        let pos = seg.position_range.clone();

        // Can only demote to equal-or-slower tier
        if to.is_faster_than(from) || to == from {
            return None;
        }

        if !self.has_budget(to, size) {
            return None;
        }

        let transfer = TierTransfer {
            key: key.to_string(),
            from,
            to,
            size_bytes: size,
            position_range: pos,
        };

        self.sub_usage(from, size);
        self.add_usage(to, size);
        let seg = self.segments.get_mut(key).unwrap();
        seg.tier = to;

        Some(transfer)
    }

    /// Promote a segment to a higher (faster) tier.
    ///
    /// Increments the segment's access counter. Returns the transfer
    /// descriptor, or `None` on failure.
    pub fn promote(&mut self, key: &str, to: Tier) -> Option<TierTransfer> {
        let seg = self.segments.get(key)?;
        let from = seg.tier;
        let size = seg.size_bytes;
        let pos = seg.position_range.clone();

        if from.is_faster_than(to) || from == to {
            return None;
        }

        if !self.has_budget(to, size) {
            return None;
        }

        let transfer = TierTransfer {
            key: key.to_string(),
            from,
            to,
            size_bytes: size,
            position_range: pos,
        };

        self.sub_usage(from, size);
        self.add_usage(to, size);
        let seg = self.segments.get_mut(key).unwrap();
        seg.tier = to;
        seg.access_count += 1;

        Some(transfer)
    }

    /// Remove a segment entirely (e.g., after eviction).
    pub fn remove(&mut self, key: &str) -> Option<SegmentMeta> {
        let seg = self.segments.remove(key)?;
        self.sub_usage(seg.tier, seg.size_bytes);
        Some(seg)
    }

    /// Look up a segment.
    pub fn get(&self, key: &str) -> Option<&SegmentMeta> {
        self.segments.get(key)
    }

    /// Find segments that should be demoted to free `needed` bytes on `tier`.
    ///
    /// Returns keys of candidate segments sorted by access count (least
    /// accessed first), stopping once the freed total ≥ `needed`.
    pub fn candidates_for_demotion(&self, tier: Tier, needed: usize) -> Vec<String> {
        let mut on_tier: Vec<&SegmentMeta> = self
            .segments
            .values()
            .filter(|s| s.tier == tier)
            .collect();

        // Least-accessed first
        on_tier.sort_by_key(|s| s.access_count);

        let mut freed = 0usize;
        let mut keys = Vec::new();
        for seg in on_tier {
            if freed >= needed {
                break;
            }
            keys.push(seg.key.clone());
            freed += seg.size_bytes;
        }
        keys
    }

    /// Touch a segment (increment access count) without moving it.
    pub fn touch(&mut self, key: &str) -> bool {
        if let Some(seg) = self.segments.get_mut(key) {
            seg.access_count += 1;
            true
        } else {
            false
        }
    }

    // ── Capacity queries ──────────────────────────────────────────────

    pub fn gpu_budget(&self) -> usize { self.gpu_budget }
    pub fn cpu_budget(&self) -> usize { self.cpu_budget }
    pub fn gpu_used(&self) -> usize { self.gpu_used }
    pub fn cpu_used(&self) -> usize { self.cpu_used }
    pub fn disk_used(&self) -> usize { self.disk_used }
    pub fn gpu_free(&self) -> usize { self.gpu_budget.saturating_sub(self.gpu_used) }
    pub fn cpu_free(&self) -> usize { self.cpu_budget.saturating_sub(self.cpu_used) }
    pub fn num_segments(&self) -> usize { self.segments.len() }

    /// Iterate over all segments.
    pub fn iter(&self) -> impl Iterator<Item = &SegmentMeta> {
        self.segments.values()
    }

    // ── Internal helpers ──────────────────────────────────────────────

    fn has_budget(&self, tier: Tier, bytes: usize) -> bool {
        match tier {
            Tier::Gpu => self.gpu_used + bytes <= self.gpu_budget,
            Tier::Cpu => self.cpu_used + bytes <= self.cpu_budget,
            Tier::Disk => true, // unbounded
        }
    }

    fn add_usage(&mut self, tier: Tier, bytes: usize) {
        match tier {
            Tier::Gpu => self.gpu_used += bytes,
            Tier::Cpu => self.cpu_used += bytes,
            Tier::Disk => self.disk_used += bytes,
        }
    }

    fn sub_usage(&mut self, tier: Tier, bytes: usize) {
        match tier {
            Tier::Gpu => self.gpu_used = self.gpu_used.saturating_sub(bytes),
            Tier::Cpu => self.cpu_used = self.cpu_used.saturating_sub(bytes),
            Tier::Disk => self.disk_used = self.disk_used.saturating_sub(bytes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_lookup() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        let meta = SegmentMeta::new("s1", 0..256, 100_000);
        assert!(store.register(meta, Tier::Gpu));

        let seg = store.get("s1").unwrap();
        assert_eq!(seg.tier, Tier::Gpu);
        assert_eq!(seg.size_bytes, 100_000);
        assert_eq!(seg.position_range, 0..256);
        assert_eq!(store.gpu_used(), 100_000);
    }

    #[test]
    fn duplicate_key_rejected() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        store.register(SegmentMeta::new("s1", 0..10, 100), Tier::Gpu);
        assert!(!store.register(SegmentMeta::new("s1", 10..20, 200), Tier::Gpu));
    }

    #[test]
    fn budget_overflow_rejected() {
        let mut store = TieredStore::new(1000, 5000);
        assert!(!store.register(SegmentMeta::new("big", 0..100, 2000), Tier::Gpu));
        assert_eq!(store.gpu_used(), 0);
    }

    #[test]
    fn demote_gpu_to_cpu() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        store.register(SegmentMeta::new("s1", 0..512, 200_000), Tier::Gpu);

        let transfer = store.demote("s1", Tier::Cpu).unwrap();
        assert_eq!(transfer.from, Tier::Gpu);
        assert_eq!(transfer.to, Tier::Cpu);
        assert_eq!(transfer.position_range, 0..512);
        assert_eq!(store.gpu_used(), 0);
        assert_eq!(store.cpu_used(), 200_000);
        assert_eq!(store.get("s1").unwrap().tier, Tier::Cpu);
    }

    #[test]
    fn demote_cpu_to_disk() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        store.register(SegmentMeta::new("s1", 0..100, 50_000), Tier::Cpu);

        let transfer = store.demote("s1", Tier::Disk).unwrap();
        assert_eq!(transfer.from, Tier::Cpu);
        assert_eq!(transfer.to, Tier::Disk);
        assert_eq!(store.cpu_used(), 0);
        assert_eq!(store.disk_used(), 50_000);
    }

    #[test]
    fn demote_to_same_tier_fails() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        store.register(SegmentMeta::new("s1", 0..100, 50_000), Tier::Gpu);
        assert!(store.demote("s1", Tier::Gpu).is_none());
    }

    #[test]
    fn demote_to_faster_tier_fails() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        store.register(SegmentMeta::new("s1", 0..100, 50_000), Tier::Cpu);
        assert!(store.demote("s1", Tier::Gpu).is_none());
    }

    #[test]
    fn promote_cpu_to_gpu() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        store.register(SegmentMeta::new("s1", 100..200, 80_000), Tier::Cpu);

        let transfer = store.promote("s1", Tier::Gpu).unwrap();
        assert_eq!(transfer.from, Tier::Cpu);
        assert_eq!(transfer.to, Tier::Gpu);
        assert_eq!(store.cpu_used(), 0);
        assert_eq!(store.gpu_used(), 80_000);
        assert_eq!(store.get("s1").unwrap().access_count, 1);
    }

    #[test]
    fn promote_increments_access_count() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        store.register(SegmentMeta::new("s1", 0..10, 1000), Tier::Disk);

        store.promote("s1", Tier::Cpu);
        assert_eq!(store.get("s1").unwrap().access_count, 1);

        store.promote("s1", Tier::Gpu);
        assert_eq!(store.get("s1").unwrap().access_count, 2);
    }

    #[test]
    fn promote_to_slower_fails() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        store.register(SegmentMeta::new("s1", 0..10, 1000), Tier::Gpu);
        assert!(store.promote("s1", Tier::Cpu).is_none());
    }

    #[test]
    fn remove_frees_budget() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        store.register(SegmentMeta::new("s1", 0..100, 50_000), Tier::Gpu);
        assert_eq!(store.gpu_used(), 50_000);

        let removed = store.remove("s1").unwrap();
        assert_eq!(removed.key, "s1");
        assert_eq!(store.gpu_used(), 0);
        assert_eq!(store.num_segments(), 0);
    }

    #[test]
    fn candidates_for_demotion_by_access_count() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        store.register(SegmentMeta::new("hot", 0..100, 30_000), Tier::Gpu);
        store.register(SegmentMeta::new("cold", 100..200, 30_000), Tier::Gpu);
        store.register(SegmentMeta::new("warm", 200..300, 30_000), Tier::Gpu);

        // Touch "hot" 3 times, "warm" 1 time
        store.touch("hot");
        store.touch("hot");
        store.touch("hot");
        store.touch("warm");

        let candidates = store.candidates_for_demotion(Tier::Gpu, 50_000);
        // "cold" (0 accesses) first, then "warm" (1 access)
        assert_eq!(candidates[0], "cold");
        assert_eq!(candidates[1], "warm");
    }

    #[test]
    fn disk_is_unbounded() {
        let mut store = TieredStore::new(100, 100);
        // Disk has no budget limit
        assert!(store.register(SegmentMeta::new("huge", 0..10000, 999_999_999), Tier::Disk));
        assert_eq!(store.disk_used(), 999_999_999);
    }

    #[test]
    fn position_range_preserved_across_tiers() {
        let mut store = TieredStore::new(1_000_000, 8_000_000);
        store.register(SegmentMeta::new("s1", 42..99, 10_000), Tier::Gpu);

        store.demote("s1", Tier::Cpu);
        assert_eq!(store.get("s1").unwrap().position_range, 42..99);

        store.demote("s1", Tier::Disk);
        assert_eq!(store.get("s1").unwrap().position_range, 42..99);

        store.promote("s1", Tier::Cpu);
        assert_eq!(store.get("s1").unwrap().position_range, 42..99);
    }

    #[test]
    fn promote_blocked_by_budget() {
        let mut store = TieredStore::new(100, 8_000_000);
        store.register(SegmentMeta::new("s1", 0..10, 200), Tier::Cpu);

        // GPU budget is 100, segment is 200 → can't promote
        assert!(store.promote("s1", Tier::Gpu).is_none());
        assert_eq!(store.get("s1").unwrap().tier, Tier::Cpu);
    }

    #[test]
    fn tier_ordering() {
        assert!(Tier::Gpu.is_faster_than(Tier::Cpu));
        assert!(Tier::Cpu.is_faster_than(Tier::Disk));
        assert!(Tier::Gpu.is_faster_than(Tier::Disk));
        assert!(!Tier::Disk.is_faster_than(Tier::Gpu));
        assert!(!Tier::Gpu.is_faster_than(Tier::Gpu));
    }
}
