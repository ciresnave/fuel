//! Distributed KV cache coordination for multi-GPU inference.
//!
//! When running pipeline- or tensor-parallel inference, the KV cache for each
//! attention layer must be kept in sync across devices. This module provides
//! coordination types and protocols — **not** the actual cache storage (which
//! lives in [`candle_inference`]).
//!
//! ## Concepts
//!
//! - **Cache shard**: One device's portion of the KV cache, identified by rank.
//! - **Sync event**: A broadcast notification that a shard has been updated.
//! - **Distributed prefix**: For prefix caching, all ranks must agree on which
//!   prefix tokens are cached before reusing entries.
//!
//! # Example
//!
//! ```rust
//! use candle_parallel::distributed_cache::{
//!     CacheShardInfo, CacheSyncProtocol, SyncEvent,
//! };
//!
//! // Build shard info for rank 0 of 4
//! let shard = CacheShardInfo::new(0, 4, 32); // rank=0, world=4, num_layers=32
//! assert_eq!(shard.layers().len(), 8); // 32 layers / 4 ranks = 8 per shard
//! assert_eq!(shard.layers(), &[0, 1, 2, 3, 4, 5, 6, 7]);
//!
//! // Protocol tracks sync state
//! let mut protocol = CacheSyncProtocol::new(4, 32);
//! protocol.record_event(SyncEvent::ShardUpdated { rank: 0, layer: 3, seq_pos: 42 });
//! assert_eq!(protocol.latest_position(0, 3), Some(42));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identifies a shard of the distributed cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheShardInfo {
    /// Rank of the device holding this shard.
    rank: usize,
    /// Total number of devices.
    world_size: usize,
    /// Total number of attention layers in the model.
    num_layers: usize,
    /// Which layers this shard is responsible for (in pipeline-parallel mode).
    layers: Vec<usize>,
}

impl CacheShardInfo {
    /// Create a shard info that assigns layers evenly across ranks.
    ///
    /// Layers are assigned contiguously: rank 0 gets layers `[0..per_rank)`,
    /// rank 1 gets `[per_rank..2*per_rank)`, etc.
    pub fn new(rank: usize, world_size: usize, num_layers: usize) -> Self {
        assert!(world_size > 0, "world_size must be > 0");
        assert!(rank < world_size, "rank must be < world_size");
        let per_rank = num_layers / world_size;
        let remainder = num_layers % world_size;
        let start = per_rank * rank + rank.min(remainder);
        let count = per_rank + if rank < remainder { 1 } else { 0 };
        let layers: Vec<usize> = (start..start + count).collect();
        Self { rank, world_size, num_layers, layers }
    }

    /// Create a shard with explicit layer assignments.
    pub fn with_layers(rank: usize, world_size: usize, num_layers: usize, layers: Vec<usize>) -> Self {
        Self { rank, world_size, num_layers, layers }
    }

    /// Rank of this shard.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// World size.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Total layers in the model.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Layers this shard is responsible for.
    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    /// Whether this shard owns a given layer.
    pub fn owns_layer(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }
}

/// Cache synchronisation events exchanged between ranks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncEvent {
    /// A shard has appended KV entries up to `seq_pos`.
    ShardUpdated {
        rank: usize,
        layer: usize,
        seq_pos: usize,
    },
    /// A prefix has been confirmed as shared across all ranks.
    PrefixConfirmed {
        /// Token hash or prefix ID.
        prefix_id: u64,
        /// Number of tokens in the prefix.
        num_tokens: usize,
    },
    /// A shard has evicted entries, invalidating positions beyond `seq_pos`.
    ShardEvicted {
        rank: usize,
        layer: usize,
        /// All positions > seq_pos are now invalid.
        seq_pos: usize,
    },
    /// All shards should flush their caches (e.g., on model swap).
    FlushAll,
}

/// Tracks distributed cache state across all ranks.
///
/// This is a metadata-only coordinator. It does not hold actual tensors —
/// tensors live in the inference engine's KV cache on each device.
#[derive(Debug, Clone)]
pub struct CacheSyncProtocol {
    /// World size.
    world_size: usize,
    /// Total layers.
    _num_layers: usize,
    /// Latest known sequence position per `(rank, layer)`.
    positions: HashMap<(usize, usize), usize>,
    /// Confirmed shared prefix IDs and their token counts.
    confirmed_prefixes: HashMap<u64, usize>,
    /// Event log (bounded).
    event_log: Vec<SyncEvent>,
    /// Maximum number of events to retain.
    max_log_size: usize,
}

impl CacheSyncProtocol {
    /// Create a new protocol tracker.
    pub fn new(world_size: usize, num_layers: usize) -> Self {
        Self {
            world_size,
            _num_layers: num_layers,
            positions: HashMap::new(),
            confirmed_prefixes: HashMap::new(),
            event_log: Vec::new(),
            max_log_size: 10_000,
        }
    }

    /// Set the maximum event log size (oldest events are dropped).
    pub fn with_max_log_size(mut self, max: usize) -> Self {
        self.max_log_size = max;
        self
    }

    /// Record a synchronisation event.
    pub fn record_event(&mut self, event: SyncEvent) {
        match &event {
            SyncEvent::ShardUpdated { rank, layer, seq_pos } => {
                self.positions.insert((*rank, *layer), *seq_pos);
            }
            SyncEvent::PrefixConfirmed { prefix_id, num_tokens } => {
                self.confirmed_prefixes.insert(*prefix_id, *num_tokens);
            }
            SyncEvent::ShardEvicted { rank, layer, seq_pos } => {
                self.positions.insert((*rank, *layer), *seq_pos);
            }
            SyncEvent::FlushAll => {
                self.positions.clear();
                self.confirmed_prefixes.clear();
            }
        }
        self.event_log.push(event);
        if self.event_log.len() > self.max_log_size {
            let drain = self.event_log.len() - self.max_log_size;
            self.event_log.drain(..drain);
        }
    }

    /// Latest known sequence position for a (rank, layer).
    pub fn latest_position(&self, rank: usize, layer: usize) -> Option<usize> {
        self.positions.get(&(rank, layer)).copied()
    }

    /// Minimum sequence position across all ranks for a given layer.
    ///
    /// Returns `None` if any rank hasn't reported for that layer.
    /// This is the "safe" position up to which all ranks are in sync.
    pub fn min_synced_position(&self, layer: usize) -> Option<usize> {
        let mut min_pos: Option<usize> = None;
        for rank in 0..self.world_size {
            match self.positions.get(&(rank, layer)) {
                Some(&pos) => {
                    min_pos = Some(min_pos.map_or(pos, |m: usize| m.min(pos)));
                }
                None => return None,
            }
        }
        min_pos
    }

    /// Whether a prefix is confirmed across all ranks.
    pub fn is_prefix_confirmed(&self, prefix_id: u64) -> bool {
        self.confirmed_prefixes.contains_key(&prefix_id)
    }

    /// Confirmed prefix token count—`None` if not confirmed.
    pub fn prefix_tokens(&self, prefix_id: u64) -> Option<usize> {
        self.confirmed_prefixes.get(&prefix_id).copied()
    }

    /// Number of events recorded.
    pub fn event_count(&self) -> usize {
        self.event_log.len()
    }

    /// Recent events (up to `n`).
    pub fn recent_events(&self, n: usize) -> &[SyncEvent] {
        let start = self.event_log.len().saturating_sub(n);
        &self.event_log[start..]
    }
}

/// Describes which ranks need cache data for a given request.
///
/// Used by the inference scheduler to decide which ranks to notify
/// when a new request arrives that can reuse cached prefixes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheRoutingHint {
    /// The prefix ID to look up.
    pub prefix_id: u64,
    /// Ranks that hold the relevant cache shards.
    pub relevant_ranks: Vec<usize>,
    /// Number of tokens that can be skipped (reused from cache).
    pub reusable_tokens: usize,
}

impl CacheRoutingHint {
    /// Create a routing hint.
    pub fn new(prefix_id: u64, relevant_ranks: Vec<usize>, reusable_tokens: usize) -> Self {
        Self { prefix_id, relevant_ranks, reusable_tokens }
    }

    /// Whether this hint has any ranks to notify.
    pub fn has_targets(&self) -> bool {
        !self.relevant_ranks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_info_even_split() {
        let shard = CacheShardInfo::new(0, 4, 32);
        assert_eq!(shard.layers().len(), 8);
        assert_eq!(shard.layers(), &[0, 1, 2, 3, 4, 5, 6, 7]);

        let shard3 = CacheShardInfo::new(3, 4, 32);
        assert_eq!(shard3.layers(), &[24, 25, 26, 27, 28, 29, 30, 31]);
    }

    #[test]
    fn shard_info_uneven_split() {
        let shard0 = CacheShardInfo::new(0, 3, 10);
        let shard1 = CacheShardInfo::new(1, 3, 10);
        let shard2 = CacheShardInfo::new(2, 3, 10);
        // 10/3 = 3r1 → first rank gets 4, rest get 3
        assert_eq!(shard0.layers().len(), 4);
        assert_eq!(shard1.layers().len(), 3);
        assert_eq!(shard2.layers().len(), 3);
        // No overlap
        let mut all: Vec<usize> = Vec::new();
        all.extend(shard0.layers());
        all.extend(shard1.layers());
        all.extend(shard2.layers());
        all.sort();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn owns_layer() {
        let shard = CacheShardInfo::new(1, 2, 8);
        assert!(!shard.owns_layer(0)); // layer 0 belongs to rank 0
        assert!(shard.owns_layer(4));  // layer 4 belongs to rank 1
    }

    #[test]
    fn protocol_track_updates() {
        let mut proto = CacheSyncProtocol::new(2, 4);
        proto.record_event(SyncEvent::ShardUpdated { rank: 0, layer: 0, seq_pos: 10 });
        proto.record_event(SyncEvent::ShardUpdated { rank: 1, layer: 0, seq_pos: 8 });
        assert_eq!(proto.latest_position(0, 0), Some(10));
        assert_eq!(proto.latest_position(1, 0), Some(8));
        assert_eq!(proto.min_synced_position(0), Some(8));
    }

    #[test]
    fn protocol_min_synced_none_if_missing() {
        let mut proto = CacheSyncProtocol::new(3, 2);
        proto.record_event(SyncEvent::ShardUpdated { rank: 0, layer: 0, seq_pos: 5 });
        // rank 1 and 2 haven't reported
        assert_eq!(proto.min_synced_position(0), None);
    }

    #[test]
    fn protocol_prefix_confirmation() {
        let mut proto = CacheSyncProtocol::new(2, 4);
        assert!(!proto.is_prefix_confirmed(0xABCD));
        proto.record_event(SyncEvent::PrefixConfirmed { prefix_id: 0xABCD, num_tokens: 64 });
        assert!(proto.is_prefix_confirmed(0xABCD));
        assert_eq!(proto.prefix_tokens(0xABCD), Some(64));
    }

    #[test]
    fn protocol_flush_all() {
        let mut proto = CacheSyncProtocol::new(2, 4);
        proto.record_event(SyncEvent::ShardUpdated { rank: 0, layer: 0, seq_pos: 10 });
        proto.record_event(SyncEvent::PrefixConfirmed { prefix_id: 1, num_tokens: 5 });
        proto.record_event(SyncEvent::FlushAll);
        assert_eq!(proto.latest_position(0, 0), None);
        assert!(!proto.is_prefix_confirmed(1));
    }

    #[test]
    fn protocol_log_bounded() {
        let mut proto = CacheSyncProtocol::new(1, 1).with_max_log_size(3);
        for i in 0..10 {
            proto.record_event(SyncEvent::ShardUpdated { rank: 0, layer: 0, seq_pos: i });
        }
        assert_eq!(proto.event_count(), 3);
    }

    #[test]
    fn protocol_recent_events() {
        let mut proto = CacheSyncProtocol::new(1, 1);
        proto.record_event(SyncEvent::ShardUpdated { rank: 0, layer: 0, seq_pos: 1 });
        proto.record_event(SyncEvent::ShardUpdated { rank: 0, layer: 0, seq_pos: 2 });
        proto.record_event(SyncEvent::FlushAll);
        let recent = proto.recent_events(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[1], SyncEvent::FlushAll);
    }

    #[test]
    fn cache_routing_hint() {
        let hint = CacheRoutingHint::new(42, vec![0, 2], 128);
        assert!(hint.has_targets());
        assert_eq!(hint.reusable_tokens, 128);
    }

    #[test]
    fn cache_routing_hint_empty() {
        let hint = CacheRoutingHint::new(0, vec![], 0);
        assert!(!hint.has_targets());
    }
}
