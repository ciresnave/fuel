//! Span-level KV cache eviction.
//!
//! Rather than evicting individual token positions, **segmented eviction**
//! groups contiguous token ranges into named *spans* (e.g. a system prompt,
//! a conversation turn, a retrieved document chunk) and manages them as
//! indivisible units.  When the KV cache exceeds capacity, entire spans are
//! evicted based on a pluggable policy.
//!
//! # Motivation
//!
//! In multi-turn chat or RAG pipelines, evicting arbitrary individual tokens
//! can leave partial context that confuses the model.  Span-level eviction
//! preserves semantic coherence by always removing complete logical units.
//!
//! # Architecture
//!
//! [`SpanRegistry`] owns the span metadata.  Spans are registered when tokens
//! enter the KV cache and removed when evicted.  The registry does not own the
//! actual KV tensors — it produces an [`EvictionPlan`] describing *which* spans
//! (and therefore which position ranges) should be evicted.  The caller is
//! responsible for applying the plan to the underlying cache.
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  KV cache positions: 0 ──────────────────► 2048 │
//! │  [system][turn-1][doc-A][turn-2][doc-B][turn-3] │
//! │   pinned  ───────────────────────────────────    │
//! │                  evictable spans                 │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use candle_inference::segmented_eviction::{SpanRegistry, SpanKind};
//!
//! let mut reg = SpanRegistry::new(2048);
//!
//! // Register spans as tokens enter the cache
//! let sys = reg.register("system", SpanKind::System, 0..50);
//! let t1  = reg.register("turn-1", SpanKind::Turn, 50..200);
//! let d1  = reg.register("doc-a",  SpanKind::Document, 200..600);
//! let t2  = reg.register("turn-2", SpanKind::Turn, 600..900);
//!
//! // Need to free 500 positions
//! let plan = reg.plan_eviction(500);
//! assert!(plan.total_freed() >= 500);
//!
//! // Apply the plan (caller removes KV rows)
//! for span_id in plan.evicted_span_ids() {
//!     reg.remove(*span_id);
//! }
//! ```

use std::collections::HashMap;
use std::ops::Range;

/// Unique identifier for a registered span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId(u64);

impl SpanId {
    fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw numeric ID.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Classification of a span, used to influence eviction priority.
///
/// Lower-priority kinds are evicted before higher-priority ones (all else
/// being equal).  `System` spans are pinned by default and never evicted
/// unless explicitly unpinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpanKind {
    /// System prompt — pinned by default, highest priority.
    System,
    /// A user/assistant conversation turn.
    Turn,
    /// A retrieved document chunk (RAG context).
    Document,
    /// Tool call output.
    Tool,
    /// Generic / uncategorised.
    Other,
}

impl SpanKind {
    /// Default eviction priority (higher = evicted later).
    ///
    /// This is used as the initial priority when no explicit value is set.
    fn default_priority(&self) -> u32 {
        match self {
            SpanKind::System => u32::MAX, // pinned
            SpanKind::Turn => 100,
            SpanKind::Tool => 80,
            SpanKind::Document => 50,
            SpanKind::Other => 30,
        }
    }
}

/// Metadata for a single registered span.
#[derive(Debug, Clone)]
pub struct SpanInfo {
    /// Unique span ID.
    pub id: SpanId,
    /// Human-readable label.
    pub label: String,
    /// Span kind / classification.
    pub kind: SpanKind,
    /// Half-open token-position range `[start, end)`.
    pub range: Range<usize>,
    /// Eviction priority (higher = evicted later).  System spans default to
    /// `u32::MAX` (pinned).
    pub priority: u32,
    /// Whether this span is pinned (immune to eviction).
    pub pinned: bool,
    /// Monotonically increasing insertion order (for tie-breaking).
    pub insertion_order: u64,
}

impl SpanInfo {
    /// Returns the number of token positions in this span.
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Returns `true` if the span covers zero positions.
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }
}

/// Describes which spans should be evicted and how many positions are freed.
#[derive(Debug, Clone)]
pub struct EvictionPlan {
    /// Span IDs to evict, in eviction order.
    span_ids: Vec<SpanId>,
    /// Corresponding position ranges (parallel to `span_ids`).
    ranges: Vec<Range<usize>>,
    /// Total positions freed by this plan.
    total_freed: usize,
}

impl EvictionPlan {
    /// Returns the span IDs to evict, in eviction order.
    pub fn evicted_span_ids(&self) -> &[SpanId] {
        &self.span_ids
    }

    /// Returns the position ranges to evict, parallel to `evicted_span_ids`.
    pub fn evicted_ranges(&self) -> &[Range<usize>] {
        &self.ranges
    }

    /// Total number of token positions freed.
    pub fn total_freed(&self) -> usize {
        self.total_freed
    }

    /// Returns `true` if no spans need to be evicted.
    pub fn is_empty(&self) -> bool {
        self.span_ids.is_empty()
    }

    /// Number of spans in the plan.
    pub fn num_spans(&self) -> usize {
        self.span_ids.len()
    }
}

/// Tracks logical spans within the KV cache and produces eviction plans.
///
/// See the [module-level documentation](self) for details.
#[derive(Debug)]
pub struct SpanRegistry {
    /// Maximum KV cache capacity in token positions.
    capacity: usize,
    /// All registered spans, keyed by ID.
    spans: HashMap<SpanId, SpanInfo>,
    /// Next span ID to assign.
    next_id: u64,
    /// Monotonic insertion counter for tie-breaking.
    insertion_counter: u64,
}

impl SpanRegistry {
    /// Creates a new registry with the given KV cache capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            spans: HashMap::new(),
            next_id: 0,
            insertion_counter: 0,
        }
    }

    /// Registers a new span covering `range` token positions.
    ///
    /// Returns the assigned [`SpanId`].  System spans are pinned by default.
    pub fn register(&mut self, label: &str, kind: SpanKind, range: Range<usize>) -> SpanId {
        let id = SpanId::new(self.next_id);
        self.next_id += 1;
        let pinned = kind == SpanKind::System;
        let priority = kind.default_priority();
        let insertion_order = self.insertion_counter;
        self.insertion_counter += 1;

        self.spans.insert(
            id,
            SpanInfo {
                id,
                label: label.to_string(),
                kind,
                range,
                priority,
                pinned,
                insertion_order,
            },
        );

        id
    }

    /// Removes a span from the registry (e.g. after eviction).
    ///
    /// Returns the removed `SpanInfo`, or `None` if the ID was not found.
    pub fn remove(&mut self, id: SpanId) -> Option<SpanInfo> {
        self.spans.remove(&id)
    }

    /// Looks up a span by ID.
    pub fn get(&self, id: SpanId) -> Option<&SpanInfo> {
        self.spans.get(&id)
    }

    /// Returns the number of registered spans.
    pub fn num_spans(&self) -> usize {
        self.spans.len()
    }

    /// Returns the total number of token positions currently covered by all spans.
    pub fn total_positions(&self) -> usize {
        self.spans.values().map(|s| s.len()).sum()
    }

    /// Returns the KV cache capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of free positions (capacity minus occupied).
    pub fn free_positions(&self) -> usize {
        self.capacity.saturating_sub(self.total_positions())
    }

    /// Pins a span so it will not be evicted.
    pub fn pin(&mut self, id: SpanId) -> bool {
        if let Some(span) = self.spans.get_mut(&id) {
            span.pinned = true;
            true
        } else {
            false
        }
    }

    /// Unpins a span so it becomes eligible for eviction.
    pub fn unpin(&mut self, id: SpanId) -> bool {
        if let Some(span) = self.spans.get_mut(&id) {
            span.pinned = false;
            true
        } else {
            false
        }
    }

    /// Sets a custom eviction priority for a span.
    ///
    /// Higher values mean the span is evicted later.
    pub fn set_priority(&mut self, id: SpanId, priority: u32) -> bool {
        if let Some(span) = self.spans.get_mut(&id) {
            span.priority = priority;
            true
        } else {
            false
        }
    }

    /// Produces an eviction plan that frees at least `needed` positions.
    ///
    /// Eviction order:
    /// 1. Pinned spans are always excluded.
    /// 2. Among unpinned spans, lower priority is evicted first.
    /// 3. Ties broken by insertion order (oldest first — FIFO).
    ///
    /// If all unpinned spans together cannot free `needed` positions, the plan
    /// evicts as many as possible.
    pub fn plan_eviction(&self, needed: usize) -> EvictionPlan {
        if needed == 0 {
            return EvictionPlan {
                span_ids: Vec::new(),
                ranges: Vec::new(),
                total_freed: 0,
            };
        }

        // Collect unpinned spans and sort by eviction priority.
        let mut candidates: Vec<&SpanInfo> = self
            .spans
            .values()
            .filter(|s| !s.pinned)
            .collect();

        // Sort: lowest priority first, then oldest insertion order first.
        candidates.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then(a.insertion_order.cmp(&b.insertion_order))
        });

        let mut plan_ids = Vec::new();
        let mut plan_ranges = Vec::new();
        let mut freed = 0usize;

        for span in candidates {
            if freed >= needed {
                break;
            }
            plan_ids.push(span.id);
            plan_ranges.push(span.range.clone());
            freed += span.len();
        }

        EvictionPlan {
            span_ids: plan_ids,
            ranges: plan_ranges,
            total_freed: freed,
        }
    }

    /// Returns an iterator over all registered spans.
    pub fn iter(&self) -> impl Iterator<Item = &SpanInfo> {
        self.spans.values()
    }

    /// Returns all span IDs, sorted by position (start of range).
    pub fn span_ids_by_position(&self) -> Vec<SpanId> {
        let mut entries: Vec<_> = self.spans.values().collect();
        entries.sort_by_key(|s| s.range.start);
        entries.iter().map(|s| s.id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_lookup() {
        let mut reg = SpanRegistry::new(1024);
        let id = reg.register("sys", SpanKind::System, 0..50);

        let info = reg.get(id).unwrap();
        assert_eq!(info.label, "sys");
        assert_eq!(info.kind, SpanKind::System);
        assert_eq!(info.range, 0..50);
        assert!(info.pinned);
        assert_eq!(info.len(), 50);
    }

    #[test]
    fn basic_eviction_plan() {
        let mut reg = SpanRegistry::new(2048);
        let _sys = reg.register("system", SpanKind::System, 0..50);
        let _t1 = reg.register("turn-1", SpanKind::Turn, 50..200);
        let _d1 = reg.register("doc-a", SpanKind::Document, 200..600);
        let _t2 = reg.register("turn-2", SpanKind::Turn, 600..900);

        // Need 500 positions — doc-a (400 positions, priority 50) should go
        // first, then turn-1 (150, priority 100) if needed.
        let plan = reg.plan_eviction(500);
        assert!(plan.total_freed() >= 500);

        // doc-a has lowest priority → evicted first
        assert_eq!(plan.evicted_span_ids()[0], _d1);
        // doc-a alone is only 400, so turn-1 should also be evicted
        assert!(plan.num_spans() >= 2);
    }

    #[test]
    fn system_spans_are_pinned() {
        let mut reg = SpanRegistry::new(1024);
        let sys = reg.register("system", SpanKind::System, 0..100);
        let _t1 = reg.register("turn-1", SpanKind::Turn, 100..200);

        let plan = reg.plan_eviction(200);
        // System span should never be in the plan
        assert!(!plan.evicted_span_ids().contains(&sys));
        // Only turn-1 (100 positions) can be freed
        assert_eq!(plan.total_freed(), 100);
    }

    #[test]
    fn eviction_respects_priority_order() {
        let mut reg = SpanRegistry::new(2048);
        let _sys = reg.register("system", SpanKind::System, 0..50);
        let other = reg.register("misc", SpanKind::Other, 50..200);    // priority 30
        let doc = reg.register("doc", SpanKind::Document, 200..500);   // priority 50
        let tool = reg.register("tool", SpanKind::Tool, 500..700);     // priority 80
        let turn = reg.register("turn", SpanKind::Turn, 700..1000);    // priority 100

        // Need a lot — should evict in priority order: Other, Document, Tool, Turn
        let plan = reg.plan_eviction(2000);
        let ids = plan.evicted_span_ids();

        assert_eq!(ids[0], other);
        assert_eq!(ids[1], doc);
        assert_eq!(ids[2], tool);
        assert_eq!(ids[3], turn);
    }

    #[test]
    fn fifo_tiebreaking_within_same_priority() {
        let mut reg = SpanRegistry::new(2048);
        let t1 = reg.register("turn-1", SpanKind::Turn, 100..300);
        let t2 = reg.register("turn-2", SpanKind::Turn, 300..500);
        let _t3 = reg.register("turn-3", SpanKind::Turn, 500..700);

        let plan = reg.plan_eviction(250);
        // Same priority → oldest first (FIFO)
        assert_eq!(plan.evicted_span_ids()[0], t1);

        // If we need more, t2 follows t1
        let plan = reg.plan_eviction(500);
        assert_eq!(plan.evicted_span_ids()[0], t1);
        assert_eq!(plan.evicted_span_ids()[1], t2);
    }

    #[test]
    fn zero_needed_returns_empty() {
        let mut reg = SpanRegistry::new(1024);
        let _ = reg.register("turn-1", SpanKind::Turn, 0..100);

        let plan = reg.plan_eviction(0);
        assert!(plan.is_empty());
        assert_eq!(plan.total_freed(), 0);
    }

    #[test]
    fn remove_span() {
        let mut reg = SpanRegistry::new(1024);
        let id = reg.register("turn-1", SpanKind::Turn, 0..100);

        assert_eq!(reg.num_spans(), 1);
        assert_eq!(reg.total_positions(), 100);

        let removed = reg.remove(id).unwrap();
        assert_eq!(removed.label, "turn-1");

        assert_eq!(reg.num_spans(), 0);
        assert_eq!(reg.total_positions(), 0);
        assert!(reg.get(id).is_none());
    }

    #[test]
    fn pin_and_unpin() {
        let mut reg = SpanRegistry::new(1024);
        let id = reg.register("turn-1", SpanKind::Turn, 0..100);

        // Turns are not pinned by default
        assert!(!reg.get(id).unwrap().pinned);

        // Pin it — should be excluded from eviction
        reg.pin(id);
        assert!(reg.get(id).unwrap().pinned);

        let plan = reg.plan_eviction(100);
        assert!(plan.is_empty());

        // Unpin — should be evictable again
        reg.unpin(id);
        let plan = reg.plan_eviction(100);
        assert_eq!(plan.num_spans(), 1);
    }

    #[test]
    fn custom_priority() {
        let mut reg = SpanRegistry::new(2048);
        let t1 = reg.register("turn-1", SpanKind::Turn, 0..200);   // priority 100
        let t2 = reg.register("turn-2", SpanKind::Turn, 200..400); // priority 100

        // Boost turn-1 priority so turn-2 is evicted first
        reg.set_priority(t1, 200);

        let plan = reg.plan_eviction(250);
        assert_eq!(plan.evicted_span_ids()[0], t2);
    }

    #[test]
    fn free_positions_tracking() {
        let mut reg = SpanRegistry::new(1024);
        assert_eq!(reg.free_positions(), 1024);

        let id = reg.register("turn-1", SpanKind::Turn, 0..300);
        assert_eq!(reg.free_positions(), 724);

        reg.remove(id);
        assert_eq!(reg.free_positions(), 1024);
    }

    #[test]
    fn span_ids_by_position() {
        let mut reg = SpanRegistry::new(2048);
        let b = reg.register("b", SpanKind::Turn, 200..400);
        let a = reg.register("a", SpanKind::Turn, 0..200);
        let c = reg.register("c", SpanKind::Turn, 400..600);

        let sorted = reg.span_ids_by_position();
        assert_eq!(sorted, vec![a, b, c]);
    }

    #[test]
    fn eviction_plan_covers_ranges() {
        let mut reg = SpanRegistry::new(2048);
        let _ = reg.register("sys", SpanKind::System, 0..50);
        let _ = reg.register("doc", SpanKind::Document, 50..300);
        let _ = reg.register("turn", SpanKind::Turn, 300..600);

        let plan = reg.plan_eviction(400);
        // Should evict doc (250 positions) and turn (300 positions) = 550 freed
        // doc has lower priority so it goes first
        assert_eq!(plan.evicted_ranges()[0], 50..300);
    }

    #[test]
    fn insufficient_evictable_space() {
        let mut reg = SpanRegistry::new(1024);
        let _ = reg.register("sys", SpanKind::System, 0..800);   // pinned
        let _ = reg.register("turn", SpanKind::Turn, 800..900);  // 100 positions

        // Need 500 but only 100 are evictable
        let plan = reg.plan_eviction(500);
        assert_eq!(plan.total_freed(), 100);
        assert_eq!(plan.num_spans(), 1);
    }
}
