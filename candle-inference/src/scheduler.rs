//! Memory-aware inference scheduler.
//!
//! In a serving context, multiple requests compete for a fixed KV cache memory
//! budget.  A naive approach admits every request immediately, which can cause
//! OOM or cascading eviction.  The **memory-aware scheduler** assigns each
//! request a *slot* with a tracked memory cost, and uses a priority queue to
//! decide admission order.  When the remaining budget is low (above a
//! configurable *eviction-pressure* threshold), lower-priority requests are
//! queued rather than admitted.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────┐  admit()  ┌────────────────────┐
//! │  Incoming  ├─────────►│  MemoryScheduler    │
//! │  Requests  │          │  ┌──────────────┐   │
//! └───────────┘          │  │ Priority Q    │   │──► active slots
//!                        │  │ (wait queue)  │   │
//!                        │  └──────────────┘   │
//!                        │  budget tracking     │
//!                        └────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use candle_inference::scheduler::{MemoryScheduler, Priority, RequestInfo};
//!
//! let mut sched = MemoryScheduler::new(1_000_000); // 1 MB budget
//!
//! let req = RequestInfo::new("req-1", 50_000, Priority::Normal);
//! let admitted = sched.try_admit(req);
//! assert!(admitted.is_some());
//!
//! assert!(sched.used_bytes() == 50_000);
//! assert!(sched.active_count() == 1);
//!
//! // Release when done
//! sched.release("req-1");
//! assert!(sched.active_count() == 0);
//! ```

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

/// Priority level for an inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Priority {
    /// Background / best-effort.
    Low = 0,
    /// Default priority.
    Normal = 1,
    /// Elevated priority (premium users, short requests).
    High = 2,
    /// Must-serve (health checks, system prompts).
    Critical = 3,
}

impl Priority {
    fn rank(self) -> u8 {
        self as u8
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank().cmp(&other.rank())
    }
}

/// Information about a pending or active request.
#[derive(Debug, Clone)]
pub struct RequestInfo {
    /// Unique request ID.
    pub id: String,
    /// Estimated memory cost in bytes (base + projected KV growth).
    pub estimated_bytes: usize,
    /// Request priority.
    pub priority: Priority,
    /// Monotonic submission order for FIFO tie-breaking.
    submission_order: u64,
}

impl RequestInfo {
    /// Create a new request.
    pub fn new(id: impl Into<String>, estimated_bytes: usize, priority: Priority) -> Self {
        Self {
            id: id.into(),
            estimated_bytes,
            priority,
            submission_order: 0, // set by scheduler
        }
    }
}

// For BinaryHeap: higher priority first, then earlier submission (lower order).
impl PartialEq for RequestInfo {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RequestInfo {}

impl PartialOrd for RequestInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RequestInfo {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then(other.submission_order.cmp(&self.submission_order)) // earlier = higher
    }
}

/// Handle returned when a request is admitted.
#[derive(Debug, Clone)]
pub struct SlotHandle {
    /// Request ID.
    pub id: String,
    /// Actual bytes reserved.
    pub reserved_bytes: usize,
}

/// Memory-aware inference scheduler.
///
/// Tracks a byte budget, active slots, and a wait queue for requests that
/// cannot be admitted immediately.
#[derive(Debug)]
pub struct MemoryScheduler {
    /// Total memory budget in bytes.
    total_budget: usize,
    /// Currently used bytes across all active slots.
    used: usize,
    /// Active slots: request ID → reserved bytes.
    active: HashMap<String, usize>,
    /// Wait queue (max-heap by priority, FIFO within same priority).
    wait_queue: BinaryHeap<RequestInfo>,
    /// Eviction-pressure threshold (fraction 0.0–1.0).  When usage exceeds
    /// `total_budget * threshold`, only `High`/`Critical` requests are admitted.
    pressure_threshold: f64,
    /// Submission counter for FIFO ordering.
    next_order: u64,
}

impl MemoryScheduler {
    /// Create a new scheduler with the given byte budget.
    pub fn new(total_budget: usize) -> Self {
        Self {
            total_budget,
            used: 0,
            active: HashMap::new(),
            wait_queue: BinaryHeap::new(),
            pressure_threshold: 0.8,
            next_order: 0,
        }
    }

    /// Set the eviction-pressure threshold (0.0–1.0).
    ///
    /// When usage exceeds `budget * threshold`, only `High` and `Critical`
    /// priority requests are admitted directly; others go to the wait queue.
    pub fn with_pressure_threshold(mut self, threshold: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be in [0.0, 1.0]"
        );
        self.pressure_threshold = threshold;
        self
    }

    /// Try to admit a request immediately.
    ///
    /// Returns `Some(SlotHandle)` if admitted, `None` if the request was
    /// queued (insufficient budget or pressure-gated).
    pub fn try_admit(&mut self, mut req: RequestInfo) -> Option<SlotHandle> {
        req.submission_order = self.next_order;
        self.next_order += 1;

        // Check if under pressure
        let under_pressure = self.usage_fraction() >= self.pressure_threshold;

        if under_pressure && req.priority < Priority::High {
            // Queue it
            self.wait_queue.push(req);
            return None;
        }

        // Check if enough budget
        if self.used + req.estimated_bytes > self.total_budget {
            self.wait_queue.push(req);
            return None;
        }

        // Admit
        let handle = SlotHandle {
            id: req.id.clone(),
            reserved_bytes: req.estimated_bytes,
        };
        self.used += req.estimated_bytes;
        self.active.insert(req.id, req.estimated_bytes);
        Some(handle)
    }

    /// Release a slot, freeing its reserved bytes.
    ///
    /// Returns `true` if the slot existed and was released.
    pub fn release(&mut self, id: &str) -> bool {
        if let Some(bytes) = self.active.remove(id) {
            self.used = self.used.saturating_sub(bytes);
            true
        } else {
            false
        }
    }

    /// Try to drain the wait queue, admitting as many queued requests as fit.
    ///
    /// Returns the handles of newly admitted requests.
    pub fn drain_queue(&mut self) -> Vec<SlotHandle> {
        let mut admitted = Vec::new();
        let mut remaining = BinaryHeap::new();

        while let Some(req) = self.wait_queue.pop() {
            let under_pressure = self.usage_fraction() >= self.pressure_threshold;
            let pressure_blocked = under_pressure && req.priority < Priority::High;

            if !pressure_blocked && self.used + req.estimated_bytes <= self.total_budget {
                let handle = SlotHandle {
                    id: req.id.clone(),
                    reserved_bytes: req.estimated_bytes,
                };
                self.used += req.estimated_bytes;
                self.active.insert(req.id, req.estimated_bytes);
                admitted.push(handle);
            } else {
                remaining.push(req);
            }
        }

        self.wait_queue = remaining;
        admitted
    }

    /// Update the reserved bytes for an active slot (e.g., as KV cache grows).
    ///
    /// Returns `true` if the update succeeded (slot exists and new size fits).
    pub fn update_usage(&mut self, id: &str, new_bytes: usize) -> bool {
        if let Some(old_bytes) = self.active.get_mut(id) {
            let old = *old_bytes;
            let new_used = self.used - old + new_bytes;
            if new_used > self.total_budget {
                return false;
            }
            self.used = new_used;
            *old_bytes = new_bytes;
            true
        } else {
            false
        }
    }

    /// Number of active slots.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Number of queued (waiting) requests.
    pub fn queued_count(&self) -> usize {
        self.wait_queue.len()
    }

    /// Total bytes currently reserved.
    pub fn used_bytes(&self) -> usize {
        self.used
    }

    /// Total budget in bytes.
    pub fn total_budget(&self) -> usize {
        self.total_budget
    }

    /// Remaining budget in bytes.
    pub fn remaining_bytes(&self) -> usize {
        self.total_budget.saturating_sub(self.used)
    }

    /// Current usage fraction (0.0–1.0).
    pub fn usage_fraction(&self) -> f64 {
        if self.total_budget == 0 {
            return 1.0;
        }
        self.used as f64 / self.total_budget as f64
    }

    /// Whether the scheduler is under eviction pressure.
    pub fn under_pressure(&self) -> bool {
        self.usage_fraction() >= self.pressure_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_admit_and_release() {
        let mut sched = MemoryScheduler::new(1000);

        let handle = sched.try_admit(RequestInfo::new("r1", 300, Priority::Normal));
        assert!(handle.is_some());
        assert_eq!(sched.active_count(), 1);
        assert_eq!(sched.used_bytes(), 300);

        assert!(sched.release("r1"));
        assert_eq!(sched.active_count(), 0);
        assert_eq!(sched.used_bytes(), 0);
    }

    #[test]
    fn budget_overflow_queues() {
        let mut sched = MemoryScheduler::new(500);

        let h1 = sched.try_admit(RequestInfo::new("r1", 300, Priority::Normal));
        assert!(h1.is_some());

        // Exceeds budget → queued
        let h2 = sched.try_admit(RequestInfo::new("r2", 300, Priority::Normal));
        assert!(h2.is_none());
        assert_eq!(sched.queued_count(), 1);
    }

    #[test]
    fn drain_queue_on_release() {
        let mut sched = MemoryScheduler::new(500);

        sched.try_admit(RequestInfo::new("r1", 300, Priority::Normal));
        sched.try_admit(RequestInfo::new("r2", 300, Priority::Normal)); // queued

        assert_eq!(sched.queued_count(), 1);

        sched.release("r1");
        let admitted = sched.drain_queue();
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].id, "r2");
        assert_eq!(sched.queued_count(), 0);
    }

    #[test]
    fn pressure_gates_low_priority() {
        let mut sched = MemoryScheduler::new(1000).with_pressure_threshold(0.5);

        // Fill to 60% → above threshold
        sched.try_admit(RequestInfo::new("r1", 600, Priority::Normal));

        // Normal priority is blocked under pressure
        let h = sched.try_admit(RequestInfo::new("r2", 100, Priority::Normal));
        assert!(h.is_none());
        assert_eq!(sched.queued_count(), 1);

        // High priority still admitted
        let h = sched.try_admit(RequestInfo::new("r3", 100, Priority::High));
        assert!(h.is_some());
    }

    #[test]
    fn critical_always_admitted_under_pressure() {
        let mut sched = MemoryScheduler::new(1000).with_pressure_threshold(0.5);
        sched.try_admit(RequestInfo::new("r1", 800, Priority::Normal));

        let h = sched.try_admit(RequestInfo::new("r2", 100, Priority::Critical));
        assert!(h.is_some());
    }

    #[test]
    fn priority_ordering_in_queue() {
        let mut sched = MemoryScheduler::new(100);

        // Fill budget
        sched.try_admit(RequestInfo::new("active", 100, Priority::Normal));

        // Queue several priorities
        sched.try_admit(RequestInfo::new("low", 50, Priority::Low));
        sched.try_admit(RequestInfo::new("high", 50, Priority::High));
        sched.try_admit(RequestInfo::new("normal", 50, Priority::Normal));

        // Release active slot
        sched.release("active");

        // Drain should admit High first
        let admitted = sched.drain_queue();
        assert_eq!(admitted[0].id, "high");
    }

    #[test]
    fn update_usage() {
        let mut sched = MemoryScheduler::new(1000);
        sched.try_admit(RequestInfo::new("r1", 200, Priority::Normal));

        assert!(sched.update_usage("r1", 400));
        assert_eq!(sched.used_bytes(), 400);

        // Exceed budget
        assert!(!sched.update_usage("r1", 1200));
        assert_eq!(sched.used_bytes(), 400); // unchanged
    }

    #[test]
    fn usage_fraction_and_remaining() {
        let mut sched = MemoryScheduler::new(1000);
        sched.try_admit(RequestInfo::new("r1", 250, Priority::Normal));

        assert!((sched.usage_fraction() - 0.25).abs() < 1e-10);
        assert_eq!(sched.remaining_bytes(), 750);
    }

    #[test]
    fn release_nonexistent() {
        let mut sched = MemoryScheduler::new(1000);
        assert!(!sched.release("nope"));
    }

    #[test]
    fn zero_budget() {
        let mut sched = MemoryScheduler::new(0);
        assert!(sched.under_pressure());

        let h = sched.try_admit(RequestInfo::new("r1", 1, Priority::Critical));
        assert!(h.is_none()); // no space at all
    }

    #[test]
    fn fifo_within_same_priority() {
        let mut sched = MemoryScheduler::new(100);
        sched.try_admit(RequestInfo::new("active", 100, Priority::Normal));

        sched.try_admit(RequestInfo::new("first", 50, Priority::Normal));
        sched.try_admit(RequestInfo::new("second", 50, Priority::Normal));

        sched.release("active");
        let admitted = sched.drain_queue();

        // Both same priority — first submitted should win
        assert_eq!(admitted[0].id, "first");
    }
}
