//! Per-graph execution-plan store — planner Stage 4a.
//!
//! Program anchor: `docs/session-prompts/load-time-incremental-
//! planner.md` (Stage 4) + architecture `04-optimization.md`
//! §Load-time incremental planning (v0.4). The 2026-06-11 decision:
//! planning moves out of `realize()` — `realize()` reduces to
//! *ensure plan coverage of the requested roots, then dispatch*.
//! This module is the memoization half of that move: the planning
//! work `prepare()` used to redo per realize call is keyed and
//! reused, and decode-loop graph growth replans only the delta.
//!
//! ## Keying + invalidation
//!
//! Plans key on **(graph identity, pinned device)** and validate
//! against the **topology generation**:
//!
//! - *Graph identity* is the `Arc<RwLock<Graph>>` allocation
//!   (pointer + a stored `Weak` re-checked by `Arc::ptr_eq` so a
//!   reallocated address can never resurrect a dead graph's plans).
//! - *Pinned device* is the realize call's `DeviceLocation` — the
//!   same graph realized on CPU and CUDA holds two independent
//!   plans.
//! - *Topology generation*: a stored plan whose
//!   [`ExecutionPlan::generation`] no longer matches
//!   [`crate::dispatch::topology_generation`] is stale — the lookup
//!   treats it as a miss (full rebuild) and counts an invalidation.
//!   This is the same signal the executor's Phase-4.3 dispatch-chunk
//!   check and the bridge's `TopologyChanged` retry already key on.
//!
//! ## Hit / extension / miss
//!
//! Coverage is judged per realize against the realize's own topo
//! order: a node *needs* a plan entry exactly when
//! [`crate::plan::node_needs_plan_entry`] says so (the same gate
//! `compile_plan` uses — the two cannot drift). Three outcomes:
//!
//! - **Hit** — every needs-entry node in the order is covered by the
//!   stored plan. No planning work runs; the stored `Arc` is
//!   returned. Repeat realizes of an unchanged graph (decode-loop
//!   steps whose only new node is the realize-root `Op::Copy`
//!   splice, which needs no entry) land here.
//! - **Extension** — the graph grew (nodes appended for the next
//!   decode step): the build callback runs `compile_plan` with
//!   [`crate::plan::PlanOptions::reuse_plan`] = the stored base, so
//!   covered nodes clone their sets and only the delta enumerates /
//!   filters / ranks.
//! - **Miss** — no stored plan (or stale generation): full build.
//!
//! Graph *mutation behind the frontier* (a shrink — impossible for
//! Fuel's append-only graphs; indicates identity confusion) is a
//! typed error, never a panic.
//!
//! ## Warm latches (Stage 4b)
//!
//! A background warm ([`PlanStore::warm_scoped`]) registers a
//! per-`(graph, device)` **planning-in-progress latch** before it
//! plans. A realize that arrives mid-warm calls [`PlanStore::plan_for`]
//! for the same key, finds the foreign latch, and *waits* on it
//! instead of planning the same nodes a second time — when the warm
//! completes, the waiter wakes up and its lookup HITs the freshly
//! stored plan. The latch owner's own `plan_for` (the warm body)
//! passes through via a thread-identity check. Latches clear on every
//! exit path (success, error, panic-unwind) via a drop guard; waits
//! carry a generous timeout so a wedged warm surfaces a typed error,
//! never a hang. Poisoned latch/state mutexes are typed errors too —
//! no panics.
//!
//! ## Commit horizon + revisions (Stage 4b)
//!
//! Once a realize's dispatch begins, the plan in use is an immutable
//! `Arc<ExecutionPlan>` snapshot — a store update never mutates it.
//! Stage 4b adds the *ahead-of-frontier adoption seam* on top:
//!
//! - [`PlanStore::submit_revision`] replaces the stored plan for a
//!   `(graph, device)` key (typed-error-validated against the live
//!   topology generation) and bumps a store-wide **revision epoch**
//!   (one atomic counter).
//! - [`PlanStore::watch`] hands the executor a [`PlanRevisionWatch`]
//!   whose [`PlanRevisionWatch::poll`] is O(1) when nothing was
//!   submitted (a single atomic read against the cached epoch) and
//!   only takes the store lock when the epoch moved.
//! - The executor (see `crate::pipelined`'s `RevisionState`) adopts a
//!   polled revision for its REMAINING work only when the revision's
//!   generation matches the active plan's and its winners over the
//!   already-executed prefix are node-for-node identical to what
//!   actually dispatched — the commit-horizon invariant. Mismatches
//!   reject the revision and the original plan finishes.
//!
//! v1 revision triggers are external (`submit_revision` callers: a
//! Judge-profile-driven re-plan, tests). Automatic re-rank triggers
//! are Stage 4c.
//!
//! ## Known staleness (documented, accepted for 4a)
//!
//! - Judge measurements landing between builds don't invalidate a
//!   stored plan (the Judge doesn't bump the topology generation).
//!   The production runtime selector (JudgeAware) re-queries live
//!   profile data at dispatch time, so picks still improve within
//!   the stored top-N; only the top-N membership itself is frozen
//!   until the next rebuild.
//! - Reused candidates keep their original inbound-transfer terms;
//!   a residency shift between realizes is repriced only for delta
//!   nodes. The bridge's residency-stitch passes run per realize
//!   against actual storage locations, so execution correctness
//!   never depends on plan-time pricing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock, Weak};
use std::time::{Duration, Instant};

use fuel_core_types::{DeviceLocation, Error, Result};
use fuel_graph::{Graph, NodeId};

use crate::plan::{node_needs_plan_entry, ExecutionPlan};

/// Per-graph store counters. Snapshot via [`PlanStore::stats`];
/// counters are cumulative for the graph's lifetime in the store.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlanStoreStats {
    /// Lookups fully served by a stored plan (zero planning work).
    pub hits: u64,
    /// Full builds with no reusable base (first realize on this
    /// (graph, device), or after a generation invalidation).
    pub misses: u64,
    /// Incremental builds — a base existed and only the delta was
    /// planned (`PlanOptions::reuse_plan` path).
    pub extensions: u64,
    /// Stored plans discarded because the topology generation moved.
    pub invalidations: u64,
    /// Total needs-entry nodes actually planned (misses count the
    /// full order's needs-entry nodes; extensions count only the
    /// delta). The decode-loop test asserts this grows by the delta,
    /// not the prefix.
    pub nodes_planned: u64,
    /// Lookups that waited on a foreign in-flight warm latch before
    /// resolving (Stage 4b). A wait followed by a HIT is the
    /// "realize-during-warm planned once" shape.
    pub latch_waits: u64,
    /// Revisions accepted by [`PlanStore::submit_revision`] for this
    /// graph (any device key).
    pub revisions: u64,
}

/// One stored plan for a (graph, device) pairing.
struct StoredPlan {
    plan: Arc<ExecutionPlan>,
    /// `Graph::len()` observed when the plan was stored — the plan
    /// frontier. Graphs are append-only, so a later observation
    /// below this value means the "same" graph identity no longer
    /// refers to the graph that was planned (typed error).
    planned_len: usize,
}

/// All store state for one graph identity.
struct GraphEntry {
    /// Re-validated by `Arc::ptr_eq` on every lookup — a dead graph
    /// whose allocation address was reused can never serve plans.
    graph: Weak<RwLock<Graph>>,
    plans: HashMap<DeviceLocation, StoredPlan>,
    stats: PlanStoreStats,
}

/// Sweep threshold: when the store map exceeds this many graph
/// entries on insert, dead-`Weak` entries are retained out.
/// Decode patterns that build a fresh graph per step (the
/// kv-context family) churn entries; the sweep keeps the map
/// bounded by the number of LIVE graphs plus this slack.
const SWEEP_THRESHOLD: usize = 32;

/// Budget for waiting on a foreign in-flight warm. Planning is
/// µs–ms-scale work; a wait that exceeds this means the warm thread
/// is wedged (or its latch leaked through a poisoned mutex during
/// unwind) — surface a typed error rather than hanging the realize.
const WARM_LATCH_TIMEOUT: Duration = Duration::from_secs(60);

/// Planning-in-progress latch for one `(graph, device)` key.
/// Registered by [`PlanStore::warm_scoped`]; foreign threads block in
/// [`WarmLatch::wait`] until the owner's drop guard marks it done.
struct WarmLatch {
    /// The warming thread. Its own `plan_for` calls pass through the
    /// latch (it IS the planning in progress).
    owner: std::thread::ThreadId,
    done: Mutex<bool>,
    cv: Condvar,
}

impl WarmLatch {
    fn new_owned_by_current_thread() -> Self {
        Self {
            owner: std::thread::current().id(),
            done: Mutex::new(false),
            cv: Condvar::new(),
        }
    }

    /// Block until the latch is marked done. Typed errors on a
    /// poisoned latch mutex and on timeout; never panics.
    fn wait(&self) -> Result<()> {
        let mut done = self
            .done
            .lock()
            .map_err(|_| Error::Msg("plan store: warm latch poisoned".into()).bt())?;
        let deadline = Instant::now() + WARM_LATCH_TIMEOUT;
        while !*done {
            let now = Instant::now();
            let Some(remaining) = deadline.checked_duration_since(now).filter(|d| !d.is_zero())
            else {
                return Err(Error::Msg(format!(
                    "plan store: waited {WARM_LATCH_TIMEOUT:?} on an in-flight \
                     warm that never completed — the warm thread is wedged or \
                     its latch leaked. Planning is µs–ms work; this is a bug \
                     in the warm caller, not a recoverable condition.",
                ))
                .bt());
            };
            let (guard, _timeout) = self
                .cv
                .wait_timeout(done, remaining)
                .map_err(|_| Error::Msg("plan store: warm latch poisoned".into()).bt())?;
            done = guard;
        }
        Ok(())
    }
}

/// Clears + notifies the owner's latch on every exit path of
/// [`PlanStore::warm_scoped`] (success, error, panic-unwind).
struct LatchClearGuard<'a> {
    store: &'a StoreInner,
    key: (usize, DeviceLocation),
}

impl Drop for LatchClearGuard<'_> {
    fn drop(&mut self) {
        // A poisoned latches mutex can't be reported from drop;
        // waiters then surface the timeout error above. (Poison here
        // implies a panic inside the store's brief lock scopes —
        // already a bug with its own report.)
        if let Ok(mut latches) = self.store.latches.lock() {
            if let Some(latch) = latches.remove(&self.key) {
                if let Ok(mut done) = latch.done.lock() {
                    *done = true;
                }
                latch.cv.notify_all();
            }
        }
    }
}

/// Shared state behind [`PlanStore`]. `Arc`-held so
/// [`PlanRevisionWatch`] handles stay valid independent of the
/// `PlanStore` value they were created from.
struct StoreInner {
    graphs: Mutex<HashMap<usize, GraphEntry>>,
    /// In-flight warm latches keyed by `(graph identity, device)`.
    latches: Mutex<HashMap<(usize, DeviceLocation), Arc<WarmLatch>>>,
    /// Bumped by every accepted [`PlanStore::submit_revision`].
    /// [`PlanRevisionWatch::poll`]'s O(1) no-revision fast path is a
    /// single atomic read of this counter.
    revision_epoch: AtomicU64,
}

/// The per-graph execution-plan store. One process-wide instance
/// ([`PlanStore::global`]) serves production; tests may build their
/// own with [`PlanStore::new`] for isolation. Cloning is cheap and
/// shares the same store (Arc-backed).
#[derive(Clone)]
pub struct PlanStore {
    inner: Arc<StoreInner>,
}

impl PlanStore {
    /// Fresh, empty store.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(StoreInner {
                graphs: Mutex::new(HashMap::new()),
                latches: Mutex::new(HashMap::new()),
                revision_epoch: AtomicU64::new(0),
            }),
        }
    }

    /// Process-wide store used by the production realize path
    /// (`fuel-core::pipelined_bridge`).
    pub fn global() -> &'static PlanStore {
        static GLOBAL: OnceLock<PlanStore> = OnceLock::new();
        GLOBAL.get_or_init(PlanStore::new)
    }

    /// Run `f` (the warm body — typically a [`Self::plan_for`] call
    /// chain) under this key's planning-in-progress latch (Stage 4b).
    ///
    /// While the latch is held, every OTHER thread's `plan_for` for
    /// the same `(graph, device)` waits for `f` to finish and then
    /// resolves against the stored result (a HIT — zero duplicate
    /// planning). `f`'s own `plan_for` calls pass through (thread-
    /// identity check). The latch clears on every exit path.
    ///
    /// A concurrent `warm_scoped` for the same key waits for the
    /// in-flight one, then runs `f` anyway — `f`'s `plan_for` resolves
    /// to a HIT/extension, so the duplicate work is the coverage walk,
    /// not a re-plan. Re-entrant warming of the same key on the same
    /// thread is a typed error (it would deadlock).
    pub fn warm_scoped<T>(
        &self,
        graph: &Arc<RwLock<Graph>>,
        device: DeviceLocation,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let key = (Arc::as_ptr(graph) as usize, device);
        let me = std::thread::current().id();
        loop {
            let existing = {
                let mut latches = self
                    .inner
                    .latches
                    .lock()
                    .map_err(|_| Error::Msg("plan store: latch map poisoned".into()).bt())?;
                match latches.entry(key) {
                    std::collections::hash_map::Entry::Vacant(v) => {
                        v.insert(Arc::new(WarmLatch::new_owned_by_current_thread()));
                        None
                    }
                    std::collections::hash_map::Entry::Occupied(o) => Some(Arc::clone(o.get())),
                }
            };
            match existing {
                None => break,
                Some(latch) if latch.owner == me => {
                    return Err(Error::Msg(
                        "plan store: re-entrant warm_scoped on the same thread \
                         for the same (graph, device) — the warm body must not \
                         warm its own key (this would deadlock)"
                            .into(),
                    )
                    .bt());
                }
                Some(latch) => latch.wait()?,
            }
        }
        let _clear = LatchClearGuard { store: &self.inner, key };
        f()
    }

    /// Wait for a foreign in-flight warm covering `(key, device)`,
    /// if any. Returns whether a wait actually happened (feeds the
    /// [`PlanStoreStats::latch_waits`] counter).
    fn wait_for_foreign_warm(&self, key: (usize, DeviceLocation)) -> Result<bool> {
        let me = std::thread::current().id();
        let mut waited = false;
        loop {
            let latch = {
                let latches = self
                    .inner
                    .latches
                    .lock()
                    .map_err(|_| Error::Msg("plan store: latch map poisoned".into()).bt())?;
                latches.get(&key).cloned()
            };
            match latch {
                None => return Ok(waited),
                Some(l) if l.owner == me => return Ok(waited),
                Some(l) => {
                    waited = true;
                    l.wait()?;
                    // Loop: a new warm may have started; re-check.
                }
            }
        }
    }

    /// Resolve a plan covering `order` for `(graph, device)`,
    /// building (fully or incrementally) via `build` only when the
    /// store can't serve the lookup.
    ///
    /// `order` is the realize's own topo order (the same nodes the
    /// build callback will hand `compile_plan`). `build` receives
    /// `Some(base)` on the extension path — callers thread it into
    /// [`crate::plan::PlanOptions::with_reuse_plan`] — and `None`
    /// on a full miss.
    ///
    /// Locking: the store mutex is never held across `build` or any
    /// graph lock; the graph read lock is taken only for the
    /// coverage walk.
    pub fn plan_for(
        &self,
        graph: &Arc<RwLock<Graph>>,
        device: DeviceLocation,
        order: &[NodeId],
        build: &mut dyn FnMut(Option<&Arc<ExecutionPlan>>) -> Result<ExecutionPlan>,
    ) -> Result<Arc<ExecutionPlan>> {
        let key = Arc::as_ptr(graph) as usize;

        // Phase 0 (Stage 4b): a foreign in-flight warm for this key
        // IS this lookup's planning — wait for it instead of planning
        // the same nodes twice. The warm thread's own plan_for passes
        // through (thread-identity check inside).
        let waited = self.wait_for_foreign_warm((key, device))?;

        let current_gen = crate::dispatch::topology_generation();

        // Phase 1 (store mutex, brief): fetch the candidate base.
        let (base, invalidated) = {
            let mut store = self
                .inner
                .graphs
                .lock()
                .map_err(|_| Error::Msg("plan store mutex poisoned".into()).bt())?;
            match store.get(&key) {
                Some(entry)
                    if entry
                        .graph
                        .upgrade()
                        .is_some_and(|g| Arc::ptr_eq(&g, graph)) =>
                {
                    match entry.plans.get(&device) {
                        Some(sp) if sp.plan.generation == current_gen => {
                            (Some((Arc::clone(&sp.plan), sp.planned_len)), false)
                        }
                        Some(_) => (None, true),
                        None => (None, false),
                    }
                }
                Some(_) => {
                    // Dead graph or a reallocated address (ABA):
                    // drop the stale entry; this lookup is a miss.
                    store.remove(&key);
                    (None, false)
                }
                None => (None, false),
            }
        };

        // Phase 2 (graph read lock): frontier sanity + coverage
        // delta against the base.
        let (graph_len, delta) = {
            let g = graph
                .read()
                .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
            let len = g.len();
            if let Some((_, planned_len)) = &base {
                if len < *planned_len {
                    return Err(Error::Msg(format!(
                        "plan store: graph shrank behind the plan frontier \
                         (len {len} < planned {planned_len}) — graphs are \
                         append-only, so a stored plan for this identity \
                         refers to a different graph state. This is a bug \
                         in graph-identity management, not a recoverable \
                         condition.",
                    ))
                    .bt());
                }
            }
            let mut delta = 0usize;
            for &id in order {
                if !node_needs_plan_entry(&g.node(id).op) {
                    continue;
                }
                let covered = base
                    .as_ref()
                    .is_some_and(|(plan, _)| plan.alternatives(id).is_some());
                if !covered {
                    delta += 1;
                }
            }
            (len, delta)
        };

        // Phase 3: hit fast-path — zero planning work.
        if let Some((plan, _)) = &base {
            if delta == 0 {
                let mut store = self
                    .inner
                    .graphs
                    .lock()
                    .map_err(|_| Error::Msg("plan store mutex poisoned".into()).bt())?;
                if let Some(entry) = store.get_mut(&key) {
                    entry.stats.hits += 1;
                    entry.stats.latch_waits += waited as u64;
                }
                return Ok(Arc::clone(plan));
            }
        }

        // Phase 4: build (full or incremental) with no locks held.
        let built = Arc::new(build(base.as_ref().map(|(p, _)| p))?);

        // Phase 5: store + counters.
        let mut store = self
            .inner
            .graphs
            .lock()
            .map_err(|_| Error::Msg("plan store mutex poisoned".into()).bt())?;
        if store.len() > SWEEP_THRESHOLD {
            store.retain(|_, e| e.graph.strong_count() > 0);
        }
        let entry = store.entry(key).or_insert_with(|| GraphEntry {
            graph: Arc::downgrade(graph),
            plans: HashMap::new(),
            stats: PlanStoreStats::default(),
        });
        if entry
            .graph
            .upgrade()
            .map_or(true, |g| !Arc::ptr_eq(&g, graph))
        {
            // The entry belonged to a different (dead) graph that
            // shared this address — reset it for the live one.
            entry.graph = Arc::downgrade(graph);
            entry.plans.clear();
            entry.stats = PlanStoreStats::default();
        }
        if invalidated {
            entry.stats.invalidations += 1;
        }
        if base.is_some() {
            entry.stats.extensions += 1;
        } else {
            entry.stats.misses += 1;
        }
        entry.stats.nodes_planned += delta as u64;
        entry.stats.latch_waits += waited as u64;
        entry.plans.insert(
            device,
            StoredPlan { plan: Arc::clone(&built), planned_len: graph_len },
        );
        Ok(built)
    }

    /// Snapshot the cumulative counters for `graph`. `None` when the
    /// store has never planned for this graph (or its entry was
    /// swept/reset).
    pub fn stats(&self, graph: &Arc<RwLock<Graph>>) -> Option<PlanStoreStats> {
        let key = Arc::as_ptr(graph) as usize;
        let store = self.inner.graphs.lock().ok()?;
        let entry = store.get(&key)?;
        entry
            .graph
            .upgrade()
            .is_some_and(|g| Arc::ptr_eq(&g, graph))
            .then_some(entry.stats)
    }

    /// The currently stored plan for `(graph, device)`, if any.
    /// Revision producers (background re-rank drivers, tests) read
    /// this, derive a revised plan, and hand it back through
    /// [`Self::submit_revision`].
    pub fn stored_plan(
        &self,
        graph: &Arc<RwLock<Graph>>,
        device: DeviceLocation,
    ) -> Option<Arc<ExecutionPlan>> {
        let key = Arc::as_ptr(graph) as usize;
        let store = self.inner.graphs.lock().ok()?;
        let entry = store.get(&key)?;
        if !entry
            .graph
            .upgrade()
            .is_some_and(|g| Arc::ptr_eq(&g, graph))
        {
            return None;
        }
        entry.plans.get(&device).map(|sp| Arc::clone(&sp.plan))
    }

    /// Submit a REVISED plan for `(graph, device)` (Stage 4b).
    ///
    /// The revision replaces the stored plan — the next realize's
    /// `plan_for` serves it — and bumps the revision epoch so every
    /// live [`PlanRevisionWatch`] observes it at its next poll. An
    /// in-flight realize adopts it only if the executor's commit-
    /// horizon checks pass (see the module doc); otherwise the
    /// original plan finishes and the revision still serves later
    /// realizes.
    ///
    /// Typed errors (validated here so the executor never sees an
    /// ill-formed revision): `revised.generation` must match the live
    /// topology generation, and a plan must already be stored for
    /// this exact `(graph, device)` — revisions revise, they don't
    /// introduce.
    pub fn submit_revision(
        &self,
        graph: &Arc<RwLock<Graph>>,
        device: DeviceLocation,
        revised: ExecutionPlan,
    ) -> Result<Arc<ExecutionPlan>> {
        let live = crate::dispatch::topology_generation();
        if revised.generation != live {
            return Err(Error::Msg(format!(
                "plan store: submit_revision generation {} does not match \
                 the live topology generation {live} — revisions must be \
                 built against the current topology (stale revisions are \
                 rejected at submit time so executors never see them)",
                revised.generation,
            ))
            .bt());
        }
        self.submit_revision_unvalidated(graph, device, revised)
    }

    /// [`Self::submit_revision`] minus the generation gate. Internal:
    /// the executor's own `rev.generation == plan.generation` check
    /// is a *second* line of defense for the submit-to-poll race
    /// window, and its tests need to plant a mismatched-generation
    /// revision without disturbing the process-global counter.
    pub(crate) fn submit_revision_unvalidated(
        &self,
        graph: &Arc<RwLock<Graph>>,
        device: DeviceLocation,
        revised: ExecutionPlan,
    ) -> Result<Arc<ExecutionPlan>> {
        let key = Arc::as_ptr(graph) as usize;
        let arc = Arc::new(revised);
        {
            let mut store = self
                .inner
                .graphs
                .lock()
                .map_err(|_| Error::Msg("plan store mutex poisoned".into()).bt())?;
            let entry = store
                .get_mut(&key)
                .filter(|e| {
                    e.graph
                        .upgrade()
                        .is_some_and(|g| Arc::ptr_eq(&g, graph))
                })
                .ok_or_else(|| {
                    Error::Msg(
                        "plan store: submit_revision for a graph with no \
                         store entry — revisions revise an existing stored \
                         plan; warm or realize the graph first"
                            .into(),
                    )
                    .bt()
                })?;
            let slot = entry.plans.get_mut(&device).ok_or_else(|| {
                Error::Msg(format!(
                    "plan store: submit_revision for device {device:?} but \
                     the stored plans for this graph cover other devices \
                     only — revisions revise an existing stored plan",
                ))
                .bt()
            })?;
            slot.plan = Arc::clone(&arc);
            entry.stats.revisions += 1;
        }
        self.inner.revision_epoch.fetch_add(1, Ordering::Release);
        Ok(arc)
    }

    /// Create a revision watch for `(graph, device)` (Stage 4b). The
    /// executor polls it at work-item boundaries; polls are O(1) (one
    /// atomic read) until a revision is submitted ANYWHERE in this
    /// store, at which point one store-lock lookup resolves whether
    /// this key has a new plan.
    ///
    /// The epoch snapshot is taken NOW — revisions submitted before
    /// the watch was created are never reported (they're already the
    /// stored plan the realize departed from).
    pub fn watch(
        &self,
        graph: &Arc<RwLock<Graph>>,
        device: DeviceLocation,
    ) -> PlanRevisionWatch {
        PlanRevisionWatch {
            store: self.clone(),
            graph: Arc::downgrade(graph),
            key: Arc::as_ptr(graph) as usize,
            device,
            seen_epoch: self.inner.revision_epoch.load(Ordering::Acquire),
        }
    }
}

/// Executor-side handle for observing plan revisions mid-realize
/// (Stage 4b). Created by [`PlanStore::watch`]; consumed by
/// `PipelinedExecutor::realize_with_plan_revisions`.
pub struct PlanRevisionWatch {
    store: PlanStore,
    graph: Weak<RwLock<Graph>>,
    key: usize,
    device: DeviceLocation,
    seen_epoch: u64,
}

impl PlanRevisionWatch {
    /// The currently stored plan for the watched key, but only when
    /// the store's revision epoch moved since the last poll (or since
    /// watch creation). `None` means "nothing new" — the O(1) fast
    /// path (a single atomic read, no locks).
    ///
    /// The epoch is store-wide, so a revision submitted for a
    /// *different* graph makes one poll return this key's (unchanged)
    /// stored plan; callers dedupe with `Arc::ptr_eq` against the
    /// plan they already hold. Degraded states (dead graph, swept
    /// entry, poisoned store mutex) answer `None` — the realize
    /// simply finishes on the plan it has.
    pub fn poll(&mut self) -> Option<Arc<ExecutionPlan>> {
        let epoch = self.store.inner.revision_epoch.load(Ordering::Acquire);
        if epoch == self.seen_epoch {
            return None;
        }
        self.seen_epoch = epoch;
        let graphs = self.store.inner.graphs.lock().ok()?;
        let entry = graphs.get(&self.key)?;
        let live = entry.graph.upgrade()?;
        let watched = self.graph.upgrade()?;
        if !Arc::ptr_eq(&live, &watched) {
            return None;
        }
        entry.plans.get(&self.device).map(|sp| Arc::clone(&sp.plan))
    }
}

impl Default for PlanStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ranker::{AlternativeSet, Candidate};
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DType, Shape};
    use fuel_graph::{topo_order_multi, Node, Op};

    fn push_node(g: &mut Graph, op: Op, inputs: Vec<NodeId>) -> NodeId {
        g.push(Node {
            op,
            inputs,
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        })
    }

    fn noop_kernel(
        _i: &[Arc<RwLock<fuel_memory::Storage>>],
        _o: &mut [Arc<RwLock<fuel_memory::Storage>>],
        _l: &[fuel_core_types::Layout],
        _p: &crate::kernel::OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn candidate() -> Candidate {
        Candidate {
            kernel: noop_kernel,
            caps: crate::kernel::KernelCaps::empty(),
            backend: BackendId::Cpu,
            device: DeviceLocation::Cpu,
            precision: crate::fused::PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: Default::default(),
            inbound_transfer_ns: 0,
            op_params: crate::kernel::OpParams::None,
            coupling: Vec::new(),
            kernel_source: "test",
        }
    }

    /// Synthetic builder: covers every needs-entry node in `order`,
    /// reusing base sets where present (mirrors the
    /// `compile_plan` + `reuse_plan` contract without the binding
    /// table).
    fn build_covering(
        graph: &Arc<RwLock<Graph>>,
        order: &[NodeId],
        base: Option<&Arc<ExecutionPlan>>,
    ) -> Result<ExecutionPlan> {
        let g = graph.read().unwrap();
        let mut alternatives = HashMap::new();
        for &id in order {
            if !node_needs_plan_entry(&g.node(id).op) {
                continue;
            }
            if let Some(set) = base.and_then(|b| b.alternatives(id)) {
                alternatives.insert(id, set.clone());
                continue;
            }
            let mut set = AlternativeSet::empty();
            set.push(candidate());
            alternatives.insert(id, set);
        }
        Ok(ExecutionPlan {
            order: order.to_vec(),
            alternatives,
            generation: crate::dispatch::topology_generation(),
        })
    }

    /// Chain graph: Const → Neg → Neg → ... (n compute nodes).
    fn chain_graph(n: usize) -> (Arc<RwLock<Graph>>, NodeId) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let tip = {
            let mut g = graph.write().unwrap();
            let mut prev = push_node(&mut g, Op::Const, vec![]);
            for _ in 0..n {
                prev = push_node(&mut g, Op::Neg, vec![prev]);
            }
            prev
        };
        (graph, tip)
    }

    fn order_for(graph: &Arc<RwLock<Graph>>, tip: NodeId) -> Vec<NodeId> {
        let g = graph.read().unwrap();
        topo_order_multi(&g, &[tip])
    }

    #[test]
    fn miss_then_hit_same_graph_device() {
        let store = PlanStore::new();
        let (graph, tip) = chain_graph(3);
        let order = order_for(&graph, tip);

        let mut builds = 0usize;
        let p1 = store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |base| {
                builds += 1;
                assert!(base.is_none(), "first lookup is a full miss");
                build_covering(&graph, &order, base)
            })
            .unwrap();
        assert_eq!(builds, 1);

        let p2 = store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |_| {
                panic!("second lookup must be a hit — build must not run")
            })
            .unwrap();
        assert!(Arc::ptr_eq(&p1, &p2), "hit returns the stored Arc");

        let stats = store.stats(&graph).expect("entry exists");
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.extensions, 0);
        assert_eq!(stats.nodes_planned, 3, "three Neg nodes planned once");
    }

    #[test]
    fn growth_extends_with_base_and_plans_delta_only() {
        let store = PlanStore::new();
        let (graph, tip) = chain_graph(4);
        let order = order_for(&graph, tip);
        store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |base| {
                build_covering(&graph, &order, base)
            })
            .unwrap();

        // Decode step: append 2 more compute nodes.
        let new_tip = {
            let mut g = graph.write().unwrap();
            let a = push_node(&mut g, Op::Neg, vec![tip]);
            push_node(&mut g, Op::Sqr, vec![a])
        };
        let order2 = order_for(&graph, new_tip);

        let mut saw_base = false;
        let p2 = store
            .plan_for(&graph, DeviceLocation::Cpu, &order2, &mut |base| {
                saw_base = base.is_some();
                build_covering(&graph, &order2, base)
            })
            .unwrap();
        assert!(saw_base, "growth path threads the stored base for reuse");
        assert!(
            p2.alternatives(new_tip).is_some(),
            "extended plan covers the appended tip",
        );

        let stats = store.stats(&graph).unwrap();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.extensions, 1);
        assert_eq!(
            stats.nodes_planned,
            4 + 2,
            "extension planned the 2-node delta, not the 4-node prefix again",
        );
    }

    #[test]
    fn device_change_is_an_independent_miss() {
        let store = PlanStore::new();
        let (graph, tip) = chain_graph(2);
        let order = order_for(&graph, tip);
        store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |base| {
                build_covering(&graph, &order, base)
            })
            .unwrap();

        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut base_seen: Option<bool> = None;
        store
            .plan_for(&graph, cuda0, &order, &mut |base| {
                base_seen = Some(base.is_some());
                build_covering(&graph, &order, base)
            })
            .unwrap();
        assert_eq!(
            base_seen,
            Some(false),
            "a different device key never reuses another device's plan",
        );
        let stats = store.stats(&graph).unwrap();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.invalidations, 0);

        // Both keys now serve hits independently.
        store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |_| {
                panic!("CPU key must hit")
            })
            .unwrap();
        store
            .plan_for(&graph, cuda0, &order, &mut |_| panic!("CUDA key must hit"))
            .unwrap();
        assert_eq!(store.stats(&graph).unwrap().hits, 2);
    }

    #[test]
    fn generation_bump_invalidates_to_full_rebuild() {
        let store = PlanStore::new();
        let (graph, tip) = chain_graph(2);
        let order = order_for(&graph, tip);
        store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |base| {
                build_covering(&graph, &order, base)
            })
            .unwrap();

        crate::dispatch::bump_topology_generation();

        let mut base_seen: Option<bool> = None;
        store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |base| {
                base_seen = Some(base.is_some());
                build_covering(&graph, &order, base)
            })
            .unwrap();
        assert_eq!(
            base_seen,
            Some(false),
            "stale generation must NOT be offered as a reuse base — the \
             Phase-4.3 contract is a full rebuild against the fresh topology",
        );
        let stats = store.stats(&graph).unwrap();
        assert_eq!(stats.invalidations, 1);
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn distinct_graphs_have_independent_entries() {
        let store = PlanStore::new();
        let (g1, t1) = chain_graph(2);
        let (g2, t2) = chain_graph(2);
        let o1 = order_for(&g1, t1);
        let o2 = order_for(&g2, t2);
        store
            .plan_for(&g1, DeviceLocation::Cpu, &o1, &mut |b| {
                build_covering(&g1, &o1, b)
            })
            .unwrap();
        store
            .plan_for(&g2, DeviceLocation::Cpu, &o2, &mut |b| {
                build_covering(&g2, &o2, b)
            })
            .unwrap();
        assert_eq!(store.stats(&g1).unwrap().misses, 1);
        assert_eq!(store.stats(&g2).unwrap().misses, 1);
        assert_eq!(store.stats(&g1).unwrap().hits, 0);
    }

    #[test]
    fn stats_none_for_unplanned_graph() {
        let store = PlanStore::new();
        let (graph, _) = chain_graph(1);
        assert!(store.stats(&graph).is_none());
    }

    #[test]
    fn build_error_propagates_and_stores_nothing() {
        let store = PlanStore::new();
        let (graph, tip) = chain_graph(1);
        let order = order_for(&graph, tip);
        let err = store.plan_for(&graph, DeviceLocation::Cpu, &order, &mut |_| {
            Err(Error::Msg("synthetic build failure".into()).bt())
        });
        assert!(err.is_err());
        assert!(
            store.stats(&graph).is_none(),
            "failed builds must not leave a store entry behind",
        );
    }
}
