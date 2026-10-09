# Incremental consumer/dependents index — replacing per-call safety-copy analysis

**Status:** DESIGN PROPOSAL, no code. Rides fuel#326 (the narrow fix this
doc's measurements are the "after" side of) and fuel#321 (the original
decode-slowness measurement this traces back to) — this doc is the
write-up those two PRs' own descriptions point to for the structural
follow-on. CireSnave has confirmed this is the real target, not a fallback
(relayed via PM, 2026-10-09 — see §0). Sequencing (PM, 2026-10-08/09):
narrow fixes first (fuel#326 — merge status: check the PR, not this line,
which will go stale the moment it merges), measure the residual, write
this doc, implement as staged PRs with a differential test gate before the
old passes are ever removed. This document is step 2; no implementation
PR exists yet.

## 0. Why this exists

fuel#321 measured Qwen3-0.6B steady-state CUDA decode at 1.81–2.03 tok/s and
left the cause unmeasured ("per-call weight residency/upload... unmeasured
hypothesis"). A `FUEL_DECODE_TRACE`-gated instrumentation pass (not merged;
lives in an unmerged branch, used only to take these numbers) found the real
cause was never GPU-side at all: `insert_safety_copies` +
`derive_ordering` together cost ~334ms of a ~520ms decode step (~64%) —
more than all GPU dispatch combined — because both re-derive their state
from scratch, over the FULL per-token graph, on **every** `realize_inner`
call, including the "plan-once" persistent-decode fast path that explicitly
skips `optimize_graph` but not this.

fuel#326 (narrow fix, scoped to `fuel-graph` only) cut two independent O(n)
costs within those two functions — see its description for the mechanism.
Measured after #326: `setup`'s share of step time fell from ~64% to ~36%,
and steady-state throughput rose to 5.36 tok/s (~2.7–3x). 36% is still above
the ~25% threshold the PM set for escalating to a structural fix, and
CireSnave has said the redesign below is the real target (not merely a
fallback if the narrow fix proved insufficient) — confirmed directly to the
PM, 2026-10-09; quote not re-transcribed here per the PM's own instruction
that a relayed ruling isn't trusted on trust — the PM has it directly.

**What #326 did NOT touch:** both functions still rebuild their reverse
("who consumes me") index from scratch every call. The graph's EDGES
(who reads what) do not change between decode tokens — only leaf *values*
change via `cache.insert`. #326 made each rebuild cheaper; it did not stop
the rebuilding.

## 1. The core idea

Maintain a `consumers: HashMap<NodeId, Vec<NodeId>>` side-table on `Graph`
itself, maintained **incrementally** at every node-push and every structural
edit, instead of rebuilt by every pass that currently needs one
(`insert_safety_copies`'s `succ`/`pred`, `derive_ordering`'s `consumers`
and alias groups). This is not a cache with an invalidation signal someone
has to remember to check — it's a structural invariant, correct by
construction, with no "is this still fresh" question to ask.

### 1.1 The invariant

For a destructive op `M` that mutates target `X` in place, `M` is always
the **last** entry in `X`'s consumer list at the moment it's appended. `X`'s
"current value" identity becomes `M`'s own `NodeId` going forward — ordinary
SSA; nothing new here, it's already how the graph reads today (anyone who
wants the post-mutation value already reads `M`'s `NodeId`, not `X`'s).

### 1.2 Constructing a new node: pin vs. copy

When a new node `W` is pushed with `X` as an input, and `X`'s consumer list
already ends in a mutator `M`:

1. Cheap, bounded check: walk **forward** from `M` through `M`'s own
   consumer chain (not backward through `W`'s ancestry, which is
   unbounded) looking for overlap with `W`'s *other* direct inputs.
2. If no overlap is found: insert `W` into `X`'s list **before** `M` — a
   free ordering pin, no copy, no bytes moved.
3. If overlap is found (or the forward walk can't shallowly disprove it):
   insert a `Copy` node in `M`'s place, rewire `M` to read the copy, and
   `W` proceeds against the original `X` directly.

Walking forward from the mutator is bounded because a mutator's forward
reach is typically small and local (a handful of ops before the result is
consumed into something that doesn't further propagate it) — unlike
`W`'s backward ancestry, which can span the entire preceding computation.
This check is shallow and conservative-on-miss, not general cycle
detection: it trades a small amount of precision (occasionally copying
where a deeper check would have found a pin) for a bounded cost, which is
the same character as #326's pre-check.

### 1.3 Aliasing (views, bundle projections, reshape/contiguize)

A view op (`y = x.transpose()`) is a genuine, distinct `NodeId` in the
graph — it can have its own shape, its own downstream consumers — but at
the *byte* level it shares `x`'s storage `Arc`. If `y` got its own
independent consumer list, a mutation appended to `x`'s list would never
be visible to anyone querying `y`'s list, even though reading `y` after
the mutation reads the exact bytes the mutation overwrote.

Resolution: views, `Op::View{slot}` bundle projections, and
`Reshape`/`Contiguize` (conservatively, matching #326's treatment — their
zero-copy-ness is a runtime fact not knowable at graph-build time) do
**not** get independent consumer lists. A read "through" one of them
resolves to its root's list transparently, chained back through the view
relation. `x`'s list — and whatever mutator ends it — governs every view
of `x` and every further view of those views, with no separate
bookkeeping per view.

### 1.4 Other structural edits

- **Rewiring** (an existing node's input repointed to a different
  `NodeId`): remove the node from its old target's consumer list, and
  insert it into the new target's list at the position matching where it
  sits in execution order — found by scanning outward (forward, then
  backward, or vice versa) from the node's own position until a `NodeId`
  already present in the new target's consumer list is found on each
  side. Worst case is a full scan; expected to be rare and local in
  practice, but **this is the one edit whose cost isn't bounded by this
  design** — see §3.
- **Fusion** (several nodes replaced by one): collect the fused group's
  external inputs (dropping inputs internal to the group), remove each
  original member from its producer's consumer list once (retaining the
  earliest removal position), and insert the fused node at that position;
  separately, redirect the fused group's own external consumers from
  whichever original member they read to the new fused node.
- **Removing a consumer from the middle of a list**: a non-issue — the
  consumer stops consuming, nothing about the producer's list ending in a
  mutator changes.
- **Removing the *last* consumer in a list**: if the entry before it is a
  `Copy` node inserted by §1.2, and nothing downstream of that copy still
  needs the pre-mutation value, the copy can be retracted and the mutator
  re-attached directly to the original target — copy elision on dead-code
  removal. Optional; not required for correctness, only for avoiding
  stale copies after an edit removes their only reason to exist.

## 2. Net effect

Eliminates both `insert_safety_copies`'s and `derive_ordering`'s per-call
costs — not by caching a result that might go stale, but by never needing
a global pass at all. The incremental cost is paid once, at each node's
construction or edit, amortized over the graph's whole lifetime — for a
decode session, that's once for the whole generation, not once per token.

## 3. Open items — each needs its own test before landing

These are the specific soundness questions raised and worked through
during design (fuel lane + CireSnave, 2026-10-09); none are resolved by
code yet.

| # | Open item | Risk if wrong | Test that would catch it |
|---|---|---|---|
| 1 | §1.2's forward-walk pin check is shallow (bounded by the mutator's own forward reach, not the new node's full ancestry) | A deep, multi-hop conflict (new node's indirect ancestor reads the mutator's output several hops back) could be missed if the forward walk from the mutator doesn't reach it | Construct a graph where the conflicting path is N hops from the mutator for increasing N; assert a copy is inserted at every N up to some bound, and document the bound beyond which the design knowingly falls back to copying unconditionally (never silently wrong — either catches it or always copies) |
| 2 | §1.3's view/alias resolution must cover all three families #326 already enumerates (single-input view ops, `Op::View{slot}` bundle projections, `Reshape`/`Contiguize`) | A view family not routed through "resolve to root's list" would reintroduce exactly the bug class #326's `alias_groups` fixes — a mutation invisible to a view's readers | Port every existing `insert_safety_copies`/`derive_ordering` test (the current ~15 covering residual connections, sibling-bundle views, transitive descendants, parallel view roots) to run against BOTH the old pass and the new incremental index on the same graphs, asserting identical `OrderingEdges`/copy-insertion output |
| 3 | §1.4 rewiring's worst-case scan is unbounded | A pathological edit pattern could make a single rewire O(graph size), reintroducing a cost this design exists to remove | Measure rewire frequency and typical scan distance on a real optimizer pass (not just decode) before deciding whether this needs its own bound, or whether it's rare enough to accept as-is |
| 4 | §1.4 fusion's bookkeeping (dropping internal inputs, redirecting external consumers) must not silently drop a consumer edge when a fused group has an internal node that ALSO has an external consumer (a group that isn't cleanly fusable) | A dropped edge is silent data corruption, the same failure class `insert_safety_copies` exists to prevent | This is fusibility analysis, not new to this design — confirm the existing fusion pass already rejects/handles non-cleanly-fusable groups before this design's bookkeeping is asked to run on one |
| 5 | The invariant (§1.1: mutator always last) must hold after EVERY kind of structural edit the optimizer performs, not just the four enumerated here | An edit type not covered silently breaks the invariant the whole design rests on | Audit every `fuel-graph` pass that mutates `Graph` structure (not just `insert_safety_copies`/`derive_ordering`) for edit shapes beyond push/rewire/fuse/remove; each new shape found gets its own row in this table before implementation starts on it |

## 4. Implementation plan (staged, no code until the PM/CireSnave sign off on this doc)

1. **Differential harness first.** Before any production code changes:
   build a test harness that runs BOTH the old (`insert_safety_copies` +
   `derive_ordering`, post-#326) and the new incremental-index logic
   side-by-side against every existing `fuel-graph` test graph plus the
   live decode harness's actual graph, asserting identical output
   (`OrderingEdges`, copy-insertion count and placement). This differential
   test is the gate for every later step — it must stay green through the
   whole migration, and the old passes are not removed until it has run
   clean against real decode traffic, not just unit-test graphs.
2. **Add the `consumers` side-table to `Graph`**, maintained incrementally
   at `push` only (rewiring/fusion/removal deferred) — the common,
   low-risk case. Differential-test against old behavior.
3. **Add the §1.2 pin-vs-copy logic**, gated behind the differential
   harness, for the `push`-only case.
4. **Add view/alias resolution (§1.3)**, with the ported test suite from
   open item #2 above passing on both old and new paths.
5. **Only then** tackle rewiring/fusion/removal (§1.4), each behind its
   own differential-tested PR, informed by whatever open items #3/#4
   investigation turns up.
6. **Cut over**: once the differential harness has run clean against the
   live decode harness for some agreed soak period, remove the old
   `insert_safety_copies`/`derive_ordering` per-call paths.

Each numbered step above is its own PR. No step starts before the
previous one's differential test is green.
