# The Fuel ↔ Lightbulb seam — mechanism, policy, and the serving contract (DRAFT 2026-07-28)

**Status.** Draft for review. **Proposes**; asserts nothing. Per
[`architecture/00-index.md` §"How phase docs relate to this set"](architecture/00-index.md),
a phase doc may propose a change to the constitution — if this is accepted, the changes it
implies to [02-layers](architecture/02-layers.md), [05-backend-contract](architecture/05-backend-contract.md),
and [09-non-goals](architecture/09-non-goals.md) land there and this doc becomes a citation.

**Purpose.** Lightbulb is an inference engine currently built on Candle; it will be ported
onto Fuel. Fuel has meanwhile shipped Increment 1 of a serving substrate
([`fuel-core/src/multi_session.rs`](../fuel-core/src/multi_session.rs): `SessionState` +
`SessionScheduler`). Both facts are good; together they create a boundary question that is
cheap to answer now and expensive to answer after the port: **which side of the seam owns
what?** This doc pins the rule, audits the as-built code against it, and lists the surface
that should exist at the seam.

**Companion.** [`frontier-paradigms-vision.md`](frontier-paradigms-vision.md) Crux 2 already
resolved the analogous question for discrete solvers ("treat it exactly like a backend — it
advertises capabilities and costs but never decides strategy"). This doc applies the same
resolution one layer up, to the serving boundary.

---

## 1. The rule

> **Fuel owns mechanism; Lightbulb owns policy. Fuel must be preemptible, accountable, and
> manageable — it must never decide whose request matters.**

Concretely, the discriminator that makes this operational:

| Kind of decision | Owner | Why |
| --- | --- | --- |
| Selection among **equivalent implementations** — serial vs. batched decode, fused vs. decomposed, CPU vs. CUDA arm | **Fuel** | Same math, same result. This *is* arm selection; it is the thing Fuel exists to do. |
| Selection among **competing requests** — whose turn, who preempts whom, who is admitted, who is evicted, SLA/priority/fairness | **Lightbulb** | Different outcomes for different callers. Fuel has no principled basis to choose, and no visibility into what the caller values. |

This is not a new principle. It is
[01-identity](architecture/01-identity.md)'s *"backends advertise capabilities and costs but
never decide strategy"* applied one layer up, with Fuel now in the advertiser's seat. The
self-similarity is the point: if the rule is right for the kernel seam, it is right for the
serving seam, and Fuel should be as unopinionated toward Lightbulb as a backend is toward
Fuel.

### 1.1 Why the line falls exactly there — cost fungibility

The deeper justification is that **a layer can only optimize over a fungible cost unit.**

| Layer | Data plane | Cost unit | What optimization is possible |
| --- | --- | --- | --- |
| Fuel | kernels / backends | FLOPs, bytes, measured latency — **fully fungible** | aggressive: fusion, placement, arm selection, capture/replay |
| Lightbulb | model executions on Fuel | tokens, KV bytes, GPU-seconds on **one** device — **fungible within the domain** | real: continuous batching, prefill/decode interleave, admission, eviction |
| (above) | heterogeneous engines | **not fungible** — no common unit exists | thin: sequencing, budget enforcement, isolation, provenance |

Fuel can compare two decode arms because they consume the same physical resource and produce
the same tokens. It cannot compare *session A's* tokens against *session B's*, because that
comparison is about value, not cost, and value is the caller's. Draw the layer boundary where
cost stops being fungible, and mechanism/policy falls out of it rather than being asserted.

---

## 2. The serving contract — what Fuel exposes and what it refuses

Four clauses. Fuel owes Lightbulb all four; Lightbulb owes Fuel none of them back.

### C-1 Capacity advertisement (Fuel exposes; Lightbulb decides)

Fuel reports what it *can* hold — free KV blocks, VRAM headroom, max additional sessions at a
given geometry, whether a batch of size *k* is admissible under the uniformity gate. Fuel
does **not** decide whether to accept a session. `add_session` returning `Err(OOM)` is the
wrong shape: by the time Fuel refuses, Lightbulb has already lost the chance to shed load,
queue, or evict a cheaper session.

> **Advertise capacity, don't adjudicate admission.** Directly parallel to
> [05-backend-contract](architecture/05-backend-contract.md)'s "slot capacity".

### C-2 Preemption with a bounded quantum (δ_int)

Lightbulb must be able to say "advance this set, then return to me" and get control back in
bounded time — and must be able to *cancel* in flight. Today a `step()` runs to natural
completion with no deadline, no cancellation token, and no yield point. Unbounded δ_int means
a long prefill can starve every other session and Lightbulb cannot intervene.

Minimum viable form: a quantum bound (n tokens, or a deadline) and a cooperative cancel
observed at realize barriers. Not thread-level preemption; a yield point per decode step is
enough.

### C-3 Checkpoint / evict / restore (δ_cp)

This is the clause that actually decides whether multi-session serving works under memory
pressure, and it is entirely absent today. `SessionState` owns a `KvCache` that can only be
created and dropped. Lightbulb needs to **evict** a session's KV to host memory (or discard it
and mark the session recomputable) and **restore** it later. Without this, admission is
permanent and the only load-shedding policy available is refusal.

Note: `CapturedRun` is already most of the way to a *plan* checkpoint. This clause is about
*state* — the KV — which is the harder and more valuable half.

### C-4 Measured cost + provenance (Fuel reports; nobody estimates)

Every step should return what it actually consumed: tokens produced, KV bytes resident,
wall-clock, which arm ran. Fuel already holds the machinery — the ledger, telemetry,
`used_batched_arm`. The discipline worth importing from the kernel seam is that **advertised
cost is a hint and measured cost is the record**; the scheduler above should trust
measurements, never a caller's or a component's self-report. This is also what makes budget
enforcement structural rather than aspirational: a component that misreports its cost
accrues a bad measured record and stops being chosen.

### What Fuel refuses (proposed [09-non-goals](architecture/09-non-goals.md) additions)

- **No fairness, priority, or SLA model.** No queue disciplines, no weighted round-robin, no
  deadline scheduling, no starvation guarantees.
- **No admission control.** Fuel says what fits; it never decides what to accept.
- **No request lifecycle.** No streaming protocol, no cancellation semantics beyond the
  cooperative token, no retry, no request IDs beyond an opaque handle.
- **No multi-tenancy or isolation policy.** Per-tenant quotas, auth, and noisy-neighbour
  mitigation are Lightbulb's, built on C-1/C-3.
- **No continuous batching policy.** Fuel provides the batched *arm* and the uniformity gate;
  deciding *which* requests to coalesce and when is Lightbulb's.

---

## 3. As-built audit — `fuel-core/src/multi_session.rs` against the rule

Increment 1 shipped a genuinely good substrate. This audit is about placement, not quality.
Findings are marked **[verified]** (read from the code) or **[judgment]**.

### Correctly on Fuel's side

- **`SchedulePolicy::{RoundRobin, Batched{max_batch}}` [verified].** Worth stating plainly
  because the name invites the opposite reading: these two arms are *semantically identical*
  — the doc comment says the batched arm is "provably equal to `RoundRobin`", with
  `RoundRobin` as the byte-exact correctness oracle. So this is **not** a scheduling policy in
  the fairness sense. It is selection among equivalent implementations of the same math —
  exactly [`frontier-paradigms-vision.md`](frontier-paradigms-vision.md)'s framing of
  `Op::Branch` ("plan-time selection among implementations of the same math… **not**
  data-dependent dispatch"). It belongs to Fuel. *The name should change* (`DecodeArm`?) so it
  stops reading as a policy slot that fairness logic can grow into.
- **`SessionState`, `ModelDims`, `BatchOutcome`, the uniformity gate, per-session error
  isolation [verified].** Mechanism, correctly placed. Error isolation in particular
  (a per-session `Err` finishes that session rather than killing the batch) is precisely the
  Isolation invariant a serving layer needs from its engine.

### Provisional — fine as an oracle, must not become the interface

- **`run_to_completion()` [verified].** Drives every session to completion with no
  preemption, no fairness, no yielding. Correct as a test/oracle driver; it is the exact shape
  Lightbulb must own. Keep it, mark it as a harness convenience, and do not let Lightbulb call
  it.
- **`add_session()` [verified].** Today it constructs and always accepts; it is really
  *session construction*, not admission. The name is the risk — admission logic will
  accrete here unless C-1 lands and the constructor is renamed to reflect that it is
  unconditional.
- **Implicit FIFO ordering [verified].** `step()` advances sessions in `Vec` order. That is a
  fairness policy chosen by omission. It is invisible, unstated, and Lightbulb cannot override
  it. Ordering should be caller-supplied (Lightbulb passes the ready set, in its order).

### Missing — the four clauses

| Clause | State | Note |
| --- | --- | --- |
| C-1 capacity | **absent** | no headroom query; OOM surfaces as an `add_session` error |
| C-2 preemption | **absent** | no quantum bound, no deadline, no cancel token |
| C-3 checkpoint/evict | **absent** | `KvCache` has no evict/restore path; this is the load-bearing gap |
| C-4 measured cost | **partial** | `StepReport` carries *what happened* (`advanced`/`finished`/`errored`/`used_batched_arm`) but not *what it cost* — no tokens, bytes, or elapsed |

### Layer drift — a separate, verifiable defect

**[verified]** `multi_session.rs` lives in `fuel-core` and takes `model: &'m LlamaModel`
(line ~348) plus `SamplingStrategy`. [`ROADMAP.md`](../ROADMAP.md)'s layer table states
Foundation (`fuel-core`) *"will never contain: tokenization, model-family assumptions,
**serving abstractions**, HF Hub client code."* Session lifecycle, sampling, and a
Llama-specific model reference are all three of the forbidden categories.

Per the working agreement ("treat doc-vs-code drift as a defect"), this should either move up
a layer (`fuel-inference`, whose exclusion list permits it) or the layer table should be
amended with an explicit, argued exception. **Recommendation: move it.** Doing so before the
Lightbulb port is materially cheaper than after, and the move is what forces the
`&LlamaModel` dependency to become a model-agnostic trait — which Lightbulb needs anyway,
since it will not serve only Llama.

---

## 4. What should exist at the seam

Illustrative shapes, not a prescribed API — implementation detail belongs in the phase doc
that builds it.

```
// C-1 — advertise, don't adjudicate.
fn capacity(&self, geometry: &ModelDims) -> Capacity;   // free blocks, headroom, max_sessions
fn batch_admissible(&self, ids: &[SessionId]) -> bool;  // the uniformity gate, queryable

// C-2 — Lightbulb supplies the ready set AND the order AND the bound.
fn advance(&mut self, ready: &[SessionId], quantum: Quantum, cancel: &CancelToken)
    -> Result<StepReport>;

// C-3 — the load-shedding primitive.
fn evict(&mut self, id: SessionId) -> Result<SessionCheckpoint>;   // KV → host / discard
fn restore(&mut self, ckpt: SessionCheckpoint) -> Result<SessionId>;

// C-4 — measured, not estimated.
struct StepReport { /* …existing… */ cost: MeasuredCost }  // tokens, kv_bytes, elapsed, arm
```

The shape of `advance` carries the whole rule: **Lightbulb decides the set and the order;
Fuel decides how to execute it and reports what it cost.**

---

## 5. Relationship to the Reasoning-Runtime sketch

For context only; nothing here depends on it.

Under the "cognitive engines as schedulable resources" framing, the layering is three nested
runtimes, each the same shape, each with a less fungible cost unit than the one below:

- **Fuel** — a runtime whose data plane is kernels. Not itself an engine; it is the substrate
  an engine is built from (Candle's slot in the stack).
- **Lightbulb** — a runtime whose data plane is model executions on Fuel. Externally **one**
  engine, internally a runtime. The engine boundary is the **unit of resource arbitration**,
  not the unit of capability: several models hosted in one Lightbulb contend for one VRAM and
  one KV pool, and only Lightbulb can arbitrate that, so they are one engine with a union
  capability profile — not three engines.
- **A reasoning runtime** — a data plane of heterogeneous engines (Lightbulb, a solver, a KB).
  Deliberately out of scope here, and premature to build: the abstraction is unfalsifiable
  until a second, genuinely non-tensor engine exists.

The relevant consequence for *this* doc is that C-1…C-4 are not speculative infrastructure
for a hypothetical layer. They are what multi-session serving needs regardless. Building them
bottom-up as serving requirements is the right sequencing under
["no consumer is a reason to sequence, not to skip"](../CLAUDE.md).

---

## 6. Open questions for review

1. **Does `multi_session.rs` move to `fuel-inference`, or does the layer table get an
   exception?** Recommendation: move, before the port.
2. **Is C-3 (KV evict/restore) in scope for Increment 2?** It is the highest-value clause and
   the only one with no partial implementation. It also overlaps the confirmed-absent
   block-pool allocator behind `Op::PagedAttn` (ROADMAP §4) — the two may be one piece of
   work.
3. **Should `SchedulePolicy` be renamed** (`DecodeArm`?) to stop it reading as a fairness
   slot?
4. **Where does sampling live?** `SessionState::sample_and_append` puts sampling strategy
   inside Fuel's session. Sampling is arguably caller policy (temperature, top-p, grammars,
   speculative acceptance all belong to the serving layer). Unresolved; it may be the fifth
   clause.
5. **What is the minimum Lightbulb actually needs to port at all?** If C-1…C-4 are a
   precondition for the port, that is a large gate. If the port can proceed against today's
   surface and adopt the clauses incrementally, sequencing is much easier. This is the
   question with the largest schedule impact and I do not have enough of Lightbulb's shape to
   answer it.
