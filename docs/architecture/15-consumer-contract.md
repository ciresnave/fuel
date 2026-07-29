# Consumer contract

**Status**: v0.1 (draft, 2026-07-28). Establishes the upward-facing half of fuel's boundary obligations: what fuel provides to the systems built *on* it, and what fuel correspondingly doesn't decide. The mirror of [05-backend-contract](05-backend-contract.md). Motivating phase doc: [`docs/fuel-consumer-seam.md`](../fuel-consumer-seam.md), which carries the per-consumer-class annexes and the as-built audit.

What consumers provide to fuel, what fuel provides back, and what fuel doesn't decide. Anchored in the same architectural principle as the backend seam ([01-identity](01-identity.md): **backends advertise; they don't decide**) applied one layer up, with fuel now in the advertiser's seat: **fuel advertises capacity and reports measurements; the consumer decides whose work matters.**

Fuel sits between two contracts. [05-backend-contract](05-backend-contract.md) governs what is *below* — backends provide kernels, capabilities, telemetry, and slot capacity, and never choose strategy. This section governs what is *above*. The symmetry is deliberate: fuel should be as unopinionated toward its consumers as a backend is toward fuel.

---

## The rule

**Fuel owns mechanism; the consumer owns policy.** Fuel must be preemptible, accountable, and manageable — and must never decide whose work matters.

The discriminator that makes this operational:

| Kind of decision | Owner | Why |
| --- | --- | --- |
| Selection among **equivalent implementations** — serial vs. batched decode, recompute vs. store activations, fused vs. decomposed, CPU vs. CUDA arm, how to shard across a supplied device set | **Fuel** | Same math, same result. This *is* arm selection; it is what fuel exists to do. |
| Selection among **competing work** — whose request runs, who is preempted, who is admitted, who is evicted, priority/SLA/fairness | **Consumer** | Different outcomes for different callers. Fuel has no principled basis to choose, and no visibility into what the caller values. |
| Selection among **different math** — which optimizer, which sampling strategy, which learning-rate schedule, what to do on a NaN step | **Consumer** | Not equivalent implementations at all. Fuel reports; it never decides. |

### Why the line falls exactly there

**A layer can only optimize over a fungible cost unit.** Fuel compares two decode arms because they consume the same physical resource and produce the same result. It cannot compare one session's tokens against another's, or a training step against an eval batch — those comparisons are about *value*, and value is the consumer's. The boundary is drawn where cost stops being fungible, so mechanism-vs-policy follows from the cost model rather than being asserted.

This is the same reasoning that puts kernel-local decisions inside a backend and placement decisions in the optimizer ([04-optimization](04-optimization.md)); it is applied once more, upward.

### Scope: this contract governs *execution* consumers

Not every consumer executes anything. Model export and conversion, weight surgery, merging, pruning, visualization, and static analysis consume the **IR**, not the runtime: they build, inspect, transform, and serialize graphs without ever realizing one. No clause below applies to them, because every clause concerns the cost, interruption, or observation of *work being done*.

Those consumers are governed by [03-ir](03-ir.md) (the base map as the stable hub) and [13-interchange](13-interchange.md) (import/export, the weight⊥graph axes). Their contract is a different one — base-map stability, round-trip fidelity, serialization guarantees. The omission here is deliberate; this contract is not to be grown to cover them.

---

## The seam has two directions

**Downward — the consumer supplies constraints**: the work set *and its order*, a resource budget, a quantum and cancellation token, required properties (determinism, tolerance), the device set.

**Upward — fuel supplies advertisements and measurements**: capacity and admissibility, which properties it can honour, measured cost, telemetry, surfaced gaps.

**The invariant — fuel never crosses the streams.** It does not promote an advertisement into a decision, and it does not relax a supplied constraint in order to win on cost.

Every clause below is an instance of one of those two flows.

---

## What fuel provides to consumers

### C-1 — Capacity advertisement

Fuel reports headroom in the consumer's admission unit — free cache blocks, activation-memory budget, maximum concurrent work at a given geometry, whether a batch of a given size is admissible. Fuel does **not** decide whether to accept work. A constructor that fails with an out-of-memory error is the wrong shape: by the time fuel refuses, the consumer has already lost its chance to shed load, queue, evict something cheaper, or shrink the request.

Directly parallel to [05-backend-contract](05-backend-contract.md)'s *slot capacity per device*.

### C-2 — Bounded quantum and cancellation

The consumer can say "advance this set, then return to me" and get control back in bounded time, and can cancel in flight. Unbounded interrupt latency means one long unit of work starves everything else with no consumer recourse. A yield point per step at realize barriers is sufficient; thread-level preemption is not required.

### C-3 — State externalization

The consumer can move fuel-held state out and back: evict, checkpoint, restore. **This clause has a fidelity axis, and the ends have materially different guarantees:**

| Fidelity | Meaning | Consumers |
| --- | --- | --- |
| **Lossy-restorable** | state may be discarded and recomputed; cheapness dominates | inference (a KV cache is recomputable from tokens) |
| **Exactly-restorable** | restore must be bit-identical *including RNG stream position*, or the resumed run diverges | training, reproducible evaluation |

Building only the lossy guarantee would satisfy inference and silently fail every reproducible consumer. The exact end depends on the RNG/generator seam (a prerequisite shared with sampling-based training and energy-based methods).

### C-4 — Measured cost and provenance

Every unit of work reports what it actually consumed: units produced, bytes resident, wall-clock, which arm ran. **Advertised cost is a hint; measured cost is the record** — the same discipline the Judge applies to backend cost annotations ([04-optimization](04-optimization.md)), applied to the consumer seam. This is what makes budget enforcement structural rather than aspirational: a component that misreports its cost accrues a bad measured record and stops being chosen.

### C-5 — Constraint admission

The consumer imposes *properties*, not resources, and those properties remove arms from fuel's selection set:

- **Determinism** — forbid non-deterministic arms (atomic reductions, non-deterministic scatter). Reproducible training and bit-exact oracle runs both live here.
- **Tolerance** — [07-tolerance](07-tolerance.md)'s per-op error budgets already have exactly this shape; this clause is the statement that they are consumer-facing.
- **Device set** — the consumer supplies which devices; fuel decides how to use them.

Fuel optimizes *within* these constraints and never around them. C-5 is what keeps fuel's aggressive arm selection safe for consumers whose correctness bar is stricter than "fast and close enough."

### C-6 — Observation and intervention

The consumer can name an intermediate value and receive it, and for some classes replace it and continue. This is not C-4: that measures what execution *cost*; this extracts what it *computed*. Observation serves quantization calibration, distillation, probing, and debugging; intervention serves activation patching, ablation, and causal tracing.

The clause carries a real tension with fuel's identity: **naming an intermediate defeats fusion across it.** An observation request therefore changes the plan, and fuel's obligation is that this be explicit — the cost is reported through C-4, the observation is never silently dropped, and fuel never silently deoptimizes without saying so.

### C-7 — Declared mutation and in-process invalidation

A consumer that mutates weights or the graph mid-process can declare it, and fuel invalidates what it derived from the prior state: captured runs, cached plans, cost-model assumptions, and arm choices valid only for the old weights.

[11-persistence](11-persistence.md) specifies invalidation thoroughly, but scopes itself to *across process restarts*. This clause is the **live** counterpart: a long-running process that mutates weights between phases, where the model hash changes and nothing rechecks it. Alternating train-infer loops are the consumer that requires it; the failure mode without it is silent degradation rather than an error.

---

## What fuel does NOT decide

The line is sharp, and mirrors [05-backend-contract §What backends do NOT decide](05-backend-contract.md):

- **Fairness, priority, and SLA.** No queue disciplines, no deadline scheduling, no starvation guarantees. Fuel advances the set it is given, in the order it is given.
- **Admission.** Fuel says what fits (C-1); it never decides what to accept.
- **Work lifecycle.** No streaming protocol, no retry, no request identity beyond an opaque handle.
- **Multi-tenancy policy.** Quotas, authentication, and noisy-neighbour mitigation are the consumer's, built on C-1 and C-3.
- **Batching, checkpoint cadence, and eviction choice.** Fuel provides the batched arm, the uniformity gate, and the evict/restore mechanism; *which* work to coalesce, *when* to checkpoint, and *what* to evict are the consumer's.
- **Anything in the "different math" row of the rule** — optimizer choice, sampling strategy, schedule, divergence response.

### Consumer-shape neutrality

**Fuel has no privileged consumer class.** It is general ML tooling — the substrate an inference engine, a trainer, a calibration pipeline, or a scientific-computing consumer is built from — and those classes are peers. The architecture is not owed a bias toward any of them, and [09-non-goals §Not orchestration-flavored architecture decisions](09-non-goals.md) excludes orchestration from Foundation symmetrically across all of them (that section previously claimed an inference center of gravity; the claim was withdrawn as wrong on 2026-07-28, [10-decisions-log](10-decisions-log.md)).

The failure mode this guards against is concrete rather than philosophical: a mechanism built to one class's guarantee and generalized later, badly. C-3 is the live example — an evict/restore API shaped by inference's lossy, recomputable KV cannot serve a consumer that needs exact restoration including RNG stream position, and the deficiency surfaces only when the second consumer arrives, after the API has hardened.

**The commitment**: mechanisms at this seam are specified against at least two consumer classes with different guarantees before they harden.

---

## Consumer classes

The classes fuel's own layering already implies ([02-layers](02-layers.md) cleaves crate boundaries around inference-only and IPC consumers). Clause applicability differs by class, which is the argument for stating the contract consumer-agnostically:

| Class | Work unit | Fuel-held state | C-1 | C-2 | C-3 | C-4 | C-5 | C-6 | C-7 |
| --- | --- | --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **A. Inference host** | step over a ready set | KV cache — *lossy* | ● | ● | ● | ● | ◐ | ◐ | ◐ |
| **B. Training host** | optimizer step / microbatch | params + moments + RNG — *exact* | ● | ● | ● | ● | ● | ◐ | ◐ |
| **C. Stateless batch** | one batch | none | ● | ◐ | — | ● | ● | ◐ | — |
| **D. Oracle / test runner** | one reference execution | none | — | — | — | ● | ● | ● | — |
| **E. IPC / remote** | a transported tensor or graph | serialized across a process boundary | ◐ | ◐ | ● | ● | ◐ | — | ◐ |
| **F. RL / alternating loop** | rollout batch, then policy update | *both* KV (lossy) and params+moments+RNG (exact) | ● | ● | ● | ● | ● | ◐ | ● |
| **G. Observation / analysis** | one instrumented execution | none | ◐ | ◐ | — | ● | ● | ● | — |

● required ◐ partial or optional — not applicable

Two columns carry the argument. **C-3** is inapplicable to C, required-but-*lossy* for A, and required-but-*exact* for B — the clause is portable, the guarantee is not. **C-7** is required by exactly one class, which is why it stayed invisible while the seam was described from the inference point of view.

Per-class detail, the as-built audit, and the folds-into table for use cases that do not earn a class live in the phase doc: [`docs/fuel-consumer-seam.md`](../fuel-consumer-seam.md).

---

## See also

- [05-backend-contract](05-backend-contract.md) — the downward mirror of this section.
- [01-identity](01-identity.md) — "advertise, don't decide", the principle both contracts instantiate.
- [09-non-goals](09-non-goals.md) — the refusals above, in the negative-space register.
- [07-tolerance](07-tolerance.md) — C-5's tolerance half, already specified.
- [11-persistence](11-persistence.md) — C-7's across-restart counterpart.
- [03-ir](03-ir.md), [13-interchange](13-interchange.md) — govern graph consumers, which this section excludes.
- [`docs/fuel-consumer-seam.md`](../fuel-consumer-seam.md) — annexes, audit, open questions.
