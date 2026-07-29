# Consumer contract

**Status**: v0.4 (draft, 2026-07-29). v0.4 (MINOR) refines v0.3's order of preference: preference 2 is **a second output from the producing op** ([12-multi-output](12-multi-output.md)), not a speculative side buffer — and adds [§Observability can be arm-dependent](#observability-can-be-arm-dependent--which-makes-it-partly-a-c-5-question), because a reduction is only expressible in-graph if the reduced value exists as a node, which can depend on which arm the optimizer picked. Verified case: attention scores are a real node on the decomposed arm (`registry/flash_attn.rs:235`) and never written on the fused flash arm. That makes a hot-path observation request partly a **C-5** arm-pruning question. v0.3 (MINOR) adds [C-6 §Two regimes](#two-regimes--and-the-obligation-differs-added-v03-2026-07-29): C-6's stated obligation ("report the cost, never silently deoptimize") is correct for *occasional* observation and **wrong for hot-path observation** — per-token, per-layer attention observation for KV eviction defeats fusion everywhere and makes capture unformable, so telling a consumer it deoptimized is not a resolution. Adds an order of preference led by **express the reduction in-graph, don't observe the intermediate**. Surfaced by the Lightbulb port session auditing its real observation sites against the clause. v0.2 (MINOR) adds [§Where the seam runs](#where-the-seam-runs-foundation-not-the-repository) — the contract binds *Foundation*, not every crate in the workspace; `fuel-inference` and `fuel-training` are consumer-side toolkits above the seam, optional by construction. Added in response to the first real consumer hitting the ambiguity (the Lightbulb port session, 2026-07-29): §15 refused admission/eviction/fairness while `fuel-inference` shipped a scheduler with priority queuing and eviction-pressure admission control. No core-claim change — the refusals bind Foundation exactly as before. v0.1 (2026-07-28) established the upward-facing half of fuel's boundary obligations: what fuel provides to the systems built *on* it, and what fuel correspondingly doesn't decide. The mirror of [05-backend-contract](05-backend-contract.md). Motivating phase doc: [`docs/fuel-consumer-seam.md`](../fuel-consumer-seam.md), which carries the per-consumer-class annexes and the as-built audit.

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

### Where the seam runs: Foundation, not the repository

**The seam runs between Foundation and the orchestration tier — not around the Fuel workspace.** "Fuel" in the refusals above means *Fuel-the-Foundation*: the graph, the optimizer, the dispatch layer, the backends. It does not mean "every crate in the fuel repository."

`fuel-inference` and `fuel-training` sit **above** the seam. They are consumer-side toolkits that happen to ship in the Fuel workspace: batteries-included default policies — eviction strategies, admission control, priority queuing, schedulers, sampling — that a consumer **may adopt, adopt selectively, or ignore entirely**. That they contain policy is not a violation of the refusals; it is the corrected [09-non-goals §Not orchestration-flavored architecture decisions](09-non-goals.md) working as intended, which names `fuel-inference` as exactly where inference-side orchestration belongs.

Three consequences worth stating, because a consumer hit this ambiguity the day the section landed:

- **Adopting a shipped toolkit is a consumer choice, never a contract obligation.** A consumer that brings its own scheduler, its own eviction policy, and its own admission control is fully conformant. Nothing above the seam is required.
- **Policy shipped above the seam is a default, not a commitment.** Foundation owes the clause guarantees; a toolkit crate owes only what its own docs claim.
- **The refusals still bind Foundation absolutely.** If admission control, fairness, or eviction *choice* ever appears below the seam — in the graph, optimizer, dispatch layer, or a backend — that is a defect regardless of how convenient it is.

The practical test for "which side is this on?": would replacing it require a consumer to fork Fuel? If yes, it is Foundation and the refusals apply. If a consumer can simply not depend on the crate, it is above the seam.

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

#### Two regimes — and the obligation differs (added v0.3, 2026-07-29)

The paragraph above is correct for **occasional** observation and wrong for **hot-path** observation. The distinction is cadence relative to the critical path, and it changes what fuel owes:

| Regime | Cadence | Examples | Fuel's obligation |
| --- | --- | --- | --- |
| **Occasional** | offline, per-run, or per-request | calibration, distillation, probing, debugging | **report the cost** (C-4) and never silently deoptimize. Paying for a broken fusion once is an acceptable trade the consumer can evaluate. |
| **Hot-path** | per-token, per-layer, on every request | attention-driven KV eviction (H2O heavy-hitter accumulation, R-KV importance pruning) | **reporting the cost is not a resolution.** An observation at this cadence defeats fusion at every layer of every step and makes a captured region unformable — the consumer does not need to be told it deoptimized, it needs a way not to. |

The second regime is a real consumer class, not a hypothetical: an inference host whose eviction policy is attention-driven must see attention scores every layer every step, which collides head-on with the capture-shaped decode path such a host is also built for. Left unaddressed, **attention-driven eviction and captured replay are mutually exclusive**, and a consumer discovers this only after building on both.

**Design guidance — ask for the reduction, not the intermediate.** A hot-path observer almost never wants the raw intermediate; it wants a *reduction over* it. H2O wants a decayed running sum of attention mass per token; R-KV wants an importance score. Those are graph computations, not observations. When the reduction is expressed as graph nodes, **nothing is observed, nothing breaks, and capture survives** — the only value realized is the small statistics tensor, and only when the policy actually consults it, which is far rarer than per-token-per-layer.

So the clause's order of preference is:

1. **Express the reduction in-graph.** No C-6 request at all. Note that a running accumulator across steps is structurally the same shape as a KV write — a runtime-offset write into a persistent buffer — so the machinery largely exists.
2. **Have the producing op emit the reduction as a second output** — [12-multi-output](12-multi-output.md) territory (`Op::View` / `Op::ScatterIntoSlot`, bundled storage), *not* C-6. See the arm-dependence note below for why this is often the only route that preserves the fast arm.
3. **Observe the intermediate**, accepting the plan change, with the cost reported per C-4. Correct for the occasional regime; a last resort for the hot-path one.
4. **Refuse and say so.** If a consumer's observation genuinely cannot leave the hot path, fuel must state that the two capabilities are incompatible rather than let it be discovered late.

#### Observability can be arm-dependent — which makes it partly a C-5 question

A reduction is only expressible in-graph if the value it reduces **exists as a node**. That is not always true, and whether it is true can depend on *which arm the optimizer picked* — so a hot-path observation request is, in part, a **C-5 constraint that prunes the arm set**.

The worked case **[verified 2026-07-29]**: attention scores are materialized on the **decomposed** arm — `registry/flash_attn.rs:235` builds `scores = scale · (q · kᵀ)` as a real `MatMul` node feeding `softmax(mask(alibi(softcap(·))))` — and are **never written** on the fused flash arm, which is the entire point of a tiled-softmax kernel (the same file notes the tiled form even produces different numerics). So "sum the attention scores" is a well-formed graph reduction against one arm and a request for a *different kernel* against the other.

Consequences for the order of preference above:

- **Preference 1 (reduce in-graph) may silently cost the fast arm.** That is a legitimate, *measurable* trade — the consumer can benchmark decomposed-plus-reduction against fused-without-observation and choose. C-5 is the clause that makes the choice explicit rather than accidental, and C-4 is what makes it measurable.
- **Preference 2 (second output from the producing op) is the only route that keeps both.** It is not exotic: `flash_attn` already carries an optional `softmax_lse` in its input signature (`flash_attn.rs:67` — "takes 4 or 5 inputs (q, k, v, [softmax_lse], [alibi])"), so auxiliary attention statistics are an established shape for these kernels. Emitting a column-sum alongside the output is a kernel-side ask, and for CUDA that means a backend request, not a fuel-internal change.

**The general rule**: before promising a consumer an in-graph reduction, check whether the value it reduces survives the arm the consumer actually wants to run on.

**[Evidence status]** design-level: derived by reading a real consumer's attention-observation sites (5 in its transformer/attention/KV-compression paths) against this clause, against `CapturedRun`'s requirements, and against `registry/flash_attn.rs`. **Not yet validated by a running port** — the reduction-in-graph route is a strong conjecture, not a demonstrated result, and the multi-output route is unbuilt.

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
