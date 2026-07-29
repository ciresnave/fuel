# The Fuel consumer seam — mechanism, policy, and the consumer contract (DRAFT 2026-07-28)

**Status.** Draft for review. **Proposes**; asserts nothing. Per
[`architecture/00-index.md` §"How phase docs relate to this set"](architecture/00-index.md),
a phase doc may propose a change to the constitution. If accepted, the core (§1–§4) is a
candidate for promotion to a numbered section — `15-consumer-contract` — as the **mirror of
[05-backend-contract](architecture/05-backend-contract.md)**, with a
[10-decisions-log](architecture/10-decisions-log.md) entry.

**Purpose.** Fuel is the tooling from which someone builds an ML system — an inference
engine, a trainer, an embedding pipeline, an oracle runner. Fuel has shipped Increment 1 of a
serving substrate ([`fuel-core/src/multi_session.rs`](../fuel-core/src/multi_session.rs)),
and Lightbulb (an inference engine currently on Candle) will be ported onto Fuel. That makes
the boundary question concrete and urgent: **which side of the seam owns what?** This doc
pins a consumer-agnostic rule, states the contract clauses Fuel owes *any* consumer, and
specializes them per consumer class in annexes.

**Fuel has two contracts, and only one is written.**
[05-backend-contract](architecture/05-backend-contract.md) governs what is *below* Fuel:
backends provide kernels, capabilities, telemetry, and slot capacity, and they don't decide
strategy. This doc governs what is *above*: what Fuel provides to a consumer, and what Fuel
correspondingly doesn't decide. The symmetry is the point — Fuel sits in the middle of a
sandwich with obligations in both directions.

**Companion.** [`frontier-paradigms-vision.md`](frontier-paradigms-vision.md) Crux 2 resolved
the analogous question downward ("treat a discrete solver exactly like a backend — it
advertises capabilities and costs but never decides strategy"). This applies the same
resolution upward.

---

## 1. The rule

> **Fuel owns mechanism; the consumer owns policy. Fuel must be preemptible, accountable, and
> manageable — it must never decide whose work matters.**

The discriminator that makes this operational:

| Kind of decision | Owner | Why |
| --- | --- | --- |
| Selection among **equivalent implementations** — serial vs. batched decode, recompute vs. store activations, fused vs. decomposed, CPU vs. CUDA arm, how to shard across a supplied device set | **Fuel** | Same math, same result. This *is* arm selection; it is what Fuel exists to do. |
| Selection among **competing work** — whose request runs, who is preempted, who is admitted, who is evicted, priority/SLA/fairness | **Consumer** | Different outcomes for different callers. Fuel has no principled basis to choose and no visibility into what the caller values. |
| Selection among **different math** — which optimizer, which sampling strategy, which LR schedule, what to do on a NaN step | **Consumer** | Not equivalent implementations at all. Fuel reports; it never decides. |

This is [01-identity](architecture/01-identity.md)'s *"backends advertise capabilities and
costs but never decide strategy"* applied one layer up, with Fuel now in the advertiser's
seat. If the principle is right for the kernel seam it is right here, and Fuel should be as
unopinionated toward its consumers as a backend is toward Fuel.

### 1.1 Why the line falls exactly there — cost fungibility

**A layer can only optimize over a fungible cost unit.**

| Layer | Data plane | Cost unit | Optimization available |
| --- | --- | --- | --- |
| Backends | kernels | FLOPs, bytes, occupancy — **fully fungible** | kernel-local |
| Fuel | kernels / backends | FLOPs, bytes, measured latency — **fully fungible** | aggressive: fusion, placement, arm selection, capture/replay |
| Consumer | executions on Fuel | tokens, samples, steps, GPU-seconds on one device — **fungible within the class** | real: batching, admission, eviction, scheduling |
| Above that | heterogeneous engines | **not fungible** — no common unit | thin: sequencing, budget enforcement, isolation |

Fuel can compare two decode arms because they consume the same physical resource and produce
the same tokens. It cannot compare session A's tokens against session B's, or a training
step against an eval batch — those comparisons are about *value*, and value is the consumer's.
Draw the boundary where cost stops being fungible, and mechanism/policy falls out of it
rather than being asserted.

---

## 2. The seam has two directions

Rather than a flat list of features, the contract is two flows and one invariant.

**Downward — the consumer supplies constraints:** the work set *and its order*, a resource
budget, a quantum and cancellation token, required properties (determinism, tolerance), the
device set.

**Upward — Fuel supplies advertisements and measurements:** capacity and admissibility,
which properties it can honour, measured cost, telemetry, and surfaced gaps.

> **Invariant — Fuel never crosses the streams.** It does not promote an advertisement into a
> decision, and it does not relax a supplied constraint in order to win on cost.

Everything in §3 is an instance of one of those two flows.

### 2.1 Scope — this contract governs *execution* consumers

Not every consumer executes anything. Model export and conversion, weight surgery, merging,
pruning, visualization, and static analysis consume the **IR**, not the runtime: they build,
inspect, transform, and serialize graphs without ever realizing one. None of C-1…C-7 applies
to them, because every clause is about the cost, interruption, or observation of *work being
done*.

Those consumers are governed by [03-ir](architecture/03-ir.md) (the base map as the stable
hub) and [13-interchange](architecture/13-interchange.md) (import/export, the weight⊥graph
axes), and their contract is a different one: base-map stability, round-trip fidelity, and
serialization guarantees. Naming the boundary here so the omission is deliberate rather than
accidental — **do not grow this contract to cover them.**

---

## 3. The clauses

Stated consumer-agnostically. Annexes specialize them; not every class needs every clause.

### C-1 — Capacity advertisement (Fuel reports; consumer admits)

Fuel reports headroom in the consumer's admission unit — free KV blocks, activation-memory
budget, max concurrent work at a given geometry, whether a batch of size *k* is admissible.
Fuel does **not** decide whether to accept work. A constructor returning `Err(OOM)` is the
wrong shape: by the time Fuel refuses, the consumer has already lost the chance to shed load,
queue, evict something cheaper, or shrink the request.

Directly parallel to [05-backend-contract](architecture/05-backend-contract.md)'s *slot
capacity*.

### C-2 — Bounded quantum and cancellation (δ_int)

The consumer must be able to say "advance this set, then return to me" and get control back
in bounded time, and must be able to cancel in flight. Unbounded δ_int means one long unit of
work starves everything else and the consumer cannot intervene.

Minimum viable form: a quantum bound (n units, or a deadline) plus a cooperative cancel
observed at realize barriers. Not thread-level preemption; a yield point per step suffices.

### C-3 — State externalization (δ_cp)

The consumer must be able to move engine-held state out and back: evict, checkpoint, restore.
**This clause has a fidelity axis, and the two ends have different implementations:**

| Fidelity | Meaning | Consumer classes |
| --- | --- | --- |
| **Lossy-restorable** | State may be discarded and recomputed; cheapness matters more than exactness | inference (KV is recomputable from tokens) |
| **Exactly-restorable** | Restore must be bit-identical *including RNG stream position*, or the resumed run diverges | training, reproducible eval |

Building C-3 to the lossy guarantee and calling it done would silently fail every
exactly-restorable consumer. The exact end has a hard dependency on the **RNG/generator seam**
already flagged in the ROADMAP as a shared prerequisite (with EBM sampling and GRPO).

### C-4 — Measured cost and provenance (Fuel reports; nobody estimates)

Every unit of work returns what it actually consumed: units produced, bytes resident,
wall-clock, which arm ran. Fuel already holds the machinery (ledger, telemetry). The
discipline worth importing from the kernel seam: **advertised cost is a hint, measured cost is
the record.** Consumers should trust measurements over any self-report. This is what makes
budget enforcement structural rather than aspirational — a component that misreports its cost
accrues a bad measured record and stops being chosen.

### C-5 — Constraint admission (consumer prunes; Fuel optimizes within)

The consumer imposes *properties*, not resources, and those properties remove arms from
Fuel's selection set:

- **Determinism** — forbid non-deterministic arms (atomic reductions, non-deterministic
  scatter/`index_add`). A trainer requiring reproducibility and an oracle runner requiring
  bit-exactness both live here.
- **Tolerance** — [07-tolerance](architecture/07-tolerance.md)'s per-op error budgets, which
  already have this exact shape but are not wired to the consumer seam.
- **Device set** — the consumer supplies which devices; Fuel decides how to use them.

Fuel optimizes *within* these constraints and never around them. C-5 is the clause that keeps
Fuel's aggressive arm selection safe for consumers whose correctness requirements are
stricter than "fast and close enough."

### C-6 — Observation and intervention (consumer names points; Fuel honours or refuses, never silently)

The consumer must be able to name an intermediate value, receive it, and — for some classes —
replace it and continue. This is **not** C-4: that clause measures what execution *cost*, this
one extracts what it *computed*.

| Mode | Meaning | Consumer classes |
| --- | --- | --- |
| **Observe** | materialize a named intermediate and hand it out | quantization calibration (activation statistics → scales), distillation (teacher intermediates), probing, debugging |
| **Intervene** | replace a named intermediate and continue execution | activation patching, ablation, causal tracing, constraint injection |

The clause has a real cost and a real tension with Fuel's identity: **naming an intermediate
defeats fusion across it.** An observation request is therefore a request that *changes the
plan*, and the contract's obligation is that this be explicit — Fuel reports the cost of
honouring the request through C-4 and never silently drops the observation, nor silently
deoptimizes without saying so.

**[verified 2026-07-28]** `RuntimeHook` appears in [`ROADMAP.md`](../ROADMAP.md),
[`frontier-architecture-gaps.md`](frontier-architecture-gaps.md), and
[`frontier-paradigms-vision.md`](frontier-paradigms-vision.md) — and in **no source file**.
The machinery is planned (Phase 9), not built, and is not yet a consumer-facing seam.
[08-pattern-harvest](architecture/08-pattern-harvest.md) is adjacent but is opt-in telemetry
about *fusion opportunities*, not consumer-directed value extraction.

### C-7 — Declared mutation and in-process invalidation

A consumer that mutates weights (or the graph) mid-process must be able to declare it, and
Fuel must invalidate whatever it derived from the prior state: captured runs, cached plans,
cost-model assumptions, and any arm choice that was valid only for the old weights.

**[verified 2026-07-28]** [11-persistence](architecture/11-persistence.md) already specifies
invalidation thoroughly — the cache header carries
`(arch_version, kernel_hashes, hw_fingerprint, judge_version, tolerance_set, model_hash)` and
any strict-field mismatch invalidates the whole cache. But its scope is explicitly *"across
process restarts."* The gap is the **live** case: a long-running process that mutates weights
between phases, where `model_hash` changes and nothing recomputes or rechecks it.

This is undefined today rather than known-broken, and it is undefined in the direction that
fails silently. The consumer that hits it first is Annex F.

### What Fuel refuses — proposed [09-non-goals](architecture/09-non-goals.md) additions

- **No fairness, priority, or SLA model.** No queue disciplines, no deadline scheduling, no
  starvation guarantees.
- **No admission control.** Fuel says what fits; it never decides what to accept.
- **No work lifecycle.** No streaming protocol, no retry, no request identity beyond an opaque
  handle.
- **No multi-tenancy policy.** Quotas, auth, noisy-neighbour mitigation are the consumer's,
  built on C-1/C-3.
- **No batching, checkpointing-cadence, or eviction policy.** Fuel provides the batched arm,
  the uniformity gate, and the evict/restore mechanism; *which* work to coalesce, *when* to
  checkpoint, and *what* to evict are the consumer's.

---

## 4. Consumer classes

Fuel's own docs already name these. [02-layers](architecture/02-layers.md) cleaves crate
boundaries around them ("Lightbulb (inference-only consumer) wants fuel-tensor without
fuel-autograd; mlmf (network IPC consumer) wants fuel-formats without fuel-loaders"), and the
ROADMAP names "Lightbulb, embeddings, retrieval, oracle test runners, quantized-only paths."

| Class | Work unit | Engine-held state | C-1 | C-2 | C-3 | C-4 | C-5 | C-6 | C-7 |
| --- | --- | --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **A. Inference host** (Lightbulb) | decode step over a ready set | KV cache — *lossy* | ● | ● | ● | ● | ◐ | ◐ | ◐ |
| **B. Training host** | optimizer step / microbatch | params + moments + RNG — *exact* | ● | ● | ● | ● | ● | ◐ | ◐ |
| **C. Stateless batch** (embeddings, retrieval, quantized-only) | one batch | none | ● | ◐ | — | ● | ● | ◐ | — |
| **D. Oracle / test runner** | one reference execution | none | — | — | — | ● | ● | ● | — |
| **E. IPC / remote** (mlmf) | a transported tensor / graph | serialized across a process boundary | ◐ | ◐ | ● | ● | ◐ | — | ◐ |
| **F. RL / alternating loop** (GRPO, RLVR, online learning) | a rollout batch, then a policy update | KV (*lossy*) **and** params+moments+RNG (*exact*) | ● | ● | ● | ● | ● | ◐ | ● |
| **G. Observation / analysis** (calibration, distillation, interpretability) | one instrumented execution | none | ◐ | ◐ | — | ● | ● | ● | — |

● required ◐ partial/optional — not applicable

Two columns carry the argument for writing this consumer-agnostically. **C-3** is empty for
class C, required-but-*lossy* for A, and required-but-*exact* for B — the clause is portable,
the guarantee is not. **C-7** is required by exactly one class, which is why it stayed
invisible while the doc was written from the inference point of view.

### 4.1 What folds in, and where

Recorded so the filtering is auditable rather than re-litigated. None of these earn a class;
each is an instance of one above.

| Use case | Folds into | Wrinkle worth remembering |
| --- | --- | --- |
| Fine-tuning / PEFT / LoRA | B (training), A (adapter serving) | shared immutable base + small mutable per-tenant delta; C-1 must account for adapter memory separately, C-3 evicts the adapter not the base |
| Hyperparameter search / experiment orchestration | *N* × B plus an orchestrator | the orchestrator is above the consumer layer, not a consumer; it is where cost stops being fungible |
| Speculative decoding / draft models | A | two models, **one** consumer — the arbitration unit is the VRAM pool; accept/reject is consumer policy |
| Distributed / multi-node training | B | collectives are Fuel's over a consumer-supplied topology; failure and elasticity response are the consumer's |
| Benchmarking / profiling | D | C-4-heavy; it is an oracle runner that cares about the measurement rather than the value |
| GPU array / preprocessing (Fuel as a tensor library) | C | wants the tensor surface without autograd — already a named crate-fission driver in [02-layers](architecture/02-layers.md) |
| Differentiable scientific computing / simulation | B-shaped | exact-restorable and iterative, but with no model and no "training"; a good check that B's clauses aren't secretly ML-specific |
| Model surgery / merging / pruning / export | **none — graph consumer** | see §2.1; governed by 03-ir + 13-interchange |

---

## Annex A — Inference host (worked example: Lightbulb)

**Unit** a decode step over a ready set. **State** per-session KV, lossy-restorable
(recomputable from tokens). **Cost** tokens, KV bytes, GPU-seconds.

**Clause specialization.** C-1: free KV blocks, max sessions at a geometry, batch
admissibility via the uniformity gate. C-2: quantum = *n* tokens or a deadline; cancel at the
decode-step barrier. C-3: evict KV to host, or discard and mark the session recomputable —
lossy is acceptable and cheapness dominates. C-4: tokens produced, KV bytes resident,
elapsed, arm used. C-5: **already live and load-bearing** — the batched arm is documented as
*ε-close* (logits within 1e-4) and token-identical, so a consumer returning logprobs to users
has a materially different requirement from one returning only tokens, and must be able to
demand the bit-exact serial arm.

### A.1 As-built audit — `fuel-core/src/multi_session.rs`

Increment 1 shipped a good substrate; this is about placement, not quality. **[verified]** =
read from the code, **[judgment]** = my assessment.

**Correctly Fuel's:**

- **`SchedulePolicy::{RoundRobin, Batched{max_batch}}` [verified].** Worth stating plainly,
  because the name invites the opposite reading: the two arms are *semantically equivalent* —
  the doc comment calls the batched arm "provably equal to `RoundRobin`", with `RoundRobin` as
  the byte-exact oracle. This is **not** scheduling in the fairness sense; it is selection
  among equivalent implementations, exactly
  [`frontier-paradigms-vision.md`](frontier-paradigms-vision.md)'s framing of `Op::Branch`
  ("plan-time selection among implementations of the same math… **not** data-dependent
  dispatch"). It stays. **[judgment]** the name should change (`DecodeArm`?) so fairness logic
  cannot grow into a slot that sounds like it invites it.
- **`SessionState`, `ModelDims`, `BatchOutcome`, the uniformity gate, per-session error
  isolation [verified].** Mechanism, correctly placed. Error isolation — a per-session `Err`
  finishes that session rather than killing the batch — is precisely the isolation property a
  consumer needs from its engine.

**Provisional — fine as an oracle, must not become the interface:**

- **`run_to_completion()` [verified]** drives every session to completion with no preemption,
  fairness, or yielding. Correct as a test/oracle driver; it is the exact shape a consumer
  must own. Keep it, mark it a harness convenience, and do not let a consumer call it.
- **`add_session()` [verified]** constructs and always accepts — it is *construction*, not
  admission. The name is the risk: admission logic will accrete there unless C-1 lands and it
  is renamed to reflect that it is unconditional.
- **Implicit FIFO [verified]** — `step()` advances sessions in `Vec` order. A fairness policy
  chosen by omission: invisible, unstated, and unoverridable. Order should be consumer-supplied.

**Clause status:** C-1 absent (no headroom query; OOM surfaces as an `add_session` error).
C-2 absent (no quantum, deadline, or cancel). C-3 absent — `KvCache` has no evict/restore path;
**the load-bearing gap.** C-4 partial — `StepReport` carries *what happened*
(`advanced`/`finished`/`errored`/`used_batched_arm`) but not *what it cost*. C-5 absent as a
consumer-facing control, though the underlying arm distinction exists.

### A.2 Layer drift — a separate, verified defect

**[verified]** `multi_session.rs` lives in `fuel-core` and takes `model: &'m LlamaModel`
(~line 348) plus `SamplingStrategy`. [`ROADMAP.md`](../ROADMAP.md)'s layer table states
Foundation (`fuel-core`) *"will never contain: tokenization, model-family assumptions,
**serving abstractions**, HF Hub client code"* — session lifecycle, sampling, and a
Llama-specific model reference hit three of those categories.

Per the working agreement ("treat doc-vs-code drift as a defect"), this should move up a
layer (`fuel-inference`, whose exclusion list permits it) or the layer table should be amended
with an argued exception. **Recommendation: move it**, before the Lightbulb port — the move is
what forces `&LlamaModel` to become a model-agnostic trait, which any inference consumer needs
anyway since none of them will serve only Llama.

---

## Annex B — Training host

**Unit** an optimizer step, or a microbatch under gradient accumulation. **State** parameters
+ optimizer moments + **RNG stream position**, exactly-restorable. **Cost** samples/sec, step
time, activation and gradient bytes.

**What is the consumer's:** optimizer choice, LR schedule, batch size, response to a NaN or
diverged step, checkpoint cadence, early stopping. All are *different math* or *value
judgments*, so the rule assigns them without ambiguity.

**What is Fuel's:** gradient checkpointing (recompute-vs-store is the same gradients at a
different memory/time point — arm selection under a consumer-supplied memory budget),
mixed-precision arm selection within a consumer-supplied tolerance, and sharding across a
consumer-supplied device set ([06-runtime](architecture/06-runtime.md) already owns data
parallelism).

**Clause specialization.** C-1: activation-memory headroom, which is what lets the consumer
choose microbatch size vs. gradient accumulation — Fuel says what fits, the consumer picks.
C-2: matters for spot-instance preemption, elastic training, and sharing a device with an
inference consumer; quantum = a step or microbatch. **C-3: this is durability, not
load-shedding** — restore must be exact and *must include the RNG stream position*, or a
resumed run silently diverges from an uninterrupted one. C-4: unchanged in shape. C-5: the
strongest case in the whole taxonomy — a reproducibility requirement prunes every
non-deterministic arm, and a mixed-precision tolerance bounds which numerical arms are legal.

**Open:** whether a trainer's C-3 and the inference C-3 share one mechanism with a fidelity
flag, or are genuinely two implementations. **[judgment]** one mechanism, two guarantees —
but this should be settled by whoever builds the second one, not asserted here.

---

## Annex C — Stateless batch (embeddings, retrieval, quantized-only paths)

**Unit** one batch. **State** none between calls. The degenerate case, and useful precisely
because it shows the contract *degrading gracefully*: this class needs C-1, C-4, and C-5 and
nothing else.

C-3 is not applicable — there is no engine-held state to externalize. C-2 is optional; batches
are short, though cancellation still matters for large ones. C-5 carries real weight here:
quantized-only consumers are explicitly trading numerics for throughput, so the tolerance
budget *is* the interface.

If a future clause cannot be omitted for this class, that is a signal the clause is really two
clauses.

---

## Annex D — Oracle / test runner

**Unit** one reference execution. **State** none. Throughput is irrelevant; correctness is
everything.

**C-5 is essentially the entire contract for this class** — require bit-exactness, forbid
non-deterministic arms, forbid approximate rewrites. That makes this class the natural
*test* for C-5: if a consumer can demand determinism and verifiably get it, C-5 works. It
also connects to the existing calibration discipline (sabotage-calibrated epsilons) and to the
kiss-ref advisory cross-check, which is an oracle consumer in all but name.

C-4 is still wanted — measured cost feeds the ledger and calibration even when the runner does
not care about speed.

---

## Annex E — IPC / remote (mlmf) — stub

Named because the ROADMAP already names it ("inter-process tensor exchange (Fuel ↔ Lightbulb ↔
mlmf using safetensors as the wire schema)", `RemoteHostStorage` Phase 7c). The wrinkle is
that state crosses a process boundary, so C-3 becomes *serialization* rather than eviction and
overlaps the interchange work in [13-interchange](architecture/13-interchange.md). Left as a
stub deliberately — writing it without a consumer would be speculation.

---

## Annex F — RL / alternating train-infer loop (GRPO, RLVR, online learning)

**Unit** a rollout batch, then a policy update. **State** *both* kinds at once — KV caches
from rollouts (lossy, discardable) and parameters + optimizer moments + RNG stream (exact).
**Cost** rollout tokens, update step time, and the ratio between them.

This class matters out of proportion to its size because it is the **only** one that runs A
and B against the same weights with a mutation between phases. Every other class holds weights
still for its entire lifetime. Already on Fuel's roadmap (GRPO/RLVR, bucket D in
[`frontier-paradigms-vision.md`](frontier-paradigms-vision.md)), so this is a scheduled
consumer, not a hypothetical.

**Why it is the C-7 consumer.** After a policy update, everything derived from the old weights
is stale: rollout KV caches (semantically — they encode the old policy's continuations),
captured runs, and any arm choice or cost estimate conditioned on the old weights. The
consumer knows the mutation happened; Fuel does not, unless told. Nothing in the current
surface provides the telling, and the failure mode is silent rather than loud — a stale
capture replays, and the run degrades instead of erroring.

**Other clause pressure.** C-3 must serve both fidelities in one process, which is the
strongest argument for one mechanism with a fidelity flag rather than two implementations
(Annex B's open item). C-5 is doubly binding: rollouts and updates may legitimately want
*different* determinism and tolerance settings, so constraints must be per-phase, not
per-process. C-2 matters because rollout batches are long and the consumer wants to bound
them against update latency.

**[judgment]** If C-7 is going to be built, this is the consumer that should specify it, and
it should be specified before the inference-side C-3 hardens — otherwise C-3 gets an
inference-shaped API that this class cannot use.

## Annex G — Observation / analysis (calibration, distillation, interpretability)

**Unit** one instrumented execution. **State** none. **Defining clause** C-6; without it this
class cannot exist at all, which is why it earns a row despite being clause-light everywhere
else.

Quantization calibration reads activation statistics to compute scales. Distillation reads
teacher intermediates. Interpretability reads, patches, and ablates. Debugging — every
consumer's fallback mode — reads whatever is going wrong. These differ in what they do with
the value, not in what they need from Fuel.

C-5 is joint-binding with C-6 here: an observation is only meaningful if the consumer also
controls the numerics that produced it. Calibrating scales against an ε-close fused arm and
then deploying against a bit-exact one (or the reverse) is a silent correctness bug, and the
consumer needs to be able to pin both.

**[judgment]** This class is the natural first consumer of C-6 because calibration is
concrete, already needed by the quantized paths Fuel supports, and produces a checkable
artifact — which makes it a better proving ground than interpretability, whose requirements
are open-ended.

---

## 5. Relationship to the Reasoning-Runtime sketch

Context only; nothing above depends on it. Under a "cognitive engines as schedulable
resources" framing, the layering is nested runtimes, each the same shape, each with a less
fungible cost unit than the one below. **Fuel is not an engine** — it is the substrate an
engine is built from (Candle's former slot). An inference host like Lightbulb is externally
*one* engine and internally a runtime; the engine boundary is the **unit of resource
arbitration**, not the unit of capability, so several models sharing one VRAM pool are one
engine with a union capability profile, not three engines.

A *trainer* is not an engine in that framing at all — it serves no requests. That is the
sharpest reason this doc must stay consumer-agnostic: **Fuel must not assume the inference
shape**, because at least one first-class consumer class doesn't have it.

The consequence for this doc is that C-1…C-5 are not speculative infrastructure for a
hypothetical layer. They are what multi-session serving and reproducible training each need on
their own merits.

---

## 6. Open questions for review

1. **Promote §1–§4 to `15-consumer-contract`?** It is the mirror of 05-backend-contract and
   would carry a decisions-log entry. Annexes stay here as a phase doc.
2. **Does `multi_session.rs` move to `fuel-inference`, or does the layer table get an argued
   exception?** Recommendation: move, before the port.
3. **Is C-3 in scope for Increment 2?** Highest-value clause, only one with no partial
   implementation, and it overlaps the confirmed-absent block-pool allocator behind
   `Op::PagedAttn` (ROADMAP §4) — the two may be one piece of work. Building it needs the
   inference *and* training fidelity requirements settled together (Annex B open item).
4. **Rename `SchedulePolicy` → `DecodeArm`?**
5. **Where does sampling live?** `SessionState::sample_and_append` puts sampling strategy
   inside Fuel's session, but temperature/top-p/grammars/speculative acceptance are arguably
   all consumer policy. Unresolved; it may be a sixth clause, or an instance of C-5.
6. **What is the minimum a consumer needs to port at all?** If C-1…C-7 gate the Lightbulb
   port, that is a large gate; if the port can proceed against today's surface and adopt
   clauses incrementally, sequencing is far easier. Largest schedule impact, and I do not have
   enough of Lightbulb's shape to answer it.
7. **Does C-6 land as Phase-9 `RuntimeHook`, or is it a separate seam?** The hook machinery is
   specified but unbuilt; if it is built for constraint-gating (its current framing in the
   frontier docs) without the consumer-facing observe/intervene surface, C-6 will need a
   second mechanism later. Cheaper to design once.
8. **Should C-7 be built now or specified now and built with Annex F?** It is undefined in a
   direction that fails *silently*, which argues for at least a loud interim: a declared
   `weights_mutated()` that conservatively invalidates everything derived, tightened later.
9. **Is one C-3 mechanism with a fidelity flag right, or two implementations?** Annex F forces
   the question because it needs both in one process. Recommendation: settle this before the
   inference-side C-3 hardens into an inference-shaped API.
