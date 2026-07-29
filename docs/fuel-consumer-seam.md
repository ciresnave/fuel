# The Fuel consumer seam — annexes, audit, and open questions (2026-07-28)

**Status.** Phase doc. **The core of this document was promoted to the constitution on
2026-07-28** as [15-consumer-contract](architecture/15-consumer-contract.md) — the rule, the
fungibility derivation, the two directions, clauses C-1…C-7, the refusals, and the consumer-class
matrix now live there and are authoritative. Per
[`architecture/00-index.md`](architecture/00-index.md#how-phase-docs-relate-to-this-set), this doc
cites rather than restates.

**What stays here**: the per-consumer-class annexes, the as-built audit of the shipped serving
substrate, the folds-into table recording which use cases *don't* earn a class, and the open
questions. These are working detail, not steady-state architecture.

**Origin.** Lightbulb (an inference engine currently on Candle) will be ported onto Fuel, and Fuel
had meanwhile shipped Increment 1 of a serving substrate
([`fuel-core/src/multi_session.rs`](../fuel-core/src/multi_session.rs)). That made the boundary
question concrete: without a stated rule, a scheduler making fairness decisions lands inside Fuel,
and the ported consumer arrives with a second scheduler that loses to it.

---

## The rule, in one line

**Fuel owns mechanism; the consumer owns policy — Fuel must be preemptible, accountable, and
manageable, and must never decide whose work matters.** The discriminator, the derivation from cost
fungibility, and the seven clauses are in
[15-consumer-contract](architecture/15-consumer-contract.md). Everything below assumes them.

---

## What folds in, and where

Recorded so the filtering is auditable rather than re-litigated. None of these earn a class; each is
an instance of one in [15's class matrix](architecture/15-consumer-contract.md#consumer-classes).

| Use case | Folds into | Wrinkle worth remembering |
| --- | --- | --- |
| Fine-tuning / PEFT / LoRA | B (training), A (adapter serving) | shared immutable base + small mutable per-tenant delta; C-1 must account for adapter memory separately, C-3 evicts the adapter not the base |
| Hyperparameter search / experiment orchestration | *N* × B plus an orchestrator | the orchestrator is above the consumer layer, not a consumer; it is where cost stops being fungible |
| Speculative decoding / draft models | A | two models, **one** consumer — the arbitration unit is the VRAM pool; accept/reject is consumer policy |
| Distributed / multi-node training | B | collectives are Fuel's over a consumer-supplied topology; failure and elasticity response are the consumer's |
| Benchmarking / profiling | D | C-4-heavy; an oracle runner that cares about the measurement rather than the value |
| GPU array / preprocessing (Fuel as a tensor library) | C | wants the tensor surface without autograd — already a named crate-fission driver in [02-layers](architecture/02-layers.md) |
| Differentiable scientific computing / simulation | B-shaped | exact-restorable and iterative, but with no model and no "training"; a good check that B's clauses aren't secretly ML-specific |
| Model surgery / merging / pruning / export | **none — graph consumer** | governed by [03-ir](architecture/03-ir.md) + [13-interchange](architecture/13-interchange.md); see [15 §Scope](architecture/15-consumer-contract.md#scope-this-contract-governs-execution-consumers) |

---

## Annex A — Inference host (worked example: Lightbulb)

**Unit** a decode step over a ready set. **State** per-session KV, lossy-restorable (recomputable
from tokens). **Cost** tokens, KV bytes, GPU-seconds.

**Clause specialization.** C-1: free KV blocks, max sessions at a geometry, batch admissibility via
the uniformity gate. C-2: quantum = *n* tokens or a deadline; cancel at the decode-step barrier.
C-3: evict KV to host, or discard and mark the session recomputable — lossy is acceptable and
cheapness dominates. C-4: tokens produced, KV bytes resident, elapsed, arm used. C-5: **already live
and load-bearing** — the batched arm is documented as *ε-close* (logits within 1e-4) and
token-identical, so a consumer returning logprobs to users has a materially different requirement
from one returning only tokens, and must be able to demand the bit-exact serial arm.

### A.1 As-built audit — `fuel-core/src/multi_session.rs`

Increment 1 shipped a good substrate; this is about placement, not quality. **[verified]** = read
from the code, **[judgment]** = assessment.

**Correctly Fuel's:**

- **`SchedulePolicy::{RoundRobin, Batched{max_batch}}` [verified].** Worth stating plainly, because
  the name invites the opposite reading: the two arms are *semantically equivalent* — the doc
  comment calls the batched arm "provably equal to `RoundRobin`", with `RoundRobin` as the byte-exact
  oracle. This is **not** scheduling in the fairness sense; it is selection among equivalent
  implementations, exactly [`frontier-paradigms-vision.md`](frontier-paradigms-vision.md)'s framing
  of `Op::Branch` ("plan-time selection among implementations of the same math… **not**
  data-dependent dispatch"). It stays. **[judgment]** the name should change (`DecodeArm`?) so
  fairness logic cannot grow into a slot that sounds like it invites it.
- **`SessionState`, `ModelDims`, `BatchOutcome`, the uniformity gate, per-session error isolation
  [verified].** Mechanism, correctly placed. Error isolation — a per-session `Err` finishes that
  session rather than killing the batch — is precisely the isolation property a consumer needs.

**Provisional — fine as an oracle, must not become the interface:**

- **`run_to_completion()` [verified]** drives every session to completion with no preemption,
  fairness, or yielding. Correct as a test/oracle driver; it is the exact shape a consumer must own.
  Keep it, mark it a harness convenience, and do not let a consumer call it.
- **`add_session()` [verified]** constructs and always accepts — it is *construction*, not admission.
  The name is the risk: admission logic will accrete there unless C-1 lands and it is renamed to
  reflect that it is unconditional.
- **Implicit FIFO [verified]** — `step()` advances sessions in `Vec` order. A fairness policy chosen
  by omission: invisible, unstated, and unoverridable. Order should be consumer-supplied.

**Clause status:** C-1 absent (no headroom query; OOM surfaces as an `add_session` error). C-2 absent
(no quantum, deadline, or cancel). C-3 absent — `KvCache` has no evict/restore path; **the
load-bearing gap.** C-4 partial — `StepReport` carries *what happened*
(`advanced`/`finished`/`errored`/`used_batched_arm`) but not *what it cost*. C-5 absent as a
consumer-facing control, though the underlying arm distinction exists.

### A.2 Layer drift — a separate, verified defect

**[verified]** `multi_session.rs` lives in `fuel-core` and takes `model: &'m LlamaModel` (~line 348)
plus `SamplingStrategy`. [`ROADMAP.md`](../ROADMAP.md)'s layer table states Foundation (`fuel-core`)
*"will never contain: tokenization, model-family assumptions, **serving abstractions**, HF Hub client
code"* — session lifecycle, sampling, and a Llama-specific model reference hit three of those
categories.

Per the working agreement ("treat doc-vs-code drift as a defect"), this should move up a layer
(`fuel-inference`, whose exclusion list permits it) or the layer table should be amended with an
argued exception. **Recommendation: move it**, before the Lightbulb port — the move is what forces
`&LlamaModel` to become a model-agnostic trait, which any inference consumer needs anyway since none
of them will serve only Llama.

### A.3 Lightbulb port survey (2026-07-29)

**Method.** Read-only structural survey of the *pre-port* codebase at `C:\Projects\Lightbulb`
(CireSnave's note: "it has not yet even started to be ported to Fuel. It will change a lot").
Structural, not behavioural — file/symbol counts and module shape, no execution. **[verified]** =
counted or read from code; **[judgment]** = assessment.

**Shape [verified].** ~66k LOC across 168 `.rs` files repo-wide, 110 under `src/`; single crate.
Built on **`candlelight`**, a Candle *fork* — not stock Candle. 44 files repo-wide touch the tensor
layer. **[flag]** what candlelight diverged from Candle is unknown to this survey and is a real port
input.

**The tensor coupling is concentrated in `model/` [verified].**

| Subsystem | files touching tensors | LOC | Character |
| --- | :-: | ---: | --- |
| `model/` | **16 / 17** | 8,895 | the tensor core — the actual port surface |
| `multi_gpu/` | 4 / 6 | — | placement / sharding |
| `cache/` | 5 / 12 | 9,399 | 5 tensor files; the other 7 are **policy** |
| `loaders/` | 3 / 3 | — | weight loading |
| `memory/` | 3 / 4 | 774 | |
| `engine/` | **1 / 25** | 15,954 | reasoning layer — reaches tensors through `speculative.rs` alone |
| `api/` | **0 / 9** | 2,288 | OpenAI-compatible server |
| `contracts/` | **0 / 6** | 1,331 | constrained generation |

**Finding 1 — Lightbulb is mostly not a tensor program [verified].** Its largest subsystem
(`engine/`, 16k LOC, 25 files) is *reasoning orchestration* — `query_analysis`, `knowledge_base`,
`context_injection`, `decomposition`, `relevance_search`, `conversation_history`, `tool_call` — and
touches the tensor layer through exactly one file (`speculative.rs`, i.e. speculative decoding, which
genuinely needs tensors). `api/` and `contracts/` are tensor-free entirely. Notably `model_runner.rs`
does *not* touch candlelight directly — the reasoning layer reaches models through `model/`'s
abstraction.

**Finding 2 — Lightbulb's shape independently corroborates the mechanism/policy line
[verified structure, judgment on significance].** It already partitions the way
[15](architecture/15-consumer-contract.md) predicts, without having been designed against it:

- `cache/{h2o_policy, streaming_policy, segmented_eviction_policy, eviction_policy}.rs` — eviction
  **policy**, four files, zero tensor coupling. Exactly what 15 says stays with the consumer.
- `cache/tiered_storage.rs` — C-3 externalization, consumer side, already built.
- `cache/{prefix_cache, cache_span}.rs` — shared-prefix reuse, i.e. the refcounted COW splice
  Increment 2 is building, from the consumer's side.
- `engine/{slot_pool, slot_monitor, memory_aware_scheduler}.rs` — the consumer's own admission and
  scheduling.
- `sampling.rs` + `contracts/` — sampling policy.

**[judgment]** The contract was derived from Fuel's side and Lightbulb sits on the predicted side of
every line. That is meaningful corroboration that the boundary is real rather than invented.

**Q5 RESOLVED — sampling is consumer policy [verified].** `src/sampling.rs` is 123 lines of
host-side post-processing over **realized** `&mut [f32]` logits: temperature scaling, top-k, top-p,
and a seeded `StdRng` draw. `contracts/` (`enum_choice`, `tagged_fields`, `validation`,
`commit_block`) layers constrained generation on top — sampling policy Fuel could never anticipate.
**Fuel should produce logits and stop.** `SessionState::sample_and_append` duplicates something the
consumer already owns and is richer at. It is scheduler-surface, so it rides the `fuel-inference`
move rather than Increment 2.

**Sub-finding — this bounds C-3-exact [verified].** Lightbulb re-seeds
`StdRng::seed_from_u64(seed)` *per sample call*; the consumer owns its RNG entirely. So the "RNG
stream position" an exact-fidelity C-3 handle must cover is **Fuel's** RNG (training-side sampling,
dropout), never the consumer's sampler. Worth pinning so the Exact arm doesn't over-scope.

**Q6 RESOLVED — the clauses do NOT gate the port.** Lightbulb keeps its own `engine/`, cache
policies, `memory/`, slot pool, scheduler, sampler, and API. The port is a **tensor-layer swap
concentrated in `model/`**, not a re-architecture around the consumer contract. C-1…C-7 are adopted
incrementally as Fuel offers them; **none block the port**.

**What does gate it — the reverse gap list (what Lightbulb needs *from Fuel*):**

1. **Eager → lazy [verified: 70 value-extraction sites (`to_vec1`/`to_vec2`/`to_scalar`), 57
   `.forward(` `Module` calls].** Most translates mechanically, since Fuel's tensor ops build graph
   nodes. The 70 extraction sites need auditing: each is either a legitimate realize boundary
   (logits → sampling) or hidden dynamic control flow that must become a graph construct or an
   explicit realize. **[judgment] the single largest port risk — and it is not a Fuel gap; it is
   Lightbulb-side work.**
2. **Model implementations.** Uses `candlelight::transformers::models::llama::{Llama, Cache, Config,
   LlamaEosToks}` *and* carries its own `custom_transformer` / `custom_attention` /
   `custom_transformer_block` (~3.3k LOC). `fuel-transformers` parity is needed for the former; the
   latter ports as ordinary graph code.
3. **`nn` surface.** `VarBuilder`, `Linear`, `linear_no_bias`, `linear_b`, `embedding`, `ops::silu`,
   `ops::softmax_last_dim`. `fuel-nn` has `VarBuilder`/`VarMap`; needs a coverage check.
4. **Quantized surface.** `QMatMul::from_qtensor`, `gguf_file`, AWQ (`awq_qwen3.rs`), and a Marlin
   FFI backend. Fuel's `qmatmul` reached a total Q4_0 decompose on main `9d6ad291` — timely. AWQ and
   Marlin are separate asks.
5. **Error type.** 49 `Error::Msg` + 28 `bail`. Mechanical, but touches nearly every coupled file.

**Finding for serving Increment 2 [judgment].** Lightbulb will almost certainly **not** adopt
`SessionScheduler` — it has its own scheduler, slot pool, and admission. It wants the **allocator**
underneath its own policies. That validates deferring the scheduler-surface reshape and may shrink
that reshape permanently. Two allocator requirements Lightbulb's cache implies:

- **Prefix sharing** — `prefix_cache.rs` + `cache_span.rs` mean shared-prefix blocks across sessions,
  which is exactly the refcounted COW splice being built. Good fit, no change needed.
- **Compressed KV** — `kv_compression.rs` is 1,998 LOC, so block sizing may not be uniform across
  sessions. **[judgment]** worth confirming the allocator doesn't assume fixed-size blocks in a way
  that forecloses compressed-KV consumers.

---

## Annex B — Training host

**Unit** an optimizer step, or a microbatch under gradient accumulation. **State** parameters +
optimizer moments + **RNG stream position**, exactly-restorable. **Cost** samples/sec, step time,
activation and gradient bytes.

**What is the consumer's:** optimizer choice, LR schedule, batch size, response to a NaN or diverged
step, checkpoint cadence, early stopping. All are *different math* or *value judgments*, so the rule
assigns them without ambiguity.

**What is Fuel's:** gradient checkpointing (recompute-vs-store is the same gradients at a different
memory/time point — arm selection under a consumer-supplied memory budget), mixed-precision arm
selection within a consumer-supplied tolerance, and sharding across a consumer-supplied device set
([06-runtime](architecture/06-runtime.md) already owns data parallelism).

**Clause specialization.** C-1: activation-memory headroom, which is what lets the consumer choose
microbatch size vs. gradient accumulation — Fuel says what fits, the consumer picks. C-2: matters for
spot-instance preemption, elastic training, and sharing a device with an inference consumer; quantum
= a step or microbatch. **C-3: this is durability, not load-shedding** — restore must be exact and
*must include the RNG stream position*, or a resumed run silently diverges from an uninterrupted one.
C-4: unchanged in shape. C-5: the strongest case in the taxonomy — a reproducibility requirement
prunes every non-deterministic arm, and a mixed-precision tolerance bounds which numerical arms are
legal.

**Open:** whether a trainer's C-3 and the inference C-3 share one mechanism with a fidelity flag, or
are genuinely two implementations. **[judgment]** one mechanism, two guarantees — but this should be
settled by whoever builds the second one, not asserted here.

---

## Annex C — Stateless batch (embeddings, retrieval, quantized-only paths)

**Unit** one batch. **State** none between calls. The degenerate case, and useful precisely because
it shows the contract *degrading gracefully*: this class needs C-1, C-4, and C-5 and nothing else.

C-3 is not applicable — there is no engine-held state to externalize. C-2 is optional; batches are
short, though cancellation still matters for large ones. C-5 carries real weight here: quantized-only
consumers are explicitly trading numerics for throughput, so the tolerance budget *is* the interface.

If a future clause cannot be omitted for this class, that is a signal the clause is really two
clauses.

---

## Annex D — Oracle / test runner

**Unit** one reference execution. **State** none. Throughput is irrelevant; correctness is
everything.

**C-5 is essentially the entire contract for this class** — require bit-exactness, forbid
non-deterministic arms, forbid approximate rewrites. That makes this class the natural *test* for
C-5: if a consumer can demand determinism and verifiably get it, C-5 works. It also connects to the
existing calibration discipline (sabotage-calibrated epsilons) and to the kiss-ref advisory
cross-check, which is an oracle consumer in all but name.

C-4 is still wanted — measured cost feeds the ledger and calibration even when the runner does not
care about speed.

---

## Annex E — IPC / remote (mlmf) — stub

Named because the ROADMAP already names it ("inter-process tensor exchange (Fuel ↔ Lightbulb ↔ mlmf
using safetensors as the wire schema)", `RemoteHostStorage` Phase 7c). The wrinkle is that state
crosses a process boundary, so C-3 becomes *serialization* rather than eviction and overlaps
[13-interchange](architecture/13-interchange.md). Left as a stub deliberately — writing it without a
consumer would be speculation.

---

## Annex F — RL / alternating train-infer loop (GRPO, RLVR, online learning)

**Unit** a rollout batch, then a policy update. **State** *both* kinds at once — KV caches from
rollouts (lossy, discardable) and parameters + optimizer moments + RNG stream (exact). **Cost**
rollout tokens, update step time, and the ratio between them.

This class matters out of proportion to its size because it is the **only** one that runs A and B
against the same weights with a mutation between phases. Every other class holds weights still for
its entire lifetime. Already on Fuel's roadmap (GRPO/RLVR, bucket D in
[`frontier-paradigms-vision.md`](frontier-paradigms-vision.md)), so this is a scheduled consumer, not
a hypothetical.

**Why it is the C-7 consumer.** After a policy update, everything derived from the old weights is
stale: rollout KV caches (semantically — they encode the old policy's continuations), captured runs,
and any arm choice or cost estimate conditioned on the old weights. The consumer knows the mutation
happened; Fuel does not, unless told. Nothing in the current surface provides the telling, and the
failure mode is silent rather than loud — a stale capture replays, and the run degrades instead of
erroring.

**Other clause pressure.** C-3 must serve both fidelities in one process, which is the strongest
argument for one mechanism with a fidelity flag rather than two implementations (Annex B's open
item). C-5 is doubly binding: rollouts and updates may legitimately want *different* determinism and
tolerance settings, so constraints must be per-phase, not per-process. C-2 matters because rollout
batches are long and the consumer wants to bound them against update latency.

**[judgment]** If C-7 is going to be built, this is the consumer that should specify it, and it
should be specified before the inference-side C-3 hardens — otherwise C-3 gets an inference-shaped
API that this class cannot use.

---

## Annex G — Observation / analysis (calibration, distillation, interpretability)

**Unit** one instrumented execution. **State** none. **Defining clause** C-6; without it this class
cannot exist at all, which is why it earns a row despite being clause-light everywhere else.

Quantization calibration reads activation statistics to compute scales. Distillation reads teacher
intermediates. Interpretability reads, patches, and ablates. Debugging — every consumer's fallback
mode — reads whatever is going wrong. These differ in what they do with the value, not in what they
need from Fuel.

C-5 is joint-binding with C-6 here: an observation is only meaningful if the consumer also controls
the numerics that produced it. Calibrating scales against an ε-close fused arm and then deploying
against a bit-exact one (or the reverse) is a silent correctness bug, and the consumer needs to be
able to pin both.

**[judgment]** This class is the natural first consumer of C-6 because calibration is concrete,
already needed by the quantized paths Fuel supports, and produces a checkable artifact — which makes
it a better proving ground than interpretability, whose requirements are open-ended.

---

## Relationship to the Reasoning-Runtime sketch

Context only; nothing above depends on it. Under a "cognitive engines as schedulable resources"
framing, the layering is nested runtimes, each the same shape, each with a less fungible cost unit
than the one below. **Fuel is not an engine** — it is the substrate an engine is built from (Candle's
former slot). An inference host like Lightbulb is externally *one* engine and internally a runtime;
the engine boundary is the **unit of resource arbitration**, not the unit of capability, so several
models sharing one VRAM pool are one engine with a union capability profile, not three engines.

A *trainer* is not an engine in that framing at all — it serves no requests. That is the sharpest
reason the contract stays consumer-agnostic: **Fuel must not assume the inference shape**, because at
least one first-class consumer class doesn't have it. (The constitution now says this directly —
[15 §Consumer-shape neutrality](architecture/15-consumer-contract.md#consumer-shape-neutrality) —
after the 2026-07-28 withdrawal of `09`'s inference-center-of-gravity claim.)

The consequence is that C-1…C-7 are not speculative infrastructure for a hypothetical layer. They are
what multi-session serving and reproducible training each need on their own merits.

---

## Open questions

1. ~~Promote the core to a numbered constitution section?~~ **Done 2026-07-28** —
   [15-consumer-contract](architecture/15-consumer-contract.md) v0.1, with `09-non-goals` v0.4
   correcting the inference-center-of-gravity claim. See
   [10-decisions-log §2026-07-28](architecture/10-decisions-log.md).
2. **Does `multi_session.rs` move to `fuel-inference`, or does the layer table get an argued
   exception?** Recommendation: move, before the port. (Annex A.2 — verified defect.)
3. **Is C-3 in scope for Increment 2?** Highest-value clause, only one with no partial
   implementation, and it overlaps the confirmed-absent block-pool allocator behind `Op::PagedAttn`
   (ROADMAP §4) — the two may be one piece of work. Building it needs the inference *and* training
   fidelity requirements settled together (Annex B open item).
4. **Rename `SchedulePolicy` → `DecodeArm`?**
5. ~~Where does sampling live?~~ **RESOLVED 2026-07-29 (Annex A.3) — consumer policy.**
   `Lightbulb/src/sampling.rs` is host-side post-processing over *realized* logits, with
   constrained generation layered on top in `contracts/`. Fuel should produce logits and stop;
   `SessionState::sample_and_append` is misplaced and rides the `fuel-inference` move. Not an eighth
   clause. Sub-finding bounds C-3-exact: the consumer owns its sampler's RNG, so the "RNG stream
   position" an Exact handle must cover is *Fuel's* RNG, never the consumer's.
6. ~~What is the minimum a consumer needs to port at all?~~ **RESOLVED 2026-07-29 (Annex A.3) — the
   clauses do not gate the port.** Lightbulb keeps its own engine, cache policies, memory, slot pool,
   scheduler, sampler and API; the port is a tensor-layer swap concentrated in `model/` (16 of 17
   files), not a re-architecture around this contract. C-1…C-7 are adopted incrementally. The real
   gates are Lightbulb-side (eager→lazy, 70 value-extraction sites) plus a Fuel-side parity list —
   `fuel-transformers`, `fuel-nn` coverage, the quantized surface (AWQ, Marlin), and the error type.
   See the reverse gap list in A.3.
7. **Does C-6 land as Phase-9 `RuntimeHook`, or a separate seam?** The hook machinery is specified
   but unbuilt (**[verified]** — it appears in `ROADMAP.md` and two frontier docs, and in zero source
   files); if it is built for constraint-gating without the consumer-facing observe/intervene
   surface, C-6 will need a second mechanism later. Cheaper to design once.
8. **Should C-7 be built now or specified now and built with Annex F?** It is undefined in a
   direction that fails *silently*, which argues for at least a loud interim: a declared
   `weights_mutated()` that conservatively invalidates everything derived, tightened later.
9. **Is one C-3 mechanism with a fidelity flag right, or two implementations?** Annex F forces the
   question because it needs both in one process. Recommendation: settle before the inference-side
   C-3 hardens into an inference-shaped API.
