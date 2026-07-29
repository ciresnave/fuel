# Consumer contract

**Status**: v0.7 (draft, 2026-07-29). v0.7 (MINOR) **scopes C-1 by tier** ahead of the memory-mapped durable tensor store: under mmap there is nothing to refuse — a mapping succeeds beyond physical RAM, residency is the kernel's decision, and **the failure mode changes from a wall to a slope**, which is strictly worse for a consumer because C-1 exists to let it act *before* a limit. Device tier unchanged; host tier becomes best-effort with **degrade-rather-than-refuse**, plus an opt-in enforceable resident cap that is [C-5's budget quantity axis](#the-resource-budget-fuel-owns-the-facts-the-consumer-owns-the-entitlement), not new machinery. Also pins ***mapped* vs *resident* bytes** as a distinction any byte-reporting surface must state. v0.6 (MINOR) **corrects C-5's third item**, which read "the consumer supplies which devices; fuel decides how to use them" — both halves wrong, and its rationale had never been written. Fuel *does* know device availability ([05](05-backend-contract.md) already requires memory-pressure, slot-count and queue-depth telemetry); what fuel cannot know is **entitlement** — that GPU 0 drives the display, that another tenant has a claim, that only half the VRAM may be consumed. And a *set* is the wrong shape: "≤ 50% of the RTX 4070's VRAM" is an ordinary requirement and is not a set of devices. The item is now a **per-device resource budget** (admissibility + quantity) with an all-visible default, and the rationale is stated. v0.5 (MINOR) records the resolution of the arm question: attention probabilities are **not obtainable on the fused arm by design** (the op's header states that avoiding the `[B,Hq,Sq,Sk]` materialization *is* its value), corrects "scores" → **`probs`** (post-softmax) as the tensor a hot-path observer needs, upgrades preference 2 from speculative to an **existing contract-checked mechanism** (`output_views` has two live consumers; `flash_attn` declares `None`), and adds [C-5 §Arm choice can prune the consumer's own policy set](#arm-choice-can-prune-the-consumers-own-policy-set--say-so-dont-let-it-be-discovered) — an arm constraint can silently disable capabilities a consumer built on that arm, and the failure mode is a policy *starved of input*, not an error. v0.4 (MINOR) refines v0.3's order of preference: preference 2 is **a second output from the producing op** ([12-multi-output](12-multi-output.md)), not a speculative side buffer — and adds [§Observability can be arm-dependent](#observability-can-be-arm-dependent--which-makes-it-partly-a-c-5-question), because a reduction is only expressible in-graph if the reduced value exists as a node, which can depend on which arm the optimizer picked. Verified case: attention scores are a real node on the decomposed arm (`registry/flash_attn.rs:235`) and never written on the fused flash arm. That makes a hot-path observation request partly a **C-5** arm-pruning question. v0.3 (MINOR) adds [C-6 §Two regimes](#two-regimes--and-the-obligation-differs-added-v03-2026-07-29): C-6's stated obligation ("report the cost, never silently deoptimize") is correct for *occasional* observation and **wrong for hot-path observation** — per-token, per-layer attention observation for KV eviction defeats fusion everywhere and makes capture unformable, so telling a consumer it deoptimized is not a resolution. Adds an order of preference led by **express the reduction in-graph, don't observe the intermediate**. Surfaced by the Lightbulb port session auditing its real observation sites against the clause. v0.2 (MINOR) adds [§Where the seam runs](#where-the-seam-runs-foundation-not-the-repository) — the contract binds *Foundation*, not every crate in the workspace; `fuel-inference` and `fuel-training` are consumer-side toolkits above the seam, optional by construction. Added in response to the first real consumer hitting the ambiguity (the Lightbulb port session, 2026-07-29): §15 refused admission/eviction/fairness while `fuel-inference` shipped a scheduler with priority queuing and eviction-pressure admission control. No core-claim change — the refusals bind Foundation exactly as before. v0.1 (2026-07-28) established the upward-facing half of fuel's boundary obligations: what fuel provides to the systems built *on* it, and what fuel correspondingly doesn't decide. The mirror of [05-backend-contract](05-backend-contract.md). Motivating phase doc: [`docs/fuel-consumer-seam.md`](../fuel-consumer-seam.md), which carries the per-consumer-class annexes and the as-built audit.

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

#### C-1 holds where fuel controls a finite budget — and memory-mapped host storage is not that

**The guarantee above is tier-scoped, and the scoping has to be stated before the durable tensor store lands.** Fuel has decided that all host storage becomes **memory-mapped to a file** — so the file tier stops being separate from the host tier: what is in host memory *is* on disk. **[verified 2026-07-29]** that decision is **not implemented** (`fuel-memory` performs no mapping) and **not in this architecture set** — [`docs/storage-unification.md`](../storage-unification.md) `:778` still says *"Storage on disk / memory mapping. Out of scope here."* So this is a design window, not a retrofit.

**Why it breaks C-1 on the host tier: under mmap there is nothing to refuse.** A mapping succeeds beyond physical RAM. What is *resident* is the kernel's page-cache decision, not fuel's allocation decision. Three consequences:

- **"Free bytes" stops being a quantity fuel controls**, so it cannot honestly be advertised as headroom.
- **`mapped` and `resident` diverge arbitrarily** under memory pressure, and only one of them is a budget.
- **The failure mode changes from a wall to a slope** — and *a slope is strictly worse for a consumer than a wall*, because C-1 exists to let it act **before** a limit. Remove the limit and the symptom becomes thrashing: gradual, silent, and invisible to admission control. There is no event to react to.

**So C-1's admission guarantee is scoped as follows:**

- **Device tier — unchanged.** VRAM is real, finite, and fuel-controlled; free-block counts mean what they say, and admission against them works.
- **Host tier under a memory-mapped store — best-effort, and fuel must say so.** Capacity there is advisory; a consumer must be able to **degrade rather than refuse**, and one that builds admission logic on host-tier headroom is building on sand. Stating this is not optional: a consumer cannot infer it, and the clause as written invites the wrong assumption.
- **An enforceable host ceiling is available as an opt-in — and it is C-5, not new machinery.** A consumer-imposed **resident cap** (held with `mlock`/`madvise`) converts the slope back into a real budget, at the cost of pinning. That is exactly [C-5's resource budget](#the-resource-budget-fuel-owns-the-facts-the-consumer-owns-the-entitlement) *quantity* axis applied to the host tier: the consumer states what it is permitted to keep resident, and fuel enforces rather than observes.

**And a distinction C-1 needs regardless of mmap: *mapped* versus *resident* bytes.** They are the same today, which is why this clause can say "bytes" unqualified. After the durable store they are not — and every consumer reading a name like `bytes_resident` will assume the stricter meaning. **Any surface reporting bytes must say which it means**; a method that reports *mapped* bytes under a *resident* name is a promise it cannot keep.

**Follow-up owed to the set:** the durable-store decision is recorded in session memory but appears nowhere in `docs/architecture/`, and `storage-unification.md` actively contradicts it. That is doc-vs-decision drift of the same class this survey has been correcting all day, and the storage sections need to catch up before the store is built.

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
- **Resource budget** — *which* devices may be used and *how much of each*. See below; this is the item that was previously mis-stated as a "device set".

Fuel optimizes *within* these constraints and never around them. C-5 is what keeps fuel's aggressive arm selection safe for consumers whose correctness bar is stricter than "fast and close enough."

#### The resource budget: fuel owns the facts, the consumer owns the entitlement

This item previously read *"the consumer supplies which devices; fuel decides how to use them."* **Both halves were wrong** (corrected 2026-07-29), and the rationale had never been written down at all — it was introduced by analogy with determinism and tolerance and then leaned on to rule out an allocator shape.

**Fuel does know device availability.** It probes topology, and [05-backend-contract](05-backend-contract.md) already requires backends to report **memory pressure** (bytes available vs total), **currently-available slot count**, and **queue depth** as continuous telemetry. Claiming fuel cannot discover system state contradicts a contract fuel already enforces downward.

**What fuel cannot know is *entitlement*** — which is a fact about the operator's intent, not about the machine:

- that GPU 0 drives the display and must be left alone;
- that another workload, tenant, or process has a prior claim;
- that this deployment may consume at most *half* the RTX 4070's VRAM, leaving headroom for something fuel will never observe.

So the split is not facts-vs-facts, it is **facts vs. policy over the facts**: *fuel observes what exists and what is in use; the consumer expresses what it is permitted to consume.* That is the same line as C-1 — fuel advertises capacity, the consumer decides admission — applied to the coarsest resource there is.

**A set is the wrong shape.** "Don't use more than 50% of the RTX 4070's VRAM" is not expressible as a set of devices, and it is an ordinary requirement. The constraint is a **per-device budget** with (at least) two axes:

- **admissibility** — may this device be used at all;
- **quantity** — how much of it: VRAM as a fraction or an absolute, and potentially slot or compute share.

**Defaults and directionality:**

- Unconstrained means **all visible devices at full capacity** — a consumer must not be forced to enumerate hardware before it can run.
- Fuel may narrow its own use for cost reasons, but must **never expand beyond** the budget.
- The consumer must **always be able to narrow** it, at any granularity the budget expresses.

That last property is the load-bearing one: it is what makes residence externally constrainable, and therefore what forbids any allocator that picks residence internally. Note it is *stronger* under a budget than under a set — an allocator must respect "≤ 50% of device 0", not merely "device 0 is permitted".

#### Arm choice can prune the *consumer's own* policy set — say so, don't let it be discovered

A constraint that removes arms can also remove capabilities the consumer built **on top of** those arms, and that consequence is invisible from either side alone.

The worked case: an inference host's eviction policies are not uniformly available. Attention-driven policies (H2O heavy-hitter, R-KV importance) need the attention probabilities, which exist only on the decomposed arm — so they are **arm-gated**. Recency, sink-window/StreamingLLM, and span-level eviction are not; they need no such observation. A consumer choosing the fused arm for throughput silently loses half its policy catalogue, and the failure mode is not an error but **a policy starved of input** — it runs, and scores nothing.

So the obligation is a reporting one: **where an arm constraint gates a capability, fuel states it at constraint-admission time rather than letting the consumer find a policy quietly receiving no data.** A capability that is *conditionally* available is a different thing from one that is available, and a consumer cannot infer the condition from the clause list.

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

The worked case **[verified 2026-07-29]**: attention probabilities are materialized on the **decomposed** arm — `registry/flash_attn.rs:285` builds `probs` (the post-softmax node; `:235`'s `scores = scale · (q · kᵀ)` is its pre-softmax input) — and are **never written** on the fused flash arm. That is not an oversight: the op's own header states that avoiding the `[B,Hq,Sq,Sk]` materialization *is* the op's value, and that lowering it to primitives "would be a footgun" — either reproducing the blowup or pretending an equivalence the tiled softmax does not have. So "reduce the attention probabilities" is a well-formed graph reduction against one arm and a request for a *different kernel* against the other.

*(Correction of record: this section first said "scores". A hot-path observer such as H2O needs **`probs`**, post-softmax — `scores` is the wrong tensor and would have been the wrong thing to ask a backend for.)*

Consequences for the order of preference above:

- **Preference 1 (reduce in-graph) may silently cost the fast arm.** That is a legitimate, *measurable* trade — the consumer can benchmark decomposed-plus-reduction against fused-without-observation and choose. C-5 is the clause that makes the choice explicit rather than accidental, and C-4 is what makes it measurable.
- **Preference 2 (second output from the producing op) is the only route that keeps both — and it is an *existing* mechanism, not speculative infrastructure. [verified 2026-07-29]** [12-multi-output](12-multi-output.md)'s `output_views` is contract-checked (`Graph::set_output_views`, `lib.rs:1813`, five invariants) and has **two live consumers**: `registry/selective_scan.rs:130` and `registry/ssd_chunk_scan.rs:119` both declare `output_views: Some(..)`. `registry/flash_attn.rs:59` declares `None`. So the second-output route is a mechanism one op has not opted into, rather than one that must be built.

  **[judgment, consumer-supplied]** the cost argument for why this need not defeat the kernel's purpose: a hot-path observer wants a reduction over the **query** axis while flash tiles over **keys**, so it is orthogonal to `softmax_lse`; the `[Sq, tile]` probs block is transiently in fast memory, so a per-tile column-sum into an `[Sk]` output is **O(Sk), not O(Sq·Sk)** — roughly `1/Sq` of the matrix. **Not measured, and kernel feasibility is the backend's call, not this section's.**

**The general rule**: before promising a consumer an in-graph reduction, check whether the value it reduces survives the arm the consumer actually wants to run on.

**Sharpened by KISS (2026-07-29), and it improves the framing.** In the KISS vocabulary attention is **not an op — it is a recipe** (`probs = softmax(QKᵀ)`; `out = probs·V`). So a hot-path observer is asking for *"a multi-output recipe whose second output is a reduction of an interior node"*, which the recipe grammar already expresses — **no new primitive**. Crucially: **a fused kernel's non-materialization of `probs` is a *lowering* choice, not a semantic one.** The recipe always has the interior node; some lowerings keep it, some fuse it away.

That relocates the difficulty without dissolving it. The clause's arm-dependence stands, but its *cause* is cleaner: observability is not a property of attention, it is a property of the chosen lowering — which is exactly why it belongs to C-5 (a constraint that prunes lowerings) rather than to C-6 (a request the op cannot satisfy). What remains is a **cost** question for the backend, not an expressibility question for the IR.

**A narrow spec gap this exposes, and it is a C-4/C-5 concern.** KISS §7.4-0001 advertises a determinism class **per op**. A secondary *reduction* output can carry a **different class from its primary** — a per-tile accumulation may be non-deterministic where the primary output is not. So a consumer constraining determinism (C-5) or accounting measured cost (C-4) against a multi-output op needs **per-output** granularity, which is not currently pinned. KISS has offered to take the clause; recorded here because the consumer-facing half is this section's.

*(Also closed: `softmax_lse` cannot serve as the reduction. It is per-query over keys; a heavy-hitter observer needs per-key over queries — orthogonal, and not derivable from one another.)*

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
