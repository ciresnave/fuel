# Fuel — non-transformer paradigms: architecture vision (2026-07-07)

**Purpose.** A durable capture of how Fuel's architecture absorbs the ML paradigms
*beyond* autoregressive transformers — energy-based / associative-memory models,
neurosymbolic execution, symbolic regression & KAN, gated compute orchestration, and
tensor-train compression. Companion to [`frontier-architecture-gaps.md`](frontier-architecture-gaps.md),
which already maps the *transformer-adjacent* frontier (hybrid SSM, MLA, sparse-MoE,
test-time compute, GRPO). This doc covers the paradigms that catalog does **not**.

**Authority.** The constitution ([`docs/architecture/`](architecture/00-index.md)) wins; the
[`ROADMAP.md`](../ROADMAP.md) owns sequencing. Like the gap catalog, this doc **tracks and
classifies — it schedules nothing on its own.** When an item here gets a phase, point its
"Home" at it.

**Central finding.** These are **not five bolt-on subsystems.** Fuel is already ~70% of the
substrate, because its three load-bearing ideas are exactly the ones these paradigms need:

1. **The DAG is a symbolic-algebra system, not just a compute graph.** The optimizer already
   treats the graph as algebraic expressions and searches for equivalence rewrites
   ([01-identity §edge 2](architecture/01-identity.md)). Symbolic regression, symbolic
   distillation, symmetry detection, and the *symbolic* half of neurosymbolic are **the same
   machine** pointed at learned/measured curves instead of authored ones.
2. **Closed primitive basis + open fused-op registry with a mandatory two-way recipe**
   ([03-ir §op vocabulary](architecture/03-ir.md)). Most "new layers" are a registry entry +
   `decompose` recipe + kernels — **zero** core change.
3. **Self-describing storage** ([`fuel-ir/src/stype.rs`](../fuel-ir/src/stype.rs), explicitly
   extensible) + lazy-realize + mmap-larger-than-RAM — which is Module 5 (tensor-train)
   almost verbatim.

The value of this vision is the **classification**, because the constitution assigns a
different cost and a different *home* to each class. Bolting on five sibling "modules" would
fail the [identity-enforcement check](architecture/01-identity.md#how-this-identity-is-enforced).

---

## The classification spine

Every item on the request sorts into six buckets, cheapest → most invasive.

| Bucket | Class | Cost | Home | Paradigms landing here |
| --- | --- | --- | --- | --- |
| **A** | **Fused op + recipe + kernels** (compose existing primitives) | Cheap, additive, constitutional | `FusedOpRegistry` + an NN-tier leaf | Modern/dense Hopfield, KAN edges, TT-layers, soft-MoE, GNN-style state graphs, differentiable constraint penalties |
| **B** | **New `Encoding`** (self-describing storage) | Cheap | [`stype.rs`](../fuel-ir/src/stype.rs) `Encoding` + FDX + kernels | Tensor-train cores (Module 5.1) |
| **C** | **New optimizer analysis/rewrite pass** (reuse the DAG-as-algebra engine) | Medium; *most Fuel-native* | `fuel-graph` / `fuel-dispatch` optimizer | `symbolic_distill()`, symmetry detection, law discovery (Module 3) |
| **D** | **New training / RL construct** (reuse autograd + optimizer + RNG seam) | Medium; a leaf | `fuel-training` / new `fuel-rl` | EnergyMinimizer, contrastive-divergence / MCMC / annealing, SymbioticLoss, GRPO/RLVR |
| **E** | **New PRIMITIVE** (build-time `Op`-enum extension — rare, precious) | High; amends the basis | `fuel-ir` `Op` enum | **Two families only** — the unified `Scan` (Crux 1) and the data-determined-extent keystone |
| **F** | **Downstream orchestration / interface** (by constitutional design **not** in-graph) | A consumer, not core | `fuel-inference` / a downstream crate + Phase-9 hooks | MCTS/beam/search, the outer relaxation/solve loop, the coarse gated router, PDDL planning search, the discrete solver, the neuro-symbolic dual-pipeline loop |

Only bucket **E** touches the closed basis. Those are the two crux decisions below.

---

## Crux 1 — one unified bounded iteration primitive (RESOLVED: adopt)

Many paradigms on the list are fundamentally **iterative / recurrent / fixed-point**, which
an acyclic input-independent DAG cannot express as-is:

- SSM / RNN (continuous signal evolution) → recurrence
- Hopfield iterative retrieval & EBM energy relaxation (Module 1) → iterate-to-fixed-point
- Diffusion denoising (Module 1, noted as EBM's evolution) → fixed-step iteration
- Contrastive divergence / MCMC / annealing (Module 1.2) → sampling loops

The constitution **already concedes exactly one** of these: the higher-order **`Scan` `Op`
(G3)** for SSM ([gaps §1](frontier-architecture-gaps.md), [10-decisions-log 2026-07-03](architecture/10-decisions-log.md)).

**Decision:** design **one** bounded iteration family up front rather than bolting on a
separate `Fixpoint` for Hopfield/EBM and a `While` for diffusion later — per the collaboration
norm *"ship the missing primitive rather than punt."* A single
`Op::Scan { body, carry, bound, /* optional early-exit predicate */ }` serves **SSM + RNN +
Hopfield + EBM relaxation + diffusion** with one basis addition.

**What keeps it constitutional (mandatory design constraints):**
- The **body is a fixed sub-graph** and the iteration count is **bounded** (a capacity, exactly
  like the KV-cache runtime-offset pattern, [03-ir §State and runtime extents](architecture/03-ir.md)),
  so the graph stays input-independent and **one plan still serves every step**.
- It ships **both halves of the recipe** ([03-ir §recipe principle](architecture/03-ir.md)):
  a total, never-panic `decompose` to the **bounded-unrolled** primitive subgraph (so the base
  map stays primitive-closed and the optimizer can lower/cover through it) and a `pattern` to
  re-fuse. An associative/chunked-scan lowering is the efficient body; the unroll is the
  fixpoint form.
- Early-exit (fixed-point convergence for Hopfield/EBM) is a predicate over the carry, evaluated
  at the realize barrier — **not** unbounded data-driven search. Unbounded "iterate until a
  dead-end" (MCTS) is **not** this op; it is Crux-2 / bucket **F**.

**When built, this is a MAJOR IR change:** an `Op`-enum extension + a
[`10-decisions-log`](architecture/10-decisions-log.md) entry + an [`03-ir`](architecture/03-ir.md)
version bump, per the working agreement. It subsumes the standalone `Scan`-for-SSM item in the
gap catalog — build them as one.

---

## Crux 2 — the pure-symbolic engine is a first-class *interface*, not an in-graph op (RESOLVED: backend-like advertising)

Module 2 (PDDL, discrete logic, SAT/constraint solving) and the search half of Module 4 are the
one place the tensor-DAG model genuinely doesn't reach: a Prolog/SAT/PDDL solver is not a
differentiable tensor computation and **must not become a primitive** (that would need a
`Custom`/opaque node, which the constitution forbids — [01-identity §what fuel isn't](architecture/01-identity.md),
[09-non-goals](architecture/09-non-goals.md)).

**Decision:** treat a discrete solver **exactly like a backend** — it *advertises capabilities
and costs but never decides strategy* ([01-identity §backends advertise, they don't decide](architecture/01-identity.md)).
Concretely:

- The neural side maps soft activations → discrete symbols with **existing primitives**
  (argmax / quantize / gather) plus a **straight-through** autograd rule (a `fuel-autograd`
  backward-rule addition, node-general — see [`grad.rs`](../fuel-graph/src/grad.rs)).
- The solver interface consumes those symbols and returns a verdict, surfaced back into the
  graph as either a **penalty node** (soft constraint, differentiable) or a Phase-9
  **`RuntimeHook`** gate (hard constraint, realize-loop).
- The solver, the PDDL importer (a `fuel-formats`-style leaf emitting a constraint graph), and
  the dual-pipeline `SymbioticLoss` loop are **downstream** (bucket **F**), never in the DAG.

"First-class" here means a first-class **hook/interface**, not absorption. Pulling search
orchestration or logic solving *into* the DAG is the one way this whole effort could violate
Fuel's identity — it is explicitly out of scope.

---

## Per-paradigm mapping

Legend: **Bucket** column uses the spine above; **Home** names the crate/leaf (NN-tier =
alongside `fuel-nn`; Use-Case tier = alongside `fuel-inference`/`fuel-training`).

### Module 1 — Physics-inspired & dense associative memory (`nn.energy`)

Proposed leaf: **`fuel-energy`** (NN tier), plus a `fuel-training` extension for the minimizer.

| Feature | Bucket | What's needed | Notes |
| --- | --- | --- | --- |
| Continuous / modern Hopfield layer | **A** | `Op::Fused` `Hopfield{beta}`; decompose → `matmul→scale→softmax→matmul` | Modern Hopfield update **is** attention; ~free |
| Dense associative memory (polynomial power) | **A** | Same fused op with a Krotov polynomial-power / log-sum-exp energy param | Capacity toggle = the exponent param |
| Iterative retrieval to fixed point | **E** (Crux 1) | The unified bounded `Scan` with a convergence early-exit | Shared with SSM/EBM/diffusion |
| `EnergyMinimizer` (∂E/∂state) | **D** | Reuse existing autograd + SGD/AdamW ([`lazy_nn_optim.rs`](../fuel-core/src/lazy_nn_optim.rs)); point the optimized variable at the **input** node, scalar-`[]` energy output | Autograd is **node-general** — ∂E/∂state is already computable; this is the pleasant surprise |
| Contrastive divergence / MCMC / annealing | **D** | The **RNG/generator seam** (already flagged in ROADMAP Deferred backlog) + sampler ops | Shared dependency with GRPO sampling — do once, first |
| Ising solver | **D**/**F** | Energy fused op + the minimizer loop; discrete anneal schedule downstream | |

### Module 2 — Hybrid neurosymbolic execution (`nesy.core`)

Proposed leaf: **`fuel-nesy`** (Use-Case tier).

| Feature | Bucket | What's needed | Notes |
| --- | --- | --- | --- |
| PDDL parser / state-space validator | **F** | A `fuel-formats`-style importer emitting a constraint graph | Downstream leaf |
| Soft→discrete symbol mapping | **A** | argmax/quantize/gather + a straight-through backward rule | Primitives exist; add the STE rule |
| Constraint-violation flagging | **C**/**F** | Optimizer pass reading realized values (soft penalty) or a `RuntimeHook` gate (hard) | |
| `SymbioticLoss` (dual pipeline) | **D** | Loss node combining neural loss + logic-compliance penalty; feedback via STE grad or the RL loop | |
| Discrete solver (SAT/logic) | **F** (Crux 2) | Backend-like advertising interface | Never a primitive |

**Reframe to sell internally:** Fuel's optimizer is *already* the "System-2" symbolic engine
(algebraic rewrites, declarative pattern-matcher with variables). Neurosymbolic largely
**exposes** that machine; it does not build a new one.

### Module 3 — Automated law discovery / KAN (`symbolic.regression`)

Proposed leaf: **`fuel-symbolic`** (NN tier for KAN layers; passes live in the optimizer).
**The most architecturally beautiful fit.**

| Feature | Bucket | What's needed | Notes |
| --- | --- | --- | --- |
| KAN layer (learnable spline/poly edges) | **A** | `Op::Fused` `BSpline`/poly with learnable-coefficient weight operands; decompose → `basis-eval → weighted-sum` | Additive; `TTLinear`-shaped |
| `.symbolic_distill()` | **C** | A search over a function library fitting a curve → emits a **closed-form subgraph** replacing the spline node | Structurally identical to the optimizer's equivalence-rewrite search |
| Symmetry / separability detection | **C** | Autograd jvp/vjp probes + Phase-9 activation hooks; test Jacobian structure | Reuses autograd |
| Law discovery (swap layers for closed form) | **C** | A rewrite pass with an **empirical-fit** objective | **This is edge-#2 applied to learned functions** — strengthens identity-check #3 |

Sequence bucket-C items **into Phase 10** (equivalence-rewrite search); they share the engine.

### Module 4 — Intelligent compute orchestration (`routing.agent`)

Mostly already-Fuel. ⚠️ **Precision required:** `Op::Branch` is **plan-time selection among
implementations of the same math**, decided by the optimizer *before* execution — it is
**not** data-dependent token/route dispatch ([gaps §3 note](frontier-architecture-gaps.md)).

| Feature | Bucket | What's needed | Notes |
| --- | --- | --- | --- |
| Coarse gated router ("skip the net, run algebra") | **F** | A downstream realize-loop: realize a cheap classifier → branch in Rust → realize the chosen sub-graph | Constitutional; not in-graph |
| Fine-grained per-token / per-expert dispatch | **E** (keystone) | **Data-determined dynamic shapes** — Fuel's own #1 unlock ([gaps keystone](frontier-architecture-gaps.md)) | No new invention; ship the keystone |
| Multi-labeled state graph | **A** + **F** | Adjacency / message-passing over constant tensors (GNN pattern) + the Crux-2 KB interface | |

### Module 5 — Hardware-aware matrix compression (`tensor.train`)

**Best fit of all — nearly free.**

| Feature | Bucket | What's needed | Notes |
| --- | --- | --- | --- |
| `TTMatrix`/`TTLinear`/`TTEmbedding` | **A** | `Op::Fused` whose `decompose` reconstructs-or-contracts the cores; cores are separate `Const` operands | Lets the optimizer choose reconstruct-then-matmul **vs** fold-input-into-core-contraction, placed across backends (edges #1+#2) — the exact decision Fuel exists to make |
| TT `Encoding` variant | **B** | Add `TensorTrain` to [`stype.rs`](../fuel-ir/src/stype.rs) `Encoding` as a **storage/streaming marker** | Note: current `Encoding` is *per-element reinterpretation* (quant); TT is *multi-operand structural*, so the **fused-op recipe carries the math**, the Encoding just marks streaming residency |
| Lazy reconstruction / expand-in-cache / protect VRAM | — | **Already Fuel's execution model** — lazy realize + mmap-larger-than-RAM + kernel fusion | `@lazy_tensor` ≈ Fuel verbatim |

---

## Suggested sequencing (advisory — the ROADMAP owns the real order)

Respecting the existing active frontier (do **not** cut the dispatch-core / `fuel-ir`
retirement critical path) and the identity checks:

1. **RNG / generator seam** (already flagged) — shared prerequisite of EBM sampling *and* GRPO.
   Do once, first.
2. **The keystone — data-determined dynamic shapes** — already Fuel's #1 unlock; delivers
   Module 4's fine-grained routing + MoE-sparse for free.
3. **Crux 1 — the unified bounded `Scan` primitive** — one basis addition unlocking SSM + RNN +
   Hopfield retrieval + EBM relaxation + diffusion. Highest lever on *this* list.
4. **Bucket-A fused ops** (Hopfield, KAN, TT-layers, soft-MoE) + **Module-5 TT `Encoding`** —
   cheap, additive, high visible payoff; land incrementally.
5. **Crux 2 — the solver-advertising interface** + the `fuel-nesy` leaf — once (3) proves the
   iteration story.
6. **Bucket-C passes** (symbolic distillation / law discovery) — sequence into Phase-10
   equivalence-rewrite search; they share the engine.
7. **Bucket-D leaves** (EnergyMinimizer, GRPO/RLVR, SymbioticLoss) on the training stack.
8. **Bucket-F** (MCTS / beam / search) stays downstream, gated on a real consumer per Phase 9 —
   provide the substrate (forkable / COW KV, batched decode), not the search theory.

## What Fuel deliberately will **not** do

- No five parallel `nn.energy` / `nesy.core` / … subsystems as peers of the DAG. Each must
  decompose into the buckets above or it fails [identity-check #1](architecture/01-identity.md#how-this-identity-is-enforced).
- No search / logic-solving pulled into the graph (Crux 2). The DAG stays acyclic and
  input-independent; iteration is the **bounded** `Scan` or a downstream loop, never a cyclic
  graph.
- No `Custom` / opaque primitive "to make room" for KAN splines or symbolic ops — every one
  decomposes to the existing basis. That is the whole point of the recipe principle.

Each item, when picked up, must still move at least one of the four
[identity-enforcement checks](architecture/01-identity.md#how-this-identity-is-enforced) *more*
true and none less — and land lazy-only, test-gated, per the working agreement.
