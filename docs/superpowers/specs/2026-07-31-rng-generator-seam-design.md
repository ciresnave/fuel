# RNG / generator seam — design

**Date:** 2026-07-31
**Status:** design approved (CireSnave). **Basis addition GATED on a KISS RFC** — see §11.
**Co-designed with:** KISS (`j9hpae4h`, steward), kiss-ref (`cb9y44nb`), Baracuda (`gnglmh0y`)

---

## 0. The gap

`ROADMAP.md` carries this as an open design gap: *"where a `Generator` lives (per-backend /
per-device / per-graph), how it threads through realize and autograd, and how backends
participate — which blocks dropout, sampling-as-a-graph-op, and stochastic training ops."*

Verified current state:

- **No RNG op exists.** No `Rand`/`Bernoulli`/`Normal`/`Uniform` in the `Op` enum. Randomness
  is entirely outside the graph.
- **Host sampling already follows §15.** `fuel-core/src/lazy_nn/sampling.rs`: *"every
  stochastic helper takes a `&mut StdRng` so the caller controls the RNG seed end-to-end."*
- **CUDA already answered "where does the Generator live" — badly for our purposes.**
  `fuel-cuda-backend/src/device.rs` holds `CudaRng(baracuda_curand::Generator)` behind
  `Arc<Mutex<..>>`, seeded `299792458`, with `set_seed()`. Per-device, backend-owned, stateful.
- **Dropout bakes a host-sampled mask into the graph as a const**
  (`fuel-core/src/lazy_nn_dropout.rs`), which names the intended end state
  `Op::BernoulliMask { p, seed }`.

**The cost of the status quo is worse than "no dropout."** From that file: *"Callers that want
a fresh mask per step must rebuild the dropout node each step with a fresh seed."* A fresh
graph per training step **defeats plan-once** — the D1→D2→D3 property and CapturedRun replay
that took a whole program to win — and embeds a full `[B,T,d]` const per dropout layer per step.

## 1. The crux: purity

A random op is not a pure function of its inputs. Fuel's DAG is input-independent so one plan
serves every step, and CSE, caching and captured replay all assume purity.

**Resolution: counter-based (stateless) RNG.** `(key, counter)` are *values*; "fresh
randomness" comes from advancing a runtime scalar, not from mutating hidden state.

KISS's steward position, to become a normative `ops.md` clause: KISS-Ops is referentially
transparent **by construction**, so counter-based is not a preference — it is *the only
formulation admissible under the grammar*. A stateful generator breaks CSE (two draws
collapse), replay (not reproducible) and plan-once simultaneously.

## 2. The primitive — one basis slot

```
Op::RandomBits { alg: RngAlg, stream: u32 }      ← the ONE new primitive
   attrs:    seed (u64), alg, stream        → node identity, base_map_hash, §6.19
   launch:   counter                        → class-8 scalar op param (§5); NOT a tensor edge
   operands: none
   output:   U32 tensor, shape declared on the Node

bernoulli_mask(p)  ⇒  RandomBits < ⌊p·2³²⌋              ← INTEGER compare
uniform_f32        ⇒  RandomBits → mantissa splice → [0,1)
normal_f32         ⇒  Box-Muller over two uniform_f32
```

Distributions are **fused ops with `decompose` recipes**, not basis members. One bucket-E slot
instead of four. KISS concurs strongly — it is how the suite already treats non-primitives
(`reduce_mean` → its §6.13 decomposition), and distributions-as-basis would bloat the floor
and duplicate what a recipe expresses.

**Output shape needs no new mechanism.** Fuel's `Node { op, inputs, shape, dtype }` carries
shape as a field, so `RandomBits` declares its shape like any node and the builder supplies it
(`dropout(x, p)` uses `x.shape()`). Because `Shape` carries **symbolic extents**, a mask over a
symbolically-shaped activation just works. A static shape *attr* would have quietly broken
dropout under the data-dependent-shape substrate — ruled out deliberately, not by luck.
(kiss-ref reaches a different answer — a shape-of-sibling edge, as their `Iota` uses — because
their recipe DAG does not carry shape. Two correct answers to different structures, not a
disagreement.)

## 3. Three properties that fall out of purity

1. **CSE is sound**, because nodes are pure — but see §4, which is the hazard purity *creates*.
2. **Captured replay is byte-exact**, because the same `(key, counter)` re-evaluates identically.
3. **Forward/backward mask consistency is free.** `RandomBits` is non-differentiable — a leaf,
   like a const — and dropout's gradient flows through the `Mul`. Backward re-evaluates the
   node and gets **identical bits**, so there is nothing to stash. PyTorch saves the dropout
   mask for backward; this design does not have to. A memory win that falls out of the purity
   chosen for replay.

## 4. Streams — the hazard purity creates

If `RandomBits` is pure, two dropout layers with the same shape, same `p`, and the same
key/counter are **structurally identical nodes, and CSE will merge them** — both layers get the
same mask. A silent correctness bug that statefulness hid by making every call different.

**Each random op carries a distinct `stream`, assigned at graph-build time as an attr**, so it
participates in node identity: CSE merges genuinely-identical uses and never merges independent
ones. Monotonically increasing per graph, deterministic (same build order → same ids →
reproducible), and stable under later rewrites because it is baked at build rather than derived
from post-optimization structure. Explicit override available for deliberate sharing.

Rejected: deriving the stream from node id or structural hash — any pass that renumbers or
reorders would silently change every mask, making a graph rewrite a numerics change.

## 5. The counter — consumer-bound, loudly

The counter is an ordinary **runtime scalar binding** supplied per realize. Fuel owns the slot;
the consumer owns the advance policy, per §15.

**Operand role — SETTLED, no grammar increment needed** (KISS steward, 2026-07-31). The counter
is **launch-scalar class 8, "scalar op params"** (Contract §6.5-0004a pinned class table:
`param{i}`, typed by the op's scalar compute dtype). Classes 1–7 are structural (per-operand
extents, strides, iteration count, base offsets, gather/index extents, workspace ptr/bytes);
class 8 is precisely the runtime-bound, value-carrying slot. Launch scalars are bound
**per-dispatch, not baked**, so a class-8 counter varies each step *without re-planning the
graph* — exactly the plan-once property this design exists to preserve. The split maps cleanly:

```
alg, key, stream  →  OpAttrs        (§6.19, node identity, base_map_hash)
counter           →  class-8 launch scalar   (runtime; NOT in §6.19)
```

A rank-0 scalar tensor operand is a second admissible modeling, but class-8 is the better fit:
the counter is an *op param*, not a data input, and keeping it off the tensor-edge graph is what
plan-once wants.

### 5.1 Node identity and the CSE rule (normative)

A class-8 launch scalar **does not participate in node identity**. That is correct for this
model, but it must be stated rather than inferred, because it is the second half of §4's story:

> `RandomBits` node identity is **`(alg, key, stream, graph-position)`**. The counter is a
> launch parameter and is **never** part of identity. CSE folds two `RandomBits` nodes iff those
> attrs *and* their derived global position match — **never on counter value**.

The consequence is the one the design wants: the *same* node, re-launched with counter *N* then
*N+1*, produces two steps' bits from one plan; and replay re-evaluates that node and gets
identical bits because the counter is re-supplied rather than remembered. Two *different* nodes
stay distinct on `stream` regardless of what counters they are ever launched with.

```
plan.bind_scalar(RNG_COUNTER, step)?;
plan.realize()?;

// unbound:
→ Err(UnboundRngCounter { stream })     // loud, typed, never silent
```

`realize` stays **idempotent** and replay stays byte-exact. The failure mode of explicit
binding — forget to advance, get identical masks every step, silently — is converted into a
typed error at realize.

Rejected: Fuel auto-advancing a per-graph counter. That reintroduces the mutable hidden state
counter-based RNG exists to avoid; realize would stop being idempotent and captured replay
would need a rewind path.

## 6. Algorithm selection

```
enum RngAlg { Philox4x32_10 }        // open: Threefry4x64_20, other Random123-family
```

**Selection is not speculative generality — it is what makes "match external convention"
satisfiable at all** (a standing project norm). Ecosystems pin different generators: PyTorch
uses Philox on CUDA, JAX defaults to Threefry. A consumer porting a model that must bit-match a
reference needs *the matching* generator. Hardcoding one would make that permanently impossible
without a basis change.

- **`alg` lives in OpAttrs, not the classify token** (KISS steward). It changes output bits for
  a given `(key, counter)`, so it is *semantic identity*: it participates in `base_map_hash` and
  §6.19 canonical serialization, like `monoid` on reduce. The classify `target_capability` token
  is hardware specialization — a different axis. Which algorithms a backend *serves* is a
  separate §6.7 Capabilities / §7.4 advertisement matter.
- **`alg` is an attr, so CSE cannot merge nodes drawn from different generators** — the same
  hazard class as `stream`, the same fix.

**Admissibility is a build-time criterion, not a doc caveat.** Only **counter-based** generators
may be variants. Baracuda's `RngKind` also exposes XORWOW/MRG32K3A/MTGP32/MT19937; those are
stateful by nature and for Fuel are **inexpressible**, not merely unguaranteed — a generator
with internal state cannot be written as a pure function of `(key, counter)`, so CSE, replay and
backward-parity break structurally, not numerically. Leaving the enum nominally "open" without
this criterion would be a trap that silently exempts dropout from every gate.

**Governing rule: the enum is cheap to design, expensive to populate.** Each variant arrives
with the *full* cross-backend parity gate. No algorithm lands on one backend only — otherwise
we recreate the promoting-cast failure where the arm exercised in tests is not the arm that runs.

## 7. Bit-identity, and why cuRAND is not used

**Hard requirement (CireSnave):** the same `(alg, key, counter, stream)` produces bit-identical
values on CPU, CUDA and Vulkan. Without it, stochastic ops are structurally exempt from every
cross-backend gate Fuel has — CPU-reference parity, the Judge's `max_ulp` ledger, kiss-ref
cross-check, sabotage-calibrated tolerances.

**`baracuda-curand` cannot serve this, and the reason is shape, not bits** (Baracuda's finding).
It wraps cuRAND's **host** API — `curandCreateGenerator` + `curandSetPseudoRandomGeneratorSeed(u64)`
— which is (a) **stateful**, advancing internally per generate call, and (b) **seed-only**, not
exposing the `(seed, subsequence, offset)` device init. A stateful host generator cannot serve a
pure op regardless of how its bits come out. Separately, cuRAND applies its own undocumented key
derivation from the `u64` seed, so a caller cannot reproduce the stream off-device.

**Decision: raw Random123 Philox-4x32-10 at the explicit `(2×u32 key, 4×u32 counter)` level, the
same ~30-line function on all three backends.** Bit-identity then holds **by definition** — same
algorithm, same inputs. The CUDA arm is a small device function, not a cuRAND wrapper.

kiss-ref's framing of why this matters most: because every backend runs the *same pure function*
rather than two implementations being reconciled, the oracle is not asserting *"these two agree"*
but *"every backend computed the same function correctly."* **Any mismatch is a backend bug,
never a reference-vs-production semantic gap.** That eliminates a whole category of "is the
reference wrong or is the kernel wrong" investigation before it can exist.

## 8. The counter-derivation clause (normative)

Philox is per-element counter-based, so bit-identity hinges on every backend deriving the **same**
per-element counter. Left implicit, two conforming backends disagree on element order and the
oracle cannot distinguish that from a real bug (kiss-ref). Four pins close it:

```
seed : u64  (build-time attr)
key[0] = (seed        ) & 0xFFFF_FFFF
key[1] = (seed   >> 32) & 0xFFFF_FFFF

linear_index = element's LOGICAL row-major (C-order, last axis fastest) position
               in the node's declared output shape
block_index : u64 = linear_index / 4
counter[0] = (block_index      ) & 0xFFFF_FFFF     // block_lo
counter[1] = (block_index >> 32) & 0xFFFF_FFFF     // block_hi
counter[2] = base                                  // runtime-bound scalar
counter[3] = stream                                // build-time stream id

out  = philox4x32_R(10, counter, key)              // Random123
word = out.v[ linear_index % 4 ]                   // Random123's PUBLISHED lane order
```

**Why a structured split rather than `base + linear_index`.** With an additive counter the
consumer must advance `base` by ≥ element count each step or streams overlap between steps —
silently, and the required stride *changes with shape*. With disjoint fields, "advance `base` by
1 per step" is correct for every shape and non-collision is **structural**. That is what a
128-bit counter is for.

**Why `ctr_lo` in `counter[0]` specifically.** Philox's convention is that `counter[0]` is the
fastest-varying word. So incrementing `block_index` by 1 is *byte-identical to one Random123
`incr()`* — carry flows `counter[0]→[1]` and never touches `base`/`stream`. A backend reaching
for the stock helper lands on the right bytes **without reading this spec carefully**; the
reverse ordering would make the natural implementation silently wrong. Make the correct thing
the default thing.

Baracuda's stronger form of the same point: the carry chain structurally cannot reach `base` or
`stream`, so distinct `(base, stream)` own **provably disjoint** counter spaces — non-aliasing
is a property of the layout, not something a reviewer re-derives.

**Logical, not physical.** `linear_index` is the element's **logical** position, *not* its
physical iteration or storage order. A backend may tile, vectorize, reorder or parallelize
freely; it must only agree on the logical index. Without this sentence a conforming backend
cannot use its natural launch geometry; with it, bit-identity costs **nothing at runtime**.
Output is a fresh contiguous row-major `U32` tensor of the declared shape, so there is no
strided or aliased case to specify.

**Non-normative implementation note.** The mapping is defined element-wise; implementations
should evaluate per **4-aligned logical block** — one Philox eval per 4 outputs, 4 coalesced
writes. A naive one-element-per-thread mapping runs the full 10-round eval 4× and discards 3
lanes: a 4× throughput cliff with zero parity signal.

**Documented bound.** The 128 bits are fully spent (64 block + 32 base + 32 stream). Per
`(seed, stream)`: `base` wraps after 2³² steps; `block` covers 2⁶⁴ blocks × 4 words = 2⁶⁶
elements. The element bound is unreachable; the step bound is astronomical **only if `base` is a
step index**. A consumer repurposing it as a global monotonic event counter across graphs and
runs could approach 2³² far sooner, so `base` is contractually a per-`(seed, stream)` **step
index**; global-counter use is out of contract.

**Residual class.** After these pins the only remaining divergence is an implementation getting
Random123's internals wrong — the `mulhilo` 32×32→64 split, the round key bumps
(`0x9E3779B9`/`0xBB67AE85`), the round count. That is not a spec ambiguity; it is an
implementation bug, and **the published Random123 vectors are load-bearing, not illustrative** —
they are what converts *"references Random123"* into *"provably computes Random123."*

## 9. Position-pure ops — a class invariant

The invariant has **two layers**, because the class has two kinds of member (KISS steward). An
earlier draft of this section stated the stream obligation as binding the whole class, which is
wrong — `Triu`/`Tril` carry no stream:

> **(a) Index rule — binds every position-pure op**, whether position-*generating*
> (`RandomBits`, `Op::Iota`) or position-*dependent* (`Triu`/`Tril`, position-derived
> encodings): derive the element's position from the **global logical row-major index in the
> unsharded logical shape**, never a partition- or rank-local index. A rank-local `Triu` mask is
> as wrong as a rank-local `Iota` value — both diverge silently under sharding.
>
> **(b) Stream rule — binds only members carrying a generator stream** (`RandomBits`): where the
> op is reproduced across ranks, **all ranks share one `stream` and one `base`**.

Splitting them keeps `Triu`/`Tril` in scope for the index rule without implying they have a
stream to share.

This is stated for the class, not for `RandomBits`, so the next such op inherits it by
construction (Baracuda). The KISS steward affirms it belongs in **KISS-Ops as a general clause**
rather than an RNG-specific one — *"put the universal rule where it's universal"* — and will
draft it. The rationale line they asked to carry with it: **gate on bits, not plausibility.**

**It is live in Fuel, not hypothetical.** `Op::Iota { len }` exists (`fuel-graph/src/lib.rs:238`),
as do `Triu`/`Tril`. Fuel's optimizer places *whole nodes* on devices and does not partition one
node's output — but `fuel-parallel/src/tensor_parallel.rs` does column/row-parallel sharding
(`TensorShard { rank, world_size }`, `shard_range`), ported to the lazy surface this week. There a
logically-single tensor is produced **across ranks, each building its own graph over its own
shard** — separate nodes in separate graphs that must nonetheless agree.

**`RandomBits` is the highest-stakes member because its wrongness is invisible.** A rank-local
`Iota` emits visibly-wrong sequential integers and someone notices. A rank-local RNG emits bits
that still look random, still have the right mean and variance, still pass every distributional
check — and simply do not match the other rank. The output carries no signal that cross-rank
identity failed. That is the whole argument for gating on published vectors rather than on
statistical plausibility.

**The payoff: this subtracts a subsystem Megatron had to build** (Baracuda). One rule gives
correct dropout in *both* tensor-parallel regimes:

- **Sharded region** (dropout after a column-parallel linear): rank *r* owns the disjoint global
  range `[r·h/tp, (r+1)·h/tp)`. Each rank draws from its global indices → the union across ranks
  is bit-identical to a single-device dropout over the full dimension. Megatron achieves this
  with a dedicated model-parallel RNG state.
- **Replicated region**: identical global indices → identical bits. Megatron's data-parallel
  RNG state.

So counter-based global-index derivation **collapses Megatron's dual-RNG-state machinery into
one rule**, with no notion of "which parallelism region is this tensor in." A real subtraction
from the TP implementation, not merely a correctness note.

**Plumbing is already present.** `TensorShard::shard_start()` (`rank * per_rank`) *is* the
global-index offset; it needs threading to the position-pure op's lowering so
`global_index = shard_start() + local_index`. Note shard **starts** are uniform but shard
**ends** are not — the last rank absorbs the remainder — so the offset must come from
`shard_start()`. Any implementation reconstructing position from an assumed-uniform extent is
wrong on the last rank only: another silent, config-dependent divergence.

## 10. Verification

`RandomBits` is a **kiss-ref floor atom**, determinism class **`ExactByte`** (KISS §6.0-0002):
exact equality, no ULP tolerance, no epsilon calibration, no drift-vs-corruption judgement. It
lands on the integer recipe lane kiss-ref shipped this week (`eval_recipe_int`, exact
`i128`/wrapping semantics).

**Dropout never leaves the exact-byte lane at any point.** `bernoulli_mask` is a `u32` compare
producing a `b1` mask, reduced/counted into `i64` — all on the integer lane. The single
highest-traffic stochastic op in training never touches a float. This is a *consequence of
choosing an integer atom over a float one*, not a lucky property, and no float-RNG framework can
claim it. Only `uniform_f32`'s mantissa splice (bit-deterministic) and `normal_f32`'s Box-Muller
touch floats, and the latter is a §6.13 recipe over two uniforms.

**Two-layer corpus** (kiss-ref owns it):

1. **Algorithm layer** — Random123's *published* vectors anchor `philox(key, counter)`. kiss-ref
   is the differential **target, never the oracle**: its implementation is *checked against*
   the published vectors, so nobody has to trust kiss-ref's word for the algorithm.

   **Exact provenance — do not paraphrase this** (verified by kiss-ref, 2026-07-31):

   | | |
   | --- | --- |
   | repo | **`DEShawResearch/random123`** — *no hyphen*; `DEShaw-Research` does not exist |
   | file | **`tests/kat_vectors`** — *no extension* |
   | blob | commit `d8a0c25e`, 2021-01-17, stable since |

   **Trap:** `tests/ut_uniform_kat_vectors.dat` also exists in that repo and is *not* this — it
   is the uniform-distribution set, not the raw generator KAT. Anchoring to it would validate
   the wrong layer.

   **The anchor must be fetched, never recited — and this is not a theoretical rule.** On
   2026-07-31 both parties independently held a *wrong* recollection of the all-`ffff` vector:
   the remembered value differed from upstream by a full 32-bit word. Had either side supplied
   it from memory, a **correct** Philox implementation would have failed its own KAT — sending
   the author to debug a right answer, or worse, to "fix" the implementation to match the wrong
   number and ship a subtly broken generator that every backend then faithfully reproduced. The
   failure mode fired; the fetch-don't-recite discipline caught it. Cite the file, never the
   digits.

   **The same discipline applies to the pointer, not only the payload.** The first version of
   this reference, given from memory, named the wrong org *and* the wrong filename — a
   plausible-looking citation that 404s. A recalled path is the same failure class as a recalled
   value, with a smaller blast radius only because it fails loudly. Verify any recalled
   specific, not merely the obviously-numeric ones.

   **And it applies to transcription — the step most likely to be botched by someone who did
   everything else right** (kiss-ref). Fetching the file and then *hand-typing* the vectors into
   a test table reintroduces the exact failure the fetch was meant to eliminate:
   hand-transcription is recall with extra steps. **Machine-generate the anchor table from the
   fetched blob** (e.g. `awk` over the file → array literals) so there is no human in the loop
   between upstream and the anchor. This matters most for the *extended* corpus: 52 vectors ×
   10 hex words is precisely the volume at which manual transcription is tedious enough to rush
   and long enough to hide one wrong digit.
2. **Mapping layer** — a separate corpus for "given declared shape and `base`, what is element
   *i*", minted by kiss-ref. **A mapping bug therefore cannot hide behind an algorithm bug**, and
   because the class is `ExactByte` the oracle names the *exact diverging index* rather than
   reporting drift — a one-shot diagnosis of a backend's index-order bug instead of a bisect.
3. **Increment-coherence vector** — same `(seed, stream, base)` at `block_index` N and N+1,
   asserting the N+1 counter bytes equal `incr(counter_N)`. This is a **structural** assertion
   where the others are *value* assertions: it tests the invariant the layout was chosen for, so
   a backend that got endianness or slot order wrong fails on the invariant rather than on a
   mystery byte mismatch. It converts §8's "the natural helper-based implementation is correct"
   rationale into something a backend fails loudly.

`RandomBits` advertising `ExactByte` plugs directly into the comparator-selection machinery KISS
landed in increment 3b: the contract advertises the determinism class, the differential harness
verifies byte-exactness.

## 11. Process gate — the RFC

**This is a basis addition populating KISS's reserved-but-empty `rnd` family**
(`spec/classify.md:695` lists `random`→`rnd` in the closed set of op-family categories;
`spec/ops.md` defines no op in it; kiss-ref has no RNG surface).

Per the KISS steward: route it through an **RFC**, as the shape-expression oracle was. They
co-shepherd the KISS side; Eric ratifies. **The `Op` enum must not land the primitive before the
RFC ratifies the basis slot** — the same "don't ship a basis nobody ratified" discipline the
`vulkan:` namespace is under.

**This reorders the increments favourably** (§12): the Philox function, the counter clause and
the conformance vectors are all ratification-independent, so the RFC arrives with a working,
vector-verified implementation behind it rather than a proposal.

## 12. Increments

The KISS steward has **green-lit this ordering from the KISS side** as the
"working, vector-verified implementation arrives *with* the RFC" discipline: build the
ratification-independent parts now, hold only the basis slot.

**Increment 1 — ratification-independent, CPU-only, no GPU slot required.**
Philox-4x32-10 as a plain function + the §8 counter derivation + conformance against the
published Random123 vectors. Born-red gate: the published vectors. No `Op` variant, no basis
change, nothing blocked on the RFC.

**Increment 2 — the basis addition** (after RFC ratification). `Op::RandomBits` + CPU kernel +
`bernoulli_mask` recipe + the dropout port that removes mask-baking and restores plan-once for
training graphs.

**Increment 3 — the other backends.** CUDA and Vulkan device functions, gated on the same
vectors, plus the §9 rank-offset threading in `fuel-parallel`.

**Increment 4 — `uniform_f32` / `normal_f32` recipes**, and sampling-as-a-graph-op (which
removes the per-token host barrier in serving).

## 13. Open items

- ~~**KISS Q4 — the operand role.**~~ **RESOLVED 2026-07-31, affirmatively — no grammar
  increment needed.** The counter is launch-scalar **class 8** (§6.5-0004a). See §5. This had
  been the design's only genuine blocker. Its resolution surfaced the §5.1 identity/CSE clause,
  which was implicit and is now normative.
- ~~**Whether the §9 class invariant belongs in KISS-Ops**~~ **RESOLVED — yes, as a GENERAL
  clause**, not RNG-specific (KISS steward: "same principle as where the fixed-width-alphabet
  rule went — put the universal rule where it's universal"). The steward will draft it.
- ~~**`Op::Iota` / `Triu` / `Tril` under tensor parallelism — UNAUDITED.**~~ **AUDITED
  2026-07-31 at the KISS steward's request. Verdict: the §9 invariant is PREVENTIVE, not
  retroactive — there is no live violation.** Evidence:
  1. **`fuel-parallel` has zero consumers** — no crate's `Cargo.toml` depends on it. Built,
     ported to the lazy surface, unwired.
  2. **`fuel-parallel` constructs no position-pure ops.** No `iota`/`triu`/`tril`/`arange`
     anywhere in its source; its TP `Linear::forward` is `apply_linear` and nothing else.
  3. **`Op::Iota` has exactly one construction path** — flash-attn's **alibi** decompose
     (relative-position bias, `fuel-core/src/lazy.rs`), which is not TP-sharded. `Triu`/`Tril`
     take **one input**: they mask an existing tensor rather than generating positions, so they
     are position-*dependent*, not position-*generating*.

  **This is "not currently reachable", not "structurally safe."** The trigger condition is
  specific: `fuel-parallel` gaining a consumer **and** a sharded tensor feeding a position-pure
  op. That argues for landing the §9 clause **before** `fuel-parallel` is wired rather than
  after — the invariant is cheap now and would be an archaeology exercise later.

  **An oracle exists for re-auditing this** (kiss-ref): their `Node::Iota` is position-pure *by
  construction* — a serial reference with no tensor parallelism, so logical-coordinate ==
  physical-output holds trivially. So Fuel's TP-threaded `Iota` (and `Triu`/`Tril` once
  reference-covered) can be **differentially tested** against kiss-ref's serial version, where a
  position-purity violation surfaces as an exact mismatch at the diverging index. The audit does
  not have to stay a reasoning exercise about threading.
- **RFC drafting.** The steward has offered to draft three §6 clause texts: the class-8 counter
  clause, the purity clause, and the general position-pure clause. Ping when the RFC starts.

## 14. Rejected alternatives

| Rejected | Why |
| --- | --- |
| **Stateful per-device generator** (what CUDA has today) | Breaks CSE, replay and plan-once simultaneously. Inadmissible under a referentially-transparent grammar, not merely undesirable. |
| **Wrapping `baracuda-curand`** | Host API is stateful and seed-only; cuRAND applies an undocumented key derivation from the `u64` seed, so the stream is not reproducible off-device. Shape mismatch decides it before bit-exactness does. |
| **Distribution ops as basis members** (`Uniform`/`Normal`/`Bernoulli`) | Four basis slots instead of one, duplicating what a `decompose` recipe expresses. |
| **Fuel auto-advancing the counter** | Reintroduces mutable hidden state; realize stops being idempotent; captured replay needs a rewind path. |
| **Stream from node id / structural hash** | Any renumbering or reordering pass silently changes every mask, making a graph rewrite a numerics change. |
| **Additive counter (`base + linear_index`)** | Requires the consumer to advance `base` by ≥ element count, with a stride that changes with shape — silent, shape-dependent stream overlap. |
| **Non-counter-based algorithms in `RngAlg`** (XORWOW, MRG32K3A, MT19937) | Inexpressible as a pure function of `(key, counter)`; would silently exempt dropout from every gate. |
| **Static shape attr on `RandomBits`** | Would break masks over symbolically-shaped activations under the data-dependent-shape substrate. Fuel's `Node` carries shape with symbolic extents instead. |
| **Bit-identity only under an explicit determinism request** | Two implementations per backend, and the arm exercised in tests may not be the arm that runs — the promoting-cast failure. |

## 15. Attribution

The design improved materially under adversarial review; these were not Fuel's:

- **KISS steward (`j9hpae4h`)** — purity as a normative grammar clause rather than a preference;
  `alg` in OpAttrs not the classify token; `ExactByte` determinism class and its tie to
  comparator selection; the RFC route and the ratification gate.
- **kiss-ref (`cb9y44nb`)** — that the shape↔counter mapping was implicit at all; the key-split
  and output-lane endianness holes; the two-layer corpus architecture and "a mapping bug cannot
  hide behind an algorithm bug"; the increment-coherence vector; the differential-target-never-oracle
  discipline.
- **Baracuda (`gnglmh0y`)** — that cuRAND's host API is the wrong *shape* regardless of bits;
  raw Random123 on all backends; the counter-word and output-lane pins; the non-aliasing argument
  for the layout; the position-pure op **class**; and the Megatron dual-RNG-state subtraction,
  which reframed global-index derivation from a hazard avoided into a capability gained.
