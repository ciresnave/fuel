# KISS RFC — `RandomBits`: a counter-based generator for the reserved `rnd` op family

**RFC:** (number to be assigned on filing to ThinkersJournal) · **Status:** Draft · **Author:** Fuel (RNG-seam agent) · **Date:** 2026-08-01
**Affects:** KISS-Classify **§6.8** `rnd` op-family (reserved at `classify.md:695`, currently **empty**) · KISS-Ops (new op + a normative purity clause) · KISS-Contract **§6.5-0004a** (class-8 launch scalar — *confirmed adequate, no amendment sought*) · **§6.19** canonical serialization · **§6.0-0002** determinism classes
**Category:** Standards-track. Populates a reserved-but-empty family; additive.

> **Pre-verified.** Unusually for a basis addition, this RFC arrives with a **working, vector-verified implementation behind it** rather than as a proposal. Two independent implementations — Fuel (`fuel-cpu-kernels/src/philox.rs`, on `main`) and kiss-ref (`feat/randombits-philox`) — were written **blind of each other** and both match the upstream Random123 KAT on all 55 vectors. The KISS steward green-lit this ordering: build the ratification-independent parts first, hold only the basis slot.

## Summary

KISS-Classify reserves a `random` → `rnd` op-family in its **closed** category set, but KISS-Ops defines no op in it and kiss-ref had no RNG surface. This RFC proposes the family's first occupant: **`RandomBits`**, a *counter-based* (stateless) generator that is a **pure function of `(key, counter)`**, plus three distributions expressed as §6.13 recipes over it rather than as additional basis members.

The central claim: **counter-based is not a preference, it is the only formulation admissible under a referentially-transparent grammar.** Everything else in this RFC follows from that.

## Motivation

A stateful generator — the conventional shape, and what every mainstream framework ships — breaks three KISS-Ops properties *simultaneously*:

- **CSE soundness.** Two draws with identical inputs are structurally identical nodes; a grammar that permits CSE will merge them, and a stateful RNG's whole contract is that they differ.
- **Byte-exact replay.** Re-executing a captured graph cannot reproduce hidden generator state.
- **Plan-once.** If freshness comes from mutating state, freshness cannot come from a re-bindable parameter — and a new graph per step defeats plan reuse entirely.

Fuel's concrete symptom today: dropout **bakes a host-sampled mask into the graph as a const**, so a fresh mask per training step requires a fresh *graph*. That is the cost of having no RNG op, and it is paid in the property the framework works hardest to preserve.

## Proposal — one basis occupant, three recipes

```
Op::RandomBits { alg: RngAlg, stream: u32 }
   attrs:    seed (u64), alg, stream      -> node identity, base_map_hash, §6.19
   launch:   counter                      -> class-8 scalar op param (§6.5-0004a)
   operands: none
   output:   U32 tensor of the declared shape

bernoulli_mask(p)  =>  RandomBits < floor(p * 2^32)      -- integer compare
uniform_f32        =>  from_bits(0x3F80_0000 | (w >> 9)) - 1.0
normal_f32         =>  basic Box-Muller over two uniform_f32
```

**One basis slot, not four.** Distributions are §6.13 recipes, exactly as the suite already treats `reduce_mean`. Making them basis members would bloat the primitive floor and duplicate what a recipe expresses.

### Normative clause 1 — purity

> An RNG op **MUST** be a pure function of its explicit key and counter. **No implicit generator state.**

Requested as an explicit `ops.md` clause rather than left implicit, so a stateful proposal is rejected by the grammar rather than by argument. The steward's position: counter-based is the only admissible formulation under referential transparency.

### Normative clause 2 — node identity and CSE

> `RandomBits` node identity is **`(alg, key, stream, graph-position)`**. The **counter is a launch parameter and is never part of identity.** CSE folds two `RandomBits` nodes iff those attrs *and* their derived global position match — **never on counter value**.

This is the subtle half. A class-8 launch scalar does not participate in node identity, which is *correct* here but must be stated: the **same** node re-launched with counter *N* then *N+1* yields two steps' bits from one plan, and replay re-evaluates to identical bits because the counter is re-supplied rather than remembered. Two **different** nodes stay distinct on `stream` regardless of the counters they are ever launched with.

`stream` is assigned at graph-build time and carried as an attr precisely so CSE cannot merge independent randomness — the hazard that *purity itself creates*, and which statefulness hid by making every call differ.

### Normative clause 3 — position-pure ops (general, not RNG-specific)

Requested at the **KISS-Ops level**, governing a class rather than this op — the steward's framing: *put the universal rule where it is universal.* Two layers, because the class has two kinds of member:

> **(a) Index rule — binds every position-pure op**, whether position-*generating* (`RandomBits`, `Iota`) or position-*dependent* (`Triu`/`Tril`, position-derived encodings): derive the element's position from the **global logical row-major index in the unsharded logical shape**, never a partition- or rank-local index.
>
> **(b) Stream rule — binds only members carrying a generator stream**: where the op is reproduced across ranks, **all ranks share one `stream` and one `base`.**

**Rationale to carry with the clause: gate on bits, not plausibility.** A rank-local `Iota` emits visibly-wrong sequential integers and someone notices within an afternoon. A rank-local RNG emits bits that still look random, still have the right mean and variance, and **pass every distributional check** — while simply not matching the other rank. The output carries no signal that cross-rank identity failed. That asymmetry is the whole argument for conformance-gating this family on published bit-vectors rather than on statistical tests.

## Algorithm and determinism

**`RngAlg` admits only counter-based generators — a build-time criterion, not a documentation caveat.** XORWOW / MRG32K3A / MTGP32 / MT19937 are stateful by nature and therefore **inexpressible** as a pure function of `(key, counter)`; admitting them would silently exempt every consumer from the guarantees above. First and only variant: **Philox-4x32-10** (Random123).

**`alg` belongs in OpAttrs, not the classify token** (steward's ruling): it changes the output bits for a given `(key, counter)`, so it is semantic identity and participates in `base_map_hash` and §6.19 — like `monoid` on reduce. Which algorithms a backend *serves* is a separate §6.7 Capabilities / §7.4 advertisement matter.

**Determinism classes (§6.0-0002) — deliberately NOT uniform across the family:**

| | class |
| --- | --- |
| `RandomBits` | **`ExactByte`** — pure integer function |
| `bernoulli_mask` | **`ExactByte`** — integer compare, never touches a float |
| `uniform_f32` | **`ExactByte`** — bitcast + a subtraction exact by Sterbenz's lemma |
| `normal_f32` | **NOT `ExactByte`** — `ln`/`sqrt`/`sin`/`cos` are not bit-identical across backends |

**The exactness boundary is the mantissa splice and Box-Muller, not the RNG.** Stating that explicitly matters: a reader who assumes the whole family is bit-exact will file a bug on the first cross-backend normal draw that differs by a ULP, and be wrong.

A consequence worth naming: because `bernoulli_mask` is an integer compare, **dropout — the highest-traffic stochastic op in training — never leaves the exact lane at any point.** That is a consequence of choosing an *integer* atom over a float one, not a lucky property, and no float-RNG framework can offer it.

## Conformance

**Anchored upstream, not on either implementation.** The authority is Random123's published vectors: repo **`DEShawResearch/random123`** (no hyphen), file **`tests/kat_vectors`** (no extension, 3 canonical) plus `tests/old_kat_vectors` (52 systematic). *Not* `ut_uniform_kat_vectors.dat`, which is the uniform-**distribution** set and is what the obvious search finds.

kiss-ref is the differential **target, never the oracle** — its implementation is *checked against* the published vectors, so no one has to trust it for the algorithm. Two-layer corpus: the **algorithm** layer (upstream vectors) and a separate **mapping** layer (given a declared shape and `base`, what is element *i*), so **a mapping bug cannot hide behind an algorithm bug**. Plus an **increment-coherence** vector asserting `counter(N+1) == incr(counter_N)` — a *structural* assertion where the others are *value* assertions, so a wrong counter layout fails by naming the invariant rather than as a mystery byte mismatch.

**The vectors are load-bearing, not illustrative.** After the derivation is pinned, the only remaining divergence is an implementation getting Random123's internals wrong; the vectors are what converts *"references Random123"* into *"provably computes Random123."*

> **Process note worth generalising.** During implementation, both parties independently held a **wrong recollection** of the all-`ffff` vector — off by a full 32-bit word. Had either supplied it from memory, a *correct* implementation would have failed its own KAT. Anchors must be **fetched, never recited**; the same applies to the *pointer* (a recalled repo path had the wrong org and filename) and to **transcription** (machine-generate the table from the fetched blob; hand-typing is recall with extra steps).

## Layer boundary — what this RFC does **not** ask for

- **No §6.5 amendment.** The counter fits **class 8, "scalar op params"**, already in the pinned class table. The steward verified this; it is recorded here so a future reader does not re-open it.
- **No grammar node in kiss-ref.** Deliberately deferred. Fuel diffs *values*, never imports a kiss-ref `FlatDag` node for `RandomBits`, so adding one would force a breaking release with no consumer — and `RandomBits` is int-lane while the distributions produce float, so a single-lane evaluator could not express them anyway.
- **No tolerance for `normal_f32`.** Deliberately unset. Tolerances are **sabotage-calibrated** — measure genuine cross-backend drift *and* the signal from a deliberately corrupted implementation, then set the bound between. A constant invented in this document would be the guessed-anchor failure one layer up. Until a second backend exists, `normal_f32` is tested **structurally** (see below).

## Testing a tolerance-free op

Three assertions cover `normal_f32` with no calibrated constant:

1. **The reflection** — `box_muller(u1 = 0.0, _) == 0.0` exactly, since `r = sqrt(-2·ln 1) = 0`. `uniform_f32` yields `[0,1)`, so `u1 = 0.0` is attainable at 2⁻²³ — *routine* at tensor scale — and without the `1.0 - u1` reflection `ln(0) = -inf` produces a NaN. The input most likely to be mishandled is also the one with an exact expected answer.
2. **The Pythagorean identity** — `z_even² + z_odd² == r² == -2·ln(1 - u1)`. The pair shares one radius and differs only by `cos` vs `sin`, so their squares sum to `r²` regardless of the angle. Catches a **cos/sin swap, a wrong-pair mapping, or a mis-derived radius while asserting no absolute value at all** — a relation among outputs rather than a comparison against a golden, so it needs no ULP budget.
3. **Position-purity** — element *i* recomputed in isolation reproduces itself.

The generalisation: **a structural assertion tests the property the design was built on and therefore needs no oracle; a value assertion needs one.** For an op deliberately tolerance-free until calibration that is the only kind of test available — and it is the more diagnostic kind regardless, since a failure names the broken invariant rather than an index.

## Backward compatibility

Purely additive. The `rnd` family is reserved and empty; no existing op, recipe, or serialization changes. Consumers that do not use RNG are unaffected. The class-8 counter uses an existing launch-scalar slot.

## Questions for the steward — ANSWERED 2026-08-01

**Q1 — clause placement. CONFIRMED: KISS-Ops level, two layers.** Layer 1 (global-logical-index)
binds *every* position-pure op — `Iota`, `Triu`/`Tril`, `RandomBits`. Layer 2 (shared
`stream`/`base` across ranks) binds only the stream-carrying members. It carries the
*gate-on-bits-not-plausibility* rationale, and it does **not** live inside the RNG op — the
steward's reasoning: *burying a universal rule in one op is how the next position-pure op
violates it*, the same reasoning that put the fixed-width-alphabet rule in §6.8 rather than
inside `vulkan:`.

**Q2 — `rnd` classify sub-structure. LEAN: no. Explicitly NOT a ruling.** The steward's lean is
that `RandomBits` is the sole `rnd` basis occupant, the distributions are non-primitive recipes
distinguished by op name, and a non-primitive decomposes to its floor — so the specialization
cell keys on the *floor* ops, not a distribution super-node. **They declined to rule from
memory** and will verify against the classify text before the RFC lands. Recorded as
lean-not-ruling so nobody downstream treats it as settled.

**Q3 — ratification. ANSWERED, and it is broader than this draft assumed.** Per the
shape-expression-oracle precedent, *ratified* requires **all** of:

1. the RFC merged into KISS `rfcs/`; **and**
2. the normative clauses merged to KISS `main` — purity, identity/CSE, position-pure; **and**
3. **`RandomBits` registered in the `ops.md` §6.1 op-set registry**; **and**
4. **its classify `rnd` occupant landed.**

Items 3 and 4 were **not** in this draft's model of the gate — it assumed the three clauses were
sufficient. Registration in the op-set registry and the classify occupant are what actually make
the basis slot *official*. Until all four are on KISS `main`, Fuel holds `Op::RandomBits` out of
its `Op` enum, the same discipline `vulkan:` is under. The steward drives the KISS-side landing
and binds kiss-ref + Baracuda where their cosign applies.

### Determinism framing — sharpened by the steward

The non-uniform class is not a special case for this family; it is an instance of an existing
rule. `normal_f32` contains a transcendental, so **§6.0-0003** makes it ULP/tolerance, and
**§6.0-0005 precedence** resolves the mix: *ULP wins over the exact-byte bits it is computed
from.* So the family needs no bespoke carve-out — it needs the precedence rule applied and the
boundary named. The framing to carry into §6.0-0002: **the exactness boundary is the mantissa
splice and Box-Muller, not the RNG.**

## Remaining open

- **Q2's verification** against the classify text (steward, before the RFC lands).
- Nothing else is open on the KISS side. Fuel's remaining increments are blocked on **hardware
  access**, not on ratification timing — a machine-wide GPU-execution hold pending a lockfile,
  unrelated to this RFC.

## Status of the implementations

| | state |
| --- | --- |
| Fuel Philox core + §8 counter derivation | **on `main`** (`4393ba0e`), 55/55 upstream vectors |
| kiss-ref core + atom + all three distributions | **GREEN**, `feat/randombits-philox` |
| Cross-diff | both match upstream ⇒ both match each other, incl. the 52 systematic inputs |
| `Op::RandomBits` in Fuel's `Op` enum | **held, pending this RFC** |
| CUDA / Vulkan device functions | not started |
| `normal_f32` tolerance | uncalibrated by design; needs a second backend |

Full design, with the normative counter-derivation clause and all four of its pins:
[`docs/superpowers/specs/2026-07-31-rng-generator-seam-design.md`](../superpowers/specs/2026-07-31-rng-generator-seam-design.md).
