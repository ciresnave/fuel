# Backend- and device-agnostic paged attention — seam design

**Status:** DRAFT. Two of three inputs outstanding (see §6). Nothing here is a
kernel request; Baracuda has been told explicitly not to build to this yet.

**Date:** 2026-08-05

---

## 0. What this document is answering

CireSnave: *"Lets proceed with the backend and device agnostic Fuel paged
attention code and see where it makes sense to land the backend-specific parts.
This may mean a new CUDA kernel in Baracuda but it may be smaller kernels to
interface better with the backend agnostic paged attention parts in Fuel. We
likely won't know for sure what needs to be built in kernels until we've at
least designed the backend and device agnostic parts."*

The sequencing constraint is the point: **the agnostic design comes first, and
the kernel shape is derived from it, not assumed.** Everything below is
organised so the kernel question is answered *last* and by evidence.

---

## 1. Finding: the agnostic layer already exists

`Op::PagedAttn` **already carries a total, never-panic, closed-basis primitive
decomposition** — `fuel-graph/src/registry/paged_attn.rs::recipe`, expressed as
portable `PatternNode` data and lowered through `decompose_via_recipe`.

```
IndexSelect(k_cache, block_table_flat)  →  Reshape  →  Permute  →  GQA-repeat
MatMul(q, kᵀ) → MulScalar(scale) → [softcap] → [alibi] → MaskedFill(-inf)
→ Fused(SoftmaxLastDim) → MatMul(probs, v)
```

Every node is in the closed primitive basis, so per G2 this is a real
decomposition rather than a basis-gap self-return. **Any backend can already run
paged attention today.**

> **The module header said the opposite** — "No primitive decomposition exposed
> at the registry layer … would defeat the design" — directly above code that
> does exactly that. Corrected in this branch. The stale claim mattered: read
> literally, it says the agnostic layer is *deliberately absent*, which would
> have made this design document start by building something that already
> exists.

### 1.1 What the floor costs, and why it is not the fast path

The recipe materialises `kv_len = max_blk · block_size` — the block table's
**padded capacity**, not the live `context_len`. It gathers dense
`[B, Hq, kv_len, D]` K and V, computes `[B, Hq, Sq, kv_len]` scores, and *then*
masks the tail with `context_lens`.

So the correctness floor's cost scales with **allocated capacity**, while paging
exists to make cost scale with **occupancy**. That is not a defect in the
recipe — a primitive-only lowering has no way to express "traverse only the live
blocks" — but it does mean:

**The agnostic layer that exists is a correctness floor. The agnostic layer that
is missing is an efficient formulation.** Those are different artifacts and
conflating them is how this ends up as a CUDA feature with a portable name.

---

## 2. Finding: no GPU backend implements PagedAttn

| fused op | cpu | cuda | vulkan |
|---|---|---|---|
| FlashAttn | 1 | **3** | 0 |
| FusedLinear | 1 | **2** | 0 |
| RmsNormLastDim | 1 | **1** | 0 |
| SoftmaxLastDim | 1 | **1** | **1** |
| **PagedAttn** | **1** | **0** | **0** |

(files mentioning the op, per backend crate)

Positive-controlled: the same query finds four other fused ops in the CUDA
backend, so `PagedAttn`'s zero is a real absence, not a failed search. The only
native implementation of paged attention in Fuel is **CPU**.

### 2.0 RESOLVED — the node runs on the HOST (measured 2026-08-05)

**`Op::PagedAttn` executes on the CPU during CUDA decode.** Per-node
`placement_of` over the real held decode plan
(`paged_decode_node_placement_report_cuda`, RTX 4070 Laptop, CUDA, 2-layer
model, 172 nodes):

```
Fused(PAGED_ATTN)   Cpu×2            <-- one per layer, ON HOST
MatMul              Cuda×15
Const               Cuda×31
BroadcastTo         Cuda×28
IndexSelect         Cuda×1
WriteSliceDoff      Cuda×4
Slice               Cuda×8
Silu                Cuda×2
Copy                Cpu×1  None×10   <-- the cross-device stitches
Permute             Cpu×2  Cuda×6    <-- neighbours dragged host-ward
Reshape             Cpu×2  Cuda×19
```

**Hypothesis (A) confirmed, (B) excluded**, and the two are not confusable:
(B) predicts *no* `PAGED_ATTN` node at all plus four recipe `IndexSelect`s per
layer. There is exactly one `IndexSelect`, it is on CUDA, and it is the
embedding lookup. **The recipe never fired** — the optimizer kept the fused node
and placed it on the only backend that implements it.

Instrument was positive-controlled two ways: `has_placements()` asserted before
reading (an all-`None` dump would read as "nothing is on CUDA"), and the
`PAGED_ATTN` identification is the registry constant compared in code, not a
number read off a label.

**Consequences that now follow, and did not before:**

1. The 186× host-ward DtoH is **the op's placement**, not `DeviceKvPool`
   plumbing. Both layers' KV caches cross the bus every token.
2. **A kernel does fix it.** §6.1's fork resolves to the kernel arm.
3. It explains the tell nobody predicted — the paged arm running *fewer* kernels
   (17,802 vs 36,718). One host op replaces the many device kernels dense
   attention would have launched.
4. Structural, not model-specific: the placement follows from "no CUDA backend
   advertises PagedAttn", which no model size changes.

### 2.1 Fact vs consequence — the reasoning this replaces

*(Retained because the discipline is the point, not because the question is
still open — §2.0 settles it by measurement.)*

The **fact** is: no CUDA/Vulkan `PagedAttn` implementation exists.

The **consequence people will want to draw** is: therefore the node runs on host
during CUDA decode and drags the KV cache back every token, which is the 186×
DtoH. **That does not follow yet**, because the optimizer has a second legal
option — lower to the §1 recipe and run every primitive on CUDA. Both behaviours
are consistent with the table above.

One piece of evidence already discriminates, and it points at host placement:
Lightbulb measured the paged arm running **fewer** kernels than contiguous
(17,802 vs 36,718). Decomposing into gather + dense SDPA would run *more*
kernels, not fewer. That is suggestive, not conclusive.

**This is exactly the trap this project has already fallen into three times** —
asserting paged placement from non-discriminating evidence, twice landing on
opposite answers. So it stays UNRESOLVED here and is settled in §6.1 by
`placement_of`, per-node, and by nothing else.

---

## 3. The seam

The efficient formulation is flash-decoding over blocks: never materialise the
dense scores; visit blocks, accumulate. Decomposed into agnostic primitives:

| # | primitive | class | today |
|---|---|---|---|
| 1 | gather-by-block-table | Indexed read | natively emittable |
| 2 | block-strided GEMV (`q · kᵀ`, `p · v`) | contraction | natively emittable |
| 3 | masked online-softmax accumulate | **reduction over a custom monoid** | **the one gap** |

Leg 3's combine, over partials `(m, l, acc)`:

```
m   = max(m₁, m₂)
l   = l₁·exp(m₁−m) + l₂·exp(m₂−m)
acc = acc₁·exp(m₁−m) + acc₂·exp(m₂−m)
```

### 3.1 Correction on record: this is a reduction, not a scan

I first drew leg 3 as a **scan**, reasoning that online softmax carries running
max/sum state between blocks. **That was wrong, and Baracuda corrected it.** The
combine above is *associative and commutative* — which is precisely why
FlashAttention/FlashDecoding merge split-KV partials in a tree. A sequential
implementation of an associative operator is a choice, not a constraint.

The error mattered beyond terminology: **drawing leg 3 as a scan would have
imposed a dependence the math does not have, forfeiting parallelism across KV
blocks** — i.e. exactly the split-KV and batch parallelism that batching exists
to obtain, in the one configuration (§5) where paged is the *only* available
path. A design-invalidating mistake, caught by a peer before it reached a seam.

The real resistance is narrower and better-defined: kernelgen's reduction Access
class admits a fixed `ReduceOp` set (sum/max/min/prod/mean/any/count). This
combine is a user-defined associative monoid over a 3-field tuple with
exp-rescale, which is not in that set. So the gap is **"reduction class cannot
yet express a custom monoid"** — backend-independent, and a much smaller thing
to build than a monolithic `paged_attn`.

---

## 4. Where the backend-specific parts land

Deliberately not decided here — that is §6's job. What the analysis *does*
constrain:

- Legs 1 and 2 look natively emittable, so **a monolithic vendored
  `paged_attn` would be vendoring two things that don't need it** to get the
  third.
- Baracuda's current CUDA `paged_decode` is a **vendored FlashInfer binding, not
  native kernelgen**. This is the concrete form of the one surviving argument
  from the original parking decision (FlashInfer-is-CUDA-only) — and it argues
  for the small-primitive decomposition on general grounds: *a seam only one
  backend can implement is not a backend-agnostic seam.*
- If leg 3 is the only gap, the seam should be drawn so that it is the **only**
  thing any backend must special-case.

---

## 5. Why this stopped being optional

Lightbulb's serving work has to choose a batching path:

- `build_batched_decode_logits` — hard uniformity gate on `cached_len`, and no
  persistent sibling (positive-controlled: 64 `DecodeSession` mentions in
  `lazy.rs`, so the absence is real). **Disqualifying for continuous batching.**
- `forward_paged_step_batched` — ragged supported per row.

So ragged continuous batching currently has **paged as its only path**. That
reclassifies paged attention from "a memory-efficiency option" to "a
prerequisite for the batching mode the serving engine needs" — which is the
argument that should drive priority, not the 10.4×.

---

## 6. Open inputs — the design does not harden until these land

### 6.1 `placement_of`, per node (mine) — ✅ DONE, see §2.0

**Answered: host-placed fused node.** The round-trip is the *op*, and a kernel
does fix it. The alternative (CUDA-placed recipe primitives, meaning the
round-trip was `DeviceKvPool` plumbing and a kernel would have been the wrong
build) is excluded.

This unblocks asking Baracuda for kernel work — but *what* to ask for still
waits on §6.2, because the k=1-vs-batched shape decides monolithic vs
primitives.

### 6.2 The decomposed arm — the real discriminator (mine)

**CORRECTION.** This section previously said the DtoH bytes/token slope
"plausibly decides monolithic-vs-primitives before anyone writes code." **It does
not, and that error propagated to Baracuda**, who sharpened it into an explicit
prioritisation rule before it was caught.

The flaw: **once ANY GPU implementation exists, the round-trip disappears
entirely.** Monolithic and small-primitives both eliminate it, equally and
completely. So the bytes/token slope characterises the *current broken state* —
an op on the wrong side of the bus — not a property separating two candidate
futures, neither of which has a round-trip at all. The 186× establishes that
**something** must be built; it is silent on **what**.

What actually decides it is **GPU-decomposed vs GPU-fused** — the fork Baracuda
originally named. §2.0 makes that arm sound like a dead letter (the recipe never
fires), but "the arm doesn't exist" is a reason to *create* it, not a reason to
stop treating it as the criterion. It is still the thing a fused kernel must
beat.

So the gating work is to **offer the decomposed arm and measure it**:

- **every recipe node CUDA-placed, fast enough** → primitives win; the 186× is
  removable with **no Baracuda kernel at all**;
- **CUDA-placed but slow** → almost certainly §1.1 (the recipe materialises
  padded capacity, not live `context_len`), which is the quantified case for a
  fused paged kernel — a *throughput* argument, not a round-trip one;
- **some primitive falls back to host** → that primitive is the ask, far
  smaller than "paged attention", and a decomposed arm cannot win until it
  exists.

### 6.2a Two axes the §1 measurement separates — and one it cannot

Refinement from Baracuda, adopted because it caught me sliding between two
claims. The §1 decomposed-arm dump separates exactly two things:

- **PLACEMENT** — §1 on host vs §1 on device.
- **FORMULATION** — §1 dense (cost ∝ *allocated capacity*, §1.1) vs §3 flash
  (cost ∝ *occupancy*).

It does **not** settle **FUSION** (monolithic vs small primitives), because
fusion is a question *within* §3 and the dump contains no §3 arm for a fused §3
kernel to be compared against. "Slow even when fully on-device" argues for the
flash **formulation**; it does not by itself argue for **fusion**. Those are
different claims and this document previously ran them together.

| question | settled by | status |
|---|---|---|
| does §1 run on device at all | §1 dump | in flight |
| how much of 10.4× is PLACEMENT vs FORMULATION | §1 dump | in flight |
| build §3 at all | the FORMULATION share above | pending |
| §3 monolithic vs §3 primitives | needs a §3 **decomposed** arm to exist | **not scoped** |
| which §3 primitive resists native emission | Baracuda, against a concrete §3 seam | last |

The fourth row is the scope discovery: "what do we ask Baracuda to build"
bottoms out in **building the §3 decomposed arm ourselves first**. Baracuda's
prediction — that the online-softmax custom monoid is the leg most likely to lack
a native binding — is recorded **against §3**, and is *not* testable by the §1
dump, which contains no online-softmax accumulate at all (one dense
`Fused(SOFTMAX_LAST_DIM)` over materialised scores). Their three-leg mapping
describes §3 throughout; it is tagged as such here so the conflation I caused
isn't repeated.

### 6.2b Batch-size sweep (Lightbulb) — shapes the ask, does not gate it

10.4× is a **k=1** number and k=1 is the case batching exists to avoid, so the
sweep decides whether a kernel targets single-token decode or batched — the
*shape* of the request. It is **downstream** of §6.2, not upstream, and it does
not choose monolithic-vs-primitives.

### 6.3 Per-primitive native-emittable read (Baracuda)

Offered, and correctly sequenced last: it needs a concrete seam to be
answerable.

---

## 6.4 Related: does every route default to its fast path?

Asked directly, and the honest answer was **no**. Audit as of 2026-08-05:

| route | fast by default? |
|---|---|
| `PagedSessionScheduler` | ✅ now (`PlanOnce`) |
| `generate_with_kv_context` / streaming | ✅ already was |
| `SessionScheduler` (contiguous multi-session) | ✅ already was |
| hand-rolled loop on `forward_with_kv_context` | ❌ → **fixed**, `forward_decode_step` |
| CUDA-graph capture | ❌ → **deliberately left**, see below |

**Fixed.** `forward_decode_step(tokens, cache, ctx)` gives plan reuse at the
*same three-argument shape* as the raw primitive, with the held plan carried on
the `InferenceContext`. The defect was never a bad default — it was that the
fast path required a call shape (`&mut Option<DecodeSession>`) you could only
write if you already knew `DecodeSession` existed. `forward_with_kv_context`
keeps its rebuild contract: the persistent path uses it as its own fallback, and
dozens of tests exercise rebuild *as the thing under test*.

**Deliberately left: CUDA-graph capture.** `forward_with_kv_context_captured` is
called from tests only — no production route captures. It is tempting to default
it, and I am not doing so, because **the number that would justify it does not
isolate it.** The measured 5,901 → 26.47 ms/token added persistence *and*
capture in one step; persistence alone is now the default everywhere, and how
much of the remainder is capture is **unmeasured**. Capture also carries real
invalidation surface — a recorded graph over fixed device addresses (which is
why the staleness fix had to retire the capture with the session).

Defaulting it on an unseparated measurement would be the same error this document
exists to avoid: **taking a true observation and attaching a mechanism nobody
checked.** The prerequisite is one more arm in Lightbulb's sweep — persistent
*without* capture vs persistent *with* — which is nearly free given the harness
already runs both configurations.

## 7. Status of claims

| claim | evidence |
|---|---|
| Agnostic decomposition exists, total, closed-basis | code read (`paged_attn.rs::recipe`, `decompose`) |
| Floor materialises padded capacity, not occupancy | code read (`kv_len = max_blk · block_size`) |
| No CUDA/Vulkan PagedAttn implementation | grep, positive-controlled against 4 other fused ops |
| Paged decode round-trips to host | **measured** (nsys, Lightbulb) — of the *path* |
| …because the PagedAttn *node* is host-placed | **MEASURED, CONFIRMED** — §2.0, per-node `placement_of` |
| The primitive recipe never fires under CUDA | **measured** — no PAGED_ATTN decomposition in the held plan |
| Leg 3 is a reduction, not a scan | Baracuda correction, accepted |
| Ragged continuous batching has only the paged path | Lightbulb, positive-controlled |
