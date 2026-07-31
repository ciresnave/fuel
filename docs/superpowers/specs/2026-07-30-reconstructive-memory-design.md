# Reconstructive ("after-image") memory — design

**Date:** 2026-07-30
**Status:** design approved; scheduled, not started
**Sequenced after:** B0.3 + B0.5 (`fuel-core` retirement), multi-session inference, the RNG/generator-seam decisions

---

## 0. Premise

Human memory may not be a lossless store that is read back. It may be an *after-image*
— a compressed trace that is **reconstructed using current state** at recall time. If so,
a machine memory built the same way should be able to hold far less than a verbatim log
while remaining useful, because the reconstructing model supplies most of the content.

The operational consequence, and the property this design is built around:

> The same cue must reconstruct **differently** as the model's weights change.

That is falsifiable, and it is what separates this from a cache. A design in which
retrieval cannot observe the weights cannot exhibit it.

## 1. What already exists in Fuel

Verified on `origin/main` (not inferred). This design is mostly *composition*, not new
substrate:

| Capability | Where | Status |
| --- | --- | --- |
| Modern Hopfield retrieval `ξ ← softmax(β·ξ·Xᵀ)·X` | `fuel-core/src/hopfield.rs:76` | built |
| Bounded fixpoint iteration with early exit | `Op::Scan { early_exit, emit: Final }` | built |
| LoRA adapters | `fuel-core/src/lazy_nn/lora.rs` | built |
| Block pool w/ lossy evict + restore + refcount | `fuel-core/src/kv_block_pool.rs` (`Fidelity`, `Externalized`, `restore`, `block_refcount`) | built |
| Block-grained affine quantization w/ sibling scale operand | `fuel-ir` `Encoding::AffineBlock` | built |
| Autograd through scans | `lower_scans_for_backward` | built |

**The one thing genuinely missing:** `hopfield_retrieve` takes `patterns: X` as a *frozen
const*. There is no store, append, or evict path. The bank exists as a **retriever** but
not as a **memory**.

## 2. Core mechanism — the residual

### 2.1 Definition

Let `θ` be the effective weights and `p(·)` the model's prediction of the content from the
cue alone. A memory stores only what the model **failed** to predict:

```
write:   r = x − p(cue; θ_write)          and the memory is stamped θ_write
read:    x̂ = p(cue; θ_stamp) + r          ← the STAMPED adapter, not current
```

`r` is the *surprise*. Content the model already predicts is never stored.

The stamp in the read equation is load-bearing: reconstructing against *current* weights
instead is the overshoot bug (§3.3).

**Consequence — the design has an opinion about what is memorable.** If `x` is the model's
own generated output then `r ≡ 0`; a model predicts its own generation perfectly. So the
residual naturally stores what arrived **from outside** — user turns, tool results,
observations. Thoughts that follow from what is already known are not stored, which is
correct.

### 2.2 Leave-one-out constraint (mandatory)

`p()` **must not consult the memory being baselined.** If the forward pass can retrieve `x`
from the bank, the model "predicts" `x` by retrieving it, `r ≡ 0`, the entry is judged
redundant and dropped — and there is then no residual left to recover it. The scheme eats
itself.

Enforcement: compute `p()` with the bank ablated, or with that entry masked, whenever
computing or re-baselining its residual.

### 2.3 The write pass is free

For content arriving from outside, the prediction falls out of work already being done:

```
turn N    : model generates its reply            (own output → r ≈ 0, skip)
turn N+1  : user message arrives → PREFILL
            prefill computes logits at every position = the model's
            teacher-forced prediction of each token given the prefix
            ⇒ per-position surprise at ZERO marginal cost
```

Hidden states pooled into `x` are likewise already in the residual stream. **No additional
forward pass is required for a fresh write.** (An extra pass *is* required to re-baseline
an existing memory against different weights — see §4.)

## 3. θ-stamps

### 3.1 A stamp is an adapter delta, not the weights

```
W = W₀ + ΔW_v
    ↑     ↑
  frozen  the LoRA adapter — the only thing that varies; this is what a stamp names
  base    (stored once, shared by every version)
```

| | per version | 50 live versions |
| --- | --- | --- |
| full weights (7B, bf16) | ~14 GB | ~700 GB — infeasible |
| LoRA adapter (rank 16) | ~20 MB | ~1 GB — fine |

**This makes the frozen base a hard requirement of the memory architecture, not merely a
catastrophic-forgetting preference.** Full fine-tuning would make every checkpoint a whole
model copy and collapse §4 under storage. Anything else in the prediction path that drifts
(e.g. a separate trainable encoder) must be pinned or versioned for the same reason.

### 3.2 Reconstruction via a stamp is exact

```
x = r + p(cue; θ_stamp)        ← swap in the stamped adapter; exact at any age
```

This is the property that makes staleness harmless (§4.4).

### 3.3 The overshoot bug — why re-baselining is mandatory

Reconstructing with *current* weights against a residual baselined at *old* weights:

```
x̂ = p(θ₆) + r
   = p(θ₆) + x − p(θ₁)
   = x + [p(θ₆) − p(θ₁)]          ← drift term
```

If consolidation worked and `p(θ₆) ≈ x`, then `x̂ ≈ x + r₁` — **you overshoot by the original
residual, and the error grows precisely as consolidation succeeds.** Correctness therefore
requires either reconstructing through the stamp (§3.2) or re-baselining (§4).

### 3.4 The residual cannot detect its own drift

Re-baselining needs `x`. Recovering `x` *from the memory itself* is a no-op:

```
x̂     = p(θ_now) + r_old
r_new = x̂ − p(θ_now) = r_old
```

So an **external anchor is mandatory**. The θ-stamp is that anchor: `x = r + p(θ_stamp)` is
exact, and it is why stamps are retained until migration completes (§4.3).

## 4. Ageing — precision decay and migration

### 4.1 Never fully evict

`‖r‖ < ε` does **not** mean *delete*. It means *this memory now costs almost nothing to
keep*. Instead of eviction, **coarsen**. Bits needed to hold `r` to absolute tolerance `ε`:

```
bits ≈ log₂(‖r‖ / ε)
```

so storage falls out automatically as consolidation shrinks the residual (‖r‖ 1.07 → ~3.4
bits/component; 0.06 → 0). Eviction becomes the limiting case of quantization rather than a
cliff, and a **floor** keeps a coarse correction forever. Nothing becomes unrecoverable; it
becomes fuzzy — which also matches the premise better than deletion does.

Substrate: `Encoding::AffineBlock { packed, block_shape, scale, zero_point }`. Decay is a
re-encode at a smaller `packed` code.

### 4.2 The floor is set by drift *detection*, not reconstruction fidelity

A heavily quantized `r_old` makes `‖r_old‖` noisy, so small `Δ‖r‖` measurements vanish into
quantization noise — exactly where it hurts, since the onset of forgetting in a
well-consolidated memory is a small increase from a small number. Therefore:

> **Floor = the minimum precision at which forgetting is still detectable.**

This makes the floor a derived criterion rather than a tuned constant.

### 4.3 Migration is a copying collector

```
for each memory m still stamped θ_old:          # idle-time, incremental
    x     = r_old(m) + p_m(θ_old)               # exact — run the OLD adapter
    r_new = x − p_m(θ_current_checkpoint)       # run the CURRENT one
    bits  = max(floor, ⌈log₂(‖r_new‖ / ε)⌉)     # re-quantize in the same pass
    emit r_new @ bits, stamped θ_current
when refcount(θ_old) == 0:
    delete the old residual file AND the θ_old adapter checkpoint
```

Rules:

- **Each migration produces a fresh, independent residual.** Never chain
  `r_new − r_old`; deltas-of-deltas compound quantization error and produce a chain that
  cannot be collapsed without replaying all of it.
- **Always migrate straight to current** — one hop per memory, never `v1→v2→v3`.
- **"Current" means the latest *checkpoint*, minted at intervals** — not θ after every
  gradient step, or each memory gets a unique stamp and version count explodes.
- **Oldest-first** during idle work, which actively drains the tail.
- **Free on refcount, not on a flag.** `θ_old`'s adapter cannot be dropped while any memory
  still references it — it is the only way to recover their `x`.
- **Crash-safety is free.** Per-memory stamps mean a half-finished migration is a resumable
  state, not corruption. Reads resolve per-stamp.
- **Re-key hook.** When option A lands (§6), migration must also recompute the addressing
  key, not just the residual. Leave the hook so adding it is not a schema change.

### 4.4 Staleness is free; back-pressure is on version count

Because reconstruction through a stamp is exact at any age (§3.2), **falling behind costs
storage and telemetry — never fidelity.** A memory 50 checkpoints behind reconstructs as
well as one migrated this morning. Migration is therefore garbage collection, not a
correctness requirement.

Two consequences:

- Bound the **number of live checkpoints**, not staleness. On hitting the bound, either slow
  checkpoint minting or force a sweep. Bounding staleness would solve the wrong problem.
- **Migration rate gates telemetry rate** (§5) — the Δ signal only arrives when a memory
  migrates. Lagging far behind coarsens the signal without invalidating it.

Load is a step function, not a trickle: minting a checkpoint makes the *entire corpus*
stale at once, so a full sweep is N memories × 2 forward passes per interval. The
back-pressure valve is what keeps that honest when it does not fit.

Cost may be largely amortized by folding migration into consolidation replay (which already
samples and runs memories). Caveat: replay samples by `‖r‖`, so well-learned low-residual
memories are rarely sampled and would strand — add a staleness bound that forces migration
regardless of sampling.

## 5. The Δ signal — measured catastrophic forgetting

Migration computes `‖r_old‖` and `‖r_new‖` for the same `x`, both exactly. The comparison is
a **byproduct**, and it is a per-memory measurement of the thing the CLS framing exists to
prevent:

```
‖r_new‖ < ‖r_old‖   → weights moved favourably for this memory
‖r_new‖ > ‖r_old‖   → this memory is being actively damaged
```

Uses:

- **Rollback criterion for a consolidation step.** Net-positive aggregate Δ means the epoch
  overfit recent data at the corpus's expense — reject the update. A validation signal that
  needs no held-out set.
- **Two distinct replay priorities.** Sampling ∝ `‖r‖` finds what was *never learned*;
  sampling ∝ `Δ‖r‖` finds what is *being lost*. Different failure modes; both wanted.
- **Learning-rate control** — a rising fraction of growing residuals means back off.
- **Interference attribution** — which memories a given training direction damages.

Known pathology: pure-surprise prioritization preferentially replays **noise** (unpredictable
because random, not because informative). Mitigate by capping priority or separating
reducible from irreducible surprise. Design this in rather than discovering it.

## 6. Coupling — where current state enters

There are exactly two doors:

| | addressing uses θ | content uses θ |
| --- | --- | --- |
| **C** decoupled | no | no |
| **A** query-side | **yes** | no |
| **B** residual | no | **yes** |

They are **orthogonal**, not nested. **B is adopted. A is deferred but hooked.**

**Interaction to respect:** if `encode()` uses live-model hidden states, addressing is
adapter-influenced *whether or not A is chosen* — so "B only" requires deliberately frozen
keys, and live-model keys require the §4.3 re-key path or retrieval quietly rots as write-
and read-time key spaces drift apart.

A is **required** for: idiolect drift in long-lived personalization; domain specialization;
per-session adapters; cross-lingual cues. A is **harmful** for: auditability — it makes
retrieval non-reproducible over time. Therefore A is a **consumer choice** long-term,
defaulted **off** in increment 1.

## 7. Mechanism / policy split (§15)

Fuel owns mechanism; the consumer owns policy. Defaults ship in optional consumer-side
crates (`fuel-inference`, `fuel-training`) and **must be replaceable**.

| Component | Owner |
| --- | --- |
| Bank storage, write, evict, re-quantize | **Fuel** |
| Residual computation (`x − p`), `‖r‖` | **Fuel** (a measurement) |
| θ-stamp + adapter-swap forward pass | **Fuel** |
| Migration *execution* | **Fuel** |
| *When* to consolidate / migrate / checkpoint | **consumer** |
| Replay sampling strategy | **consumer** |
| Acting on Δ (rollback, LR, admission threshold) | **consumer** (Fuel measures) |

Replay sampling being consumer-side and host-side is what keeps this **off** the open
RNG/generator-seam gap — it needs no sampling-as-a-graph-op.

## 8. Bucket classification

Per `docs/frontier-paradigms-vision.md`:

| Piece | Bucket | Cost |
| --- | --- | --- |
| Residual write/read path | **A** (compose existing primitives) | cheap |
| Generic block pool extraction | **A** (refactor) | cheap |
| Precision-decay ladder rungs | **B** (`Encoding`/`DType` extension) | cheap |
| Consolidation / migration scheduling | **F** (consumer) | out of core |
| Δ-signal consumption | **F** (consumer) | out of core |

**No bucket-E work — this adds nothing to the primitive basis.** Every operation needed
(`Op::Scan` retrieval, subtract, norm, write-slice, `AffineBlock` re-encode) already exists.
This must be verified deliberately before scheduling, because it is the claim that sets the
cost tier.

## 9. Data model

**Shape is data, not a type parameter.** Everything varying between models is shape and
dtype, which in Fuel are runtime values validated at graph-build time. `KvGeometry` sets the
precedent — a runtime struct, not a generic — and a generic base pool must take a geometry
*value* or it cannot hold heterogeneous kinds.

```
MemoryGeometry { dims, dtype, encoding, block_shape, .. }   // the data shape
trait MemoryPolicy { .. }                                    // the behaviour, default impl
                                                             // in fuel-inference/fuel-training,
                                                             // consumer-replaceable
```

Policy hooks: what is worth storing, pooling, admission threshold (`‖r‖ > ε`), floor policy.

**Base pool extraction.** `KvBlockPool` is the right *shape* but a KV-typed API
(`KvGeometry`, `kv_bytes_resident`, `StateKind`). Extract a generic block-pool core that
**both** `KvBlockPool` and the memory bank build atop. It must not land in `fuel-core`
(retiring) — hence the B0.3/B0.5 sequencing.

**Content and key are separate, and may come from different places.** `x` is what is
reconstructed; the *addressing key* is what the bank is looked up by. Increment 1 pairs a
live-model `x` with a **frozen** key (A off, reproducible retrieval); option A later moves the
key onto the live model and turns on re-keying at migration (§4.3, §6). Conflating the two is
what makes the "is A optional?" question look confusing.

**`x` = the per-position hidden-state span**, `[T, d]`, block-paged; `p` = the leave-one-out
forward pass over the cue alone. Rationale: fixed per-position shape, variable `T` handled by
blocks, encoder is free (the model's own states, nothing extra to version), reuses KV
geometry directly. Increment 1 uses a **single pooled vector** (`T = 1`) — a geometry change,
not a redesign.

## 10. Increment 1 — scope and gate

**Scope:** the write path and bank only. No consolidation loop, no training, no CUDA, no A.

**Born-red acceptance test (round-trip exactness):**

1. Write a memory under adapter `v1`.
2. Drift the adapter to `v2`.
3. Reconstruct via the **stamped** adapter → assert `x` recovered within tolerance.
4. Reconstruct via the **current** adapter → assert it does **not** — this is §3.3's
   overshoot bug turned into a test.

Gates the whole θ-stamp mechanism with zero training infrastructure and no GPU.

Follow-on increments, in order: precision decay + floor → migration (copying collector) →
Δ-signal telemetry → option A + re-keying.

## 11. Open items

- **Precision ladder (investigate first).** `AffineBlock` is parameterized by
  `packed: DType`, but the available sub-byte codes have **not** been surveyed — `DType::F4`
  is the only one named in the `Encoding` docs. If the ladder is effectively F32 → F4 → gone,
  "graceful degradation" is three steps, not a curve, and the §4.2 floor may not be
  expressible. Adding rungs is bucket B (cheap) but is real work, not free reuse.
- **Online/test-time update path.** Whether `fuel-training` supports inference-time weight
  updates is **unverified**. Not on increment 1's path; required before consolidation lands.
- **Verify the bucket-E claim** (§8) before scheduling.

## 12. Rejected alternatives

| Rejected | Why |
| --- | --- |
| **Option C** (fully decoupled, as originally proposed) | Retrieval is invariant to θ by construction, so the reconstructive property cannot exist. Contradicts the premise: current state would enter only *after* reconstruction. |
| **Hash-style verification sketch** | An alarm with no remedy — too small to reconstruct from, so it can detect drift but never repair it. |
| **Compressed-sensing sketch** (`s = Px`, sparse recovery) | Elegant and would work — `s − P·p(θ)` measures `P·r_new`, and JL bounds make detection sound — but correctness would rest on a sparsity assumption that can silently fail. Keep as a later storage optimization, not a correctness dependency. |
| **Full eviction at `‖r‖ < ε`** | Leaves no way to recover or re-baseline the memory (§3.4). Replaced by precision decay with a floor. |
| **Chained deltas between baselines** | Compounds quantization error per hop; produces a chain that cannot be collapsed. |
| **Epoch-pinned migration (v1→v2→v3)** | Bounds live adapters at 2 but migrates stragglers repeatedly. Always-to-current is one hop per memory, and adapters are cheap. |
| **`x` as an associated type** | The variation is shape and dtype — already runtime values validated at build time. A type parameter puts the check in the wrong place and prevents one pool holding heterogeneous kinds. Replaced by `MemoryGeometry` + a policy trait. |
| **Full fine-tuning instead of adapters** | Makes each checkpoint a whole model copy; collapses §4 under storage. |

## 13. Correction to the source proposal

The original writeup (`Hopfield fast buffer + inference-time LoRA + nightly consolidation`)
is PyTorch-shaped, where library and application blur. Three corrections carried into this
design:

1. **"Zero catastrophic forgetting" is false.** A bounded-capacity Hopfield bank has
   retrieval interference — beyond capacity, spurious and metastable mixture attractors, not
   clean recall. So "prune" is on the correctness critical path, not housekeeping.
2. **The consolidation daemon is policy, not mechanism** (§7) — a background job that runs
   "when idle" or "at midnight" is bucket F by Fuel's own classification.
3. **The coupling was never named.** Passing the prompt through an in-model Hopfield layer
   would get option A *incidentally* if adapters sit upstream — so the proposal is arguably
   A-by-accident rather than C-by-design. Because it is unnamed, nothing protects or tests
   it, and the tell is that pruning stayed a judgment call. Had the residual been present,
   pruning would have fallen out as a measurement.

## 14. Why this is Fuel-shaped

`‖r‖` is one measured scalar driving what would otherwise be five independent heuristics —
admission, replay priority, learning rate, convergence, and ageing — plus `Δ‖r‖` as a
second, distinct trend signal. That is the same posture as the rest of Fuel: **measure at
the seam, decide in one place, and let the consumer own policy.**

A longer-term possibility, not scoped here: consolidation as a **graph rewrite** (bucket C)
rather than a fine-tune. Distilling episodic traces into a compact generalization is
structurally what `symbolic_distill()` does, and the optimizer-as-intelligence is Fuel's
actual differentiator. A nightly fine-tune script is the version anyone could build.
