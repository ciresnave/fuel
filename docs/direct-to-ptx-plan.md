# Direct-to-PTX — what it would be, and what the evidence supports

**Status**: for CireSnave's decision, 2026-08-27. Measured at `73e05b71`.
**Sources**: Baracuda's PTX-importer headroom survey (`baracuda docs/measurements/ptx-importer-headroom/README.md`, merged); Unpopped's representational analysis; Fuel-side loader verification.

---

## 0. The question that was asked, and the two questions it contains

CireSnave asked two things that have been travelling as one, and **only one of
them is refuted:**

> **(A)** *"As for PTX, it removes any assumptions made by the creators of CUDA.
> **Would going straight to PTX bypass the need for a C compiler?**"*
>
> **(B)** *the 426 hand-tuned CUDA kernels may not have had the low-level levers
> available, and something could re-emit them further tuned.*

**(A) is about toolchain independence. (B) is about tuning headroom.** They imply
different artifacts — an **emitter** for (A), an **importer** for (B) — and the
evidence lands almost entirely on (B).

**The answer to (A) is yes, and nothing in the survey touches it.**

---

## 1. The importer is refuted, twice, independently

### The representational refutation (Unpopped)

`StructureKey` has **no slot** for tiling, staging, or unroll. A lifter drops what
it cannot represent; a rewriter passes through what it cannot recognise. Their
`CUDA_RESIDUE` list already declines hand-optimised kernels **by name, with a
test pinning it.**

### The empirical refutation (Baracuda)

**9 of 426, static, sm_80, ship-exact PTX/SASS/`ptxas -v`.** The residual headroom
partitions by **owner**, and only the first row is evidence for an importer:

| | category | reachable by an importer? |
|---|---|---|
| **(a)** | load/store **vectorization** — `.v2`/`.v4` absent 8/9, only flash vectorized | **yes — and it is the weakest possible evidence FOR the proposal** |
| **(b)** | **register spills / pressure** — flash capped at 255 regs | **no. An importer re-runs `ptxas` on the imported PTX** |
| **(c)** | **`cp.async` pipelining** — int8 GEMM stages synchronously; flash uses LDGSTS | not a peephole — a pipeline restructure with commit/wait barriers and double-buffering |
| — | *already captured* — read-only cache **9/9 in the PTX**; tensor cores where applicable (flash HMMA, GEMM `mma.sync.m16n8k32.s8`) | not headroom at all |

**Row (a) is the fatal one, and it is fatal in the direction that looks like
support.** The vectorization *is* absent — and **nvcc emitted scalar having seen
the C++ types, alignment and access patterns, and declined.** The importer reads
*lowered scalar PTX*, with **strictly less information than nvcc had**, downstream
of the decision it would reverse. Baracuda explicitly did **not** verify that
widening would be safe for any kernel — only that nvcc declined it.

**Row (b) is the one that changes the plan's shape rather than its size:**
*not a lever CUDA C is hiding — a lever nobody holds.*

**Two projects reached the same crux from opposite ends.** *(Recorded as my
synthesis: Baracuda measured only the empirical leg and did not read Unpopped's
analysis. Neither verified the other's.)*

---

## 2. How much the null is worth, stated as its author states it

**The selection is the claim, not the conclusion.** The sample is **purposive, not
random** — deliberately spanning both ends of the headroom-likelihood axis:
memory-bound scalar kernels where vectorization headroom is **most** likely
(`binary_add`, `gemv_dense`, `softmax`, `rms_norm`) **and** the gold-standard
hand-tuned kernel where it is **least** (flash-attn backward), plus int8 tiled
GEMM, arbitrary-mask attention, dequantize, topk.

**So the null is moderately strong: it looked where headroom should be and found
it already captured or unreachable.** The honest cut, in their words: *purposive
not random, 9-of-426, and I picked shapes I understand — a family I didn't sample
(a conv, an SSD/scan) could differ. A 9-of-426 null is a real result and it is NOT
a survey of 426.*

### ⚠️ The instrument nearly reported the opposite, and was checked

Their first read-only-cache grep looked for the SASS token **`.CI`**; on this
toolchain it is spelled **`.CONSTANT`** (PTX `ld.global.nc`). **First pass:
"read-only cache 0/9 → headroom." Truth: 9/9, already in the PTX.**

**A correct grep for the wrong spelling returned a false absence that pointed the
design at "build the importer."** Caught mid-survey by verifying the actual load
notation before trusting the count. **Weight the null knowing that.**

### The denominator was settled by measuring, not by restating

A prose-says-9 / table-shows-8 mismatch was a real defect — a sweep glob spelled
the object `rmsnorm*` where it ships `rms_norm_*`, so the ninth silently matched
nothing. **Fixed by measuring the real ninth** (read-only 498/498, 0 spill bytes
verified by recompile, partial 64-bit vectorization) **rather than dropping the
claim to 8.** It reinforced the table.

---

## 3. The emitter is a different proposition and the survey does not touch it

**Baracuda's view, flagged by them as a view rather than a measurement** — their
survey scoped to the importer:

> **The importer's fatal crux — downstream of nvcc's decision with less
> information — does NOT apply to an emitter**, because an emitter starts from a
> high-level IR that **still has** the types, shapes and alignment nvcc had. **It
> is upstream, not downstream.**

So the information-availability argument that kills the importer **does not reach
the emitter.** The emitter's real question is different and harder: **code-gen
quality** — can a generator emit as well as nvcc plus hand-tuning? *That is
unmeasured.*

**And one nuance in the emitter's favour on the hard-wall category:** an emitter
also feeds `ptxas`, so it also cannot directly control register allocation — **but
it has leverage an importer does not**: `launch_bounds` / `maxrregcount`, and
generating structure that pressures registers less. **Category (b) is a hard wall
for the importer and a soft one for the emitter.**

### What the emitter is actually worth, and it is not the tuning argument

**The emitter does not have to beat nvcc to justify itself. It has to remove a
toolchain dependency**, which is question (A) and is answered yes.

Today a `--features cuda` build requires **nvcc → `cl.exe` → `vcvarsall`**, with a
documented failure catalogue in this repo's own working agreement: mixed-toolset
header/compiler mismatches that die deep in the stdlib and read as CUDA bugs; a
`cmd /c` invocation that returns exit 0 having compiled nothing; a ~56-minute
kernel forge; `ptxas` allocation failures under concurrency that read as code
defects. **This session lost hours to that class.**

**Emitting PTX and loading it removes the C compiler from the runtime path
entirely.** The loader already exists — **verified**: `load_ptx` / `get_function`
are present in the published `baracuda-driver` `0.0.1-alpha.78`. So the missing
piece is the emitter, not the loader and not the driver.

---

## 4. What would change either answer, and what it costs

| question | measurement | cost | verdict if it fails |
|---|---|---|---|
| **importer** | **sm_89 recompile** of the mma/fp8-shape kernels. Baracuda's box *is* sm_89; the build capped **sm_80**, so those shapes are **genuinely unmeasured, not measured-and-absent** | **≈ an afternoon** — recompile the sm_89-specialized GEMM/attention set, re-run the same sweep | **the only extension that could find a lever the sm_80 sample cannot see** |
| **importer** | a wider or random sample | any | **would NOT move it** — the crux is representational and identical in every row. More kernels emitting `ld.global.nc` or lacking `.v4` does not change the argument |
| **emitter** | **A/B one or two shapes**: generate candidate PTX, run it on the 4070 against the shipped kernel | a build, not a read | **the only thing that speaks to code-gen quality**, which is the emitter's actual open question |

**Note the asymmetry**: the importer's remaining uncertainty is cheap to close and
unlikely to reverse. The emitter's is a build, and it is the one that matters.

---

## 5. Recommendation

**1. Do not build the PTX importer.** Refuted representationally and empirically,
by two projects, from opposite ends. **Record the refutation rather than the
absence of a decision**, so it is not re-proposed in six months by someone reading
the original premise.

**2. Treat the emitter as open, and justify it on toolchain independence rather
than on tuning.** That is question (A), the answer is yes, and it does not depend
on winning a code-generation contest against 426 hand-tuned kernels. **If it also
tunes well, that is upside, not the case.**

**3. Before committing to the emitter, buy the one measurement that speaks to its
real risk** — Baracuda's offered A/B on one or two shapes. **A generator that
loses badly on a shape Fuel actually runs is the failure mode**, and it is
cheaper to find on two kernels than on a program.

**4. Optionally close the importer question completely** with the sm_89 recompile.
An afternoon, and it converts *"unmeasured at this arch"* into a fact. Worth it
only if the importer is likely to be re-proposed; the refutation stands without
it.

---

## 6. What this plan deliberately does not claim

- **It does not claim the emitter will produce competitive kernels.** Nobody has
  measured that. Section 3 says so twice because it is the load-bearing unknown.
- **It does not claim 426 kernels have no headroom.** It claims **9 purposive
  samples at sm_80 found the reachable headroom already captured or owned by
  `ptxas`**, and that the one reachable category is the weakest possible evidence
  for an importer.
- **It does not treat the two refutations as independent confirmation of each
  other.** They converge, but Baracuda measured only the empirical leg and did not
  read Unpopped's analysis. **Agreement between two parties who did not check each
  other is convergence, not corroboration.**
