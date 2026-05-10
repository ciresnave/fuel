# Tolerance: per-op error budgets and approximate optimization

**Status**: v0.3 (draft, 2026-05-09). v0.3 changes: calibration tooling picks comparators using the `PrecisionGuarantee` structure (per [05-backend-contract](05-backend-contract.md#per-kernel-precision-guarantees)) instead of a distinguished reference backend. v0.2 added the "Tolerance discovery and calibration" section.

The architectural model for *controlled approximation*: how users, model definitions, and runtime callers specify acceptable numerical deviation, how the optimizer reasons about cumulative error along candidate routes, and what optimization space tolerance unlocks.

This is one of fuel's five competitive edges (see [01-identity §The five competitive edges](01-identity.md#the-five-competitive-edges)). Most production ML frameworks treat numerical precision as a binary or global mode; per-op tolerance budgets that the optimizer reasons about are uncommon. Fuel commits to making them work end-to-end.

---

## Concept

A **tolerance budget** is the maximum acceptable deviation between the strict-equivalence result of a computation and the result an optimization may produce. Budgets are specified by users, model definitions, or runtime callers; they're consumed by optimization rules that introduce non-strict transformations; the optimizer prunes candidate routes that exceed the available budget.

By default, every computation runs strict-equivalence — tolerance defaults to zero, no approximate optimizations fire, results are bit-reproducible (subject to floating-point determinism caveats discussed below). Tolerance is opt-in everywhere except in narrow, explicitly-marked default paths (none planned for v1).

Three forms of tolerance specification, listed in increasing strictness:

- **Strict**: zero tolerance. Strict-equivalence rewrites only. The default at every level when no other tolerance is specified.
- **Relative**: a per-element relative-error bound, e.g. `Tolerance::Relative(1e-3)` for "at most 0.1% deviation per element." Useful when the absolute scale of values is unknown or varies.
- **Absolute**: a per-element absolute-error bound, e.g. `Tolerance::Absolute(1e-5)`. Useful when the scale is known (e.g., logits, probabilities).

Combinations (relative *and* absolute, max of either) and named profiles (`Tolerance::Mild`, `Tolerance::Aggressive` mapping to specific bounds) are reasonable extensions but are implementation-side conveniences over the same underlying machinery.

## What tolerance unlocks

The optimization space reachable under non-zero tolerance budgets includes (but isn't limited to):

- **Mixed-precision lowering.** Replace F32 ops with BF16 or F16 in regions where the accumulated error stays within budget. Each dtype-conversion candidate carries an error contribution annotation; the optimizer commits the lowering only if the route's cumulative error fits.
- **FP reassociation.** `(a + b) + c → a + (b + c)` is mathematically equivalent for real numbers but not for IEEE floats. Under non-zero tolerance, the optimizer can reorder addition chains to expose parallelism, vectorization opportunities, or numerically-better-conditioned associations.
- **Fast/approximate intrinsics.** `expf` → `__expf`, `sqrt` → `rsqrt`-based identities, polynomial approximations for `tanh`, `sigmoid`, `gelu`, softmax. Hardware-specific fast paths (CUDA's `__sinf`, ARM Neon approximations) are kernel variants the optimizer can swap in.
- **On-the-fly quantization.** A computation that loads an F32 weight matrix can be replaced with one that uses a pre-quantized version when the budget allows the quantization error. Particularly valuable when the same F32 weight is used many times — quantize once, dequantize on each use, eat the quantization error to save bandwidth.
- **Sparse approximations.** Drop near-zero contributions in attention (top-k attention), in matmul (block-sparse approximations for low-magnitude blocks), in softmax (truncated softmax). All are tolerance-controlled.
- **Algebraic rewrites that aren't strictly identity.** `softmax(logits / t) ≈ softmax_temperature(logits, t)` for small `t`. `log(softmax(x)) → log_softmax(x)` is exact in real arithmetic but in IEEE floats the fused form is more numerically stable (sometimes meaning the *fused* form is the strict path and the unfused is the approximate one — direction of error matters, see below).

The list isn't exhaustive. The point is the *category*: any rewrite the optimizer would otherwise reject for not preserving strict equivalence becomes a candidate when the user has signaled willingness to trade controlled error for compute or memory.

## Hierarchical specification

Tolerance budgets are specified at four levels, with finer levels overriding coarser ones:

1. **Graph default**: the tolerance applied to the whole DAG when no override is given. Set when the graph is created or when `realize()` is called. Sensible default for a model: `Tolerance::Strict`.
2. **Subgraph override**: a region of the DAG (e.g., one transformer block, one residual branch) tagged with a tolerance that overrides the graph default within that region. Useful for "this layer is precision-critical but the rest can be loose."
3. **Per-op override**: an individual op tagged with a tolerance that overrides the surrounding subgraph. Useful for "this softmax must be strict but the surrounding linears can be fuzzy" — common in attention where the softmax determines correctness and surrounding compute determines throughput.
4. **Per-call override**: at `realize()` / `materialize()` / `item()` time, the caller may override the graph-baked tolerance. Useful for serving paths where the same compiled graph runs under different SLAs — tight tolerance for premium requests, loose tolerance for batch traffic.

Per-call override is what makes top-N route preservation truly useful for tolerance: the optimizer can preserve a strict-route and a fuzzy-route as alternatives; the runtime route picker chooses based on the per-call tolerance.

The hierarchy is *additive only at coarser levels*: a per-op override of `Strict` cannot be loosened by a per-call override; the tightest budget along the chain wins. This is the safe direction. The opposite (loosen-from-call) would let runtime telemetry override model-author intent, which violates the model-author's safety contract.

## Error model: best-effort upper bounds, evolving to empirical

The optimizer needs to know how much error each candidate rewrite contributes so it can compare cumulative error along a route to the available budget. Two models, used in sequence:

**v1: best-effort annotations.** Each rule (lowering, fusion, algebraic rewrite, dtype change) declares an upper-bound error contribution as an annotation: `error_bound: ErrorBound`. The optimizer sums or composes these along a candidate route; routes whose cumulative bound exceeds the available budget are pruned. Annotations are conservative — pessimistic upper bounds, derived from numerical-analysis literature or empirical worst-case observations.

The optimizer is *safe by default* under this model: real error is bounded above by the annotated bound, so a route the optimizer admits is guaranteed (modulo annotation correctness) to satisfy the budget. The cost is that the optimizer sometimes leaves wins on the table — a real route's actual error may be much smaller than the annotated bound, and the optimizer doesn't know it.

**v2: empirical error from the Judge.** The Judge already measures latency per (op, dtype, size_class, backend). The same machinery extends to measure *actual error vs. a reference oracle* per (op, dtype, size_class, backend). The optimizer queries empirical error data instead of annotations; pruning becomes more accurate; previously-pessimistically-rejected routes become available.

V2 requires an oracle-grade comparator for every op (so error has something to be measured against). Comparators are kernels with `bit_stable_on_same_hardware: true` and tight `max_ulp` declared in their `PrecisionGuarantee` (per [05-backend-contract §Per-kernel precision guarantees](05-backend-contract.md#per-kernel-precision-guarantees)). The always-built backend's coverage commitment ensures at least one such kernel exists for every primitive op. V1 ships first; V2 promotes once Judge has the bandwidth to measure.

A note on composition: error is path-dependent. Two ops each with 1% error don't always compose to 2% error — sometimes errors cancel, sometimes they amplify, depending on the specific ops and inputs. The annotation model uses pessimistic compositional rules (additive-or-multiplicative-bound depending on op category); the empirical model can be richer (measure actual end-to-end error along common subgraphs). Both are approximations of the true mathematical behavior; both are provably-safe upper bounds when bounds are conservative.

## Direction of error and one-sided budgets

Some ops have asymmetric error behavior. Quantization always loses information (unsigned-error). Some approximations consistently overestimate or underestimate (signed-error). Tolerance budgets v1 use unsigned bounds (the most pessimistic case); future extensions can specify signed budgets when the model author knows the direction.

A subtle point: *the strict path is not always the more-numerically-accurate path*. `log(softmax(x))` in the unfused form computes softmax to F32, then `log()`, accumulating error in both ops; the fused `log_softmax(x)` form reorders to avoid the intermediate exponential overflow and is *more* numerically stable. So "strict equivalence to the unfused form" is not the same as "minimum numerical error." The optimizer's tolerance budget is about deviation from the user-specified reference behavior, not about minimizing absolute numerical error. Some rewrites under non-zero budget can produce *more* accurate results than the strict path. This is fine and expected; tolerance is about controlled deviation, not error minimization.

## Backend implications

Backends advertise per-kernel precision characteristics via the `PrecisionGuarantee` structure on each registered kernel (see [05-backend-contract §Per-kernel precision guarantees](05-backend-contract.md#per-kernel-precision-guarantees)). The structure carries multiple optional bounds:

- **`bit_stable_on_same_hardware: true`** — strictest commitment; bit-reproducible deterministic execution. These kernels qualify as comparators for calibration tooling and serve as anchors for cross-backend equivalence tests.
- **Tight `max_ulp` (e.g., ≤ 1)** — correctly-rounded or near-correctly-rounded; suitable as oracle-grade comparators alongside `bit_stable` kernels.
- **Bounded `max_relative` / `max_absolute`** — approximate kernels (TF32 matmul on Ampere, fast-math intrinsics, mixed-precision accumulation, polynomial approximations of transcendentals). Usable only when the user's tolerance budget permits.
- **All-`None`** — uncharacterized; conservative consumers treat as "no commitment, assume worst case." The kernel is admissible only under loose tolerance budgets where worst-case bounds still fit.

Backends do not decide whether their approximate kernels are admissible — the optimizer does, by checking each kernel's `PrecisionGuarantee` against the route's available budget. Backends just expose what their kernels are and what error they contribute.

The empirical Judge's V2 refinement (per [Error model](#error-model-best-effort-upper-bounds-evolving-to-empirical) above) measures actual error per kernel against an oracle-grade comparator and updates the registered `PrecisionGuarantee` values from data over time. Static annotations are starting points; empirical measurement converges them toward truth.

## Validation

A user who has set a non-zero tolerance budget needs a way to verify the budget is reasonable for their model. Architectural support:

- **Strict baseline materialization.** A user can request a `realize()` with `Tolerance::Strict` to produce the reference output, then compare against the fuzzy output produced under their normal budget. The framework provides element-wise comparison helpers.
- **Per-route inspection.** When the optimizer has admitted a non-strict route, the route's annotated cumulative error bound is queryable. A user can see "the route I'm using has a worst-case bound of 0.4%; my budget was 1%."
- **Drift monitoring under empirical-error v2.** Once Judge produces empirical error data per route, drift between predicted and measured error is logged. Large drift signals an annotation that's wrong (too optimistic or too pessimistic), prompting recalibration.

These are user-visible affordances, not optional debug surface — they're how users verify their model still produces acceptable outputs under the budget they chose.

## What this rules out

A few non-features, called out explicitly to keep the scope clean:

- **No global `fast_math` flag.** Tolerance is hierarchical and explicit, never a global mode that silently changes results everywhere. Users who want "loose mode" set the graph default; users who want "strict mode" leave it default. There's no flag that makes a strict graph behave loosely.
- **No automatic tolerance learning.** The optimizer doesn't decide what tolerance is acceptable; the user/model author does. The optimizer reasons under whatever budget it's given.
- **No numerical-stability fixes that violate strict equivalence under default settings.** A rewrite that produces *more accurate* results but isn't bit-equivalent to the strict path requires a non-zero tolerance to fire, even if it's "better" in absolute terms. The user's reference behavior is whatever they wrote; the optimizer doesn't unilaterally improve on it.
- **No per-element tolerance.** Budgets are per-op or per-region. A fine-grained "this specific element can be off by X" is overkill for the use cases we're targeting (mixed-precision regions, approximate kernels, dtype lowering). If a real consumer needs per-element granularity, the model is revisited.

## Composability with the rest of the architecture

Tolerance integrates with every other architectural concept:

- **IR ([03-ir](03-ir.md))**: tolerance metadata attaches to subgraph regions and ops in the DAG; the IR carries the budget alongside the structure.
- **Optimization ([04-optimization](04-optimization.md))**: rules in the OptimizationMap declare error contributions; the rule driver checks cumulative budget before firing a non-strict rule. Top-N route expansion includes both strict and tolerance-permitted candidates.
- **Backend contract ([05-backend-contract](05-backend-contract.md))**: backends advertise per-kernel error characteristics. The optimizer consumes them.
- **Runtime ([06-runtime](06-runtime.md))**: the route picker honors per-call tolerance overrides when selecting among preserved routes.
- **Pattern harvest ([08-pattern-harvest](08-pattern-harvest.md))**: harvested telemetry includes tolerance budgets along with op sequences. Some workloads have effectively-strict tolerance; others run aggressively loose. Pattern harvest can prioritize fused-op development for whichever regime its users actually live in.

The single architectural property that ties tolerance to everything else: **the budget is a first-class node-level annotation, not a flag and not a global**. Carry it through the system the same way you'd carry dtype or shape.

## Tolerance discovery and calibration

Setting per-op tolerance manually is impractical for any model bigger than a toy. Most users default to `Strict` and never benefit from the optimization machinery this section describes. Discovery automates what only experts do today: run the model through representative test inputs while tweaking per-op tolerances; learn the maximum tolerance per location that doesn't degrade output quality below a user-set threshold; save the discovered recipe alongside the model.

This is opt-in. Without calibration, tolerance defaults to `Strict` everywhere; the architecture's tolerance machinery exists but isn't exercised. Calibration unlocks it.

### Workflow

1. **User provides representative test inputs and a quality metric.** The metric defines what "correct" means for the user's use case (see "The metric problem" below).
2. **Calibration runs the model with various per-op tolerance settings**, comparing outputs against a strict-mode baseline using the chosen metric.
3. **The discovered recipe** — a per-op (or per-region) tolerance budget map — is saved as a sibling artifact alongside the model file. See [11-persistence](11-persistence.md) for file format and invalidation criteria.
4. **At inference**, the recipe is loaded and applied as the graph's tolerance specification, replacing the default `Strict`. The user can still override per-call.

### The metric problem

"Did the output change?" is use-case dependent and hard to answer generically:

- **Classification**: top-k accuracy preservation, or KL divergence of probability distributions.
- **Generative LLM**: token-sequence overlap (BLEU/ROUGE), perplexity preservation, or task-benchmark scores.
- **Embedding model**: cosine similarity to reference embeddings.
- **Image generation**: FID / LPIPS / human evaluation.

Fuel ships a small library of standard metrics (accuracy, KL, perplexity, embedding distance) and lets users register custom metrics for use cases the library doesn't cover. Calibration uses whichever metric is configured. **Recommendation: require multiple metrics to all pass** (e.g., perplexity AND a task benchmark AND embedding similarity). Single-metric calibration can mask real degradation that the chosen metric doesn't capture.

### The test-distribution problem

Calibrated recipes are valid for inputs *similar* to test data. A model calibrated on news text may degrade on code generation. The framework can warn ("you calibrated with 50 inputs; consider 500+ for stable results") but can't fix the distributional concern — users have to provide test data representative of their production inputs. Production-traffic drift can render previously-calibrated recipes wrong; users monitor production output quality and re-calibrate when it drifts.

### Search algorithm

Three options, increasing in sophistication:

- **Greedy per-op loosening**: start at strict, loosen one op at a time, accept if quality stays above threshold. Simple; O(n_ops × n_test_inputs) inferences.
- **Sensitivity-first**: identify which ops are most quality-sensitive (small perturbation → large output change), tighten only those, loosen the rest aggressively. Faster convergence; needs a sensitivity-analysis pre-pass.
- **Bayesian optimization**: model the quality-vs-tolerance landscape; sample efficiently. Best results with bounded budget; significant infrastructure to implement.

Architectural commitment: the search algorithm is pluggable. v1 ships greedy + sensitivity-first as a pragmatic baseline. Bayesian as future work, registerable through the same algorithm interface.

### Granularity

Per-op is the finest, also the largest search space. Per-layer (every transformer block uses the same tolerance) is more manageable. Per-region (encoder/decoder/output projection have separate tolerances) is coarse but practical. **Hierarchical** is the right default — start per-region, refine per-layer where sensitive, refine per-op only where needed. Discover at the right level of detail per use case.

### Output: tolerance recipes as sibling artifacts

A calibration run produces a **tolerance recipe** — a serialized per-location tolerance map plus calibration metadata (metric used, test-set size, quality threshold achieved, fuel version, model hash). The recipe lives in a sibling file alongside the model, similar to the optimization cache but with different invalidation semantics:

- **The recipe is hardware-independent.** Same recipe applies on any backend; only the model is the variable. (Contrast with the optimization cache, which is hardware-specific.)
- **The recipe is invalidated only when the model changes.** Backend updates, hardware changes, fuel version bumps don't invalidate it (the recipe records what the model can tolerate, not how it executes).
- **The recipe records its provenance.** Who calibrated it, when, with what metric, against what test-set size. Production deployments can audit recipes the same way they audit model files.

See [11-persistence](11-persistence.md) for the file format and how the recipe sits alongside the optimization cache.

### Community sharing

Tolerance recipes are not sensitive: the test inputs that produced them might be (don't share inputs), but the discovered tolerance values themselves are just numbers per op. This makes recipes natural to share.

Architectural commitment: **opt-in community sharing of tolerance recipes**, using the same server infrastructure as [08-pattern-harvest](08-pattern-harvest.md). Specifics:

- **What's shared**: the recipe (per-location tolerance values + metadata) keyed by (model_hash, metric_name, calibration_quality_threshold). Test inputs are NOT shared.
- **Trust signals**: count of users who've validated a recipe; aggregated quality scores from independent users; provenance.
- **Validation on download**: framework runs a quick sanity check (small test set on user's data) before applying a downloaded recipe. Users see "this recipe was calibrated by N users with metric X; quick validation says it works on your inputs to within Y%."
- **Defaults**: high-trust recipes (validated by many users) ship as suggested defaults for popular models. Low-trust recipes are surfaced as suggestions with caveats.
- **Opt-in**: same opt-in flag as pattern-harvest. Off by default; users who opt in both contribute and benefit.

This creates a virtuous cycle: more users running calibration → more recipes contributed → better starting tolerances for everyone → more users benefit from non-default tolerances → more demand for the optimization machinery the architecture supports.

### Calibration non-features

- **No automatic tolerance inference at inference time.** Calibration is an offline workflow, not a runtime adaptation. The runtime route picker doesn't decide tolerance based on observed inputs; it applies whatever recipe (or strict default) the user configured.
- **No cross-model recipe transfer.** A recipe is keyed by model_hash. A recipe calibrated for LLaMA-7B doesn't apply to LLaMA-13B even if the architectures are similar. Cross-architecture transfer is an interesting research direction but not the v1 commitment.
- **No silent recipe override.** A user who has a recipe loaded but wants strict execution for a specific call uses the per-call tolerance override (already supported in the hierarchy). The recipe isn't automatically suspended for safety-critical paths.

---

## See also

- [01-identity §5](01-identity.md#5-per-op-tolerance-budgets-that-unlock-approximate-optimizations) — tolerance as competitive edge.
- [04-optimization](04-optimization.md) — how rules consume tolerance budgets.
- [05-backend-contract](05-backend-contract.md) — what backends advertise about kernel precision.
- [06-runtime](06-runtime.md) — per-call tolerance overrides at the route picker.
- ROADMAP §"Phase 7.5 graph optimizer architecture" → "Out of scope: approximate optimizations" — the original deferred-pending-budget-semantics note that this section now answers.
