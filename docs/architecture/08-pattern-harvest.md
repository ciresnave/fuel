# Pattern harvest and shared community telemetry

**Status**: v0.2 (draft, 2026-05-09). v0.2 changes: (1) the section now covers the four-flow community-telemetry infrastructure (patterns, tolerance recipes, hardware fingerprints, kernel-stat summaries) — pattern harvest is one of four; (2) the primary opt-in mechanism is a first-use prompt (capturing the "people who don't care" segment without going to silent opt-out); (3) per-flow privacy commitments pinned for each data type.

Opt-in telemetry that tells the project's maintainers which op sequences fuel users actually run, so the fused-op catalog can grow toward what real workloads need fused — not what's familiar from prior frameworks.

This is one of fuel's five competitive edges (see [01-identity §The five competitive edges](01-identity.md#the-five-competitive-edges)). Fuel learns what to fuse next from collective user experience; competitors guess from intuition. Over years, that compounds.

---

## What's harvested

When a user opts in (off by default; explicit flag required), fuel records two kinds of data from the base map (the fully-decomposed primitive DAG, see [03-ir](03-ir.md#the-base-map-fully-decomposed-primitive-dag-permanently-retained)):

- **Longest unfused op sequences.** Subgraphs of N adjacent primitive ops that don't currently match any registered fused op's pattern, ranked by length.
- **Most-frequently-repeated sequences.** Op subgraphs (at their longest matchable form) that appear often across the user's workload.

The "top 10 longest" + "top 10 most repeated at their longest length" framing keeps the data volume bounded (20 sequences per harvest report, not raw graph dumps).

The data is sequence-shaped (op kinds + per-op feature constraints + structural input bindings), not value-shaped. Test inputs and intermediate activations are never harvested — only the *structure* of the computation.

## Why the base map

Harvest reads the base map specifically, not the user-facing form or the optimized form:

- **User-facing form** would bias toward what fuel already supports as fused ops (users who built with `Tensor::softmax_last_dim()` would show that as one node, hiding the underlying sequence).
- **Optimized form** would also bias — the optimizer already fused what it could.
- **Base map** is canonical: every model, regardless of how the user built it, decomposes to the same primitive form. Sequences detected at the base map's level are real opportunities the optimizer couldn't exploit because no fused-op kernel exists for them.

This is the structural reason base map retention (per 03-ir) matters for harvest: without it, harvest would systematically miss the patterns that *should* be fused.

## What's not harvested

The architecture's privacy commitment is sharp:

- **No input data.** Test inputs, intermediate activations, model outputs — never sent.
- **No model weights.** Parameter values stay local.
- **No personally-identifying information.** No usernames, no machine identifiers, no IP addresses recorded with the data (the transport may see them; the harvest payload doesn't embed them).
- **No model identification beyond hash.** Users opt in to share that "model X uses sequence Y often"; they don't opt in to share which specific user is running model X.

What's in the harvest payload:

- Op-sequence structure (op kinds, parameters, structural input bindings).
- Frequency counts (how often this sequence appeared in the user's workload during the harvest window).
- Length metrics (sequence length, span in the DAG).
- Anonymized version stamp (fuel version, hardware class — "x86_64 + CUDA 12" rather than specific GPU model).
- Opt-in identifier (a generated, user-controllable opaque token; not tied to any account).

## How harvest is enabled

Four ways, in priority order:

1. **First-use prompt** (primary mechanism): the first time fuel is used in a new environment (per-user, per-installation), a one-time prompt appears asking the user to enable or disable community telemetry. The prompt explains what's collected (across all four data flows) and provides a link to the privacy commitments. The user's choice is respected for that environment; no further nag. This captures users who would contribute if asked but wouldn't actively opt in via configuration. Industry-standard pattern (Homebrew analytics prompt, rustup installer telemetry question, mise first-run config).
2. **Per-call flag**: the user passes `harvest: true` to a specific realize. Fine-grained control; useful for one-off calibration runs.
3. **Per-process flag**: an environment variable or config option enables harvest for the whole process lifetime. Useful for production telemetry collection.
4. **Configuration file**: a per-installation config that pre-resolves the prompt, with per-call override. Useful for organizations that want to contribute systematically (or definitively opt out across an entire deployment).

In all four cases the default is **off until the user makes a choice**. There is no configuration that silently enables telemetry without an explicit user action — the first-use prompt is opt-in, not opt-out. Headless environments where prompting isn't possible default to disabled (no telemetry without explicit configuration). "Opt-in" is rigorous.

Users who opt in can opt in to all four flows or to specific subsets. Default for opt-in users is "all flows" with the option to disable individual ones via configuration. Most users won't customize per-flow; some will (e.g., a privacy-conscious user who wants to contribute hardware-fingerprint data but not pattern data).

## How harvested data is used

The harvest server aggregates submissions across opted-in users and produces:

- **Ranked sequences for fused-op development priorities.** "These are the 50 most-impactful unfused sequences across our user base; these are the ones to write fused kernels for next."
- **Suggested fusion candidates.** Sequences that appear together often enough to warrant new fused ops in the registry.
- **Trend signals.** Patterns whose frequency is rising over time (new model architectures gaining traction) inform proactive fused-op work.

The architectural payoff: **fuel's fused-op catalog grows toward what users actually need, ranked by aggregate impact.** Each fused op shipped is justified by data; competitors decide what to fuse based on intuition or what was easy.

## Shared infrastructure with tolerance recipes

The community-telemetry infrastructure carries four data flows, each with the same opt-in story but different payload schemas and aggregation logic on the server side:

| Flow | What's shared | Privacy stakes | Server-side use |
| --- | --- | --- | --- |
| **Pattern harvest** | Op-sequence structure (op kinds, parameters, structural input bindings); frequency counts; length metrics. From the base map. | Medium — model-revealing in the structural sense (a never-published architecture appearing in telemetry). | Ranks unfused sequences for fused-op development priorities. |
| **Tolerance recipes** (per [07-tolerance §Community sharing](07-tolerance.md#community-sharing)) | Discovered per-op tolerance budgets keyed by `(model_hash, metric_name, calibration_quality_threshold)`. | Low — tolerance values are numbers per op, not model-revealing on their own. | Ships as suggested defaults for popular models; surfaces with trust signals. |
| **Hardware fingerprints** (per [11-persistence §Cache generation and distribution](11-persistence.md#cache-generation-and-distribution)) | "Hardware fingerprint X exists" — neither model-revealing nor workload-revealing. | Lowest — the same fingerprint is shared by many users; not personally identifying. | Auto-populates the named target sets the cache-generation tool ships with. |
| **Kernel-stat summaries** (per [05-backend-contract §Dynamic telemetry](05-backend-contract.md#dynamic-telemetry-reported-continuously)) | Locally-aggregated per-(op, dtype, size_class, backend, device) summary statistics: median, P95, P99, sample count. Never raw timestamped traces. | Medium — workload-shape implied by which cells have measurements; mitigated by aggregation and model-anonymization (no model identification in the upload). | Refines the cache-generation tool's static-cost annotations toward measured reality; serves as starting baseline for new users' local Judges on similar hardware. |

All four flows share:

- Same first-use opt-in prompt (one decision, four data flows; per-flow override available for fine-grained control).
- Same anonymized identifier (a generated, user-controllable opaque token; not tied to any account).
- Same anonymization rules (no PII, no model identification beyond hash where shared, no IP-address embedding in payload).
- Same transport.
- Different payload schemas, different aggregation logic, different server-side products.

Architecturally they're sibling features on one community-telemetry pipeline. Implementation-side they share the upload mechanism; only the data-collection and aggregation differ per flow.

## Validation, trust signals, and downstream use

Aggregated data has provenance:

- **Submission count per sequence.** A sequence reported by 100 users has different weight than one reported by 1.
- **Diversity of contributors.** Distinct opt-in identifiers contributing the same observation matters; one contributor reporting the same sequence 1000 times is one signal, not 1000.
- **Time stamps.** Patterns that appear consistently over many months are different from one-off bursts.
- **Hardware-class breakdown.** A sequence prevalent on CUDA-12-class hardware may inform GPU-side fused ops; one prevalent on CPU may inform AOCL/MKL kernel work.

Suggested fused-op candidates from harvest data are *suggestions*, not auto-implementations. Maintainers review the data, decide what to prioritize, write the fused kernel. The architecture supports this loop; it doesn't automate kernel writing.

Future direction: as the OptimizationMap rule infrastructure matures, harvested data could feed an *automated* fused-op-candidate pipeline (offline e-graph saturation over harvested patterns, surfacing equivalence classes that warrant new rules). v1 keeps the loop manual; the data substrate supports automation when it's ready.

## What this rules out

- **No silent telemetry collection.** Even with the first-use prompt as the primary opt-in mechanism, the prompt is *opt-in* — no data is sent until the user explicitly enables. Headless environments where prompting isn't possible default to disabled. There is no scenario where fuel collects data without an explicit user action enabling it. (See [09-non-goals](09-non-goals.md) for the full categorical rejection of opt-out telemetry.)
- **No mandatory telemetry.** Fuel never harvests without explicit opt-in. Production deployments that don't opt in are first-class supported; harvest is a contributor benefit, not a tax.
- **No usage analytics beyond op patterns.** Fuel doesn't track session duration, realize counts, error rates, or any other product-analytics-style data. The harvest is *narrowly scoped* to op-sequence telemetry.
- **No proprietary-data leakage paths.** The op-sequence structure is the public-API surface of fuel; sharing structure doesn't reveal anything about the user's model that the model file's existence didn't already reveal. Inputs and weights are explicitly excluded.
- **No retroactive opt-in.** A user who didn't opt in for past sessions has no harvest data to retroactively share. The data was never collected.

## Operational concerns

Two real concerns to acknowledge:

1. **Server infrastructure is non-trivial.** Receiving submissions, aggregating, deduplicating, surfacing prioritized lists, maintaining over time — this is real infrastructure work the project has to commit to. If the server side lapses, the harvest mechanism stops being useful even if the client side keeps reporting.

2. **Trust in the project.** Users opt in to share data with the project's maintainers. The project's commitment is to (a) document what's collected, (b) honor the privacy commitments above, (c) use the data only for the stated purpose (fused-op prioritization). Breaches of these commitments would (justifiably) destroy user trust in the harvest mechanism. The architecture documents the contract; the maintainers honor it.

These aren't reasons to skip the feature — they're reasons to do it carefully. The competitive edge it unlocks is real and worth the operational responsibility.

---

## See also

- [01-identity §The five competitive edges](01-identity.md#the-five-competitive-edges) — pattern harvest as competitive edge #4.
- [03-ir §The base map](03-ir.md#the-base-map-fully-decomposed-primitive-dag-permanently-retained) — the canonical form harvest reads from.
- [04-optimization](04-optimization.md) — the optimizer that consumes registered fused ops once new ones land.
- [07-tolerance §Community sharing](07-tolerance.md#community-sharing) — sibling feature using the same server infrastructure.
- [11-persistence](11-persistence.md) — sibling artifacts (cache, tolerance recipe) that harvest is *not* but shares plumbing with.
