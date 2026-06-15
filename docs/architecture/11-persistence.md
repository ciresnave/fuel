# Persistence

**Status**: v1.2 (2026-06-14). **v1.2 reconciles persistence with the 2026-06-14 "plan IS the graph" redirection** (see [10-decisions-log](10-decisions-log.md)): the native `.fuel` artifact holds the **whole graph** — base map + storage + (after optimization) the optimized multi-path paths — so locally there is no separate cache *file*; the base map is the portable *portion*, and foreign-hardware loads validate-and-scoped-re-optimize ([03-ir §Persisting the unified graph](03-ir.md#persisting-the-unified-graph-base-map--optimized-paths), [06-runtime §Scoped re-optimization](06-runtime.md#scoped-re-optimization)). The separate per-target *distribution* caches below remain valid for **shipping** pre-optimized plans to hardware you don't have locally. Weights/storage are mmap-backed (larger-than-RAM); the Judge baseline ships **bundled in-package** (2026-06-13). v1.1 (2026-06-08) adds the optional **runtime snapshot** (L3) artifact — designated durable runtime state (KV-caches, optimizer state) for resuming a live computation — plus the three-layer *model / +plan / +snapshot* save framing, and records the decision against saving all activations (bandwidth-bound reload loses to on-device recompute). v1.0 (2026-05-09) changes: (1) cache files are mmap'd at process startup, not read into memory; the cache format's mmap-friendly layout becomes load-bearing rather than aspirational; (2) cache updates use write-new-file-and-swap, not in-place writable-mmap modification; (3) per-decision-point dependency records support scoped re-optimization (per [06-runtime §Scoped re-optimization](06-runtime.md#scoped-re-optimization)). v0.2 added "Cache generation and distribution" section covering the `fuel cache generate` CLI tool, named target sets, sibling-file convention with HF Hub / GitHub auto-discovery, multi-version DAG-format support, opportunistic migration during background re-optimization, community-aggregated empirical data refining static annotations, and auto-populated named target sets from opt-in telemetry.

How fuel persists computed artifacts across process restarts: the optimization cache (which lets reload skip optimization), the tolerance recipe (which captures discovered per-op tolerance budgets), and the architectural commitments around format, invalidation, distribution, and offline pre-optimization.

This section cross-cuts most of the other architecture documents. The optimization cache embeds decisions made by the optimizer (04), references kernels advertised by backends (05), and is consumed at startup by the runtime (06). The tolerance recipe embeds decisions made during calibration (07) and shares server infrastructure with pattern harvest (08). Persistence is where these become deployment-ready artifacts.

---

## Artifacts and their lifecycles

Because the plan IS the graph, the primary artifact is **the native `.fuel` file itself** — it holds the whole graph: base map + storage + (after optimization) the optimized multi-path paths. Locally there is no separate cache *file*; finalizing a model writes the unified `.fuel`, and reloading it is fast via load-time validation + scoped re-optimization ([03-ir §Persisting the unified graph](03-ir.md#persisting-the-unified-graph-base-map--optimized-paths)). Two further artifacts have distinct lifecycles and so are **separate siblings**, plus one distribution artifact:

- **Tolerance recipe** (`model.fuel-tolerance` or similar): per-op tolerance budgets from calibration. **Hardware-independent** (keyed to the base map), so it ships once for any hardware — a different lifecycle from the per-hardware optimized paths, hence a separate file.
- **Runtime snapshot** (L3, optional): designated durable runtime state (KV-caches, optimizer state) for *resuming* a live computation. The most ephemeral lifecycle of all. See [Runtime snapshots](#runtime-snapshots-resuming-designated-durable-state-l3).
- **Per-target distribution caches** (optional): for *shipping* pre-optimized plans to hardware the producer doesn't have locally, the cache-generation tool emits the optimized portion as separate per-(hardware, backend) artifacts (see [Cache generation and distribution](#cache-generation-and-distribution)). These are the explicit-distribution analogue of the optimized paths the unified `.fuel` carries locally.

The three-layer mental model still holds — **L1 model** (base map + weights; the portable portion of the `.fuel`), **L2 + plan** (the optimized paths; in the unified `.fuel` locally, or a per-target cache for shipping), **L3 + snapshot** (resume state) — it is just unified into one file locally rather than always split.

## What the persisted graph contains (and a distributable cache extracts)

Because the plan is the graph, the optimized paths are not a separate blob — they are part of the graph the `.fuel` serializes. What that comprises:

1. **The base map** — the fully-decomposed primitive DAG ([03-ir](03-ir.md#the-base-map-fully-decomposed-primitive-dag-permanently-retained)); the portable portion, and re-optimization's restart point.
2. **The optimized multi-path paths** — the bounded per-device Pareto paths with decision points at branches ([04-optimization](04-optimization.md)): rule applications, kernel-variant/placement choices, layout fixups, transfer/cast insertions, cumulative-error and cost annotations, conditional cost adjustments. Hardware-dependent; validated on load.
3. **Kernel selections per path node**, encoded as `(backend_id, op_kind, dtypes, kernel_revision_hash)` tuples — *not* `KernelRef` function pointers (process-local). On load each tuple re-resolves to a live `KernelRef` via the binding-table catalog (optional pre-resolve, else lazy — [06-runtime](06-runtime.md#kernel-resolution-optional-pre-resolve-else-lazy)).

A **distributable per-target cache** is exactly items 2-3 extracted for one (hardware, backend) target (the base map is shared / shippable separately). What is *never* persisted: function pointers (process-local); the Judge's profile data (separate — and a baseline ships bundled in-package); tolerance recipes (separate sibling).

## What gets stored: the tolerance recipe artifact

A calibration run (per [07-tolerance §Tolerance discovery and calibration](07-tolerance.md#tolerance-discovery-and-calibration)) produces:

- **The discovered tolerance map**: per-location tolerance budgets keyed by (node-id-or-region in the base map).
- **Calibration metadata**: which metric was used, test-set size, quality threshold achieved, fuel version, model hash, calibration timestamp.
- **Provenance**: who calibrated (opaque opt-in identifier; never tied to user account), how (which search algorithm), against what (reference backend, hardware fingerprint optional).

The recipe's keys reference the base map's structure, so the recipe is portable across hardware: same base map → same locations → same discovered budgets. (Hardware-dependent variation in *cost* is handled by the optimization cache; *what error a model can tolerate* is a model property, not a hardware property.)

## Storage in the native file: mmap, write-back, tiers

The `.fuel` holds the graph's **storage** as well as its structure, by storage class ([03-ir §Storage classes](03-ir.md#storage-classes-and-sessions)):

- **Shared storage (weights)** is mmap-backed — loaded as a zero-copy view paged on demand, never copied wholesale. This is the prerequisite for **running models larger than RAM** (only the working set is resident; the plan prefetches ahead of the execution frontier per [06-runtime §Cross-tier prefetch](06-runtime.md#cross-tier-prefetch-the-plan-is-the-schedule)). Weights are written back **only when they change** (training), at explicit checkpoints — never write-through per step.
- **Session state** (KV-caches) is written back **only on an explicit snapshot** (L3); an ephemeral session discards it.
- **Transient activations** are never persisted.

So finalizing after training updates the weights in place (a checkpoint), finalizing after optimization writes the optimized paths, and neither forces re-shipping the other — write-back cadence matches each class's lifecycle, and a crash mid-write can't tear the file because writes land at `msync` checkpoints, not continuously.

## Format: memory-mappable, sibling-file, schema-versioned

Both artifacts use the same format conventions:

- **Memory-mappable binary layout.** Read with `mmap` for zero-copy load (subject to the cross-platform mmap caveats every framework deals with). Pointers within the file are relative offsets, never process-absolute.
- **Endianness recorded** in the header for portability (though in practice every target fuel cares about is little-endian).
- **Schema version in the header** for forward-compatibility. Newer fuel commits to reading older DAG-format-version cache files for at least the previous N major versions (concrete N is policy: v1 commits to "all versions for the first 2 years; then a deprecation policy with at least 12 months' notice before dropping a format"). Format additions should be backward-compatible where feasible (newer fields ignored by older readers); where a new feature requires old fields to behave differently, the format does need a hard version bump. Strict-version-or-recompile remains the fallback when readers can't bridge. Most distributors continue shipping format-v1 caches; format-v2 is opt-in for distributors who need newer features.

  Newer fuel reading an older-format cache: cache is valid (executes correctly using the older-format reader). Background re-optimization (per [06-runtime](06-runtime.md)) opportunistically migrates the cache to the current format when it produces a refined plan. Migration is a side effect of the re-optimization that was going to run anyway; no separate migration pass needed.
- **Per-section offset table** for variable-size data (op params, layouts, side-tables). Header says where each section begins; readers seek by offset.

The format is fuel-specific, hand-rolled. We considered flatbuffers / rkyv / capnproto and chose against them: the data is fuel-shape-specific, the schema isn't trying to be cross-language, and a hand-rolled format keeps load logic simple and dependency-free. Schema versioning with strict policy keeps maintenance cost bounded.

## What is unified vs sibling

- **Unified into the native `.fuel`**: the graph — base map + storage + optimized paths. The plan is the graph, so its optimized portion rides in the same file locally (no separate cache file to manage; one artifact to snapshot, copy, or evolve via training). A distribution *stripper* can remove the optimized paths to ship just the portable base map.
- **Separate siblings** (distinct lifecycles): the **tolerance recipe** (hardware-independent — ships once for all hardware) and the **runtime snapshot** (run-dependent — resumes one session); plus **per-target distribution caches** when *shipping* pre-optimized plans for hardware the producer lacks locally.

The reasoning is lifecycle, not file-count dogma: data sharing the graph's lifecycle (its own optimized paths) lives in the graph; data with a divergent lifecycle (a hardware-independent recipe, a one-run snapshot, a foreign-target cache) is a sibling so it can be shipped or invalidated independently. (When importing from a *foreign* model file such as safetensors, fuel still writes its native `.fuel` as the load target; the foreign file is the import source, not the runtime artifact.)

## Invalidation: cache vs recipe

The two artifacts have different invalidation rules. Both store header data that the runtime checks against the running process at load time.

### Optimization cache invalidation

The cache is valid if and only if all of these match the runtime environment:

- **Architecture version** (the registry's schema; the `OpEntry` shape; the rule schema). Bumps with material architecture changes. Mismatch → invalidate.
- **Backend kernel-revision hashes**. Each registered kernel has a hash; the cache embeds the set of hashes for kernels referenced in its plan; mismatch → invalidate. (See [05-backend-contract](05-backend-contract.md) for backend obligations around revision hashes.)
- **Hardware fingerprint**. GPU model + memory size + driver version + multi-device topology. Changes meaningfully affect optimization decisions. Mismatch → invalidate.
- **Profile data version**. The Judge's empirical-data version when the cache was built. Two policies: (a) invalidate eagerly when newer profile data is available (newer data might pick different routes), or (b) accept staleness and let the runtime route picker apply current telemetry on top of the cached top-N alternatives. The architecture commits to (b) — top-N preserves alternatives, telemetry adapts at pick time, modest profile-data drift doesn't require regenerating the whole cache.
- **Tolerance configuration**. If the cache was built for `Tolerance::Strict` and the user is now running with non-strict, different alternatives are admissible. Multi-key approach: cache stores alternatives across multiple tolerance levels (separate plan trees per common tolerance setting); runtime picks the right plan tree based on current tolerance configuration.
- **Model file hash**. Obvious; if the model has changed, the cache is invalid.

So the cache header carries: `(arch_version, kernel_hashes, hw_fingerprint, judge_version, tolerance_set, model_hash)`. Any mismatch on a *strict* field → recompile. Profile-data drift is handled by the route picker, not by invalidation.

### Tolerance recipe invalidation

The recipe is valid if:

- **Model file hash** matches.
- **Architecture version** matches (the recipe references base-map structure; if the IR has changed substantially, structure changes too).

The recipe is **hardware-independent**: changes to backends, hardware, kernel revisions, profile data, tolerance configuration do *not* invalidate the recipe. The recipe records what error the model can tolerate; that property is intrinsic to the model.

This is the structural reason the two artifacts are separate files: their invalidation criteria diverge enough that one cache file would have to invalidate too eagerly (lose tolerance work when only a kernel hash changed) or too lazily (use a cached plan when the hardware has changed).

## Hardware fingerprint sensitivity is a tunable

The optimization cache's hardware fingerprint is the most sensitive invalidation knob:

- **Too strict** (e.g., fingerprint changes on every driver patch) → cache rarely hits, defeating the point.
- **Too lax** (e.g., treats different GPU models as equivalent) → stale plans run on hardware they weren't optimized for.

The right calibration emerges from measurement. The default starts on the **strict side** — driver version included, GPU model included, memory size included, NUMA topology included for multi-GPU. As real deployment data shows which changes actually shift optimization decisions, the policy can be relaxed where it's safe.

The fingerprint is computed deterministically (same hardware, same fuel version → same fingerprint). It's not a security measure; it's a freshness check. Caches from a known-good environment can be safely reused; caches from unknown environments are invalidated.

## Re-resolution on use (lazy, not at load)

The cache file is mmap'd at process startup (per [06-runtime §What the runtime persists](06-runtime.md#what-the-runtime-persists)); the runtime does not read or walk the cached plan at load time. `KernelRef` resolution happens *lazily*: when the route picker chooses an alternative at a decision point, the runtime resolves `KernelRef`s for nodes in that alternative just-in-time via `binding_table.lookup(op_kind, dtypes, backend)`. One HashMap lookup per node, ~100 nanoseconds; trivial cost amortized over execution.

Combined with mmap, this means startup is essentially instant for cache hits. Only the cache header and per-decision-point index get touched before the first realize. Pages for never-picked alternatives may never load at all.

When `KernelRef` resolution fails (a tuple's `kernel_revision_hash` doesn't match any currently-registered kernel — a backend has been updated since the cache was built), the affected decision point is invalidated. Per the scoped re-optimization model, only that decision point's alternatives need re-generation; other decision points keep their cached alternatives. If many decision points fail to resolve at once (e.g., a major backend version change), the cumulative invalidation can amount to whole-cache discard, in which case the runtime falls back to running optimization from the cached base map.

## Per-decision-point dependency records for scoped re-optimization

Each decision point in the cache stores a small dependency record alongside its alternatives:

- Which kernels (by `kernel_revision_hash`) its alternatives reference.
- Which (backend, device) placements its alternatives use.
- Which cells `(op, dtype, size_class, backend, device)` of profile data its costs depend on.

These records are what enables [scoped re-optimization](06-runtime.md#scoped-re-optimization): when a trigger fires (device removed, kernel updated, profile data refined), the runtime intersects the trigger with each decision point's dependency record to compute the affected scope. Re-optimization runs only on affected decision points; unaffected decision points are untouched.

The dependency record is small (typically tens of bytes per decision point); cumulative storage cost is negligible compared to the alternatives themselves.

## Distribution: cache as a deployment artifact

Once persistable, the optimization cache becomes a natural artifact for **shipping pre-optimized models**. A model author can:

1. Run fuel offline against a target hardware fingerprint.
2. Produce the optimization cache.
3. Ship `model.safetensors + model.safetensors.fuel-cache` as a bundle.
4. End users with matching hardware fingerprints get instant optimization-skip.
5. End users with non-matching fingerprints fall back to runtime optimization from scratch (still using the shipped base map, which is hardware-independent and saves the decomposition step).

This is a feature TensorRT charges for and torch.compile is still working out. Fuel ships it as a natural consequence of the architecture, not as a separate productization.

The architecture supports this; it doesn't mandate it. Many model distributions won't ship caches (the producer doesn't know consumers' hardware in advance). Fuel works fine without shipped caches; users build them locally on first run.

## Cache generation and distribution

To make distribution friction-free, fuel ships a `fuel cache generate` CLI tool (and equivalent library function) that produces cache files for many target environments in one command:

```bash
fuel cache generate \
    --model llama-2-7b.safetensors \
    --target-set common-2026 \
    --output-dir ./cache/
```

`--target-set common-2026` is a named bundle defined in fuel itself: "the top N consumer GPUs from the Steam hardware survey + A100/H100 + Apple M-series + AOCL CPU baseline + MKL CPU baseline" or similar. Fuel ships and updates these target-set definitions over time as hardware popularity shifts. Distributors don't have to know which specific hardware to target — they pick a named set; fuel resolves it to ~10-20 (hardware, backend) pairs.

**The tool runs statically.** It doesn't need actual hardware to emulate. Static cost annotations + per-target hardware fingerprints + cost-model layers 1-2 are enough to produce a plan. Empirical Judge data (layer 3, runtime telemetry) refines the plan when end users actually run it on real hardware. So the generation tool can run anywhere — a contributor's laptop produces caches for hardware they don't own.

**Cardinality bound.** 10-20 (hardware, backend) pairs × 1-2 fuel-DAG-format versions ≈ 10-40 cache files per model. Each typically 1-10MB (the plan, not the weights). For a 14GB Llama-7B model file, the cache bundle adds ~50-200MB total — under 2% overhead. Manageable.

**Lower-friction default.** Distributors who don't want to think about specifics use:

```bash
fuel cache generate --model my-model.safetensors --defaults
```

Generates for the project-curated default target set. Distributors who care about specific niche hardware add `--target rtx-3060,m1-ultra` etc.

### Static annotations refined by community-aggregated empirical data

When community-aggregated kernel-stat summaries are available for a target hardware fingerprint (per [08-pattern-harvest §Shared infrastructure with tolerance recipes](08-pattern-harvest.md#shared-infrastructure-with-tolerance-recipes)), the cache-generation tool fetches them and uses them to refine layer-1 cost annotations. Per-cell community medians replace FLOP-counting estimates where data exists, with confidence intervals tied to sample count. Cells without community data fall back to FLOP-counting.

The result: caches generated against hardware fingerprints with community data are calibrated against actual measured behavior on similar hardware, not just theoretical bounds. Cold-start optimization quality on common hardware approaches what local empirical refinement would produce, before the user has run anything locally.

Distributors who don't want to fetch community data can pass `--no-community-priors`; the tool uses static-only annotations. The privacy implication of fetching community priors is small (the fetch reveals what hardware fingerprints the distributor is generating for, not what model), but is documented.

### Remote loader integration with auto-discovery

Fuel's loader supports model loading from remote sources, with sibling-cache auto-discovery:

```rust
let model = fuel::load("hf://meta-llama/Llama-2-7b")?;
let model = fuel::load("github://author/repo/model.safetensors")?;
let model = fuel::load("https://example.com/model.safetensors")?;
let model = fuel::load("./local/model.safetensors")?;
```

The `hf://` and `github://` URI schemes are sugar over the underlying transport. fuel-loaders uses the existing `hf-hub` Rust crate for HF Hub; GitHub is HTTPS GET on `raw.githubusercontent.com`.

When loading from a remote source, fuel:

1. Identifies the user's environment fingerprint (hardware, backends, fuel version, tolerance config).
2. Looks in the same remote location for cache files matching that fingerprint. Naming convention: `{model-name}.fuel-cache.{hw-fingerprint-short}.{backend-set-short}.{fuel-version-short}`.
3. If found, downloads and validates against the strict invalidation criteria above.
4. If validation passes, uses the downloaded plan; skips local optimization (or starts background re-optimization per [06-runtime](06-runtime.md)).
5. If no matching cache or validation fails, falls back to local optimization (with concurrent execute hiding most of the cost).

The user sees a one-line log message: `"using cache plan from hf://meta-llama/Llama-2-7b matching your hardware (rtx-4090.cuda.v1)"` — transparent without being noisy.

### Named target sets are auto-populated

Fuel needs to know which hardware fingerprints to include in named target sets like `common-2026`. The architecture commits to **opt-in auto-population from telemetry**: when fuel encounters hardware not in any current target set, an opt-in user (per [08-pattern-harvest](08-pattern-harvest.md)) reports the fingerprint to the project's server. After N distinct opt-in identifiers report the same fingerprint cluster (default N=20 over a 30-day window), the server adds it to the next target-set update.

Caveats:

- **Aggregation by hardware-model first.** Same GPU + different driver versions = different full fingerprints. Auto-population aggregates at the GPU-model level for inclusion decisions; full fingerprint differences still drive cache-validation matching.
- **Maintainer review for malicious submissions.** Server-side rate-limiting per identifier; sanity checks (claimed fingerprint should map to plausible hardware specs).
- **Eviction policy.** Hardware that was popular three years ago shouldn't stay in the curated set forever. The project commits to periodic review and eviction.
- **Updates ship with fuel.** Newly-included fingerprints appear in the next fuel release's target-set definitions, OR as a downloadable data file refreshed without a full version bump (implementation choice).

The privacy stakes for fingerprint reporting are lowest of the four telemetry flows (the data is "this hardware exists" — neither model-revealing nor workload-revealing nor personally identifying since many users share fingerprints).

## Tolerance recipes shipped alongside

Tolerance recipes face the same distribution opportunity: a calibrated recipe shipped alongside a model gives end users discovered tolerances on first run. Hardware-independence makes this easier than caches — one tolerance recipe works for any user's hardware.

The shipping convention: same sibling-file layout, same schema-version-and-hash invalidation. A tolerance recipe shipped with a model becomes the default for users who don't override. Per-call overrides remain available.

This pairs naturally with the [community sharing](07-tolerance.md#community-sharing) framework — a maintainer or contributor can calibrate once, contribute to the project's server, and the recipe becomes available to any user via download. Recipes shipped in-tree alongside model files are convention; recipes downloaded from the project's server are ergonomic.

## Concurrent execute and persistence interaction

A cached optimization plan is, by definition, complete: every decision point's alternatives are precomputed, the optimization frontier has reached the end of the graph. So when a cache hits, **concurrent optimize-and-execute is moot for that realize** — the runtime can skip optimization entirely and dispatch from the cached plan immediately.

This is the cache's primary use case: cold-start TTFT optimization. Concurrent execute helps the *first-ever* realize on a new graph; the cache helps every *subsequent* realize after the first one's optimization has been persisted.

The two are complementary:

- **First realize, no cache**: concurrent execute starts dispatch as soon as the optimization frontier crosses the first nodes.
- **Subsequent realize, cache hit**: cache load skips optimization; runtime dispatches from the precomputed plan directly.
- **Cache hit but graph extended (autoregressive decoding)**: load the cache for the original graph; concurrent-optimize the appended portion as decoding proceeds.

## Runtime snapshots: resuming designated durable state (L3)

The optimization cache lets a process skip *optimization* on reload; it does not capture *runtime state* — the tensors a live computation has accumulated. A third, optional artifact, the **runtime snapshot**, captures designated durable state so a process can *resume* a paused computation instead of restarting it. This completes a three-layer save model:

- **L1 — model** (base map + weights): portable, hardware-independent. The native `.fuel` artifact ([13-interchange](13-interchange.md)).
- **L2 — + plan** (the optimization cache above): hardware-dependent; hot-loads by skipping optimization.
- **L3 — + snapshot** (designated durable state): process/run-dependent; resumes live state.

"Save with vs without the plan / runtime state" is **not a flag inside one file** — it is *which sibling artifacts a caller writes*. L1 alone is a cold-loadable model; L1+L2 hot-loads on matching hardware; L1+L2+L3 resumes a session. Keeping them as separate siblings is the same lifecycle argument as the cache vs recipe split: the model is valid everywhere, the plan is valid per-fingerprint, the snapshot is valid for one run.

### What a snapshot contains: designated durable state, not all activations

A snapshot persists state that is **durable and expensive to recompute**, explicitly enumerated by the producer — never a blind dump of the executor's realized-node cache:

- **KV-caches** — a serving session's accumulated attention state (backed by `Op::WriteSlice` / `Op::WriteSliceRotating`). Persisting these checkpoints a conversation.
- **Optimizer state** — training moments / momentum / step counters.
- **Producer-designated long-lived intermediates** — a costly partial result a producer explicitly marks resumable.

### Why not save every activation

Persisting the full set of realized activations to make a hot load "come up running" does **not** speed launch, and usually slows it:

- **Input-dependent activations are invalid across launches.** A forward pass's intermediates are a function of *that* input; a fresh launch with any other input cannot reuse them. (If the input is identical, the *output* is the thing to cache, not the intermediates.)
- **Reload is bandwidth-bound; recompute stays on-device.** Activations are large — often larger than the weights. Loading them disk → host → device is bandwidth-bound at every hop, while recompute keeps the work on-device where compute is cheap relative to transfer. This is the same trade that makes **gradient checkpointing** recompute activations rather than store them: when bandwidth is the constraint, recompute wins, and disk is a slower tier than the device memory those schemes already avoid.
- **The real launch-speed levers are already in L1+L2.** mmap'd weights (near-instant load) + the plan cache (skip optimization) + lazy `KernelRef` resolution. The first forward pass's compute is input-dependent and unavoidable; saved activations don't shorten it.

So the architecture commits to **designated durable state**, not all-activations.

### Optional: materialized derived constants

The one place the "precompute and save" intuition pays off is **input-independent** derived values not already constant-folded into L1/L2 — e.g. a quantized model dequantized to a wider dtype at first use, or precompute tables the optimizer didn't fold. These can be persisted as an optional **derived-weights** variant of the model artifact, trading disk for skipped one-time preprocessing. This is a *weight* artifact (input-independent, reusable across runs), not an activation snapshot, and it is opt-in.

### Snapshot invalidation

The snapshot has the **most ephemeral** lifecycle of the three artifacts, so it is the most strongly separated:

- **Model hash** + **base-map / graph identity** must match — a snapshot is meaningless against a different graph.
- **Hardware fingerprint** matches when the snapshot holds device-resident tensors (it usually does).
- Written on **explicit checkpoint** only — never write-through per step (the same no-write-amplification rule the cache follows).

## Concerns to honor

Three real costs to acknowledge:

1. **Schema versioning is a maintenance commitment.** Every IR shape change breaks cache compatibility. Strict-version-or-recompile is the conservative default. If cache-compat becomes a deployment headache (lots of users complaining about cache misses after fuel updates), migration tooling for in-place upgrade becomes a real requirement. Until that data exists, strict-version is the right policy.

2. **Hardware fingerprint sensitivity is iterative.** Default tight; relax based on measured data. Fingerprint policy will evolve over fuel's lifetime; the architecture commits to *having* a fingerprint, not to a specific fingerprint definition.

3. **The cache becomes a deployment artifact users may not understand.** "Why does fuel re-optimize on this machine but not that one?" is a question users will ask. The architecture's answer: the fingerprint changed; the cache doesn't apply. Documentation has to make this transparent; surfacing the fingerprint difference (e.g., "cache invalidated because driver updated from 535.x to 545.x") helps users diagnose.

## What this rules out

- **No automatic cross-hardware translation.** A CUDA cache cannot become a Vulkan cache automatically. Cross-target cache distribution requires per-target builds; the architecture doesn't pretend otherwise.
- **No silent partial-cache use.** If any strict invalidation field mismatches, the cache is invalid as a whole. Partial reuse ("the plan is mostly right; just rewrite a few decisions") is risky and out of scope.
- **No write-through cache during execution.** The runtime doesn't update the cache on every realize. Caches are written when the optimizer commits a new optimized form (initially or after re-optimization). Updating during execution would create write-amplification and make the cache file racy.
- **No cache encryption.** The cache contains structural information about the model (op sequences) but not weights. Sensitive deployments that don't want to leak structure can disable cache file generation; encryption is out of scope.

## Where this lives in code

The persistence layer is a Foundation-layer concern (per [02-layers](02-layers.md)). Implementation:

- `fuel-graph` produces the in-memory base map and optimized form.
- A `fuel-cache` module (likely in fuel-graph or a sibling crate) handles serialization, schema versioning, mmap layout, fingerprint computation.
- The runtime ([06-runtime](06-runtime.md)) calls cache load on startup and triggers cache write when the optimizer commits.
- Backends contribute kernel-revision hashes ([05-backend-contract](05-backend-contract.md)) that flow into the cache header.

Implementation detail (specific Rust types, file extension conventions, exact hash function) is not architectural; it lives in the relevant crates' design docs.

---

## See also

- [03-ir §The base map](03-ir.md#the-base-map-fully-decomposed-primitive-dag-permanently-retained) — the in-memory artifact this section persists.
- [04-optimization §Per-decision-point alternatives](04-optimization.md#per-decision-point-alternatives) — the optimized form's structure.
- [05-backend-contract](05-backend-contract.md) — kernel-revision hashes, backend identity, hardware fingerprint inputs.
- [06-runtime](06-runtime.md) — cache load and re-resolve on startup.
- [07-tolerance §Tolerance discovery and calibration](07-tolerance.md#tolerance-discovery-and-calibration) — the calibration workflow that produces tolerance recipes.
- [08-pattern-harvest](08-pattern-harvest.md) — sibling community-sharing infrastructure.
- ROADMAP §"Phase 7.5 — Core simplification" — the broader cleanup phase persistence work would join.
