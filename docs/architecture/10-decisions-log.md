# Decisions log

**Status**: v1.0 (2026-05-09).

Material architectural changes over time. Each entry records what changed, when, why, and which sections were affected. The log is the audit trail for the architecture; it answers "why does the architecture say X?" when X is non-obvious.

---

## Format

Each material decision gets one entry. Entry shape:

```text
## YYYY-MM-DD — Short title

**Sections affected**: 01, 03, 04 (whichever were revised)
**Phase / PR**: Phase 7.6, or PR #N, or "design pass — no code"
**Bumped to**: 01 v1.1 → v1.2, 03 v1.0 → v2.0, etc.

**What changed**: one paragraph summarizing the architectural shift.

**Why**: the underlying motivation. Often a real consumer pressure, an
incident, a design pass that surfaced a tension, or a contradiction the
existing architecture couldn't resolve.

**Alternatives considered**: 1-3 alternatives the project examined and
rejected, with brief reasons.

**Implications going forward**: what this decision constrains or unlocks
for future work.
```

The format keeps entries small (a paragraph or two each) but informative enough that someone reading the log a year later can understand the decision without re-deriving the context.

## What counts as material

A change is material — and gets a log entry — if it:

- Changes a section's *core claim* (warrants a MAJOR version bump per the index's versioning convention).
- Removes or substantially redirects a previously-stated commitment.
- Adds a new commitment that constrains future work.
- Resolves a previously-deferred decision (one of the audit's cross-cutting questions getting a definitive answer).

A change is *not* material — and doesn't need a log entry — if it:

- Fixes typos, link rot, or formatting issues.
- Adds clarifying language without changing meaning.
- Refines an existing commitment without changing its scope.

The distinction is judgment. When in doubt, log it; entries are cheap, missing context is expensive.

## What doesn't go here

The decisions log records *architecture-set* changes — modifications to the documents in `docs/architecture/`. Other kinds of decisions live elsewhere:

- **Phase work plans, sub-task ordering, work-item scoping**: ROADMAP.md.
- **Per-feature design decisions during implementation**: per-phase design docs in `docs/` (e.g., `docs/storage-unification.md`, `docs/fused-op-registry.md`).
- **Code-level design decisions, type-shape choices, file layouts**: code reviews, commit messages, PR descriptions.
- **Memory entries** about session-by-session context: `~/.claude/projects/.../memory/`.

Cross-references are fine — an architecture-decision-log entry can link to the phase doc that motivated it. But the architecture log isn't a replacement for those other records.

---

## 2026-05-09 — Architecture set v1.0 established

**Sections affected**: 00 through 11 (all eleven sections established or revised).
**Phase / PR**: design pass — no code.
**Bumped to**: 00 v1.0, 01 v0.2, 02 v0.2, 03 v0.2, 04 v0.3, 05 v0.2, 06 v1.0, 07 v0.3, 08 v0.2, 09 v0.1, 10 v1.0, 11 v1.0.

**What changed**: the v0.x → v1.0 drafting period (2026-05-08 to 2026-05-09) established the foundational architecture set in `docs/architecture/`. Eleven sections were drafted and iteratively revised across two batch revision passes. Twenty-four material architectural decisions were made during the drafting; the architecture set replaces the previous piecemeal phase-by-phase design approach with a single durable description that all phase work anchors to. Triggered by the architectural audit (`docs/architecture-audit.md`, 2026-05-08) which mapped eight in-flight architectural threads, surfaced five cross-cutting questions, and recommended this consolidation over continued phase-by-phase work.

**Why**: through May 2026 fuel had accumulated multiple in-flight architectural threads (Phase 7.5 storage unification, Phase 7.6 FusedOpRegistry, Phase 6b Judge, scheduler-driven residency, GradientRule migration, CUDA depth migration, Layout-on-Node, binding table) that were partially mutually inconsistent. The architectural audit identified that piecemeal continued work would compound rather than resolve the inconsistencies. The user chose to pause feature work and produce a foundational architecture set — the constitutional document the rest of fuel anchors to.

**Twenty-four decisions captured during v0.x → v1.0**:

1. **DAG-first identity.** Fuel commits to "the DAG is the source of truth for every decision." All strategic decisions (placement, fusion, kernel-variant choice, slot assignment, tolerance trade-offs) live at the DAG level; backends advertise capabilities and execute what the optimizer hands them. (01-identity.)

2. **Five competitive edges.** Cross-backend placement aware of fusion catalogs; algebraic-equivalence rewrites; top-N route preservation; pattern-harvest-driven fused-op development; per-op tolerance budgets. (01-identity.)

3. **The bet stated as compounding.** "Bigger fused kernels matter and will keep mattering. Above the kernel layer, optimization techniques that span ops, span backends, adapt at runtime, and trade controlled error for compute will keep finding wins backend-internal fusion can't reach." Both layers compound. (01-identity.)

4. **Op-shape A: closed enum primitives + `Op::Fused(id, params)` arm.** No separate `NodeKind` discriminator type. After exploring four options (A two-tier; B newtype + collapsed registry; C parallel types with NodeKind; D flat enum with all ops), the user chose A for long-term goals — preserves open-registry property for fused ops while keeping the type system reflective of the architecture. (03-ir.)

5. **Pre-resolved KernelRef per node.** The binding table is a planning-time catalog, not a runtime lookup; the executor calls function pointers directly. Resolves audit Q-A definitively. Lazy resolution at use time pairs with mmap'd cache. (03-ir, 06-runtime.)

6. **Three coexisting artifacts**: user-facing form, base map (canonical primitive DAG, retained as permanent artifact), optimized form (top-N alternatives per decision point with pre-resolved kernels). (03-ir.) *(**Amended 2026-06-14** by the "plan is the graph" redirection below: the optimized form is not a separate top-N artifact — it is the **same graph transformed in place** into bounded Pareto paths + decision points. User-facing form and base map stand; "top-N alternatives" is superseded by the per-device frontier. See [03-ir](03-ir.md).)*

7. **Per-decision-point alternatives, not N global routes.** The optimizer preserves up to N alternatives per decision point (default N=3); decisions can be coupled via conditional cost adjustments. The runtime route picker resolves alternatives at dispatch time using current telemetry, mixing and matching across decision points. Strictly more flexible than top-N complete plans. (04-optimization.) *(**Superseded 2026-06-14** by the "plan is the graph" redirection below: alternatives attach to **branch points** and the retained paths per device are bounded by a **Pareto frontier + crowding cap**, not a fixed N. The "per decision point, not N global routes" direction stands; the fixed `N=3` does not. See [04-optimization](04-optimization.md) and item 6's superseded note.)*

8. **Three forms of parallelism named explicitly**: pipeline (sliding window between optimizer and executor), data (independent subgraphs run concurrently), within-kernel (SIMD/intra-kernel concurrency owned by backends). (04-optimization, 05-backend-contract, 06-runtime.)

9. **Slot-capacity advertisement model.** Even same-device parallelism flows from DAG-level decisions, not backend-internal policy. Backends advertise total + currently-available slots; runtime allocates work to slots. Honors the "backends advertise; they don't decide" principle even at same-device granularity. (05-backend-contract, 06-runtime.)

10. **Concurrent execute is per-rule, per-route.** Rules self-declare frontier-compatibility (`Concurrent` | `WholeGraph`); routes built from `Concurrent` rules can execute concurrently with optimization; routes including `WholeGraph` rules require whole-graph optimization first. Per-realize concurrency policy: `Auto` / `Required` / `Forbidden`. (04-optimization.)

11. **Tolerance as fifth competitive edge with hierarchical specification.** `Strict` | `Relative(x)` | `Absolute(x)`; graph-default → subgraph override → per-op override → per-call override; tightest-wins composition. Best-effort upper bounds annotations now → empirical Judge measurement later. (07-tolerance.)

12. **Tolerance discovery and calibration.** Opt-in workflow that runs models through user-supplied test inputs to discover maximum acceptable per-op tolerance budgets. Metrics library (accuracy, KL, perplexity, embedding distance, custom); search algorithms (greedy, sensitivity-first, Bayesian); hierarchical granularity (per-region → per-layer → per-op); discovered recipes shareable on community server. (07-tolerance.)

13. **Reference backend dissolved into per-kernel `PrecisionGuarantee` structure.** Each kernel declares `{bit_stable_on_same_hardware, max_ulp, max_relative, max_absolute, notes}`. The always-built backend (fuel-cpu-backend by convention) commits to providing at least one `bit_stable` kernel for every primitive op as the coverage guarantee. Cleaner than special-status backend. (05-backend-contract.)

14. **Pattern harvest opt-in with first-use prompt.** Industry-standard pattern (Homebrew analytics prompt, rustup installer, mise first-run) captures "people who don't care" segment without going to silent opt-out. "No silent telemetry" added as explicit non-goal. (08-pattern-harvest, 09-non-goals.)

15. **Four-flow community telemetry infrastructure.** One opt-in pipeline carries: pattern harvest (op sequences) + tolerance recipes (per-op error budgets) + hardware fingerprints (auto-populates target sets) + kernel-stat summaries (refines static cost annotations toward measured reality). Per-flow privacy commitments documented. (08-pattern-harvest.)

16. **Cache generation and distribution tooling.** `fuel cache generate --target-set common-2026 --defaults` produces caches for many target environments in one command. Static-cost annotations refined by community-aggregated empirical data when available. Named target sets auto-populated from opt-in fingerprint telemetry. (11-persistence.)

17. **Remote loader integration.** `fuel.load("hf://...")`, `github://`, `https://`; auto-discovers sibling cache and tolerance-recipe artifacts at the model's location. fuel-loaders is the implementing crate. (02-layers, 11-persistence.)

18. **Multi-version DAG-format support + opportunistic migration.** Newer fuel reads at least the previous N format versions; format additions are backward-compatible where feasible; background re-optimization migrates older caches to the current format as a side effect of producing refined plans. (11-persistence, 06-runtime.)

19. **Background re-optimization with per-decision-point atomic swap.** Downloaded caches are starting points; background optimization with local empirical Judge data refines them via merge logic (cached alternatives + newly-discovered alternatives, re-ranked by local cost, top N retained per point). Atomic swap is per-decision-point, not whole-DAG. Same primitive used by concurrent optimize-and-execute. (06-runtime.)

20. **Local Judge baseline initialization.** The Judge can optionally start from a community-aggregated profile for the user's hardware fingerprint before refining with local measurements. (06-runtime.)

21. **Precision-filter pass before cost ranking.** Alternatives at each decision point are filtered by their `PrecisionGuarantee` against the user's per-call precision requirement and cumulative tolerance budget; cost ranking ranks the survivors. (04-optimization.)

22. **Scoped re-optimization based on trigger.** Each decision point records its dependencies (kernels, devices, profile cells); triggers compute the affected scope by intersection. Most triggers (device removed, kernel updated, profile data refined) affect a small subset of decision points, not the whole graph. (06-runtime, 11-persistence.)

23. **mmap'd cache with lazy KernelRef resolution.** Cache files are mmap'd at process startup, not read into memory; only the cache header gets touched immediately. KernelRef resolution is lazy at decision-point pick time. Startup is near-instant for cache hits. Write-new-file-and-swap on cache update; mmap-fallback to read-into-memory where mmap is unavailable. (06-runtime, 11-persistence.)

24. **Architecture set as constitutional document.** The eleven sections in `docs/architecture/` are authoritative; phase docs cite them. Phase docs propose changes to the architecture set; if accepted, the set is updated and phase docs anchor to the updated section. The decisions log (this file) records material architectural changes going forward. (00-index, 10-decisions-log.)

**Alternatives considered, then rejected**:

- *Continue piecemeal phase work without consolidation.* Audit identified compounding inconsistencies; rejected.
- *Full top-down architectural revamp without an audit first.* Higher risk; the audit gave information to make targeted decisions; consolidation rather than revamp was sufficient.
- *Op-shape B (newtype `Op` + collapsed registry).* Cleaner uniformity but loses pattern-matching ergonomics for primitives (the common case in algebraic rewrites); user chose A after weighing the trade.
- *Op-shape D (flat enum with all ops).* Reconsidered late in the design pass; rejected because the open-registry property for fused ops is genuinely valuable for long-term automation and downstream contributions.
- *Reference backend as distinguished oracle.* Replaced with per-kernel `PrecisionGuarantee`; same correctness anchor without the backend-special-status awkwardness.
- *Opt-out telemetry by default.* Rejected; legal exposure (GDPR/CCPA), trust costs, architectural-alignment costs outweigh data benefit. First-use prompt captures most of the "people who don't care" segment.
- *Server-side cached optimization plans aggregated by fuel-the-project.* Rejected; infrastructure cost, versioning headaches, hardware-fingerprint variability, trust boundary. Replaced with community-distributed caches alongside model files (sibling-file convention; HF Hub / GitHub auto-discovery).
- *e-graph saturation as the primary optimizer engine.* Rejected; performance characteristics unsuitable for per-realize hot path. Possible offline rule-discovery tool. Declarative + callable rule patterns serve the optimizer's needs.

**Implications going forward**:

- Phase work re-anchors to the architecture set. Phase 7.6 (FusedOpRegistry) implementation now follows architecture v1.0's commitments rather than the original Phase 7.6 design doc.
- The decisions log becomes load-bearing — every material architectural change going forward gets a log entry.
- The 24 decisions captured here form the baseline; future decisions are documented against this baseline.
- Implementation work (in code, in phase docs) is sized against the architecture's commitments, not against earlier inconsistent intermediate states.

**Related artifacts**: `docs/architecture-audit.md` (the audit that triggered consolidation); session memory entry `project_architecture_doc_set_v0_2.md` (initial drafting state); `project_phase_7_6_paused_for_audit.md` (the in-flight phase work that paused for the audit).

---

## 2026-05-09 — Fused-op registry crate-split clarified

**Sections affected**: 03 (IR).
**Phase / PR**: Phase 7.6 step 1 implementation, commit `408ff57a` on `feature/storage-unification`.
**Bumped to**: 03 v0.2 → v0.3.

**What changed**: clarified the fused-op registry's cross-crate split. Architecture v1.0 §03-ir's "What lives where" table already named the split correctly ("fuel-graph (metadata) + fuel-storage (BackendImpl payloads)"), but `docs/fused-op-registry.md` v2 wrote the type-shape as a single `FusedOpEntry` struct in `fuel-graph` carrying a `SmallVec<[(BackendId, BackendImpl); 4]>` field. That doesn't compile: `BackendImpl` carries `KernelRef`, which lives in `fuel-storage`, and `fuel-storage` already depends on `fuel-graph` (not the reverse). Step-1 implementation surfaced the contradiction. Resolution: graph-side metadata (id, name, family, pattern, decompose, backward, shape/dtype rules) lives in `fuel-graph::registry::FusedOpEntry`; kernel-side payloads (`BackendImpl`, `CostEstimate`, `PrecisionGuarantee`, `KernelRevisionHash`) live in `fuel-storage::fused::FusedKernelRegistry`; the two halves are joined by `FusedOpId` at runtime. `docs/fused-op-registry.md` bumped to v3 to match. No architectural commitment changed — only the implementation-side rendering of an existing commitment.

**Why**: the dependency graph forces the split. Putting the registry entry in `fuel-graph` is right (rule code + lowering rules need access to `decompose` and `pattern` callables), but `KernelRef` cannot reach `fuel-graph` without inverting an existing crate dependency. A single struct in either crate fails the architecture's "metadata + payload" partition.

**Alternatives considered**:

- *Move the whole registry to `fuel-storage`.* Rejected — rule code in `fuel-graph::opt` needs `decompose` and `pattern`, and `fuel-graph` cannot import from `fuel-storage`.
- *Add a third crate `fuel-fused-registry` that both depend on.* Rejected — the metadata side genuinely belongs with the graph (next to `Op`, the lowering rules, the autograd backward dispatch); only `KernelRef` forces the split. A third crate would create artificial fragmentation.
- *Generic `FusedOpEntry<I = ()>` parameterized by impl type.* Rejected — the fuel-graph-side entry never carries impls (it can't), so the type parameter is always `()`; trait-objects or unit-types add complexity without buying anything.

**Implications going forward**: future fused-op migrations (Phase 7.6 step 4) register the metadata-side entry in `fuel-graph::registry::<op>::entry()` and the kernel-side `BackendImpl`s in `fuel-storage::fused::FusedKernelRegistry::register(id, backend, impl_)`. The `register_fused!` macro proposed in step 6 now spans both crates and does the join by id. Step 9's binding-table planning-time refactor reads from the kernel-side registry by id when resolving `KernelRef`s.

**Related artifacts**: commit `408ff57a` (Phase 7.6 step 1); `docs/fused-op-registry.md` v3; session memory entry `project_phase_7_6_step_1_shipped.md`.

---

## 2026-05-22 — Planning-surface reconcile pass + bridge-retirement trajectory recorded

**Sections affected**: none (architecture set unchanged). ROADMAP + session-prompts + memory cross-references updated.
**Phase / PR**: organizational cleanup — no code.
**Bumped to**: no section version bumps. This is a *cross-reference* decision, not an architecture-set change.

**What changed**: a "Current frontier" stanza, a "Recently shipped (last 30 days)" pointer, and a "Next 1-3 sessions" priority list were added to the top of `ROADMAP.md`. Phase 7.5 work item B2, Phase 7.6 steps 1-3 + 6 + partial-9, were updated from `[ ]` to `[x]` / `[~]` to reflect the shipped/in-progress states already recorded in memory. A new Phase 7.6 step 9c subsection captures the typed-storage-retirement audit summary and adds a bridge-retirement-trajectory subsection that maps the path from this session's `VulkanBackendDevice` bridge to the architecture v1.0 destination (graph-level `Op::Copy` / `Op::Alloc`, dispatch-erased `Device` tag, retired `DynBackendStorage` trait). 10 session prompts whose work has shipped were moved to `docs/session-prompts/shipped/` with a README explaining the archive policy; 3 active prompts remain in the top-level directory (`baracuda-cutlass-alpha-13-integration.md`, `fill-op-primitive-set.md`, `onemkl-v0-2-followups.md`).

**Why**: through May 2026 the planning surfaces drifted apart. ROADMAP entries for Phase 7.5 / 7.6 were drafted before the architecture set existed; many sub-tasks completed and were tracked in memory rather than reflected back into ROADMAP. Multiple "Phase" numbering schemes nested (ROADMAP Phase 7.6 / audit Phase E.3 / bridge-retirement Phase 2-7) made the "you are here" arrow invisible — an LLM session had to reconstruct current state from ~3 audit memos + the in-flight memo every time. The user surfaced the drift; the answer was *use the existing planning surfaces better* rather than create a new central plan (per the [02-layers](02-layers.md#stopping-rule-for-new-crates) stopping rule applied to planning artifacts).

**Alternatives considered**:

- *Create a new top-level "master plan" doc.* Rejected — duplicates ROADMAP's role and violates the stopping rule.
- *Defer the reconcile until Phase 7.6 ships.* Rejected — the drift is actively costing context-reconstruction time per session; the reconcile is cheap and pays back immediately.
- *Delete shipped session prompts rather than archive.* Rejected — historical record of *why* a session was framed a certain way is useful when revisiting decisions.

**Implications going forward**:

- The "Current frontier" stanza is the new "you are here" surface; update at session end. One-line maintenance per session.
- Each shipped session prompt's archive in `shipped/` documents its corresponding `project_*_shipped.md` memory entry; cross-referencing is by filename / phase number.
- Bridge-retirement trajectory under Phase 7.6 step 9c is now the authoritative map of what code dies when. Future sessions touching the typed-storage / `Device` / `DynBackendDevice` surface should consult that section before writing new bridge-shaped code.
- The four [01-identity enforcement-check questions](01-identity.md#how-this-identity-is-enforced) become the per-session architecture-alignment gate: every active workstream should be runnable through them at session start, with the result recorded in the session prompt.

**Related artifacts**: this session's `ROADMAP.md` edits; `docs/session-prompts/shipped/README.md`; memory entries `project_phase_7_6_step_9c_parity_audit.md` (updated this session) and `project_vulkan_v3_fanout_shipped.md` (updated this session) for the Vulkan Device-wiring follow-ups.

---

## 2026-06-08 — Model interchange architecture established

**Sections affected**: 13 (interchange, new), 02 (layers), 00 (index).
**Phase / PR**: design pass — no code.
**Bumped to**: 13 established at v0.1, refined same-session to v0.2 (StableHLO promoted to import + export; *representation ≠ op* disposition framing); 02 v0.2 → v0.3; 00 index updated (new section 13 row + cross-link map).

**What changed**: established the model import/export architecture as new section [13-interchange](13-interchange.md). The core commitments: (1) external formats decompose along two *independent* axes — weight payload and graph payload — so interchange splits into weight interchange and graph interchange, reused at different rates; (2) the base map ([03-ir](03-ir.md)) is the single hub — fuel's primitive `Op` vocabulary *is* the interchange vocabulary, no second neutral IR; (3) the native on-disk graph format is **not new** — it reuses [11-persistence](11-persistence.md)'s base-map serialization and DAG-format-version machinery, shipped standalone with weights external; (4) crate structure is three core+leaf tiers — format (`fuel-formats` + IR-free `fuel-format-*`), interchange (`fuel-interchange-weights` + `fuel-interchange-graph` cores + per-format `fuel-format-interchange-*` binding leaves), model (`fuel-model-core` registry + `fuel-model-*`) — with the per-format node↔weight binding as the only format-local glue; (5) the model registry uses link-time distributed registration (`inventory`/`linkme`); (6) high-demand models are split into their own crates now, the long tail extracted lazily; (7) host-language source is ingested by *tracing* (which collapses into the graph-import path), while a dev-time `fuel-codegen` scaffolder emits draft parametric `fuel-model-*` crates from source AST, sharing the graph-interchange op-map.

**Why**: the user wants fuel to read/write models from/to as many external formats as practical, and to stop hand-coding each new architecture from scratch. A design dialogue established that the popular "four distribution categories" framing (HF source / ONNX-IR / TorchScript-GGUF / JIT-codegen) conflates weight+tag formats with graph formats and miscategorizes JIT/codegen (which fuel *is*, not ingests). Re-deriving the taxonomy around the weight⊥graph axes — and grounding it in fuel's existing base-map and persistence commitments — produced a structure where most of the "weight side" is already built (`fuel-formats` + the architecture zoo) and the genuinely new work is the graph op-mapper plus a registry.

**Alternatives considered**:

- *One fused weight+graph interchange crate.* Rejected — it couples the IR-free format tier to `fuel-graph` and breaks the "safetensors-only consumer pulls almost nothing" guarantee. The parse/map seam keeps the format tier IR-free.
- *A second neutral IR for interchange to map into.* Rejected — the base map already is the hub; a second intermediate adds a translation hop and a vocabulary to maintain for no gain.
- *A separate native DAG distribution format.* Rejected — duplicates [11-persistence](11-persistence.md)'s base-map serialization; the base map is already hardware-independent and shippable, so the interchange format reuses it.
- *Big-bang split of all ~65 existing models into per-model crates up front.* Rejected against the [02-layers stopping rule](02-layers.md#stopping-rule-for-new-crates) as a speculative split; high-demand models are pre-split (near-certain consumers), the tail stays lazy.
- *Static-AST parsing of host-language source as the import mechanism.* Rejected — a model's compute graph is a runtime property (depth/branches resolve from config at instantiation), so static parsing yields a scaffold, not a correct graph. Tracing is the correct automatic path and it *is* the graph-import path; static AST is reserved for the scaffolder's draft output.
- *TorchScript / TensorRT `.plan` as interchange targets.* Rejected — TorchScript is deprecated upstream and unparseable outside libtorch; `.plan` is a kernel-baked non-portable engine. Import the source (ONNX) instead.

**Implications going forward**:

- The prerequisite structure is the **tier seam + registry** (the interchange cores, the IR-free format tier, `fuel-model-core` with `inventory`), justified now because the importer and scaffolder are real consumers. The per-model crate explosion rides the scaffolder and lazy extraction, *after* one importer validates the seam — it is not a precondition.
- ONNX is the flagship import+export target; `.pt2` (Core-ATen) and **StableHLO (import + export via MLIR FFI — the JAX/TF/XLA convergence point, and the only clean path to JAX)** follow; weight formats reuse the existing `fuel-formats` parsers plus the model registry.
- The interchange importer follows a **disposition model** (*representation ≠ op*): a source op maps to a primitive, a decomposition, a fused op, an **import-time lowering** (control flow → predication/unrolling, dynamic shape → specialization), or **another Fuel layer** (multi-output bundles, scheduler, weight-interchange quant). Only constructs with no graph representation (unbounded data-dependent side-effecting loops, unknown `custom_call`) hard-reject. The worked example + Fuel completeness audit is `docs/interchange/stablehlo-to-fuel-op-map.md` (119 StableHLO ops; ≈100 covered/handled; gaps = sort/top-k, pooling, FFT, inverse-trig, product-reduce — add only under real consumer pressure). The `L` import-lowering toolkit is shared across ONNX / ATen / StableHLO importers.
- The former `fuel-onnx` IO-layer placeholder ([02-layers](02-layers.md)) resolves into `fuel-format-onnx` (parse) + `fuel-format-interchange-onnx` (map).
- Implementation sequencing, format dossiers, Rust-crate landscape, and caller-migration tranches live in the migration plan (`docs/session-prompts/model-interchange-import-export-plan.md`), per the set's phase-doc convention.

**Related artifacts**: [13-interchange](13-interchange.md) (new section); `docs/session-prompts/model-interchange-import-export-plan.md` (migration plan); this session's [02-layers](02-layers.md) v0.3 revision and [00-index](00-index.md) edits.

---

## 2026-06-08 — Runtime-snapshot persistence artifact (L3); save-all-activations rejected

**Sections affected**: 11 (persistence), 13 (interchange — one-line cross-link).
**Phase / PR**: design pass — no code.
**Bumped to**: 11 v1.0 → v1.1.

**What changed**: added a third, optional persistence artifact — the **runtime snapshot** — capturing designated durable runtime state (KV-caches, optimizer state, producer-marked long-lived intermediates) so a process can *resume* a live computation. Framed the full save surface as three layers: **L1 model** (base map + weights; the native `.fuel` artifact), **L2 + plan** (the optimization cache; hot-load by skipping optimization), **L3 + snapshot** (resume live state). "Save with vs without the plan/runtime state" is *which sibling artifacts a caller writes*, not a flag inside a monolithic file.

**Why**: a user requirement to save "everything in the graph including in-flight data," with an explicit question of whether saving *all activations* would make hot-load launch faster. Analysis: it would not. Input-dependent activations are invalid across launches (a new input can't reuse them; if the input is identical, cache the output). Reloading large activations is bandwidth-bound at every disk→host→device hop while recompute stays on-device — the same trade that makes gradient checkpointing recompute rather than store. The real launch-speed levers (mmap weights, plan cache, lazy `KernelRef`) already live in L1+L2. So the snapshot persists *designated durable state*, not the executor's full realized-node cache.

**Alternatives considered**:

- *Save every realized activation for fastest hot-load.* Rejected — no launch-speed gain (bandwidth-bound reload ≥ on-device recompute; input-dependent activations unreusable) and large disk cost.
- *Merge model + plan + snapshot into one file.* Rejected — divergent lifecycles (weights shared everywhere, plan hardware-dependent, snapshot run-dependent); merging forces re-shipping weights when the plan changes and over-invalidates. Sibling files, per the existing 11-persistence decision.
- *Drop the snapshot concept (leave durable state to app code).* Rejected — KV-cache / optimizer-state checkpointing is a real serving/training need; a defined artifact + invalidation is worth the architecture line.

**Implications going forward**:

- The genuinely launch-relevant precompute case (input-independent derived values not already constant-folded — e.g. dequantized weights) is an optional **derived-weights** variant of the *model* artifact, not an activation snapshot.
- F2 (Serde on Fuel-IR) implements L1 first; L2 is the existing cache design; L3 lands when a resumable-state consumer (serving KV-cache persistence or a training checkpoint) needs it.

**Related artifacts**: [11-persistence §Runtime snapshots](11-persistence.md#runtime-snapshots-resuming-designated-durable-state-l3); [13-interchange](13-interchange.md); `docs/session-prompts/model-interchange-import-export-plan.md`.

---

## 2026-06-13 — Baseline Judge data is bundled in-package (supersedes opt-in download for the baseline path)

**Sections affected**: 06 (runtime — Local Judge baseline initialization), 08 (pattern-harvest — telemetry posture, clarification only).
**Phase / PR**: design decision — no code yet.
**Bumped to**: 06 (minor, when the section is next revised); recorded here ahead of the section edit.

**What changed**: the original plan (decision 20, "Local Judge baseline initialization") had a fresh install *optionally download* a community-aggregated baseline profile for its hardware fingerprint. This reverses that for the baseline path: **a baseline Judge dataset is bundled in the Fuel package itself** so a fresh install starts with empirical priors with no network access required. Local profiling + the new online/idle-time measurement path (see ROADMAP §"Online Judge cost feedback" / the expected-vs-real dispatch check) still refine it on the user's exact hardware over time. Telemetry **upload** stays strictly opt-in (unchanged); this decision is about what ships *down* in the box, not what flows *up*.

**Why**: Fuel should "just work" for almost all users, including those with no/limited internet access (rare in the U.S., common in places like Nigeria). An opt-in *download* of baseline data would make a Fuel-based program effectively unusable offline (the cold-start cost model would be static-only). Bundling the data removes that dependency. The user explicitly accepted the change from the earlier opt-in-download plan to achieve broad usability.

**Relationship to the "not bundled hardware-specific cache distribution" non-goal (09)**: no conflict. That non-goal rejects auto-distributing the *optimization cache* (a plan keyed to an exact hardware fingerprint, where a mismatch is silently wrong). Baseline *Judge data* is different: it is empirical priors across hardware classes used only to *seed* the local Judge before local measurement refines it — a starting point, not an authoritative hardware-locked plan. It degrades gracefully (a near-miss hardware class is still a better prior than static-only) rather than being silently wrong.

**Open sub-questions** (for when this is built): which hardware classes to ship, the size/compression budget of the bundled dataset, and whether it is a default feature that can be disabled for minimal builds. The design-discussion conclusion that online/idle-time measurement may make generic baseline data "largely unnecessary" stands as a reason to keep the bundled set modest — it accelerates cold start, it is not the long-term source of truth.

**Related artifacts**: ROADMAP §"Post-wipe resume addendum" follow-ups (online Judge feedback; expected-vs-real dispatch check); [06-runtime](06-runtime.md) §Local Judge baseline initialization (to be revised when implemented).

---

## 2026-06-13 — Lifecycle overview section (14) + terminology reconciliation

**Sections affected**: new 14 (lifecycle); 00 (index — added to the set table + reading-order spine).
**Phase / PR**: doc — no code (one code rename proposed, not yet applied).
**Bumped to**: new section 14 v0.1.

**What changed**: added [14-lifecycle](14-lifecycle.md), the first document that narrates the whole path end to end (load → build graph → plan → realize → inference/training loop) with a single canonical glossary. It is the orientation/spine doc; 00-index's reading order now points to it first. The set was previously decomposed strictly by concern, so no document walked the full flow — which let terminology drift.

**Why**: a working session surfaced a concrete misunderstanding (an implementation detail, the realize-internal worker thread, was mistaken for a pipeline stage). Aligning users/developers/agents needs one shared, code-accurate narrative + fixed vocabulary.

**Terminology reconciled** (the overview is canonical):

- The realize-internal worker that lowers planned nodes to WorkItems — code symbol `compiler_thread_body`, previously called the "compiler thread" — is renamed **"work-item producer"** (it produces `WorkItem`s; it does not compile machine code and is not a pipeline stage). The code rename of `compiler_thread_body` → `work_item_producer` is approved but not yet applied. `compile_plan` keeps its name, leaving exactly one "compile" in the vocabulary.
- "the plan" is canonical for the `ExecutionPlan` (a.k.a. "optimized form / optimized DAG").
- "runtime selector (Picker 2)" is canonical for the surface 06 calls the "route picker" and older code calls the "Router"; "plan-time ranker (Picker 1)" is the `compile_plan` ranking pass.

**Today-vs-intended captured in 14** (so the overview isn't read as fully-built): eager-copy load (not mmap-resident); no load-time planning wired (warm runs per-forward); decode re-plans per token (Stage 5 memoization unbuilt); binding-table lookup happens in the work-item producer at realize time, not at plan-build time.

**Related artifacts**: [14-lifecycle](14-lifecycle.md); [00-index](00-index.md) (table + reading order); the work-item-producer rename is queued against `fuel-dispatch/src/pipelined.rs:734`.

---

## 2026-06-14 — Major redirection: "the plan IS the graph" (multi-path optimization)

**Sections affected (to be revised to match this entry)**: 03 (ir), 04 (optimization), 06 (runtime), 09 (non-goals), 11 (persistence), 13/02 (load vs import), 14 (lifecycle). This entry is the AGREED anchor; the section rewrites implement it. Until a section is revised, this entry wins.
**Phase / PR**: design redirection from an extended owner design review (the 14-lifecycle doc surfaced that the built code had strayed from the intended architecture). Validated by a standalone frontier-pruning prototype (`C:/Projects/frontier-prototype`, not in the workspace).
**Status of code**: most of the built optimizer/executor (per-node `AlternativeSet` + separate `ExecutionPlan` + per-node resolve + per-WorkItem dispatch) is a *staging post*, not the destination. Migration is staged so the system stays runnable throughout.

**The core change**: the optimized graph and "the plan" are ONE structure, not two. There is no separate `ExecutionPlan` object; the graph itself, after optimization, is a **multi-path structure** carrying alternative routes that diverge and reconverge, with decisions localized to **branch points**, not every node.

**The agreed decisions** (each reverses or extends prior architecture where noted):

1. **The plan is the graph.** Optimization annotates/transforms the graph in place into a multi-path structure; the surviving N-best routes ARE the optimized model. (Revises 03/04's three-artifact "optimized form as separate annotation" framing toward a single evolving structure; base map = the unoptimized graph, retained as the portable artifact.)
2. **Multi-path, decisions at branches.** Alternatives are real divergent/reconverging paths; decision points = branch points (few), not per-node. The built "every kernel-bearing node gets an AlternativeSet, resolved per node" over-generalized "decision point" and is the drift to undo.
3. **The graph is the model, input-independent.** Built at load/import — NOT at `forward()`. Fully built (ideally optimized) before the first input. "No graph until forward runs" is a defect. Nodes need not own storage to exist.
4. **Load vs import.** `map_from_file()` (or similar) LOADS a finalized native Fuel graph+storage (mmap'd). `from_gguf()`/`from_safetensors()`/HF etc. are IMPORTS (conversion → base map). The native `.fuel` format (L1) is the load path; everything else converts.
5. **Finalize-to-disk is the default run mode, not a precondition.** Any in-memory model can be finalized to an mmap'd file — yielding fast reload, snapshotting, training-persistence, AND larger-than-RAM in one mechanism — but the base map runs before finalize. Write-back at explicit checkpoints (`msync`), never write-through-per-step.
6. **Storage classes via "session index."** A node's lifetime/sharing class is inferred from its op (`Op::Const` → shared/weights; cache-write target → session-state with explicit override; else → transient), with multiple storages per node indexed by **session** for session-state. Weights are shared across sessions (not session-indexed). Transient never persists and never crosses to the file, but DOES cross devices mid-realize (D2D) on multi-device paths. Shared written-back only if changed (training); session written-back only on snapshot.
7. **`optimize_graph` = pathfinders + rankers + optimizers, run lock-step.** Pathfinders add candidate paths (algebraic reshaping; dependency tracking for parallelism; windowed scanner over fused + primitive ops). Rankers measure each path: **timing** (the Judge — see #8), **precision** (digits), **accuracy** (ULP/rounding/monotonicity), **memory** (per-tier vector, see #10). Optimizers merge/discard sub-optimal paths (per-op tolerance; duplicate-path convergence; path-timing) and never strand the last path for a (device,backend) nor touch a path in an active inference/training cycle. Tie-break order: precision → accuracy → memory.
8. **Bounding the path explosion (prototype-validated).** Naive per-device Pareto pruning EXPLODES (frontier passes 3000 by region 14 of 128). The bound comes from two complementary things: **(a) the right axes** — ONE central time metric (median/average for throughput, p99 for latency; **drop `t_min` as a selection axis** — "fastest best-case" is not a selection goal; the Judge measures the full distribution but the optimizer optimizes on one mode-selected metric), memory as discrete **tiers**, precision/accuracy as discrete levels → a naturally small (~100), FLAT, **lossless** frontier with no cap; and **(b) a crowding-distance-capped beam** (NSGA-II style, per ending-device) as a hard backstop — keep≈32/device matches the no-cap optimum on all tested runtime queries, and bounds even adversarial continuous-axis cases. Per-device bucketing keeps multiple paths per device (time/memory tradeoffs; multi-device-spanning paths) and never strands a device.
9. **Runtime picks within the surviving frontier by live telemetry** (device load, free memory tier) at branch points — this is competitive edge #3 (top-N route preservation) realized structurally. Kernel-variant choice is largely baked at optimize time; device/path choice adapts at runtime.
10. **Three-tier memory (disk / host / device) tracked as a vector, not a scalar.** A path has separate host-RAM and device-VRAM (and disk) footprints; dominance and runtime selection are per-tier (which tier is the binding constraint decides which path wins). The plan IS the prefetch schedule across tiers: disk→RAM (mmap demand-paging + `madvise`) for larger-than-RAM, RAM→VRAM (H2D ahead of frontier) for larger-than-VRAM.
11. **`Storage` must support mmap-backed zero-copy views** (paged), not only owned buffers — prerequisite for larger-than-RAM and the native format. Today's eager-copy load defeats this and is to be replaced.
12. **Pre-resolving kernels is an optional post-optimize step** (startup-time vs TTFT trade): resolve all `KernelRef`s up front to take the lookup off the hot path, or resolve lazily for fastest first-token.
13. **Terminology** (per [14-lifecycle](14-lifecycle.md)): the realize-internal worker is the **work-item producer** (code rename of `compiler_thread_body` pending); "the plan" = the optimized graph; "runtime selector (Picker 2)" = the route picker; `compile_plan` keeps its name as the one "compile."

**Why** (the bet, restated): a single multi-path graph optimized offline by pathfinders/rankers/optimizers, dispatched as sequences between few decision points with runtime adaptation per request, is the substrate the five competitive edges actually need — not a per-node alternative side-table resolved one op at a time (which the owner correctly judged "almost eager with a queue"). The prototype shows the feared blowup is avoidable and cheap (~100 paths) with the right axes + a backstop cap.

**Prototype evidence**: `C:/Projects/frontier-prototype` — naive Pareto explodes @region 14 (all seeds); coarse quantization / collapsing min/max only delay it; recommended axes (1 central time + 3 memory tiers + discrete precision/accuracy) → flat ~100-path lossless frontier; capped beam keep=32/device → matches no-cap optimum on all 5 runtime queries; adversarial backstop holds.

**Open items** (to settle during the section rewrites): the exact `keep` to standardize; how a node declares its storage class (inference-from-op + explicit override is the lean); the disk-spill traversal's locality requirement for `optimize_graph`; and the staged-migration sequence against the runnable system.

---

## See also

- [00-index §Versioning convention](00-index.md#versioning-convention) — when to bump section versions.
- ROADMAP.md — phase-level work tracking.
- `docs/architecture-audit.md` — the cross-thread audit that triggered the v0.1 architecture-set drafting.
