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

**Related artifacts**: the cross-thread architectural audit that triggered this consolidation (doc `docs/architecture-audit.md`, removed 2026-06-20 as superseded — its Q-A/Q-B/Q-C/Q-E resolutions are folded into the sections + this entry; Q-D, where cross-cutting types live after the fuel-core fission, remains an open question intrinsic to that future phase); session memory entry `project_architecture_doc_set_v0_2.md` (initial drafting state); `project_phase_7_6_paused_for_audit.md` (the in-flight phase work that paused for the audit).

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

## 2026-06-20 — Adaptive runtime fusion: the recipe principle, two-tier extensibility, and the Fuel-strategist / backend-synthesizer JIT loop

**Sections affected (revised to match this entry)**: 03 (ir), 04 (optimization), 05 (backend-contract), 08 (pattern-harvest), 09 (non-goals), 14 (lifecycle) — MAJOR; 01 (identity), 02 (layers), 06 (runtime), 11 (persistence), 12 (multi-output), 00 (index) — MINOR. This entry is the AGREED anchor; the section edits implement it. Until a section is revised, this entry wins.
**Phase / PR**: design pass from an extended owner design review on the kernel-boundary program (branch `feat/kernel-contracts-dlpack`). No code yet; this sets the destination the FKC declarative-fusion spec (`docs/specs/fkc-fusion-patterns.md`) and the telemetry plan (`docs/session-prompts/baracuda-telemetry-plan.md` §9) implement.
**Bumped to**: 03 v0.4 → v0.5, 04 v0.5 → v0.6, 05 v0.4 → v0.5, 08 v0.2 → v0.3, 09 v0.2 → v0.3, 14 v0.5 → v0.6; 01 v0.2 → v0.3, 02 v0.4 → v0.5, 06 v1.2 → v1.3, 11 v1.2 → v1.3, 12 v1.0 → v1.1, 00 v1.0 → v1.1.

**The core change**: Fuel acquires an **adaptive runtime-fusion loop** — it detects fusion opportunities a model author never wrote, asks a backend (Baracuda first) to JIT-synthesize a kernel for them during idle time, and adopts the result cost-guided — *without* surrendering the constitution ("the optimizer that reads the DAG is where the intelligence lives; backends advertise but never make strategic decisions"). Making that safe required pinning down the **recipe principle** and a **two-tier extensibility model**, and reconciling several blanket "frozen at startup" claims that predate the loop. Eight decisions:

1. **The recipe principle (G1).** Every fused op carries a primitive recipe in **two inverse directions**: a `decompose` (fused → primitive subgraph; *lowers* it onto the base map) and a `pattern` (recognize that primitive subgraph; *re-fuse*). **Both are mandatory.** A fused op with no recipe is an **opaque island** — invisible to base-map analysis (the co-occurrence / missing-fusion telemetry can't see across or inside it) and impossible to re-fuse. The two halves are the same data viewed in opposite directions (the DecompositionMap / OptimizationMap derive one-to-one from the registry, per [04-optimization](04-optimization.md)).

2. **`decompose` is TOTAL + never-panic + primitive→self (G2).** `decompose` never `panic!`s (the never-panic constitution rule). A **primitive decomposes to itself** — the recursion's fixpoint, already the identity form `decompose = |_g, id, _p| id` at `fuel-graph/src/registry.rs:823`. The **base map is the fixpoint of `decompose` over every node** (lower until `decompose(x) == x`; a primitive is just a node no lowering rule fires on — exactly the `optimize_graph` rewrite model). A panicking `decompose` is **always a bug**: the op is either a true primitive (must return self) or a non-primitive missing its recipe (a bug / basis gap), distinguished by **basis membership, never by the return value**; a non-basis op that fails to decompose is a **surfaced opaque-op gap** (a base-map flag → the missing-fusion / inventory telemetry), never a crash and never silently masquerading as primitive. This is **load-bearing for optimization *itself***, not just JIT: optimization = lower-to-base-map + find-best-cover, so an op that won't decompose *breaks the optimizer*, not merely a downstream feature. The three current panicking decomposes (`nf4_matmul.rs:120`, `flash_attn`, `selective_scan`) are bugs to fix, not a permanent category. A model's recipe (`decompose` + `pattern`) **always ships with the fused op** — never deferred "until intermediates fit" (that produces an opaque island).

3. **The primitive basis is build-time-closed (G3).** The primitive `Op` set is a compile-time Rust enum (`fuel-graph/src/lib.rs`); there is **no generic opaque / `Custom` node** in the lazy graph. So an externally-supplied (provider / JIT) op **cannot become a new primitive at runtime**. It must either **decompose into the existing primitive basis** (and carry a `pattern` to replace that sequence), or — if it needs a primitive Fuel lacks (e.g. a higher-order `Scan` for SSMs) — prompt a **Fuel-side, build-time `Op`-enum extension** the provider cannot do itself. The primitive vocabulary is a **hard shared contract** with providers. JIT adds kernels / recipes *over* existing primitives, never new primitives.

4. **Two-tier runtime extensibility (G4) — the reconciliation of the "frozen at startup" claims.** The blanket non-goals ("fused-op registry / OptimizationMap populated at startup, frozen thereafter; no runtime extensibility") were each motivated by an **untrusted-user** security / stability surface. They are re-scoped into a three-way split:
   - **Build-time-closed (stays):** the primitive `Op` enum (per G3) and **untrusted user-installable rules** (the [09-non-goals](09-non-goals.md) rejection holds — arbitrary user code in the optimizer is still out).
   - **Tier 1 — already runtime-extensible:** the **kernel binding table** (implementations). `extend_global_bindings` (`fuel-dispatch/src/dispatch.rs:5098`) write-locks `OnceLock<RwLock<KernelBindingTable>>`, appends (append-only, multi-sibling `SmallVec`), re-runs `finalize()`, and calls `bump_topology_generation()` to invalidate cached routes. JIT-ing a kernel for an **existing op identity** lands here today; this was never the frozen part.
   - **Tier 2 — the new goal:** trusted, **Fuel-orchestrated, cost-gated** runtime registration of a **new fused-op identity** (the fused-op *metadata* registry — `OnceLock<FusedOpRegistry>`, dense Vec, closed `FusedOpParams` enum, bare-`fn`-pointer entries — becomes runtime-updatable: append-only, **stable never-reused** `FusedOpId`s). The **mechanism is the declarative form**: a runtime fusion can *only* be declarative (pattern + recipe + shape/dtype/cost carried as **data**, run by generic interpreter fns), because you cannot ship Rust `fn` pointers or add enum variants at runtime. So implementing the **stubbed declarative pattern engine** (`PatternKind::Declarative => false` at `fuel-graph/src/opt.rs:434`) is the **prerequisite** for Tier 2 — one generic `FusedOpParams::Declarative(Arc<Recipe>)` variant + generic interpreters + the append-only registry.

5. **Missing-fusion telemetry (G5).** Today there is **no** missing-fusion signal: Fuel can see fusions it *performed*, not fusions it *wanted-but-lacked* ("no rule fired" is identical across every primitive node, `opt.rs:256-273`). It needs a **new graph-layer hook** and depends on the base emission seam (`structure_key` is a 13-line stub; no `DispatchRecord` is emitted yet). Sequencing (canonical in [08-pattern-harvest](08-pattern-harvest.md) + `baracuda-telemetry-plan.md` §9): **closed-world `FusionMissRecord`** (a recognized fusion-eligible chain realized as N primitives because the kernel was absent — reason `NoBackendKernel`, against **known** `FusedOpId`s) is the v1 **headline**, built FIRST (its consumer — append a `BindingEntry` — already exists, Tier 1). **Open-world `SequenceRecord{fused_as: None}`** (a frequent realized op chain matching **no** known identity — discovered by *observation*, not subgraph *enumeration*) is **deferred**, because its consumer is the Tier-2 runtime declarative registration. We never enumerate the subgraph space and never search for a whole-model fusion.

6. **Whole-model / megakernel is a real but NARROW technique (G6).** Supported when profiling justifies, but the **last / highest-risk target, never the default**. It wins *something* over even an ideal CUDA Graph (inter-kernel GPU scheduling bubbles + cross-boundary software pipelining) and removes host-round-trip risk, and *can* host multiple strategies via a persistent-megakernel. But the benefit curve **turns over** because of **fixed launch geometry** (one grid/block per launch), **kernel-global register allocation** (the worst region's register footprint is imposed on every region → bandwidth-bound regions are forced to the compute region's low occupancy; internal sub-kernels do **not** fix this — the register file is partitioned at launch by the kernel's static max), and **per-shape JIT/specialization combinatorics**. "Bigger fusion = better" is **not monotonic**. Captured-run replay (CUDA Graphs / pre-recorded command buffers, per [11-persistence](11-persistence.md)) is the cheaper step below it that already captures most of the launch-overhead win without fusing compute.

7. **The closed-loop adaptive optimizer (G7).** The synthesis. The **base map** (guaranteed total by G2) is the substrate both the optimizer and the JIT loop read. **Fuel is the STRATEGIST**: it chooses *which* sub-base-map regions to request (sending **partial** base maps, never the whole map), controls *when* (idle-time, host- and all-device resource-aware — Fuel is the only layer that sees the whole machine), and makes the cost-guided **adopt / reject** call. **A backend (Baracuda) is the SYNTHESIZER**: it builds the best kernel for a Fuel-*chosen* region, applying hardware knowledge *within* it — Fuel choosing the region **is** the fusion decision; **no backend-side opportunity-finding** (the constitution holds; this is not backend-internal fusion, per [09-non-goals](09-non-goals.md)). The loop is **explore/exploit**: **co-occurrence telemetry** (frequency-counted realized chains) is the **exploration prior** that orders which regions to JIT first; empirical **"winning"** (a kernel/path entering an optimized plan under cost-guided selection) is the **exploit posterior** — ground-truth fitness — and **win-rate flattening is the STOP signal** (the benefit-curve knee) that bounds JIT requests. Neither replaces the other.

8. **Kernel-cache pruning (G8).** Prune **rarely** — likely not at all initially. When forced, evict kernels that **lose across *every* model** first ("loses no matter which model considers it" = a never-useful proof). Gate eviction behind a **developer-set max-kernel-drive-space cap** — only evict under space pressure, so a kernel is never discarded while there is room (a currently-*shadowed* kernel that might win under a different cover, or in a different model, stays). Do **not** prune on a single model's losses — "winning" is relative to the current kernel set (shadowing). Canonical home: [11-persistence](11-persistence.md).

**Why**: the kernel-boundary program (FDX + FKC) opened the door to backends that JIT-generate kernels for Fuel's specific needs. The natural consumer of the missing-fusion telemetry is exactly that loop. But the loop is only safe if (a) every op has a recipe so the base map is total (else the optimizer has blind islands), (b) the basis stays build-time-closed so providers can't smuggle in primitives, (c) the extensibility is split so the *trusted Fuel-orchestrated* path is enabled without opening the *untrusted user* path, and (d) Fuel keeps the strategy while the backend keeps the synthesis. The prior docs' blanket "frozen" language conflated trusted/untrusted and primitive/fused-metadata, blocking the loop on paper; this entry separates them.

**Alternatives considered, then rejected**:
- *Let the backend find fusion opportunities and JIT them autonomously.* Rejected — that migrates strategy into the backend, violating the constitution; Baracuda synthesizes a Fuel-chosen region instead.
- *Send Baracuda the whole base map.* Rejected — wasteful and a strategy leak; Fuel sends partial, optimizable sub-regions, on Fuel's resource-aware schedule.
- *Greedy "largest region that fits first."* Rejected — that optimizes launch-count (the least-valuable axis) and is the lowest-frequency, highest-compile-cost, highest-fit-risk target; order by frequency-weighted profitability (small repeated motifs first), with empirical winning as the ground-truth re-prioritizer.
- *Treat a foreign black-box kernel as a new "primitive by declaration."* Rejected — there is no runtime primitive registration and no opaque node; a provider op must decompose into the existing basis or prompt a build-time basis extension.

**Implications going forward**: (1) the **stubbed declarative pattern engine** is promoted from a Phase 7.6 convenience to the **prerequisite for the JIT loop** — it gates Tier-2. (2) The **total/never-panic `decompose`** standardization (fixing the three panicking decomposes + a typed contract) is promoted from the never-panic backlog to a **correctness requirement for the optimizer**. (3) The **closed-world `FusionMissRecord`** is the build-first telemetry piece; open-world co-occurrence is gated behind Tier-2. (4) Backends gain a **JIT-on-request** contract surface (Fuel sends a partial base map + budget; the backend returns a kernel + its FKC contract incl. `PrecisionGuarantee`); the route picker cost-gates adoption and the binding table ingests it. (5) Kernel-cache growth gains a **drive-space-capped, loses-everywhere** eviction policy.

---

## 2026-07-01 — FKC cost unification: register GPU capabilities + per-backend throughput (honest cross-device placement)

**Sections affected**: 04
**Phase / PR**: FKC cost unification Part A (`1acf3222`) + Part C (this change)
**Bumped to**: 04 v0.6 → v0.7

**What changed**: Two coupled fixes to the Layer-1 cost model that together make cross-device placement honest. **Part A** registers GPU `BackendCapabilities` (derived from the binding table) and the per-op cost functions for GPU backends, so `compute_static_costs` no longer *skips* an uncapped GPU candidate and leave it priced at zero. **Part C** replaces the backend-agnostic `1 FLOP ≈ 1 ns` prior in the composite-nanosecond figure with a **roofline over each candidate backend's throughput**: `composite_ns ≈ max(flops ÷ compute_throughput, bytes ÷ mem_bandwidth) + kernel_overhead_ns`, with `compute_throughput_flops_per_ns` / `mem_bandwidth_bytes_per_ns` added to `BackendCapabilities`. The placement DP reads the authoritative registered figure; the candidate rank (no caps in hand) uses a matching per-backend prior derived from the same constants. `kernel_overhead_ns` is never scaled, so Layer-2's measured latency (packed into overhead) is unaffected.

**Why**: A concrete crash — a CPU-pinned realize spilled onto an *unseeded* GPU because the GPU candidate was priced at zero (Part A's bug) and, once priced, was still modeled at CPU throughput so it could never legitimately out-price the CPU (the gap Part C closes). The root cause was that "backends advertise capabilities/costs" (§constitution) wasn't actually wired into the FLOP→time conversion. This realizes the throughput refinement earlier drafts deferred to "Phase 1.5."

**Alternatives considered**: *A reachability filter that prunes unseeded devices from placement.* Rejected — it hides the mis-pricing rather than fixing it, and forecloses seed-on-demand. *Storing the throughput on each `Candidate` / `CostEstimate`.* Rejected — 40+ field-explicit construction sites, and `bytes_moved` is also read raw for the VRAM tiers; resolving the rate from `candidate.backend` at the composite site is cleaner. *Calibrated per-device numbers.* Deferred — the priors are deliberately directionally-correct only; Layer 2 (Judge) supplies calibration.

**Implications going forward**: (1) `BackendCapabilities` is now the single authoritative home for per-backend throughput; a real backend or the Judge can refine it per cell. (2) The composite figure is throughput-aware everywhere it is consumed (placement DP, candidate rank, dispatch-time selectors). (3) Part B (route *every* kernel's cost through its FKC contract so the FLOP/byte counts themselves are contract-sourced) is the remaining leg of the cost-unification program.

---

## 2026-07-02 — Fused-op Layer-1 cost composed from its decomposition (no zero sentinel)

**Sections affected**: 04
**Phase / PR**: Fused-op cost-from-decompose (this change); follows FKC cost unification Part A/C (`1acf3222` / `0e178b9b`)
**Bumped to**: 04 v0.7 → v0.8

**What changed**: A fused / synthesized op with no declared or measured cost registered a zero-cost sentinel (`fused_unknown_cost` for FKC-imported ops; the runtime-adopted path likewise), and a zero-priced candidate wins spuriously — the exact mis-pricing Part A fixed for GPUs, now recurring for every fused op that lacks a Judge cell. This adds a **Layer-1 default that composes the fused op's cost from its own `decompose`**: `flops = Σ` over the decompose subgraph's primitive nodes of their per-node Layer-1 FLOPs (arithmetically exact for algebraic fusions); `bytes_moved =` the fused op's own **boundary I/O** (declared inputs + final output — fusion elides the intermediates, so the *tight* estimate, not `Σ` intermediate bytes); `kernel_overhead_ns =` **one** launch overhead (the fused kernel launches once). It is an **optimizer-level default** — `fuel_dispatch::fused_cost::{cost_from_decompose, fused_layer1_cost}`, computed where the graph + `decompose` are both in hand, NOT inside the bare `cost(shapes, params, caps)` fn pointer (which has no graph). It fires **only for the sentinel** (identity fn-pointer compare, as `fill_unset_*` already does for the primitive `unknown_cost`): a fused op WITH a declared cost or a Judge measurement is priced by that, unchanged — so the layering is **measured › declared › composed-from-recipe › (never) zero**. The runtime-adoption path (`adopt_runtime_fused`) now stamps the same sentinel so runtime/JIT-synthesized fused ops (where sentinel-zero was most likely) benefit too.

**Why**: The recipe principle (every fused op carries a total, never-panic `decompose` whose fixpoint is the primitive base map — 2026-06-20 decision) means the exact primitive-equivalent of every fused kernel is already known, and every primitive already has a Layer-1 cost fn. Composing those is a real, bounded number available immediately (no Judge data required) — the bridge "until the Judge can verify it during real runs." It is directionally safe under carry-forward placement: for algorithm-changing fusions (flash-attention trades recompute for O(n) memory) the decompose FLOPs are an approximation the Judge later refines *downward* — a missed win until measured, not an optimistic *irreversible* bad commit.

**Alternatives considered**: *Populate a real cost fn at registration that closes over the decompose (spec shape B).* Rejected — the `cost:` field is a bare `fn` pointer that can't capture the decomposition map; it would need the same signature-widening the deferred Task-F trampoline needs. The use-site sentinel fallback (shape A) is zero registry churn. *A declared `cost:` expression as the default (Task F).* Kept as an optional Layer-1 *override* for authors who want to beat the composition's precision — this decision demotes it from prerequisite to refinement. *Loose `Σ decompose.bytes` upper bound for `bytes_moved`.* Available as a conservative fallback; the tight boundary-I/O estimate is used because every input is already in hand and it is closer to the true fused value.

**Implications going forward**: (1) Every fused op — FKC-imported, static-registered, or runtime-adopted — now has a nonzero, directionally-safe Layer-1 cost the moment it exists; the zero-cost landmine is closed across the fused surface. (2) Task F (contract-declared `cost:` trampoline) is now an *optional* precision refinement, not a prerequisite. (3) A param-carrying interior primitive in a decompose prices at its shape-derivable floor under the v1 fold (`OpParams::None`); the Judge is the corrector, and a full `op_to_op_params` reconstruction is a possible follow-up if a consumer needs tighter interior pricing. (4) The fused-registry cost read is not yet consumed by the ranker's `compute_static_costs` (which prices primitives via the binding table); `fused_layer1_cost` is the accessor the fused-op cost consumer will call when it lands.

---

## 2026-07-03 — The three flagged `decompose`s resolved: NF4 + concrete-`k_len` flash recipes land; symbolic-`k_len` flash + SSM scan are documented basis gaps

**Sections affected**: 04 (optimization) — MINOR (status note only); no core-claim change.
**Phase / PR**: never-panic / recipe-principle backlog (G2). Branch `feat/kernel-contracts-dlpack`. Code: `fuel-graph/src/registry/{nf4_matmul,flash_attn,selective_scan}.rs`; parity + gap-posture tests in `fuel-core/src/lazy.rs`.

**What changed**: The 2026-06-20 G2 entry (and CLAUDE.md, and 04) flagged "three current panicking decomposes (`nf4_matmul.rs:120`, `flash_attn`, `selective_scan`)." **First correction: they were not, in fact, panicking** — a prior G2 pass had already converted every panic to a self-return (the never-crash posture). So the residual work was never "stop the crash"; it was "supply the *recipe* so the base map isn't stranded with an opaque island" (the load-bearing-for-the-optimizer half of G2). That work is now done, per op:

- **`nf4_matmul` → total primitive recipe.** `dequantize(w_packed, absmax) → matmul`, built with **no data-carrying `Const` and no device handle** (a `decompose` fn has neither): `Cast(U8→F32)` + `lower = wf − 16·⌊wf/16⌋` / `upper = ⌊wf/16⌋` nibble-unpack (exact for `1/16 = 2⁻⁴`), `Unsqueeze→Concat→Reshape` interleave to codes `[N, K]`, a **codebook lookup as an indicator sum** `Σᵢ LUTᵢ·relu(1−|c−i|)` (pure elementwise, exact because codes are exact small integers), broadcast per-block `absmax`, `Transpose`, `MatMul`. Parity test matches the fused CPU byte-kernel numerically.

- **`flash_attn` concrete-`k_len` → total recipe; symbolic-`k_len` → documented gap.** `k_len` is `Option<DynScalar>`. `None` (vanilla) already decomposed. **`Some(Concrete(kl))` was static all along but returned self** (the `k_len.is_some()` short-circuit was too coarse); it now `Slice`s K/V to the live prefix and runs the SDPA recipe **bottom-right-aligned** — `q_pos_offset = kl − Sq` threaded through every causal / sliding-window / ALiBi band (`recompute_probs` + `alibi_bias` gained the offset param; the two existing callers pass 0, keeping the backward byte-identical). **`Some(Sym(_))` is a genuine registry-layer basis gap**: slicing to a *symbolic* length needs a primitive the basis lacks (`Op::Slice` carries a static `usize`; nothing materializes a `DynScalar` into a length-mask tensor inside a `decompose`, which never sees the per-realize `SymEnv`). The symbolic decode *oracle* is emitted one layer up by the optimizer's `decode_flash` arm, which **does** hold the `SymEnv` — so returning self here is correct-by-design, not a punt.

- **`selective_scan` → the constitution's canonical basis gap (G3).** No recipe over today's `Op` basis is both total *and* numerically valid: the `O(seqlen)` unroll is an unbounded, un-re-fusable explosion (not a recipe), and the diagonal-SSM `CumSum` closed-form `h[t] = exp(a·D[t]) ⊙ cumsum_t(exp(−a·D[s]) ⊙ x[s])` **overflows** because Mamba's `a = −exp(a_log) < 0` makes `exp(−a·D[s]) = exp(|a|·D[s])` blow up — exactly why the kernel is chunked. Held as a never-crash surfaced gap; the precise ask to close it is a **build-time `Op`-enum extension** (a higher-order `Scan` / associative- or chunked-scan primitive), per G3.

**The signature question (why no `Result`).** G2 demands never-panic but the DecompositionMap signature `fn(&mut Graph, NodeId, &FusedOpParams) -> NodeId` cannot fail. Resolution, consistent across all three: **returning self *is* the typed gap-marker.** The fixpoint driver (`opt.rs` `run_pass` / `LoweringRule::rewrite`) already treats `new_id == id` as "no progress" and records no remap, so the lowering loop **terminates** and the node is left as `Op::Fused` — a surfaced opaque-op gap an inventory pass can find. A primitive→self and a gap→self are byte-identical to the driver and distinguished only by **basis membership** (exactly G2's rule), so **no signature widening is needed** and none was done. The gap-posture test asserts termination + node-survival empirically.

**Why (not force a lowering)**: the recipe is the *math oracle* + the substrate the optimizer covers; a wrong lowering (numerically-unstable cumsum) or an unbounded one (seqlen unroll) is worse than an honest surfaced gap — "partial-with-precise-gaps is valid; never force a wrong lowering."

**Alternatives considered**: *NF4 codebook via `Const`-tensor + `IndexSelect`* — rejected: a `decompose` fn has no device handle to build a data-carrying `Const`, and it adds an integer-index dtype dependency; the indicator sum needs only confirmed elementwise primitives and is provably exact. *Generalize `recompute_probs` vs. inline a second SDPA for the offset case* — chose the shared-function offset param (one code path, backward stays byte-identical). *Widen `decompose` to `Result`* — rejected: G2 makes totality a build-time invariant and self-return already expresses the gap; widening would ripple to every decompose + the fixpoint driver for no gain.

**Implications going forward**: (1) Two of the three flags clear; the remaining `decompose` gaps (`flash_attn` `Sym` `k_len`, `selective_scan`) are now *documented, tested* basis gaps with a named missing primitive each — not backlog bugs. (2) When a `DynScalar`-length `Slice`/mask primitive lands (or the `decode_flash` symbolic oracle is folded into the registry), `flash_attn`'s `Sym` path decomposes with the same offset machinery. (3) The SSM `Scan` `Op` extension (G3) unblocks `selective_scan` (and `ssd_chunk_scan`).

---

## 2026-07-04 — Frontier-readiness audit: the 2025–26 research-edge gaps cataloged (tracking note)

**Sections affected**: none — MINOR status/tracking note only; no core-claim change. This entry *records where a set of gaps is now tracked*; it decides nothing.
**Phase / PR**: documentation. Branch `feat/kernel-contracts-dlpack`. Artifact: [`docs/frontier-architecture-gaps.md`](../frontier-architecture-gaps.md); registered in `ROADMAP.md` → Deferred backlog → "Frontier-architecture gaps."

**What / why**: A six-track sweep assessed Fuel against the current ML research frontier — hybrid SSM/Transformer, Multi-head Latent Attention & QKV pruning, hyper-sparse MoE & soft routing, test-time compute (inference-scaling / search-on-generation), and GRPO / verifiable post-training — to answer "how much of the edge is Fuel already built to support," and to ensure no missing capability is forgotten. Recurring finding: Fuel typically has the *expressible* form (often the model itself — MLA is fully built as a lazy DAG; LFM2/Based hybrids port; all six MoE models run) but not the *efficiency payoff*.

**The keystone**: three payoffs — SSM autoregressive decode, MoE sparsity, MLA compressed KV cache — all gate on the **data-determined** half of symbolic extents (per-op-produced runtime counts over fixed-capacity buffers). The input-determined half is shipped (Phase D persistent decode); the data-determined half is designed-not-built ([`data-dependent-shapes-design.md`](../session-prompts/data-dependent-shapes-design.md)) and is *also* already required by Phase 8.5. Highest-leverage unlock.

**Effect on this log's own documented gaps**: the two `decompose` basis gaps from the 2026-07-03 entry above — symbolic-`k_len` `flash_attn` and the SSM `Scan` op (`selective_scan` / `ssd_chunk_scan`) — now have a **scheduled tracking home**; they were documented here but had never been entered on the path. Their posture is unchanged (never-crash surfaced gaps, one named missing primitive each). The stale "bugs to fix" phrasing in `ROADMAP.md`'s Phase 7.5 recipe-principle narrative was corrected in the same pass.

**Implications going forward**: (1) The catalog is a backlog *index*, not a plan — the ROADMAP owns sequencing, the constitution wins on posture. (2) Newly-tracked orphans (no prior planning-doc home): the higher-order `Scan` op; SSM decode init-state + GPU scan dispatch; MoE sparse per-token dispatch + balancing losses + soft-MoE; MLA decode cache + KV-container generalization + weight-absorption + two-projection attention; batched multi-sequence decode + forkable/COW KV cache; and GRPO + RLVR. (3) Test-time-compute *search orchestration* is confirmed **out of layer by design** (a Phase 9 downstream concern), not a gap to close inside Fuel — only its substrate pieces (batched decode, forkable KV) are Fuel's to build.

---

## 2026-07-04 — FKC reaches all three backends; contracts can pin a cost fn (trampoline); optimize-time kernel-variant selection reads the Judge

**Sections affected**: 04 (optimization) — MINOR (variant-bake status note); 05 (backend-contract) / the FKC spec — MINOR (the `cost.cost_fn` field). No core-claim change; all three are capability completions along existing commitments.
**Phase / PR**: FKC program + the Judge Layer-2 arc. Branch `feat/kernel-contracts-dlpack`. Code: `fuel-dispatch/src/fkc/{cuda_link,lower,register,schema}.rs`, `variant_bake.rs`, `telemetry/`, `fuel-core/src/judge/`, `fuel-ir/src/dispatch.rs`.

**What shipped (four related capability completions):**

- **FKC is now 100% across all three real backends.** CPU + Vulkan (13 families) + **CUDA (31/31 families, ~429 keys)** register from `docs/kernel-contracts/**` via the `CudaLinkRegistry` (`cuda_ep!`, the beachhead-then-sweep pattern proven on CPU/Vulkan). Every one-time deferral resolved (WriteSlice, forward-Pad, the fused registry + dtype-fan, cast-110, Vulkan FlashAttn, CUDA flash_decoding). The backends are no longer hand-registered; the contract corpus is the single source of truth for the binding table.

- **The cost-trampoline (Task-F): an FKC contract can pin a real cost fn.** `CostBlock` gains `cost_fn: Option<String>`; `LinkRegistry` gains `resolve_cost_fn(name) -> Option<CostFn>` (the exact analog of `resolve_primitive`'s symbol→`KernelRef`), so a contract names a registered cost fn and the importer stamps *that* (not the `unknown_cost` sentinel) — `fill_unset_cost_for_backend` then leaves it. This is what let `flash_decoding` migrate while preserving its custom `cost_flash_decoding_cuda` infeasibility gate (returns `flops == u64::MAX` for `seq_q != 1` / `head_dim > 128`), which an unconditional `unknown_cost` → fill_unset upgrade would have destroyed. Unresolvable names are a typed `UnknownCostFn` error, never silent. Demotes the general expression-trampoline (adoption-plan §2.3) to an optional refinement.

- **Same-device kernel-variant selection is baked at optimize time, reading the Judge.** `variant_bake.rs` collapses a same-device `Op::Branch` (e.g. the decomposed attention region vs. a CUDA flash arm) to the cheaper arm at optimize time — the runtime route picker resolves *placement* arms only (keyed on arm 0's op), so a same-device kernel-variant choice must be baked, per 04-optimization ("baked at optimize time"). The bake reads **measured** latency (`decode_arm_composite_ns_judged`, Layer-2-first + Layer-1 fallback) so a fused arm that *ties* on Layer-1 FLOPs but wins on measured latency (no materialized attention matrix, one launch) is selected; ties / unknown / capability-missing keep arm 0 (the oracle), so the no-Judge path is byte-identical. A Judge *measurement is capability evidence* (the cell exists only because the kernel ran during profiling), so a measured cell is admissible past the Layer-1 availability gate.

- **The Judge gained decode-shape Layer-2 coverage**, which the bake above consumes: f16/bf16 dtypes, decode-shaped ladders (skinny GEMV + softmax rows), and `OpKind::FlashAttn` as a profiled op (`SizeClass::attention`). This surfaced and fixed **a latent correctness bug**: `SizeClass` was an aspect-blind `log2(total_elements)` and the Judge *producer* keyed a matmul on `m·n` (output) while the ranker *consumer* keyed on `m·k` (LHS input) — square matmuls agreed by accident, **every non-square matmul in every model had an unreachable/poisoned Judge cell**. Fixed via one shared `SizeClass::matmul(m,n,k)` aspect key both sides derive (`SizeClass::for_op`), `PROFILE_REPORT_VERSION` 2→4.

**Why**: these complete the cost-unification program's original intent — a cost model that is *honest* (no zero sentinels, real per-backend throughput, measured Layer-2 where available) and *contract-sourced* (the contract, not hand-written Rust, is the source of truth for bindings **and** their costs) — across every backend Fuel actually runs.

**Alternatives considered**: *a sibling `CostRegistry`* rather than a `resolve_cost_fn` trait method — rejected: one resolution surface (the `LinkRegistry`) is cleaner and already per-backend. *A `SizeClass` enum with a `MatMul` variant* — rejected for blast radius (~120 `SizeClass(N)` literals across four crates); widening `u8→u32` + packing the aspect key kept them compiling. *Wiring the bake into the runtime picker* — rejected: the picker resolves placement only and regenerates each realize (it would clobber a baked variant); collapse-to-winner at optimize time touches the hot path not at all.

**Implications going forward**: (1) A live CUDA flash win now needs only a **bf16 CUDA decode path** (the flash kernel is f16/bf16; today's F32 decode graph gates the arm out) + a live Judge profile — the mechanism (emitter → bake → Judge) is proven end-to-end. (2) The cost-trampoline generalizes: any contract can pin a cost fn; the expression-string trampoline is now optional polish. (3) With all backends contract-sourced, a new kernel's *entire* dispatch surface (binding, caps, precision, cost) is declarative — the FKC program's thesis.

---

## 2026-07-08 — The runtime-fused kernel sidecar is TRANSITIONAL; end-state = a generalized binding key

**Sections affected**: 03 (IR) / 04 (optimization) — none revised yet (the sidecar is implementation, not a claim change); this entry exists to prevent the implementation from silently *becoming* a claim.
**Phase / PR**: JIT-on-request adoption (branch `jit-integration`). Code: `fuel-dispatch/src/runtime_fused_kernels.rs` (+ `runtime_fused_arm.rs`, `runtime_fused_pathfinder.rs`, `jit_adopt.rs`, `jit_cuda_load.rs`; `fuel-graph/src/runtime_fused.rs`).

**What changed**: adopting a JIT-synthesized (Tier-2) fused op required a place to bind its kernel at runtime. The binding table is keyed by the compile-time `OpKind` enum, which cannot admit a runtime-allocated `FusedOpId` — so the adoption path ships a **parallel `FusedOpId`-keyed kernel sidecar** (`runtime_fused_kernels`), a matching metadata sidecar in `fuel-graph`, a dedicated `is_runtime` resolution arm in `compile_one`, and a separate `PassRegistry::default_passes_with_runtime_fusion()` constructor. **All four are transitional** — the same status `ExecutionPlan` holds — and none of them is licensed to grow new consumers.

**Why**: the standing dispatch-architecture direction is *"registry runtime-mutable (no sidecar); build new dispatch/fusion infra into graph+registry"*. The sidecar is a letter-violation accepted knowingly (flagged by the data-dependent-shapes session at coordination, 2026-07-08) because the alternative — generalizing the binding-table key — is a cross-cutting change to a surface two other active programs are mid-rebuild on, while the JIT loop was already hardware-verified end-to-end against the sidecar.

**Alternatives considered**: *generalize the binding key now* — rejected for sequencing, not correctness (it is the end-state); *a synthetic `OpKind` per adopted op* — rejected: `OpKind` is a closed compile-time enum and faking entries poisons every exhaustive consumer; *keying runtime kernels into the plan* — rejected: the plan is itself being deleted ("plan IS the graph").

**Implications going forward**: the end-state is to **generalize the binding-table key to `{Static(OpKind) | RuntimeFused(FusedOpId)}`** so runtime entries live in the ONE registry. When that lands: the kernel sidecar folds in; `compile_one`'s `is_runtime` arm collapses into the terminal lookup; `default_passes_with_runtime_fusion` collapses back into `default_passes` (the test-hermeticity split exists only because the sidecar is process-global); and `OpKind::RuntimeFused` remains as the key's runtime discriminant. Until then, treat any new code that reads the sidecar directly (rather than through `fused_kernel_available` / the `compile_one` arm) as a review flag.

---

## 2026-07-08 — Binding key generalized; the runtime-kernel sidecar is FOLDED

**Sections affected**: none revised (this executes the end-state the previous entry named; the standing "registry runtime-mutable, no sidecar" claim now holds for runtime kernels).
**Phase / PR**: branch `binding-key-gen` (after the dd-shapes merge freed the `pipelined.rs`/`plan.rs` surface). Code: `fuel-dispatch/src/kernel.rs` (`BindingKey`), `runtime_fused_kernels.rs` (now a facade), `pipelined.rs` (`compile_one`'s arm reads the table).

**What changed**: the binding table's key is now `BindingKey { Static(OpKind) | RuntimeFused(FusedOpId) }` (`From<OpKind>` keeps every existing registration/lookup call site source-unchanged). A runtime kernel registers through `extend_global_bindings` under its `RuntimeFused` id — **the second registry is gone** (and with it the per-adopt `Box::leak` of the dtype tuple; the table owns its `KernelDTypes`). `compile_one`'s `is_runtime` arm resolves through the ordinary `lookup_with_caps`, so an absent kernel and a dtype mismatch are one honest `NoBackendForOp` miss; the arm keeps only the runtime-specific obligations (backend-pin guard, single-output reject, `Runtime{scalars} → JitScalars`). Static-audit iterators (`iter_keys`/`iter_precision`/`iter_cost`) expose Static entries only; `fill_unset_cpu_cost` skips runtime rows (they keep the `unknown_cost` sentinel — never a lying zero).

**What is STILL transitional** (scoped honestly, contra the previous entry's optimism):
- **The metadata sidecar** (`fuel-graph/src/runtime_fused.rs`, id → region) remains — folding it means a runtime-mutable `FusedOpRegistry`, a separate program.
- **`default_passes_with_runtime_fusion`** therefore also remains: its hermeticity rationale was always the *metadata* scan (the pathfinder iterates `runtime_entries()`), which is still process-global.
- **The `compile_one` arm** is thinner but not collapsed: full collapse needs `op_to_binding_key` at the terminal lookup + the plan path (G4).
- **G4 (plan-path pricing)**: a runtime arm is sparse-skip unpriced in `compile_plan` (safe — arm 0 is the runnability fallback). Pricing derives from the recipe (`decompose_region`) keyed by the `RuntimeFused` id — no stored closure needed; the fused-cost sentinel attachment the sidecar carried is retired with it.

**Implications going forward**: new runtime-kernel consumers go through the binding table (`BindingKey::RuntimeFused`) or the facade — never a new side structure. G4 + the terminal-arm collapse + the metadata-registry fold are the named remainder.

---

## 2026-07-08 — Elementwise NaN conventions pinned to torch parity (relu/maximum/minimum NaN-propagating)

**Sections affected**: none — this pins a numerics *convention* (and fixes a CPU/CUDA divergence + several FKC-doc misdescriptions); no numbered architecture-doc core claim changes. The convention itself was previously undocumented at this level, living only as scattered "NaN-as-missing" prose in `docs/kernel-contracts/**`.
**Phase / PR**: cross-project coordination with Baracuda (kernel-contracts / NaN-semantics audit). Code: `fuel-cpu-backend/src/chassis/{unary,binary}.rs` (`Relu`, `Maximum`, `Minimum`), `fuel-cpu-backend/src/dyn_impl.rs` (`all_unary_relu`, `cpu_binary_op`'s `Maximum`/`Minimum` closures), `fuel-core/src/op.rs` (`UnaryOpT for Relu`, `BinaryOpT for Maximum`/`Minimum` — unreachable for real computation today, flipped for consistency). Tests: `fuel-cpu-backend/src/byte_kernels.rs`, `fuel-core/src/lazy.rs`.

**What changed**: `Relu`, `Maximum`, and `Minimum` are now **NaN-propagating** on CPU, matching PyTorch (`torch.relu(nan) == nan`; `torch.maximum`/`torch.minimum` return NaN if *either* operand is NaN). Previously the CPU implementations used bare `f32::max`/`f32::min`/`x.max(0.0)`, which is IEEE `maxNum`-style **NaN-as-missing** (returns the non-NaN operand). This is a **behavior change on the production CPU dispatch path** (the `chassis::binary::{Maximum,Minimum}` / `chassis::unary::Relu` op markers that `fuel-cpu-backend::byte_kernels`'s thunks — and therefore `fuel-dispatch`'s registered CPU kernels — route through), not just a doc correction. `dyn_impl.rs`'s eager `CpuStorage` path (the `DynBackendStorage` `unary_op_dyn`/`binary_op_dyn` implementation, used e.g. by the Judge's op-kind reference evaluation) is fixed in lockstep — its prior `Maximum`/`Minimum` closures (`if a < b { b } else { a }`) were additionally **asymmetric**: NaN comparisons are always `false`, so the old code silently propagated NaN only when the *left* operand was NaN and scrubbed when only the right was — an unintentional bug this fix also closes. `ReluInplace` flipped for free (it reuses `chassis::unary::Relu`). Reduction ops (`ReduceMax`/`ReduceMin`/`max_dim`/etc.) and softmax internals are explicitly untouched — this pin is elementwise-only. The scrubbing (NaN-as-missing) behavior remains available under the separate `Fmax`/`Fmin` op family, unchanged.

**Why**: CireSnave's standing collaboration norm — "match external convention for well-known ops (PyTorch/CUDA semantics) over internal consistency" (`CLAUDE.md` § Collaboration norms) — applies directly: Fuel has no prior NaN-semantics commitment of its own, PyTorch does, and matching it is lower-surprise for anyone porting a model. The pin was triggered by a cross-project audit from Baracuda (the CUDA-kernel sibling project), which found Fuel's CPU core already **diverged from its own CUDA backend**: `binary_maximum_fp.cu` / `binary_minimum_fp.cu` were *already* NaN-propagating while CPU's `Maximum`/`Minimum` scrubbed — so this fix is also a cross-backend consistency fix, not just a PyTorch-parity one. Baracuda's audit additionally found the FKC docs (`docs/kernel-contracts/dispatch/elementwise-binary.fkc.md`) **mislabeled CUDA's own maximum/minimum kernels as NaN-as-missing** — a pure doc bug (the CUDA kernel was never scrubbing) — corrected in the same pass.

**Baracuda coordination**: relu is the one op that could *not* be aligned symmetrically today — CUDA's `unary_relu_fp.cu` (bound via `baracuda_dispatch.rs` → `bk::unary_relu_f32` etc.) uses `fmaxf`, which scrubs. Baracuda has committed to shipping a NaN-propagating relu kernel in **alpha.76** (this pin coordinates with that commitment — an "advertised capability withhold" is lifted once the propagating kernel lands, per the FKC capability-advertisement model). Until then, CPU relu is NaN-propagating and CUDA relu is NaN-as-missing: a **documented, dated, transitional divergence**, not an oversight. `fuel-core/src/lazy.rs::relu_cuda_still_scrubs_nan_pending_alpha76_rebind` is a live (`#[ignore]`'d, `--features cuda`) test that pins the current CUDA-scrubs behavior and fails loudly (with flip instructions in its doc comment) the moment alpha.76 changes it — a deliberate "pin the gap" test, in the same spirit as the flash_attn / selective_scan gap-posture tests from the 2026-07-03 entry above. Per `CLAUDE.md`, the CUDA relu kernel binding itself is **not** touched by this change (Baracuda owns baracuda; Fuel-internal changes stop at the FFI boundary).

**Correction — no separate `Fmax`/`Fmin` op family exists.** The originating task briefed this as "scrubbing semantics remain available as the separate `Fmax` family (do not remove or alter `Fmax`/`Fmin` if they exist)." They do not exist as a Fuel op: grepping the workspace, `Fmax`/`Fmin` appear only in `fuel-core/src/mkl.rs` (and its `fuel-cpu-backend` mirror) as **Intel MKL's own C-function names** (`vsFmax`/`vdFmax`/`vsFmin`/`vdFmin`) — the `#[cfg(feature = "mkl")]` vectorized-acceleration path (`F32_VEC`/`F64_VEC` `bin_op!` hooks) of this *same* `Maximum`/`Minimum` `BinaryOpT` impl, not a separate op or a separate scrubbing home. This pin's `is_nan()` guard is on the scalar fallback methods only; the `mkl`-feature vectorized path still calls MKL's native `vsFmax`/`vdFmax` (C99 `fmax`/`fmin` semantics — NaN-as-missing) unchanged, so *if* the `mkl` feature were ever built (never enabled in this repo's tests per `CLAUDE.md`) and *if* `fuel-core/src/op.rs`'s `Maximum`/`Minimum` were ever wired into a live `Map1`/`Map2`-style dispatch (today they are not — see the "Implications" bullet below), the vectorized and scalar paths of the same op would silently disagree on NaN. Flagged as a latent landmine rather than fixed in this pin: `crate::mkl::{vs_max,vd_max,vs_min,vd_min}` are out of this pin's scope (native MKL bindings, not Fuel-authored scalar math), and the whole `mkl`-feature surface is currently unreachable for computation regardless.

**Alternatives considered**: *Canonical-NaN output* (collapse any NaN operand to a single canonical `NaN` bit pattern) — rejected as the default: payload preservation was achievable at negligible cost (`is_nan()` check + early return of the original operand, not a re-encode) for every CPU dtype in this pin's scope (f32/f64 native, bf16/f16/f8e4m3 via the existing f32 round-trip), so there was no correctness/perf reason to discard the payload. *Leave CUDA relu's divergence undocumented / silently accept it* — rejected: a silent CPU/CUDA split on a common op is exactly the kind of gap the constitution requires surfacing (never-panic / no-silent-gap posture), hence the dated pin-the-gap test rather than a TODO comment. *Also flip Vulkan's `MIN`/`MAX` macro (`metal/elementwise.fkc.md`, `vulkan/elementwise.fkc.md` — both currently documented NaN-naive)* — out of scope for this pin: neither backend was part of Baracuda's audit or this task's mandate, and changing their kernels without in-repo verification of their actual current behavior would be an unverified claim, not a fix; their FKC docs are left exactly as they were.

**Implications going forward**: (1) When baracuda alpha.76 ships, flip `relu_cuda_still_scrubs_nan_pending_alpha76_rebind`'s assertion (documented inline) and collapse `docs/kernel-contracts/{cpu,dispatch}/elementwise-unary.fkc.md`'s CPU/CUDA relu split back into one NaN-propagating claim. (2) Vulkan's and Metal's NaN conventions for `Maximum`/`Minimum`/`Relu` remain an open, separately-scoped question — their FKC docs still say NaN-naive/NaN-as-missing and should be independently audited before anyone relies on Vulkan/Metal matching this pin. (3) `fuel-core/src/op.rs`'s `UnaryOpT`/`BinaryOpT` marker-struct math (`Relu`, `Maximum`, `Minimum`) was flipped even though it is currently unreachable for real computation (`Tensor::{relu,maximum,minimum}` route through `Storage::{unary,binary}_impl` → op-name redispatch → `fuel-cpu-backend/src/dyn_impl.rs`, not through these trait methods) — flagged here so a future CPU `Map1`/`Map2`-style blanket impl (mirroring what CUDA/Metal already have) doesn't silently inherit stale scrubbing math.

---

## See also

- [00-index §Versioning convention](00-index.md#versioning-convention) — when to bump section versions.
- ROADMAP.md — phase-level work tracking.
