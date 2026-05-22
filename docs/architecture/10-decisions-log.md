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

6. **Three coexisting artifacts**: user-facing form, base map (canonical primitive DAG, retained as permanent artifact), optimized form (top-N alternatives per decision point with pre-resolved kernels). (03-ir.)

7. **Per-decision-point alternatives, not N global routes.** The optimizer preserves up to N alternatives per decision point (default N=3); decisions can be coupled via conditional cost adjustments. The runtime route picker resolves alternatives at dispatch time using current telemetry, mixing and matching across decision points. Strictly more flexible than top-N complete plans. (04-optimization.)

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

## See also

- [00-index §Versioning convention](00-index.md#versioning-convention) — when to bump section versions.
- ROADMAP.md — phase-level work tracking.
- `docs/architecture-audit.md` — the cross-thread audit that triggered the v0.1 architecture-set drafting.
