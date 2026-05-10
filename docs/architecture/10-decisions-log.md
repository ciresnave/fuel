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

## See also

- [00-index §Versioning convention](00-index.md#versioning-convention) — when to bump section versions.
- ROADMAP.md — phase-level work tracking.
- `docs/architecture-audit.md` — the cross-thread audit that triggered the v0.1 architecture-set drafting.
