# Fuel architecture: index

**Status**: v1.0 (2026-05-09). The architecture set is the durable description of what fuel is and how it's structured. Code, ROADMAP, and per-phase design documents anchor to this set. When this set and any phase document conflict, this set is authoritative; the phase document is updated to match.

**Audience**: future-you, future-me, contributors (human or model) trying to understand "what is fuel trying to be?" Not new users. Not API reference. Not tutorials.

**Living document**: each section is independently versioned; material changes are recorded in [10-decisions-log.md](10-decisions-log.md). The v1.0 establishment entry summarizes the 24 architectural decisions made during the v0.x → v1.0 drafting period.

---

## The set

| # | Section | Summary |
| --- | --- | --- |
| 01 | [identity](01-identity.md) | What fuel is, what it isn't, what makes it competitive |
| 02 | [layers](02-layers.md) | Crate boundaries, layer model, dependency direction |
| 03 | [ir](03-ir.md) | DAG, base map, primitive Op enum, fused-op registry, layouts |
| 04 | [optimization](04-optimization.md) | DecompositionMap, OptimizationMap, per-decision-point alternatives, sliding window |
| 05 | [backend-contract](05-backend-contract.md) | What backends provide (kernels, capabilities, telemetry, slot capacity); what they don't decide |
| 06 | [runtime](06-runtime.md) | Route picker, dispatch lookahead, data parallelism, telemetry-driven decisions |
| 07 | [tolerance](07-tolerance.md) | Per-op error budgets, hierarchical specification, approximate optimizations, calibration |
| 08 | [pattern-harvest](08-pattern-harvest.md) | Opt-in telemetry to guide fused-op development |
| 09 | [non-goals](09-non-goals.md) | What fuel deliberately doesn't try to be |
| 10 | [decisions-log](10-decisions-log.md) | Material architectural changes over time |
| 11 | [persistence](11-persistence.md) | Optimization-cache and tolerance-recipe sibling artifacts; format, invalidation, mmap |

---

## Reading order

**For a new contributor wanting the picture in 30 minutes**:

1. → 01 identity (5 min)
2. → 02 layers (3 min)
3. → 03 ir (8 min)
4. → 04 optimization (10 min — the longest)
5. → 09 non-goals (3 min)

That's the spine. The remaining sections are there when you need them.

**For someone designing a new phase**:

1. Read 01 to ground purpose.
2. Read 03 + 04 to ground IR and optimization model.
3. Read 05 if your phase touches a backend; 06 if it touches runtime; 07 if it touches numerical precision.
4. Cross-check against 09 (non-goals) before committing to scope.
5. After the phase ships, append a row to 10 (decisions log) if material decisions changed.

**For a code reviewer asking "is this aligned with fuel's architecture?"**:

- The relevant section number is usually obvious from the PR description.
- If it's not, the change probably touches multiple sections; review against each.

---

## Cross-link map

The architecture sections aren't independent. Below is the dependency graph — an arrow A → B means "A's content is grounded in concepts B defines."

```text
01 identity ──┬──→ 03 ir
              ├──→ 04 optimization
              ├──→ 05 backend-contract
              └──→ 07 tolerance

02 layers   ──→  03 ir

03 ir       ──┬──→ 04 optimization
              ├──→ 05 backend-contract
              ├──→ 06 runtime
              └──→ 11 persistence

04 optimization ──┬──→ 05 backend-contract
                  ├──→ 06 runtime
                  ├──→ 07 tolerance
                  ├──→ 08 pattern-harvest
                  └──→ 11 persistence

05 backend-contract ──┬──→ 06 runtime
                      ├──→ 07 tolerance
                      └──→ 11 persistence

06 runtime ──→ 11 persistence

07 tolerance ──┬──→ 06 runtime
               ├──→ 08 pattern-harvest    (shared community-telemetry infra)
               └──→ 11 persistence         (tolerance-recipe sibling artifact)

08 pattern-harvest ──→ 11 persistence       (shared community-telemetry infra)

09 non-goals          (referenced by 01, 04, 07; mostly self-contained)
10 decisions-log      (records changes to all of 01-09, 11)
11 persistence ──→ 03, 04, 05, 06, 07, 08   (cross-cuts everything that produces a persistable artifact)
```

If you change 03 (the IR), expect to revise 04, 05, 06, 11. If you change 04 (the optimization model), expect to revise 06, 07, 08, 11. If you change 01 (identity), the whole set may need re-anchoring.

---

## Versioning convention

Each section's header records its version. Bump when the change is material — a redirection, a new architectural decision, a removed concept. Don't bump for typos, clarifications, link fixes. The version is `vMAJOR.MINOR`:

- `MAJOR` increments when a section's *core claim* changes (e.g., "fused ops live in a registry, not in Op" is a major change to 03).
- `MINOR` increments when content is added or refined without changing core claims.

The decisions log (10) records every MAJOR bump with one paragraph of context (what changed, when, why, related PRs).

---

## How phase docs relate to this set

This set defines the steady-state architecture. Phase documents (currently in `docs/`: `storage-unification.md`, `fused-op-registry.md`, `architecture-audit.md`) describe in-flight work that moves the codebase toward this steady state.

The relationship is:

- A phase doc may *propose* a change to this set. The proposal is reviewed; if accepted, the set is updated; the phase doc references the updated section.
- A phase doc may *cite* this set for context. Cite by section number (`see 03-ir.md`). Don't restate; link.
- When a phase ships, its decisions log entry (in 10) records what (if anything) changed in this set as a result.

Implementation detail (file layouts, type signatures, code shapes) lives in phase docs, not here. This set says *what* fuel is and *why*, not *how* to implement it.

---

## Out of band: this is a draft

v0.1 means: the structure is committed, the content is in active drafting. Sections will land one or two at a time. Until v1.0 (the first complete set), expect:

- Some sections in this index reference docs that don't yet exist.
- Sections that do exist may be partial.
- The decisions log is empty until v1.0 (no decisions to record yet relative to a baseline).

When all 11 sections exist, this header gets bumped to v1.0 and from that point the decisions log becomes load-bearing.
