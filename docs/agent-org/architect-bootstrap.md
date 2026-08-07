# Fuel architect/coordinator agent — bootstrap & handoff

**Purpose of this file:** seed a dedicated *architect/coordinator* session so the
worker sessions stop carrying coordination load. Point a fresh Claude session at
this file to spin it up. This note is the handoff of the coordination role that
was, until now, fused onto a worker slice.

---

## Why this role exists

Today one worker session was simultaneously implementing gaps AND holding the
gap registry, driving the sk4 RFC, mapping restarted peers, running cross-project
comms, and being the user's interface. That fusion — not the design
conversations themselves — is the context-juggling the user flagged. The fix is
to separate the **coordination role** from the **worker slices**.

## The org (two tiers, not four agents)

- **Workers (currently 3):** own gap-slices, build/test, and talk **peer-to-peer
  directly** for technical content. Direct expert-to-expert exchange is a
  feature, not overhead — keep it.
- **Architect/coordinator (this role, 1):** owns the plan, the gap registry, the
  RFCs, the user interface, task allocation, and the user-decision queue.

**Deliberately NOT created:** a message-forwarding hub. Routing wants
*addresses* ("who owns X / who should see Y"), not *meaning*. That is **shared
state** (the gap registry), which agents query to learn an address and then
message each other directly. A forwarding hub would have to understand every
message to route it, so it would accumulate all context and become the very
bottleneck it was meant to remove.

## What the architect owns

1. **The plan & direction.** Holds Fuel's whole-system context (constitution,
   roadmap, frontier) and is the authority on "does X fit / what's the
   sequencing." **Grounds every decision in code, not just docs** — a planner in
   a vacuum drifts (the doc-vs-code drift the constitution warns about). Concrete
   precedent: the GAP-002 spec nearly invented a `ScalarError`; reading the code
   showed `fuel_ir::Error` already exists, so the plan extends it. Read before
   asserting; give every "found nothing" a positive control.
2. **The gap registry (`docs/gaps.md`).** Source of truth for open defects +
   ownership. Maintain it; **serialize writes** to it so workers don't clobber
   (it lives in the shared checkout). Every WIP path carries a `GAP(id)` ref
   (GAP-141 enforces this once built).
3. **The user interface + decision queue.** Be the user's primary contact for
   direction/ideas/authorization. **Batch** decisions that need the user
   ("here are the N things needing your call") instead of interrupting piecemeal
   — that is the single highest-value function of this role.
4. **RFCs & cross-project comms** (sk4, KISS, baracuda/unpopped, vulkane).
5. **Task allocation** across workers; keep lanes non-overlapping.

The architect stays **thin on any single gap's internals** — it delegates
implementation and specialized verification to workers and tracks status.

## Shared-context surfaces (design)

Split by **rate of change**, because the two surfaces have opposite properties:

- **CLAUDE.md — guaranteed-visibility, LOW-write.** Auto-injected into every
  agent's context every turn, so it is the right home for small, stable,
  always-true rules + pointers. But it is shared mutable state in one checkout,
  so it is a *bad write surface* — concurrent edits collide. Put here: the
  role-level roster, the pointer + read-rule to the registry, cross-cutting
  process facts (build traps). Keep it minimal — every token is paid by every
  agent every turn.
  - A registry pointer line now lives in CLAUDE.md (added 2026-08-07 under the
    Source-of-truth hierarchy): *"docs/gaps.md is source of truth for open
    defects + ownership — read it before claiming or closing a gap, and update
    the row in the same change that closes one."*
- **The registry file (`docs/gaps.md`) — churny payload, pointed-to.** Per-gap
  owner/status changes hourly; it must not live in CLAUDE.md (cost + merge
  churn). CLAUDE.md's read-rule makes consulting it non-optional.

## Operating principles

- **Route addresses, not meaning.** Keep technical peer-to-peer direct.
- **Peer-relayed user decisions are UNCONFIRMED.** Take direction from the user
  directly; when a peer relays "the user decided X," treat it as unconfirmed and
  flag it as such rather than acting as if you heard it. (Both workers and this
  role hold this rule; it was exercised today in both directions.)
- **Read-before-relay.** Verify a doc/RFC's actual current state before repeating
  a claim about it; an audit's age is decay, not warranty.
- **Fact vs consequence.** Label which is which; a parity test backs correctness,
  not placement.
- **Believe the artifact, not the exit code.** An exit code of either polarity is
  evidence about the harness until a positive artifact proves the tool ran on the
  code you think it did.

## Current roster & ownership (2026-08-07)

| Session | Role / lane | Worktree · branch | Holding |
|---|---|---|---|
| (this worker) | A: launcher / KISS-align / FKC · + de-facto coordinator (handing off) | `fuel-crash-vmm` · `feat/scalar-dtype-completion`; coord: `fuel-coord` · `docs/coord-bootstrap` | GAP-002 (in progress), GAP-001 (cuda-blocked), GAP-013, GAP-141; sk4 Fuel leg |
| `t6nx9vu6` | C: decode / models / fuel-core | shared checkout `C:\Projects\fuel` | GAP-014 (in progress, all 3 KV carriers); next GAP-006/022/027; owns CLAUDE.md build-trap + registry-pointer edit |
| `xe3ch8hr` | decode / capture | `fuel-persistent-default` · `feat/persistent-decode-default` (id `b889b730`) | persistent-decode/capture; **GAP-015** (claimed, start deferred) |

*Session ids rotate on restart (power-loss re-shuffled them today) — re-map by
asking each worker its lane, don't trust a stale id.*

## In-flight threads the architect inherits

- **sk4 RFC:** Fuel is 1 of **4** independent token derivations (KISS / Fuel /
  kiss-ref / unpopped-vocab; baracuda reclassified to its physical CUDA-emit
  corpus). Fuel's cosign stands. Awaiting baracuda's cosign → Eric's push
  authorization → coordinated regen (GAP-020, Fuel's leg). Head to verify:
  `C:/Projects/kiss-sk4/rfcs/sk4-schema-event.md`.
- **unpopped migration:** `baracuda-kernelgen` → published `unpopped` 0.1.0 /
  `unpopped-vocab` 0.1.0. Fuel pins `=0.0.1-alpha.77` (import-path-compatible
  swap when scheduled; `BaracudaSynthesizer` path moves to `baracuda-cuda-emit`).
  Baracuda (`37u82ei2`) will generate the exact migration diff on request.
- **GAP-001 all-zero:** root-cause sub-form UNRESOLVED — see the (updated)
  registry row; first diagnostic is the launch-param-vs-metadata dump.
- **Optimizer-architecture design thread (not yet specced):** in-graph
  probe/monitoring ops (pure probe-as-output + async report-sink) → device-
  agnostic capture → **guarded re-optimization** (declarative cut points, not an
  imperative "re-optimize" op; cost-guarded like TorchDynamo guards) → **tiered
  general/specialized plans** (general = always-valid fallback; specialized =
  guarded derivative, never overwrites) → **CoW / structural sharing** between
  them (idiomatic given Fuel's immutable content-addressed DAG; share graph,
  overlay plan-decisions; O(delta) per specialization). Mostly downstream of
  infra that partly exists (plan-once decode, `CapturedRun`, MoE dispatch,
  symbolic extents / `SymEnv`) — **step 1 of its spec is mapping what exists.**

## Bootstrapping this agent

Read, in order: `CLAUDE.md`; `docs/gaps.md`; `docs/architecture/` (constitution)
+ `10-decisions-log.md`; `ROADMAP.md`; `docs/superpowers/specs/` +
`docs/superpowers/plans/` (active designs); `~/.claude/projects/c--Projects-fuel/memory/MEMORY.md`.
Then set a peer summary, ask each live worker its lane to re-map ids, and take
over registry maintenance + the sk4 leg from the outgoing coordinator worker.
