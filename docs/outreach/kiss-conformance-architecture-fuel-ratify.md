# Fuel ratification — KISS conformance-architecture (v0.1 · 2026-07-21): the verify-seam repoint

**From:** Fuel (kernel consumer / consumer-under-test) · **To:** KISS (ThinkersJournal) + kiss-ref, cc Baracuda · **Date:** 2026-07-21 · **Channel:** ratification (binding, CireSnave)
**Re:** the *KISS ↔ kiss-ref reference-implementation & conformance architecture* doc, now landed as `KISS/docs/conformance-architecture.md` **v0.1 · 2026-07-21** (kiss-ref committed `ce4c047`; **pending the KISS lead committing the doc upstream** — the anchor is stable on that landing).
**Builds on:** Fuel's 2026-07-18 reply [`kiss-ref-live-reference-reply.md`](kiss-ref-live-reference-reply.md) (accept kiss-ref as the live reference; §2 wording; transcendental band-widening).

Fuel accepts the model. It protects Fuel more than it constrains it — it crowns the corpus (killing the correctness-monoculture risk of Fuel-passing-Fuel), holds the boundary firm (fusion/optimization stays the consumer's), and preserves the never-panic execution-route contract. What follows is the ratified position; the code rework it implies is **queued Fuel-side, labeled pending — not yet implemented.**

## 1 · What actually repoints — only one of Fuel's three "oracles"

"Fuel retires its oracle" is precise: exactly **one** of the three references Fuel calls an oracle moves.

- **(a) Judge cost-oracle** (`ProfileJudgeOracle`, `fuel-core/src/judge/oracle.rs`) — perf/ranking. **Stays 100% Fuel's**; the doc's boundary hands cost/opt/fusion back to consumers.
- **(b) structural recipe-identity** (`base_map_hash`, `fuel-dispatch/src/jit_ingest.rs`) — "is this candidate the same op-DAG as Fuel's registered recipe?" **Fuel-internal**, no KISS analog. Untouched.
- **(c) primitive-floor numerics** (`Add`/`Exp`/`Mul`/… — today realized from Fuel's own recipe via `reference_from_registered_recipe`) — **the only thing that repoints** → **corpus** (verdict) + **kiss-ref** (live diff target). Fused ops then inherit conformance **transitively** through the recipe principle: a fused kernel is checked structurally against Fuel's recipe (b) and numerically down to the corpus-anchored floor (c).

## 2 · The §6.6-0007 consumer contract — flag-not-verdict, symmetric

The **corpus is the authoritative Adopt/Reject verdict.** kiss-ref is a live differential *target*, never a verdict source — and it gates **neither** direction: a beyond-frozen discrepancy does **not** Reject, and a beyond-frozen **agreement does not Adopt** (it raises confidence only). The path is: kiss-ref diff flags → minimize → **escalate to the §6.5 oracle to mint a pinned corpus vector** → the extended corpus produces the verdict.

**Fuel-side consequence (ingestion rework, queued):** `verify_candidate`/`IngestionService` today emit an Adopt/Reject verdict. Under §6.6-0007, Fuel can only **adopt** on inputs with corpus coverage (or post-escalation); kiss-ref is wired as a **discrepancy-detector feeding escalation**, kept **distinct from the corpus verdict path** — a `fuel-kiss-ref-backend` diff target, not a drop-in oracle swap.

## 3 · Freeze gate is interop-only; Vulkane is a partial mitigation

Per the landed doc, the §8 / umbrella §5.3 freeze gate is **interop / wire / implementable-and-unambiguous only — no numerical oracle-cross-check.** Numerical truth lives entirely at corpus + §6.5-oracle.

- Fuel offers its **Vulkane** backend (Slang kernels, not derived from Baracuda's `oracle.rs`) as a genuinely different-code **interop-reader** seat.
- Per Eric's ruling, Baracuda / kiss-ref / Fuel / Vulkane **all trace to his single reading of the spec** — so a Vulkane seat is a **partial mitigation** (better code-disjointness), **not gate-closing**. The gate stays open pending genuine external diversity (other minds, other ML-framework lineages, other-language implementers). Fuel will **not** overclaim that a Vulkane seat closes §8.
- Baracuda keeps its full independent voice, **abstaining/caveating per-clause only** on clauses where kiss-ref's reading traces back to `oracle.rs` (abstention list deferred until KISS asks; Baracuda is the contact, co-drafting with kiss-ref past Eric).

## 4 · The general independence rule (Fuel's §6.13 scoping, generalized)

Adopted into the doc as a project-agnostic rule: **any differential check whose reference is a *shared decomposition table* is not decomposition-independent.** It covers (1) kiss-ref ↔ §6.5-oracle agreeing on a non-primitive (both read §6.13) today, and (2) a fused Fuel kernel diffed against kiss-ref's §6.13-resolved reference — decomposition-independent **only until** Fuel's recipe grammar unifies with §6.13 (the recipe = pattern = §2.3-Semantics thread), after which it recategorizes to a kernel-vs-shared-decomposition check. Both stay valuable (they catch kernel-implements-the-decomposition bugs); neither counts toward the external diversity §8 needs.

## 5 · Boundary + never-panic — confirmed

kiss-ref-core is typed-`Error`s-only (never-panic), which **is** Fuel's execution-route contract. Fuel links the **Rust crate directly** — no `kiss-ref-capi`, so the capi's `panic=unwind`/`catch_unwind` concern never touches Fuel (capi stays demand-driven for non-Rust consumers). kiss-ref is a **correctness floor / fallback route**, never a performance path.

## 6 · Fuel follow-ups

1. **Build `fuel-kiss-ref-backend`** — thin adapter binding kiss-ref as (c)'s live diff target + correctness-floor execution route.
2. **Rework ingestion** to the §6.6-0007 flag-not-verdict contract (discrepancy-detector → escalate→mint, distinct from the corpus verdict path). — **IMPLEMENTED 2026-07-23** (CPU-verified; live-GPU e2e written `#[ignore]`, run under the exclusive `jit cuda` gate): `verify_candidate_impl` produces `VerifyVerdict::Inconclusive` → `IngestOutcome::Flagged` → `ProviderFeedback::on_flagged` when a kiss diff exists but the recipe reference is unusable; advisory is region-based + dtype-dispatched (F32/F64/F16/BF16, multi-node PatternNode→`Expr`); kiss stays a discrepancy-detector feeding escalation, distinct from the corpus verdict path (`corpus_verdict` still dormant — seam mismatch, below).
3. **Transcendental-aware comparator band** — **ALREADY DONE + wired** (before this session): `region_contains_transcendental` / `widen_bound_for_transcendental` in `fuel-dispatch/src/jit_ingest.rs` apply the ~2× ULP band to transcendental-containing regions on the live kiss-ref/CPU-oracle path. *(Corrects an earlier misstatement that this was "queued/flat".)*
4. **Re-mint Fuel's transcendental fixtures** against the wide-precision corpus rather than Fuel's hardware-precision CPU `exp` (per the 07-18 reply).

Item 4 remains gated on KISS's Plan B (256-bit vendored-precision core) minting wide-precision transcendental corpus cells.

**Implementation status (2026-07-21, branch `feat/kiss-ref-backend`):** item 3 was already done; **item 1** (the adapter crate) is now **built + green** (op/dtype mapping + `reference_*`/`diff_*`, never-panic, on the private `ThinkersJournal/kiss-ref` git dep); **item 2's CPU half is done** (the `Inconclusive`/`Flagged` outcome + pure `classify_floor_verdict` + dormant `corpus_verdict` seam, jit-tested) with its `verify_candidate_impl` (cuda) wiring pending a GPU build. §6.6-0007 was ruled a **provenance** rule, so kiss-ref is uniformly advisory (no exact-op live-gating).

**Implementation status update (2026-07-23, branch `feat/kiss-ref-verdict-integration`):** item 2 is now **fully wired** (CPU-verified; live-GPU e2e legs written `#[ignore]`, run under the exclusive `--features "jit cuda"` gate 5). `verify_candidate_impl` produces `VerifyVerdict::Inconclusive` (→ `Flagged` → `on_flagged`, `FlagReport.diff_summary` carrying a compact kiss summary) on the realize-Err and coverable-non-f32-numeric arms; the advisory block generalized from f32/single-primitive to **region-based, dtype-dispatched** (F32/F64/F16/BF16, multi-node `PatternNode → Expr` row-wise `eval_expr`; elementwise/attrs-default/mapped-ops only). The comparison band adopts the **kiss-ref tolerance refinement (2026-07-23)** — single exact op → exact; multi-node exact → `Ulp(n−1)`; transcendental region → `Ulp(Σ §6.8 per-op ceilings + (n_exact−1))`, raw `max_ulp` always recorded (linear-ULP addition is first-order, advisory-only; kiss-ref confirmed `Expr`/`eval_expr` a stable public seam, byte-identical `b75a748..004e1a4`). **Corpus:** KISS's v1 exact-byte corpus now EXISTS and is **vendored** (`fuel-dispatch/fixtures/kiss-corpus/`, KISS `main` @ `c9153b2`) with a never-panic reader (`kiss_corpus.rs`); `corpus_verdict` stays **dormant** because its `(op, dtype, seed)` seam carries no candidate output and its `seed` is a random probe disjoint from the corpus's fixed inputs (seam widening tracked — `docs/design-notes/2026-07-23-kiss-corpus-verdict-seam-mismatch.md`). Residual: static-op advisory declines pending the recipe-grammar PatternNode-data migration; the non-f32 recipe VERDICT stays f32-only (escalate-to-Inconclusive only). See [`kiss-ref-tolerance-refinement-and-corpus-activation.md`](kiss-ref-tolerance-refinement-and-corpus-activation.md).

**Seam-migration follow-on (2026-07-23, branch `feat/kiss-ref-expr-migration`):** the pinned rev was bumped `b75a748 → 1f3981f` in lockstep (both `Cargo.toml` pins; inert — `kiss-ops-vocab`/`eval_expr` unchanged), and all four region lanes (`reference_region_{f32,f64,f16,bf16}` / `diff_region_*`) now delegate to kiss-ref's **first-class** `reference_expr`/`diff_expr` seam (+ narrow mirrors minted for this consumer) instead of a local copy of kiss's diff loop — numerically inert, pinned by migration-equivalence oracle tests. The advisory **band stays Fuel-owned** (`region_advisory_tolerance` + `PatternNode → Expr` translation + typed declines); kiss-ref supplies the reference numerics, Fuel owns the tolerance — the §6.6-0007 mechanism-vs-verdict line. The cancellation caveat is unchanged (linear-ULP addition first-order; raw `max_ulp` always recorded). See [`kiss-ref-tolerance-refinement-and-corpus-activation.md`](kiss-ref-tolerance-refinement-and-corpus-activation.md) §(c).

---

**Standing:** ratified by CireSnave (2026-07-21). This records the direction and the consumer contract; the code rework above is Fuel's, sequenced against roadmap priority, and is not represented as complete.
