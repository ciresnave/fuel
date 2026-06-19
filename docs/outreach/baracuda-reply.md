# Fuel → Baracuda — formal reply: Judge timing-retention RESOLVED, Open-Q-1 = YES

**Status: DRAFT — not sent.** This is the **formal reply companion** to
[`baracuda-dlpack-fkc-ask.md`](baracuda-dlpack-fkc-ask.md), Fuel's outbound two-way-contract
proposal (FDX + FKC + telemetry). That proposal deferred its formal answer — and Baracuda's
**Open Question 1** — pending a check of whether Fuel's Judge retains per-candidate timings.
**That check is done. The answer is YES.** This document resolves the §6 deferral.

It is framed per the same working agreement as the proposal it answers: a **cross-project
proposal, not a unilateral edit** to a sibling repo. Baracuda is a sibling project (same author,
`ciresnave`); nothing here has been written into any Baracuda repo. This is Fuel's settled
position on the one question that gated the reply, plus the now-answerable follow-ups.

**Author side:** Fuel (`fuel-core-types`, `fuel-dispatch`, `fuel-core` judge subsystem).
**Counterpart:** Baracuda kernel-specialization / AOT-matrix team.

---

> **Maintainer note (action on the companion doc).**
> [`baracuda-dlpack-fkc-ask.md`](baracuda-dlpack-fkc-ask.md) **§6** currently lists "Fuel's
> formal reply" and "the Judge timing-retention check" under **DEFERRED**. With this reply,
> that deferral is **RESOLVED**. Update §6 to mark the Judge-retention dependency **RESOLVED
> (2026-06-18)** and link this document
> ([`baracuda-reply.md`](baracuda-reply.md)) as the formal answer. The §6 bullet
> "Per-(shape, impl) timings retained, or only the winner? — **DEFERRED**" in §4 likewise moves
> to **YES** (see §2 below). Leave the *telemetry-subsystem* (emission layer) item in DEFERRED —
> that one is still open; only retention closed.

---

## 1. What changed since the proposal

The proposal's deferral rested on one premise that turned out to be stale: that the Judge was
"mid-rebuild" and "f32-square-only," so its timing-retention shape was unknown. **That premise
was wrong as of the verification done for this reply.** Reading the code as ground truth (per
Fuel's "the sketch is the design, the code is the truth" rule), the Judge **already retains
per-`(op, dtype, size_class, backend, kernel_source)` timings — including losing alternatives —
as `u64` nanoseconds**, and has since per-alternative measurement shipped (Phase 6b). The
retention question that gated the reply is therefore answerable now, affirmatively, with no new
Judge work required.

What the proposal got right stands unchanged: the boundary shape (FDX + FKC carry every join
token), the `ImplId` basis tuple, the miss definition, and the negative-strides decision. This
reply does not revise any of those; it resolves the one open dependency and answers the
follow-up open questions that were blocked behind it.

---

## 2. Open Question 1 — ANSWERED: YES, per-(shape, impl) timings are retained

> *Baracuda Open-Q-1: "Do you retain per-(shape, impl) timings, or only the winner?"*

**Yes — per-alternative, not winner-only.** Your `candidates[]` is **feasible**; it does not
require Fuel to build new retention. The retention exists in two concrete artifacts, both
verified against the code:

### 2.1 `ProfileReport` — persistent, one entry per measured alternative (including losers)

The persisted Judge artifact is a flat table of `ProfileEntry`, one **per measured
alternative**, explicitly including alternatives that lost. Each entry carries the full key plus
a `u64` nanosecond latency and a `kernel_source` tag that distinguishes sibling impls at the
same `(op, dtypes, backend)` decision point:

```rust
// fuel-core-types/src/dispatch.rs:655-683
pub struct ProfileEntry {
    pub op:            OpKind,
    pub dtype:         DType,
    pub size_class:    SizeClass,
    pub backend:       BackendId,
    pub device_index:  u32,
    /// Median wall-clock time per invocation over `iterations`.
    pub latency_ns:    u64,
    pub iterations:    u32,
    pub max_rel_error: f32,
    /// Distinguishes which kernel sibling produced this measurement
    /// when multiple alternatives register at the same
    /// (op, dtypes, backend) key. "" is the pre-v2 default.
    pub kernel_source: String,
}

// fuel-core-types/src/dispatch.rs:689-692
pub struct ProfileReport {
    pub version: u32,           // PROFILE_REPORT_VERSION == 2  (dispatch.rs:32)
    pub entries: Vec<ProfileEntry>,
}
```

`ProfileReport` persists as atomic JSON (`save`/`load`, `dispatch.rs:697-719`). The adapter
module states the loser-retention intent directly: the report holds *"one entry per
`(op, dtype, size_class, backend, device, kernel_source)` cell, including LOSING alternatives"*
([`fuel-core/src/judge/oracle.rs:14-17`](../../fuel-core/src/judge/oracle.rs)), and the cost
composer is built from the **report, not the winners-only `DispatchTable`**, precisely *"because
the cost composer's Layer-2 refinement needs latencies for EVERY candidate it ranks — a
losing-but-close alternative must carry its own measured number, not inherit the winner's"*
([`oracle.rs:18-23`](../../fuel-core/src/judge/oracle.rs)). That is exactly the property your
`candidates[]` needs.

### 2.2 `ProfileJudgeOracle` / `HashMapJudge` — in-memory, keyed per impl

The in-memory query surface is keyed on the same five axes — `kernel_source` is **part of the
key**, so siblings do not collide:

```rust
// fuel-dispatch/src/ranker/judge.rs:72-75
pub struct HashMapJudge {
    entries: std::collections::HashMap<(OpKind, DType, SizeClass, BackendId, String), u64>,
}

// fuel-dispatch/src/ranker/judge.rs:53-60  (the JudgeOracle trait method)
fn measured_latency_ns(
    &self,
    op: OpKind,
    dtype: DType,
    size_class: SizeClass,
    backend: BackendId,
    kernel_source: &str,
) -> Option<u64>;
```

`fuel-core`'s `ProfileJudgeOracle::from_report` indexes every entry for exact-match lookup
([`fuel-core/src/judge/oracle.rs:65-92`](../../fuel-core/src/judge/oracle.rs)). The
sibling-non-collision property is a shipped, passing test —
`sibling_kernel_sources_do_not_collide`
([`oracle.rs:170-195`](../../fuel-core/src/judge/oracle.rs)) — which asserts two impls at the
identical `(op, dtype, size_class, backend)` cell resolve to **distinct** latencies and that an
unmeasured sibling **misses** (`None`) rather than borrowing a neighbour's number. That is the
exact guarantee that makes `ImplId`-keyed `candidates[]` honest: each impl carries its own
measured number.

### 2.3 The upgrade, stated plainly

The proposal's §4 answer to Open-Q-1 was **DEFERRED**. The corrected answer is:

> **YES — Fuel retains per-`(op, dtype, size_class, backend, kernel_source)` timings, including
> losers, as `u64` nanoseconds. `candidates[]` is feasible.**

with two caveats, neither of which blocks the feed:

1. **Coverage is currently narrow, but the schema is complete and coverage is actively
   broadening.** What is *retained* is fully keyed (§2.1/§2.2); what is *populated* today is a
   bounded profiling matrix — **F32 only** (the measurement loop hardcodes `DType::F32` at
   [`fuel-core/src/judge/mod.rs:476` and `:512`](../../fuel-core/src/judge/mod.rs), guarded by
   `assert_eq!(dtype, DType::F32, "judge: only f32 wired for now")` at
   [`mod.rs:730`](../../fuel-core/src/judge/mod.rs)) — over an offline-profiled **square-matmul
   size ladder** (no GEMV / decode-shaped cells), a fixed primitive set, with no online
   exploration. So today many decode-regime cells (GEMV, non-F32, quantized) **miss** the oracle
   (`None` — the correct "no measurement" signal, never a fabricated number). **This is explicitly
   transient:** Fuel's Judge is slated for extensive expansion — more dtypes (it "will not be
   F32-only for long"), judging every op that supplies no declared cost, and flash-vs-decomposed
   arm comparison. Crucially, **the emission layer is built coverage-agnostic** — it reads whatever
   the oracle holds, so `candidates[]` **densifies automatically** as the Judge's matrix grows,
   with **no telemetry-format or wire change**. Plan for a feed that starts sparse and fills in
   over time, not a fixed-coverage snapshot.

2. **The EMISSION layer is the remaining build — not retention.** What is *not* yet built is the
   `DispatchRecord` / `MissRecord` **JSONL writer** over `ProfileJudgeOracle` plus the FKC
   best-admissible-match-is-generic miss signal, the opt-in flag, and the `ImplId` /
   `StructureKey` join. That work is tracked in
   [`docs/session-prompts/baracuda-telemetry-plan.md`](../session-prompts/baracuda-telemetry-plan.md)
   *(authored 2026-06-18 on this branch)*. Retention — the part you depend on and could not specify yourself — is **done**;
   emission is a self-contained Fuel feature that reads an existing store.

> **Divergence note (sketch vs code).** The program-state sketch and the carried memory said
> the Judge was "f32-square-only / mid-rebuild" with latencies as "f32 squares." The code
> contradicts both: latencies are **`u64` nanoseconds** (`ProfileEntry.latency_ns: u64`), and
> per-alternative retention **already shipped** (Phase 6b; test
> `sibling_kernel_sources_do_not_collide` passes). Only the *dtype population* (F32-only) part
> of the stale memory survives, and even that is a measurement-loop gap, not a schema or
> retention gap. This reply follows the code.

---

## 3. The other open questions — now answerable

With retention known, the follow-ups the proposal left "Judge-dependent" resolve:

1. **Granularity — aggregated histograms, not per-dispatch records.** Fuel's dispatch rate in
   decode is high; per-dispatch records would be a firehose. The Judge already stores at cell
   granularity (`(op, dtype, size_class, backend, kernel_source) → median latency`), so the
   natural emission is **aggregated per-key histograms** — which matches both your stated
   preference and Fuel's storage shape. No per-dispatch retention is needed or offered.

2. **`est_speedup` — inferred from the fallback record, not estimated at miss time.** Because
   the report retains the *losing* alternatives' latencies alongside the winner's, the speedup a
   specialized kernel would have to beat is **derivable from the data already present** (the
   generic fallback's `latency_ns` vs. the cell's best). We will infer it from the fallback's
   `DispatchRecord` rather than fabricate an estimate at miss time — and we would rather **drop
   the field entirely** than hold extra dataset to compute it (your action item 4). The retained
   loser timings make inference cheap, so the field is feasible if you want it.

3. **Sampling — feasible.** Since aggregation happens over a bounded per-key store rather than a
   per-dispatch log, sampling (rate-limit or reservoir over emitted records) is a
   straightforward knob on the emission layer, not a constraint on retention. We will expose it
   in the telemetry plan if the histogram volume warrants it.

---

## 4. Re-confirmed boundary shape (unchanged from the proposal)

The retention answer does not move any of the committed boundary decisions. For completeness,
Fuel **re-confirms**, as still-committed:

- **FDX is the tensor-description half.** Standard versioned DLPack for the ecosystem; FDX
  (DLPack + nullable `*const FDXSidecar`) for Fuel. The base `DLTensor` is never a lie (FDX
  honesty invariant). (`docs/specs/dlpack-extension.md` §3.)
- **`ImplId` = the FKC kernel-identity tuple** `(BackendId, op, dtypes, kernel_source,
  kernel_revision_hash)` — **no new identifier**, mapped directly onto your
  `{ Baracuda | Vendor | FuelNative }` via `kernel_source`. This is the same `kernel_source`
  axis the Judge already keys on (§2.2 above), so a telemetry record's impl id and the Judge's
  measurement key are the **same field**, by construction. (FKC §4.11.)
- **A "miss" is "the best admissible match at this key is a *generic* contract."** It falls out
  of FKC planner matching; no bolt-on detector. (FKC §4.2, §4.12.)
- **Negative strides are first-class**, keeping the `flipped` demand axis visible so a
  flip-specialized kernel's demand surfaces in the miss histogram instead of being normalized
  away. (FDX §3.2.1, FKC §4.1.1.)

---

## 5. What Fuel asks of Baracuda (mirror, carried forward)

Unchanged from the proposal's §5, restated so this reply stands alone:

1. **Adopt FDX as the Fuel-facing tensor description, standard DLPack as the ecosystem-facing
   one.** Minimum: versioned standard DLPack on the external boundary; accept a nullable
   `*const FDXSidecar` on the Fuel ABI. Review the struct shapes *before* FDX freezes (still
   DRAFT).
2. **Confirm `structure_key`'s input contract accepts FDX operand descriptions**, so Fuel never
   reimplements your key. (FDX §4.1.)
3. **Co-define and freeze the `ImplId` wire encoding** on the basis tuple `(BackendId, op,
   dtypes, kernel_source, kernel_revision_hash)`. The basis is settled; the wire bytes are
   joint. (FKC §4.11.)
4. **Register the negative-strides-first-class decision** and confirm your `OperandKey.flipped`
   derivation matches FDX's signed-stride description. (FDX §3.2.1.)
5. **Agree the miss signal is "best admissible match = generic contract"** so neither side
   builds a redundant detector. (FKC §3.3 / §4.12.)

These are the same five asks; the difference this reply makes is that ask (3)'s `kernel_source`
discriminant is now demonstrably the *live* Judge key, not just a spec proposal — strengthening
the "no second identity surface" argument with running code.

---

## 6. Process note (working-agreement framing)

This reply is **DRAFT, on branch `feat/kernel-contracts-dlpack` (unmerged; `main` untouched)**.
Nothing here has been written into any Baracuda repo. FDX and FKC remain DRAFT on the same Fuel
branch. The `ImplId` basis, the `structure_key`-input contract, and the telemetry-record shapes
are offered for Baracuda's review **before** either side freezes its half — consistent with
Fuel's rule that cross-project changes are proposed, never landed unilaterally on a sibling.

Next steps: Baracuda's review of the answers above; the joint `ImplId` wire-encoding freeze; and
building the emission layer over the now-confirmed retention per
[`docs/session-prompts/baracuda-telemetry-plan.md`](../session-prompts/baracuda-telemetry-plan.md).
The retention dependency that gated this reply is **closed**.

---

### References
- Companion proposal (resolves §6 of): [`baracuda-dlpack-fkc-ask.md`](baracuda-dlpack-fkc-ask.md).
- Judge retention — code (ground truth):
  - [`fuel-core-types/src/dispatch.rs:655-692`](../../fuel-core-types/src/dispatch.rs)
    (`ProfileEntry` / `ProfileReport`; `latency_ns: u64`; `kernel_source`),
    `:32` (`PROFILE_REPORT_VERSION == 2`), `:697-719` (`save`/`load`).
  - [`fuel-dispatch/src/ranker/judge.rs:53-75`](../../fuel-dispatch/src/ranker/judge.rs)
    (`JudgeOracle` trait + `HashMapJudge` five-axis key).
  - [`fuel-core/src/judge/oracle.rs:14-23, 65-92, 170-195`](../../fuel-core/src/judge/oracle.rs)
    (`ProfileJudgeOracle::from_report`; loser-retention rationale; the passing
    `sibling_kernel_sources_do_not_collide` test).
  - [`fuel-core/src/judge/mod.rs:476, 512, 730`](../../fuel-core/src/judge/mod.rs)
    (F32-only measurement-loop caveat).
- Boundary specs (DRAFT): [`docs/specs/dlpack-extension.md`](../specs/dlpack-extension.md) (FDX),
  [`docs/specs/kernel-contract-format.md`](../specs/kernel-contract-format.md) (FKC).
- Telemetry emission plan:
  [`docs/session-prompts/baracuda-telemetry-plan.md`](../session-prompts/baracuda-telemetry-plan.md).
