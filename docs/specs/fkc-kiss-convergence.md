# FKC ⇄ KISS-Contract convergence

**Status:** analysis + proposed plan, 2026-08-07. ⚠️⚠️ **CURRENCY PASS 2026-09-06, measured at `561ee255`. THE ANALYSIS HOLDS; THE STATUS LINE AND THE SEQUENCING DID NOT. Read this block before §5.**

> **WHAT SURVIVED, AND IT IS THE LOAD-BEARING CLAIM.** §0 says `docs/specs/kernel-contract-format.md` contains **zero** references to KISS. **Re-measured: still 0, a month later.** The doc supplied its own positive control and it still arms — the same query returns **1, 1, 12** in `fuel-dispatch/src/fkc/{mod,shape_expr,verify/ulp}.rs`. **The contract format has still had no reconciliation pass, which is the finding this document exists for.**
>
> ⚠️ **A CURRENCY PASS THAT REPORTS ONLY DECAY TEACHES A READER TO DISCOUNT THE WHOLE DOCUMENT.** The parts below that expired are administrative; the argument is intact.

> ⚠️⚠️ **FALSE AT HEAD: "no KISS RFC has been opened."** GAP-039 records **RFC 1 of 4 OPENED 2026-08-08 — KISS#124** — *the day after this document was written.* **§6 is therefore no longer purely a proposed ask: one quarter of it is live with an external party.**
>
> **The sentence was true for about twenty-four hours and has read as current ever since.** It is an ABSENCE CLAIM in a STATUS HEADER, which is the worst combination available: an absence needs re-measuring to stay true, and a header is the one line every reader takes on trust without checking.

> **§5 SEQUENCING, ROW-BY-ROW AT `561ee255`. Three of six entries have moved:**
>
> | § 5 item | row | state now |
> |---|---|---|
> | 1 | GAP-035 | **OPEN** — live Fuel work |
> | 2 | GAP-036 | ⚠️ **CLOSED, struck — BOTH INCREMENTS SHIPPED** |
> | 3 | GAP-037 | **OPEN but HELD**, on KISS#125 + the `revhash.rs` trap |
> | 4 | GAP-038 | ⚠️ **CLOSED by PR #130**, 2026-09-06 |
> | 5 | GAP-039 | **OPEN**, held on KISS — the RFC above |
> | prereq | GAP-048 | ⚠️⚠️ **a FOUR-COLUMN row with NO STATUS CELL** |
>
> ⚠️⚠️ **GAP-048 IS THE ONE THAT MATTERS AND IT IS NOT A STALENESS PROBLEM.** It is named here as a **hard blocker at the moment of declaring conformance** for item 1 — and its row has no status cell at all, so **IT CANNOT BE CLOSED, ONLY DELETED.** A prerequisite whose format forbids the outcome "done" is a permanent blocker on item 1 by construction, and nothing about reading §5 reveals that.

> **Not corrected in place below, deliberately.** The 2026-08-07 text is the record of what was believed when the plan was made, and rewriting it would destroy the evidence that item 2 and item 4 were *predicted* correctly and then *shipped*. **The block above is the verdict; the text below is the history.**

Fuel-side only. **§6 was the proposed ask when written; see the currency block — RFC 1 of 4 is now open.**
**Author:** Fuel architect/coordinator. **Registry:** GAP-035 … GAP-039.

---

## 0. The question, and the short answer

*Are Fuel and KISS in sync? Does KISS specify the kernel interface the same way Fuel does?
If not, which is better in which parts?*

**They are not in sync, and the split is unusually clean:**

- **The code is converging.** Three narrow places: the shape-expression oracle (Fuel
  **authored** `dims(...)` / `with_dim(...)`, now registered extensions in KISS-Ops §6.20),
  the ULP metric (repointed to `kiss-ref-core`), and dtype spelling (sk4, 3 of 6 cosigns).
- **The format specification has never been reconciled.** `docs/specs/kernel-contract-format.md`
  contains **zero** references to KISS. Positive control: the same query finds KISS in
  `fuel-dispatch/src/fkc/{mod,shape_expr,verify/ulp}.rs`, so the search reaches.

So convergence is happening bottom-up on *vocabulary*, through RFCs, while the *contract
format* — the thing that actually defines the kernel interface — has had no reconciliation
pass at all. FKC v1 was written independently and never repointed.

This is not a criticism of either document. FKC was written to solve Fuel's problem
(auto-registration onto a dispatch surface with honest cost) and it does. KISS was written
to solve the cross-vendor problem. They overlap by ~70% and each is better at what it was
built for. The work is to make the overlap explicit rather than parallel.

---

## 1. Structural comparison

KISS-Contract §6 defines a seven-section document. Mapping Fuel's FKC onto it:

| KISS-Contract §6 | Fuel FKC | Verdict |
|---|---|---|
| 6.3 Identity | §3.3 identity + `ImplId` §4.11 + `revision_hash` §4.7 | aligned |
| **6.4 Semantics** — resolvable op DAG to a primitive floor | **ABSENT** | **KISS materially better** |
| 6.5 Interface (full ABI in one place) | `entry_point` → `link_registry` → `KernelRef` (§12.6) | **divergent by design** |
| 6.6 Dispatch | §12.1 dispatch key → `(OpKind, KernelDTypes, BackendId)` | aligned |
| 6.7 Capabilities (incl. cost) | §4 — five-flag layouts, fast-path predicates, symbolic-extent tolerance, aliasing | **Fuel materially richer** |
| 6.8 Guarantees (precision, determinism, `audited_status`) | §4.8 precision, §4.9 determinism | **KISS better on `audited_status`** |
| 6.9 Provenance (who produced the kernel) | FKC `provenance` = *cost* provenance | **name collision, different concepts** |

**The §6.5 divergence is deliberate on both sides and should stay.** KISS specifies a wire
ABI because it crosses a vendor boundary; FKC resolves `entry_point` through a
`link_registry` to a `KernelRef` because it is registering in-process. These are different
problems and neither should adopt the other's answer wholesale. Convergence here means
*FKC declaring which ABI it implements*, not replacing its indirection.

**The §6.9 collision is a real hazard.** Both documents use the word `provenance` for
different things — KISS for *who produced the kernel* (a supply-chain claim), FKC for
*where the cost numbers came from*. A shared document that uses one word for two concepts
will be misread. Fuel's concept should be renamed `cost_provenance` on convergence, which
is what KISS already calls it (§6.8-0006).

---

## 2. Where Fuel is better, and why

1. **Cost.** Both have it — and both independently arrived at `declared` / `measured`
   provenance (Fuel spells it `judge_measured`). But Fuel's is a **vector** contribution
   (compute / bandwidth / overhead + per-tier memory), optionally symbolic, with two
   compile targets and a Judge bootstrap loop that **flips provenance when it refines**.
   KISS's is a class plus an expression. Fuel's optimizer needs the richer form, and the
   provenance-flip is a genuinely good idea nobody else has.
2. **Layout as a five-flag capability set**, not one bool (contiguous-only / strided /
   broadcast-via-stride-0 / non-zero-start-offset / in-place) — explicitly so the planner
   can price contiguize-vs-strided-vs-materialize honestly.
3. **Awkward-layout fixups are themselves FKC kernels** (§4.3). Self-consistent: the fixup
   is priced by the same mechanism as the work it enables, so a contract can never hide a
   decision behind "handled internally."
4. **Precision is a hard pre-filter applied *before* cost ranking** (G4). An ordering
   commitment KISS does not make, and the correct one — a kernel that cannot meet the
   precision requirement should never reach the cost comparison.
5. **Import = registration** (G5), zero hand-written glue. Operational rather than
   conceptual, but it is what makes a contract *load-bearing* instead of documentation.
   Fuel learned this the expensive way (see §3.2).

---

## 3. Where KISS is better, and why it matters more than it looks

### 3.1 Semantics as a mandatory, resolvable op DAG — the big one

Fuel's contract **names** an op (`op_kind` / `fused_op`); KISS's **carries its meaning**,
as a mixed-abstraction DAG in which every non-primitive node is resolvable down to a
KISS-Ops primitive floor, with an acyclic strictly-decreasing termination guarantee.

Naming suffices inside Fuel, which owns the registry. **Across a vendor boundary it does
not — and Fuel has already paid for this.** Baracuda's `rope_apply` is *interleaved*;
Fuel's `FusedOps::ROPE` is *rotate-half*. Same name, different function. It was caught only
by a numeric oracle, after a fused registration had to be reverted. A KISS-Contract §6.4
semantics DAG makes that a **contract-read-time** mismatch instead of a runtime one.

Fuel already has the raw material: `decompose` / `pattern` recipes, the base map, a
build-time-closed primitive basis, and the recipe-identity property (`base_map_hash`
equality). It simply is not in the contract.

### 3.2 Clause IDs mapped 1:1 to tests, with the build failing on an untested MUST

This is structurally the countermeasure to Fuel's single most expensive recurring defect:
**existence ≠ enforcement.** The FKC audit found ~24 gaps, mostly correctly-named
validators nobody invoked. GAP-016 (filed 2026-08-07) records ~612 `panic!` sites against a
"never panic on production paths" hard rule that had **never been counted**. GAP-141's own
scoping found the registry's founding case (GAP-001) sits in a class the first increment
does not even check.

**Fuel enforces by vigilance; KISS enforces by construction.** Adopting the clause-ID
discipline is the highest-return item in this document, and its return is *internal* —
it would pay for itself even if no cross-project interop ever happened.

### 3.3 `audited_status` is *derived*, with a single home

Fuel shipped 85 CUDA kernels carrying `audited: false` — never verified, nobody noticed,
for months. KISS makes `audited_status` a **derived** value (§6.8-0008/-0009/-0010) rather
than an authored field. A derived status cannot be fabricated by an author who is in a
hurry.

### 3.4 Two version axes

KISS mandates **wire/ABI schema version** and **published-crate semver** as independent
axes, with a per-change bump-rule table. FKC has one `fkc_version` (§11), conflating "the
bytes changed" with "the code changed." The two-axis split is strictly better and cheap to
adopt.

### 3.5 A real threat model

KISS §10 states plainly that it is a **code-distribution protocol** with no authentication,
no integrity, no sandboxing, and that `revision_hash` does **not** detect substitution. It
then names *Fuel's own* verify-before-adopt ledger as the pattern worth standardizing
(§10.5 item 6). FKC has no threat model at all, despite Fuel shipping the ingestion service
that accepts provider kernels and loads them.

Fuel is **ahead on the practice** and **absent on the statement**. That asymmetry is worth
closing in both directions: adopt the threat-model section, and upstream the ledger pattern.

---

## 4. The proposed shape

Not "Fuel adopts KISS wholesale" and not "KISS adopts FKC."

> **FKC v2 declares itself a KISS-Contract *profile*:** it takes the seven-section spine and
> the clause-mapped conformance discipline, and keeps Fuel's richer capability/cost
> vocabulary as **registered extensions** on the promotion path.

The extension registry already exists and Fuel already owns two entries (`dims`, `with_dim`),
so the mechanism is proven rather than hypothetical.

---

## 5. Sequencing (registry rows)

Ordered by return, not by size.

| # | Work | Row | Why here |
|---|---|---|---|
| 1 | **Add a Semantics section to FKC** carrying the recipe DAG | GAP-035 | Unblocks real Baracuda/Unpopped/Vulkane interop; the rope incident is the paid-for proof. Fuel already has the material. |
| 2 | **Adopt clause IDs + 1:1 conformance mapping** | GAP-036 | Highest internal return. Targets the defect class that keeps costing real money. Pairs with GAP-141. |
| 3 | **Derive `audited_status`; rename `provenance` → `cost_provenance`** | GAP-037 | Two small changes closing a proven failure (85 unaudited kernels) and a name collision. |
| 4 | **Two version axes + bump-rule table** | GAP-038 | Cheap, mechanical, prevents a class of skew. |
| 5 | **Upstream Fuel's better parts as KISS extensions** | GAP-039 | Five-flag layouts, cost-as-vector + provenance-flip, precision-before-cost ordering, fixups-as-kernels, the verify-before-adopt ledger (§10.5 item 6 explicitly invites it). |

**Item 5 SPLITS IN TWO, on a firsthand read of KISS-Classify §6.8 (2026-08-07).** The
capability-vocabulary half is **not an RFC at all**: §6.8-0004 makes each namespace's
capability-set vocabulary the *namespace maintainer's* to own, and KISS pins only grammar +
byte-exact matching. What Fuel needs there is a **namespace registration** (§6.8-0003 — a
merged PR to the ThinkersJournal registry, published as a versioned machine-readable file
bundled with the Conform suite), which is a smaller and much more likely-to-succeed ask.
The genuinely RFC-shaped parts are the ones touching KISS-Contract §6.7 itself: five-flag
layouts, cost-as-vector + the provenance-flip, precision-before-cost ordering,
fixups-as-kernels, and the verify-before-adopt ledger.

**A new prerequisite for item 1, from GAP-048.** Declaring FKC a KISS-Contract profile
implies KISS-Ops conformance, and Fuel is currently missing the second half of two §6.15
op pairs (`rem_trunc`; the IEEE-suppressing `fmax`/`fmin`). Not a live defect — nothing
computes a wrong answer — but a **hard blocker at the moment of declaring conformance**,
so it wants resolving (add the ops, or record a deliberate profile deviation) *before*
item 1 lands rather than after.

Items 1–4 are Fuel-internal and need no external agreement. **Item 5 is the only one
requiring a KISS RFC**, and it is deliberately last: Fuel should arrive with its own house
in order, and with the extensions already implemented, since KISS's promotion path requires
≥2 dissimilar implementations plus a conformance test before a core promotion anyway.

---

## 6. Why KISS is the right substrate for this family

The project family maps onto KISS's roles almost exactly:

| Project | KISS role |
|---|---|
| Lightbulb | consumer |
| Fuel | consumer **and** provider (it dispatches, and it ingests provider kernels) |
| Baracuda, Vulkane | providers |
| Unpopped | provider-side vocabulary + emitter |
| kiss-ref | the "second dissimilar implementation" the freeze gate (umbrella §5.3) demands |

That last row matters more than it reads: KISS cannot freeze a sub-standard without two
structurally dissimilar implementations that do not share lowering code. Fuel + kiss-ref
already satisfy that shape for the parts Fuel implements. **Fuel is not merely a consumer of
this standard; it is one of the two legs that lets the standard freeze at all.**

---

## 7. Stated limits of this analysis

- Read: KISS `spec/umbrella.md` in full, `spec/contract.md` section structure + §2.3/§6.7/§6.8
  clauses, **`spec/conform.md` §6.1–§6.2 firsthand (added 2026-08-07)**, Fuel
  `docs/specs/kernel-contract-format.md` structure + §1/§11, and
  `docs/kernel-contracts/README.md`. **Not** read in full: `spec/{classify,ops,grammar,
  announce,synth,consume,emit}.md`.
- **§3.2 is no longer second-hand and it UNDERSTATED the requirement.** KISS-Conform
  mandates bidirectional totality (both directions build-gating), a matrix that is
  **derived and may not be hand-authored** (§6.1-0004), a **generate-time** gate that makes
  an under-covering suite unbuildable rather than merely failing (§6.2-0005), and a hard
  separation of structural gate from run-time verdict (§6.2-0006). Fuel's GAP-141 gates
  are run-time checks of a structural property — functional, but the weaker shape.
- **KISS-Ops and KISS-Classify are no longer second-hand either (read 2026-08-07):** KISS-Ops
  §6.3 (primitive floor), §6.14 (termination) and §6.15 (non-mergeable distinctions);
  KISS-Classify §6.8 (target-capability descriptor). Still second-hand: `spec/{grammar,
  announce,synth,consume,emit}.md`.
- **§3.1 VERIFIED unchanged.** The primitive floor is real and exhaustively enumerated
  (§6.3-0001), and the termination guarantee is stronger than I paraphrased: integer levels
  with `level(floor)=0`, a mandated-acyclic decomposition graph, and strict decrease along
  every edge (§6.14-0001…0003), plus §6.14-0006 closing the subtle case where an op is
  embedded in a structural atom's scalar body.
- **The capability caveat is RESOLVED, and it went the OTHER WAY from the direction I
  hedged toward.** I guessed the gap might be *smaller* because some capability vocabulary
  might live in KISS-Classify. It does not, and the reason is a deliberate design choice:
  **KISS-Classify §6.8-0004 says KISS pins only the token grammar and the byte-exact match
  rule and MUST NOT pin any namespace's capability-set vocabulary**, which "freezes
  independently". A `target_capability` is `<namespace>:<capability-set>` matched byte-exact
  with no subset/prefix/implication logic (§6.8-0002). So the verdict stands and is
  *reinforced* — KISS declines to own this vocabulary by design.
- **This changes the SHAPE of item 5, not just its size** — see §5.
