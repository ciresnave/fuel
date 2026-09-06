# Status-vocabulary conversion — the decision table

**What this is.** The record of every judgment made converting `docs/gaps.md`'s
status cells to the controlled prefix set in
[`gaps-status-vocabulary.md`](gaps-status-vocabulary.md). It lands with the
conversion because it is the first thing anyone will ask for when a prefix looks
wrong.

⚠️ **THIS BUYS GREPPABILITY, NOT ACCURACY.** Not one defect found in the
2026-08-28 close-convention audit would have been caught by a prefix. Cite this
file as *"N rows carry prefix X"* and never as *"N gaps are verified closed"*.

## The two rules the conversion followed

**1. It changed the prefix FORM, never the VERDICT.** Where a row's lead was
already a set member it was left alone — *even where the body appears to
contradict it*. Re-verdicting a row is a **disposition** change, the same
reasoning that ruled out striking rows during a formatting pass. Those
disagreements are reported, not silently fixed.

**2. The prefix is PREPENDED; the original cell is preserved verbatim.** So
`**PARTIAL** — **FIXED** — one Vulkan-gated pickup remains` is mildly redundant
and loses nothing. The old vocabulary is also the record of what the row said
before the pass.

⚠️ **Why a lead-only mapping was rejected.** `GAP-095` reads
`FIXED** — one Vulkan-gated pickup remains`. Mapping its leading token stamps it
`CLOSED`, meaning *nothing remains*, in a cell that names what remains nine words
later — a false greppable head, worse than the bespoke form it replaced, because
`grep -c '| CLOSED'` then counts it as done. **The lead does not determine the
prefix, because the cell keeps talking.** Measured: 20 of the 54 closed-leading
rows carry remainder language, and the defect runs in **both** directions —
`GAP-214` and `GAP-223` lead `OPEN` with bodies that say `✅ CLOSED`.

## The table

| GAP | prefix | why | cell lead before conversion |
|---|---|---|---|
| `GAP-222` | `CLOSED` | FIXED; no remainder named. | ✅ FIXED `00b8974d` (`CpuInvoker::with_seeded_output` + an explicit |
| `GAP-224` | `CLOSED` | SHIPPED, both halves, first caller wired. | BOTH HALVES SHIPPED `ccf677d9` — `fuel_test_support::require_gpu_r |
| `GAP-019` | `PARTIAL` | RESOLVED upstream, but the Fuel-side mitigation is pending. | RESOLVED (mitigation pending in Fuel)  |
| `GAP-012` | `CLOSED` | FIXED. | FIXED (waste only)  |
| `GAP-015` | `CLOSED` | FIXED, gated and born-red verified. | FIXED — built, gated on the reproducer, born-red verified.  |
| `GAP-020` | `PARTIAL` | Hold released and two coordinates done, but the judge-cache half is stated UNVERIFIED and the row is deliberately kept. | ⚠️ THE HOLD OUTLIVED ITS CONDITION BY TWELVE DAYS, AND HALF ITS SUBJ |
| `GAP-032` | `OPEN` | HELD -> OPEN; suffix optional on conversion. | HELD  |
| `GAP-036` | `CLOSED` | SHIPPED, both increments. Struck; consistent. | BOTH INCREMENTS SHIPPED Child row: GAP-251 — filed to home thi |
| `GAP-046` | `WON'T DO` | DECLINED -> WON'T DO. | DECLINED — with reason, not omitted  |
| `GAP-243` | `PARTIAL` | 44 of 49 sites adopted; 5 remain. | MOSTLY LANDED — 44 of 49 sites adopted; the CUDA sweep is VERIFIED. |
| `GAP-059` | `CLOSED` | FIXED; no remainder named. | FIXED  |
| `GAP-075` | `CLOSED` | FIXED; no remainder named. | FIXED  |
| `GAP-076` | `CLOSED` | FIXED. Struck; consistent. | FIXED  |
| `GAP-095` | `PARTIAL` | Lead says FIXED and the same cell names what remains. | FIXED — one Vulkan-gated pickup remains  |
| `GAP-096` | `CLOSED` | ANSWERED -> CLOSED. | ANSWERED — DERIVED (CireSnave 2026-08-15); `fill_unset_cpu_precisi |
| `GAP-231` | `PARTIAL` | CLOSED for paths/lines/commands, OPEN for counts. | Swept and gated by `nduiwcsu`. CLOSED for paths / line numbers / |
| `GAP-148` | `OPEN` | ASSENTED -> OPEN per the Absorbs table. | ASSENTED — pin complete — Fuel-side latent issue OPEN (see GAP-076 |
| `GAP-158` | `CLOSED` | RESOLVED as not-a-defect. | RESOLVED — NOT-A-DEFECT  |
| `GAP-162` | `OPEN` | SCHEDULED -> OPEN; suffix optional. | SCHEDULED  |
| `GAP-163` | `CLOSED` | DELIVERED -> CLOSED. | DELIVERED  |
| `GAP-164` | `CLOSED` | FIXED. | FIXED  |
| `GAP-166` | `PARTIAL` | Unit A closed, Unit B deferred behind an MLA port. | UNIT A CLOSED `2699fbad` — geometry honest, `collapse_uniform` LIV |
| `GAP-168` | `PARTIAL` | MERGED, and the cell names a remaining op-family sub-scope. | `Bool` CUT MERGED `7236e76e` — 4 backends green, all owed items di |
| `GAP-176` | `CLOSED` | The cross-check FIRED here on its first run against real data, which is how this row got read at all. Both residues are homed: (C) at GAP-185, (D) at GAP-258 filed 2026-09-02. By the homed-residue test the closure is genuine, so THE STRIKE WAS ALWAYS CORRECT and only the lead was stale. Ruled by the architect. Sequencing matters: filing (D) BEFORE converting is what makes CLOSED defensible - converting first would have given PARTIAL plus unstrike, re-opening finished work. | POPULATION A CLOSED `c0b5372e` (102 -> 0); aarch64 population CLOSED |
| `GAP-180` | `OPEN` | FIX UNDECIDED -> OPEN per the Absorbs table. | MECHANISM IDENTIFIED, FIX UNDECIDED — lockfile TRACKED `70334bb9` (jus |
| `GAP-194` | `PARTIAL` | Correctness half closed; CUDA confirmation UNVERIFIED. | CORRECTNESS HALF CLOSED `e785188e`; WIRING `c270a957` (CPU-inert,  |
| `GAP-196` | `CLOSED` | Doc corrected, field renamed, semantics test-locked. | DOC FIXED — module doc `:5` corrected to the true semantic (`1` =  |
| `GAP-197` | `PARTIAL` | HANDLED for four sites; OPEN as a standing check. | HANDLED `c4b38ed9` for the four sites; OPEN as a standing check |
| `GAP-199` | `WON'T DO` | RULED - DELETE -> WON'T DO per the Absorbs table. | RULED — DELETE the 5 rubato examples (lockfile discipline: the dep |
| `GAP-200` | `CLOSED` | All three layers accounted for; the trailing warning is a CAUTION, not residue. Architect confirmed layer 2 landed. Strike correct, lead stale. | LAYER 1 CLOSED `2537d941` (resolution works, verified in a cold `C |
| `GAP-201` | `PARTIAL` | FIXED, and OPEN as the structural question. | FIXED (pushing with GAP-200 layer 2); OPEN as the structural que |
| `GAP-205` | `PARTIAL` | Panic fixed and hardening done; 2 semantics items documented-not-fixed and deferred. | PANIC FIXED `abf478e5`; HARDENING DONE `44193322` — first tests ev |
| `GAP-208` | `PARTIAL` | Vulkan restored; CPU-fused (77) and CUDA (142) still UNAUDITED. | VULKAN SIX RESTORED `6ed74f54` (24 records under `[T, Bool, T]`, v |
| `GAP-216` | `PARTIAL` | "(a) CLOSED; (b) OPEN" -> PARTIAL per the Absorbs table. | (a) CLOSED — dead, delete; (b) OPEN, SEVERITY A — shape-gate the c |
| `GAP-219` | `CLOSED` | Fixed, and the follow-on sweep came back clean. | FIXED `5691f723`  ✅ FOLLOW-ON SWEEP DONE 2026-08-20 AND IT CAME  |
| `GAP-228` | `PARTIAL` | The (a) half done; the type change unallocated (GAP-291). | DONE — the (a) half; `audited: bool` still cannot express the dist |
| `GAP-228` | `PARTIAL` | Increment row; 60 remain. | DONE — 60 of the original 320 remain: attention 40, SSM 8, fused 8 |
| `GAP-228` | `PARTIAL` | Increment row; 40 remain. | DONE — 40 remain, ENTIRELY attention (FlashAttn / BackwardK / Back |
| `GAP-228` | `PARTIAL` | Increment row; 32 remain. | DONE — 32 remain: FlashAttnBackwardK/Q/V 24 (partial mirror, await |
| `GAP-228` | `PARTIAL` | Increment row; 8 remain. | DONE — 8 remain: PagedAttn only, separate surface by ruling (q + 2 |
| `GAP-228` | `CLOSED` | Final increment: CPU UNAUDITED 320 -> 0. | DONE — CPU UNAUDITED 320 -> 0; 623/623 backed without the fill  |

## Follow-on: the 4 Meta rows, converted once the arity fix made them visible

⚠️ **These were never exempt on their merits — they were INVISIBLE to the
vocabulary check.** It skips any row without exactly 5 cells, and these had 4
because an **Area** cell was missing on the *left*, which shifted every later
field. Repairing the arity did not change their status; it made their status
**readable**. **A reviewer seeing four new vocabulary rows appear in an arity
commit must not read it as the arity fix breaking something** — it is the same
shape as `~~GAP-176~~` becoming legible to the strikethrough cross-check.

| GAP | prefix | why | cell lead before conversion |
|---|---|---|---|
| `GAP-141` | `PARTIAL` | INC-1 DONE -> PARTIAL per the Absorbs table; increment 1 landed, increment 2 scheduled. | INC-1 DONE @ `999f5b67` (`fuel-ir/tests/gap_refs.rs`, runs under t |
| `GAP-142` | `OPEN/standing` | STANDING -> OPEN/standing. The suffix is SOURCED from the row, not invented: a process rule is permanently open by design and a flat OPEN would read as unfinished work forever. | STANDING  |
| `GAP-143` | `OPEN/standing` | As GAP-142. | STANDING  |
| `GAP-144` | `OPEN/standing` | As GAP-142. | STANDING  |
