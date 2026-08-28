# A controlled status vocabulary for `docs/gaps.md`

**Status**: PROPOSAL, 2026-08-28. Not applied. Scoped on the architect's allocation after a
close-convention audit read every status cell in the file.

---

## Read this before the proposal: what it buys, and what it does not

**This convention buys GREPPABILITY, not ACCURACY.**

**Not one defect found during the 2026-08-28 audit would have been caught by it.** `GAP-198`'s
`RE-OPENED` was present and correct. All eight partially-closed rows stated their remainder
accurately in their leading phrase. `GAP-186`'s un-homed residual sat in a cell that correctly said
`CLOSED`. Every real finding that night came from **reading a row**, and no prefix set substitutes
for that.

It is still worth doing, for a measurable reason rather than an aesthetic one: **"how many gaps are
open" blocked a report to the project owner three times in one night**, because the answer depends
on a construct the table does not carry. A controlled prefix makes that question answerable by
`grep`.

**A convention sold as hygiene and read as verification is the defect this registry keeps
producing.** Whoever cites this file must be able to say *"N rows carry prefix X"* and must never
say *"N gaps are verified closed"* on the strength of it.

---

## The measured starting point

At `ea032a60`. **Every number here comes from one method** — a row takes the schema of the nearest
`| ID |` header *in its own table*, and a non-pipe line ends the table.

| population | rows |
|---|---|
| total | **253** |
| 5-column (`ID \| … \| Gap \| Status`) — **inside the convention** | 141 |
| 4-column (`ID \| File:Line \| Tier \| Gap`) — no status cell | 92 |
| **no `\| ID \|` header above them at all — schema undetermined** | 20 |

⚠️ **That last row is a finding, not a rounding detail.** Twenty GAP rows sit in table fragments
introduced by a `---` rule with no header. The hook's header check reports *"headers DISAGREEING
with their rows: NONE"* and **cannot see them — there is no header to disagree with.** They were
found only because two schema-counting methods disagreed 94 vs 92: the looser one silently
attributed them to a *different* table's header several lines away. **Two constructs disagreeing was
the only detector.**

Of the 141 rows that have a status cell, **125 lead with a recognisable verdict token**:

```
OPEN 73 · CLOSED 27 · FIXED 6 · DONE 6 · STANDING 3 · MEASURED 2 · RESOLVED 2
HELD 1 · DECLINED 1 · ASSENTED 1 · SCHEDULED 1 · DELIVERED 1 · RE-OPENED 1
```

The other **16** open with a bespoke phrase. These are the full set, and they are why a two-word
collapse would destroy information:

`BOTH HALVES SHIPPED` · `BOTH INCREMENTS SHIPPED` · `INC-1 DONE` · `UNIT A CLOSED` ·
`LAYER 1 CLOSED` · `POPULATION A CLOSED` · `CORRECTNESS HALF CLOSED` · `CUT MERGED` ·
`DOC FIXED` · `PANIC FIXED; HARDENING DONE` · `VULKAN SIX RESTORED` · `HANDLED … OPEN as a
standing check` · `RULED — DELETE …` · `MECHANISM IDENTIFIED, FIX UNDECIDED` ·
`THE HOLD OUTLIVED ITS CONDITION BY TWELVE DAYS` · `(a) CLOSED; (b) OPEN`

---

## The proposed set

Seven prefixes. The prefix is the **greppable head**; free prose follows and keeps everything the
current forms carry. **Nothing is lost in conversion** — that is why this is not a collapse to two
words.

| Prefix | Means | Absorbs | Strikethrough |
|---|---|---|---|
| `CLOSED` | nothing remains | FIXED, DONE, RESOLVED, DELIVERED, SHIPPED, MERGED, RESTORED, HANDLED, ANSWERED | **yes** |
| `PARTIAL` | some remains, **and the prose says what** | PARTIALLY CLOSED, UNIT A CLOSED, LAYER 1 CLOSED, POPULATION A CLOSED, HALF CLOSED, INC-1 DONE, `(a) CLOSED; (b) OPEN`, `CLOSED for X … OPEN for Y` | no |
| `OPEN/<state>` | work outstanding; state is required | OPEN, HELD, STANDING, SCHEDULED, ASSENTED, `FIX UNDECIDED` | no |
| `RE-OPENED` | was closed, is not; the history matters | RE-OPENED, NOT CLOSED | no |
| `WON'T DO` | decided against | WON'T DO, DECLINED, `RULED — DELETE …` | **yes** |
| `VOID` | the row's **premise** is gone | MISFILED, SUBJECT DELETED | **yes** |
| `MEASURED` | the row's output **is** a measurement; nothing fixed, nothing owed | MEASURED | **yes** |

### Why `OPEN` takes a required second word

`OPEN` is already **73 rows — the largest and least differentiated bucket in the file.** Folding
`HELD`, `STANDING` and `SCHEDULED` into it makes the commonest prefix the least informative, and
those are not synonyms for open:

- `OPEN/held` — blocked on something external; **not available for work**.
- `OPEN/standing` — permanently open **by design**; a standing check that never closes. Under a
  flat `OPEN` it reads as unfinished work forever.
- `OPEN/scheduled` — allocated, with an owner.
- `OPEN/unallocated` — open and nobody has it. The default.

One greppable head, four states preserved. `grep -c '| OPEN/'` still answers the question that
motivated this.

### Why `MEASURED` is its own prefix and not `CLOSED`

Rows exist whose deliverable was a number, not a change. Nothing was fixed, so `CLOSED` overstates;
nothing is owed, so `OPEN` is wrong. It takes strikethrough because the work is done.

---

## The rows outside the convention: exempt by SCHEMA, and count the exemption

**The convention applies to 5-column tables. 4-column tables fold status into the `Gap` cell and are
outside it.**

Exempt **by schema, never by a named list** — a named list rots and gets appended to, while a schema
rule is re-derivable at every commit.

⚠️ **The exemption must be COUNTED, not merely stated.** `scripts/check-gaps-table.py` now prints,
on every commit:

```
rows OUTSIDE the status convention (4-col, no status cell): 92 of 253
rows with NO HEADER above them (schema undetermined):       20
```

**112 of 253 rows — 44% — are outside the convention.** A convention covering 56% of a file,
described in prose as covering the file, is the *green-reads-as-coverage* failure — the same one
both doc-vs-code guards had to close in their own scope statements. **Making the hole visible is this
proposal's deliverable; closing it is a separate decision nobody has taken**, and it would be a
92-row mechanical edit against a file whose delimiter handling has already produced two false
measurements in one night.

The **20 headerless rows** are a smaller and more tractable fix: give those fragments a header. That
would also restore the hook's header check over them, which currently reports clean because it
cannot see them.

---

## Enforceability, stated plainly

**The hook CAN check:**
- a 5-column row's status cell **starts with** a member of the set — exact, mechanical, no judgment;
- **the strikethrough cross-check: a struck row whose prefix is `PARTIAL` or `OPEN/*` is a
  contradiction.** This is the one genuine accuracy test available here, and it exists only because
  the strikethrough mapping above is fixed by construction rather than left to convention;
- the distribution of leading tokens, printed every commit, so **a new form appears as a number that
  moved** rather than as prose nobody re-reads.

**The hook CANNOT check:**
- whether the prefix is **true**. See the first section: this is the whole limitation.
- the 92 statusless rows, or the 20 headerless ones.

---

## Ship condition

**The set and the hook check land in the SAME change, or neither lands.**

An unenforced convention in this file reverts to its prior state; that is observed behaviour, not a
risk estimate. The worked example is from the same night this was scoped:
`BARACUDA_FORGE_THREADS` was mandated in `CLAUDE.md` for weeks, and the `cargo check --workspace`
path ignored it the entire time with nobody being careless — because `baracuda-kernels-sys` forges
unconditionally, so a workspace check is a CUDA build without saying so. The fix that worked was
`[env]` in `.cargo/config.toml`, not a clearer rule.

> **A convention that only a careful reader honours is one the careless PATH defeats, not the
> careless person.**

---

## Not in scope

- **Conversion of the 125 leading tokens.** A fresh-session job with a clear spec.
- **Adding a status column to the nine 4-column tables.** Named above as an open decision.
- **Giving the 20 headerless fragments a header.** Smaller, and it restores an existing check.
- **The 4 Meta rows that carry four fields instead of five** — they omit the Area, which shifts
  every later field left. Left untouched pending a look; they may be the interesting ones.
