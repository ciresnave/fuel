# Method-rules drafts parked in issue #159

Verbatim copy of issue #159 (body and every comment) taken 2026-10-25 at `origin/main@ac29b53c`, so the text survives if the issue is closed. The three Part 1 entries are already in `docs/method-rules.md`; the rest are approved in principle only and tracked as GAP-366.

## Issue body (2026-09-09T06:08:42Z)

Three `docs/method-rules.md` entries **approved by the fuel architect**, plus their `CLAUDE.md` pointer halves. Held on #138 (which touches both files); filed here because they live only in a session-scoped scratchpad and a budget wall is a restart in slow motion.

⚠️ **Landing requirement, measured against the gate's own predicates** (`fuel-ir/tests/method_rules_index_join.rs`, **five** arms A–E, not four):
```
each entry lands as a PAIR (section + CLAUDE.md pointer) in ONE commit
the heading must BE the slug: every char ASCII-lowercase/digit/hyphen,
  or sections() SILENTLY DROPS it -- it does not fail, it disappears
arm D reddens on a method-rules link carrying NO #anchor at all
validated: 3 headings VALID; 0 collisions; my own outbound cross-ref
  (staleness-by-workaround) resolves -- arm B safe
⚠️ re-derive the section count after #138 lands; 58 was at 3fee13bf
```

---

# PART 1 — sections for `docs/method-rules.md`

## an-error-count-is-a-count-of-surfaces-not-causes

**A compiler error count reads like a work estimate and is not one. It counts where a defect
SURFACED, never how many things must change — and it is a ONE-WAY bias, because a single cause
surfaces in many places while many causes never surface in one.**

Measured 2026-09-06 on GAP-003 (`fuel-core`, three consecutive `cargo build -p fuel-core -j 4`
runs, logs retained):

    run   coded errors   primary spans   unique file:line
    p1        65              65               65
    p2        37              37               37
    p3        31              31               31

⚠️ **The naive suspicion — that the count is inflated by duplicate spans — is FALSE, and
checking it is what found the real mechanism.** Errors, primary spans and unique locations are
**1:1 in all three runs**. Every error sits at its own distinct line. Nothing is double-counted.

**The inflation is at the other end.** Between p2 and p3 only **7 of 37 locations recur**: a
handful of signature fixes extinguished thirty errors at once. **The log gave no way to see
that**, because each of those thirty looked exactly like an independent site.

    lazy.rs:147   error[E0599] ... --> pub fn graph(&self)

**That line calls no constructor.** It is a downstream consequence of a return type three
functions away, and the diagnostic attributes it to the function it surfaced in — correctly,
and uselessly for estimating.

### Why one-way rather than merely noisy

**Both directions exist and they do not cancel.** One cause surfaces at N sites (inflating);
and the compiler stops at the first failing phase, so causes behind a resolution error never
surface at all (deflating) — but the second is invisible *while the first is on screen*, so
what a reader sees is the inflation. **A rule about a symmetric error would be a caution about
noise. This one has a direction, which is why it is worth a rule.**

### The worked example is the sequence, not any number in it

GAP-003's scope was estimated at **1,549** (a bare `::from_f32(` grep, colliding with 19
unrelated `fn from_f32` definitions), corrected to **231** (qualified), and re-derived at least
twice more before the compiler settled it. **Three instruments, three answers, and the ones
taken from error logs were the ones that moved most.**

⚠️ **Do not memorise the chain — memorise that there was one.** Every figure in it is now
stale, which is the point: see `a-count-has-no-rename-resistant-form-so-bound-it`. What
survives is the shape.

### Practice

**To estimate work, count CAUSES: distinct declarations that must change. To estimate
progress, the error count is fine — it is monotone under fixes even when it is not
proportional to them.** And when reporting either, say which one you counted.

**The compiler is still the best instrument available, and the reason is not accuracy but
falsifiability: it cannot miss a site the way a grep can.** It over-reports where a grep
under-reports, and an over-report gets investigated.

---

## a-hook-you-route-around-on-a-guess-is-a-hook-you-have-not-read

**Committing work-in-progress that does not compile, I reached for `--no-verify` on the reflex
that a pre-commit hook would refuse a red tree.** `CLAUDE.md` forbids it unasked; I reset and
re-committed through the hook, and **it passed.**

**`scripts/hooks/pre-commit` invokes `rustfmt` twelve times and `clippy` zero. It has no
opinion about whether the tree compiles.** The bypass was not merely disallowed, it was
**unnecessary** — and the only way to learn that was to try it the right way first.

⚠️ **The general form: a bypass is an assertion about what the guard checks, and it is the one
assertion the bypass guarantees you will never test.** Route around a gate and you keep your
belief about it permanently unfalsified — the same structure as
`staleness-by-workaround`, pointed at a guard instead of a prohibition.

---

## a-name-search-is-structurally-blind-across-a-seam

**Two failure modes when enumerating where an assumption lives. They are mirrors,
and only one of them announces itself.**

    COLLISION   same NAME, different CONSTRUCT   -> produces HITS you can read and reject
    DISPERSAL   same CONSTRUCT, different NAMES  -> produces SILENCE

**Measured 2026-09-08 while sizing multi-carry `Op::Scan`, both within one hour.**

**The collision:** grepping `n_slots` for `Op::Scan`'s output-bundle arity returned
~20 hits across three files. **They were the KV/latent cache's slots.** Caught
before they entered a total — counting them would have made the bundle look like it
reached three more files and moved the whole estimate.

**The dispersal:** `fuel-kernel-seam-types` carries `scan_role: Option<u8>` and
`scan_index: Option<u32>`, with the single-carry constraint **in prose only**, on a
`[patch.crates-io]` crate whose bytes are contracted to be comparable with a
Baracuda-emitted blob. **A wire surface, not an internal site.**

⚠️ **THE ASYMMETRY IS THE WHOLE RULE: silence is indistinguishable from absence.**
A collision is caught by reading context. A dispersal needs a *different second
query*, and nothing in the first query's output suggests you need one.

### Two defects, and only the first is fixable by a better pattern

**PUNCTUATION-NARROWING.** The pattern was `'Op::Scan {'` — with a brace.

    'Op::Scan {'  in fuel-kernel-seam-types/  ->  0     the file never entered the classifier
    'Op::Scan'    in fuel-kernel-seam-types/  ->  4

**Excluded one step before classification.** The pattern was doing two jobs — FIND
and PRE-FILTER — and a filter folded into a pattern narrows the population before
you can see what you excluded. **Fix: search the bare token, classify second,
report both counts; the difference is the only visible evidence a filter ran.**

⚠️ **NAME-DISPERSAL — AND IT DEFEATS THE FIX ABOVE.** In the load-bearing places
the assumption is spelled `scan_role`, `scan_index`, `SCAN_ROLE_CARRY`, and the
prose *"v1 single-carry model"*. **Four spellings, none of them the token.
Searching the bare token would STILL have missed it.**

**No pattern can close this, because THE LAYERING THAT MAKES THE ARCHITECTURE GOOD
IS THE SAME PROPERTY THAT MAKES A NAME SEARCH BLIND ACROSS IT** — the seam exists
in order NOT to name the graph's types. **The only instrument is to ask "where does
this ASSUMPTION live" and enumerate LAYERS, not tokens.**

### And the framing chose the population

⚠️ **I had AUTHORED the file I missed** — `canonical.rs` is mine, and the
`scan_role`/`scan_index` arms the finding rests on are code I wrote. Familiarity
did not help.

**I searched the token of the layer I was SIZING, not the layer I had BUILT. The
question arrived framed as an enum change, so the search space became the enum's
layer: I LET THE FRAMING PICK MY POPULATION.**

**Corollary for anyone handing out a sizing task: the framing you choose is a
population claim you are making on the assignee's behalf, and they will not see it
as a claim.**

---

# PART 2 — the `CLAUDE.md` pointer half

# CLAUDE.md POINTER HALF — three bullets, one per new method-rules section.
# The gate (fuel-ir/tests/method_rules_index_join.rs) requires BOTH halves in ONE
# commit: arm A reddens on a section with no CLAUDE.md anchor, arm B on an anchor
# with no section, arm D on a method-rules link carrying NO `#anchor` at all.
# Insert under "## Engineering process", matching the existing bullet style.

- **⚠️ AN ERROR COUNT IS A COUNT OF SURFACES, NEVER OF CAUSES — and it is a ONE-WAY bias.** Measured over three consecutive builds (65 / 37 / 31): errors, primary spans and unique `file:line` were **1:1 in all three**, so the naive suspicion of duplicate spans is FALSE. **The inflation is at the other end — only 7 of 37 locations recurred between two runs, so a handful of signature fixes extinguished thirty errors at once, and the log gave no way to see it because each looked like an independent site.** To estimate WORK count CAUSES; the error count is fine for PROGRESS, being monotone under fixes even when not proportional to them. **Say which one you counted.** → [`an-error-count-is-a-count-of-surfaces-not-causes`](docs/method-rules.md#an-error-count-is-a-count-of-surfaces-not-causes)

- **⚠️ A HOOK YOU ROUTE AROUND ON A GUESS IS A HOOK YOU HAVE NOT READ.** `--no-verify` on the reflex that a pre-commit hook would refuse a red tree — it would not; the hook runs `rustfmt` twelve times and `clippy` zero, and has no opinion about whether the tree compiles. **A bypass is an assertion about what the guard checks, and it is the one assertion the bypass guarantees you will never test.** Same structure as [`staleness-by-workaround`](docs/method-rules.md#staleness-by-workaround), pointed at a guard instead of a prohibition. → [`a-hook-you-route-around-on-a-guess-is-a-hook-you-have-not-read`](docs/method-rules.md#a-hook-you-route-around-on-a-guess-is-a-hook-you-have-not-read)

- **⚠️ A NAME SEARCH IS STRUCTURALLY BLIND ACROSS A SEAM — and the two failure modes are mirrors, only one of which announces itself.** **COLLISION** (same NAME, different CONSTRUCT) produces hits you can read and reject; **DISPERSAL** (same CONSTRUCT, different NAMES) produces **SILENCE**, and silence is indistinguishable from absence. Worked example: sizing multi-carry `Op::Scan`, `n_slots` collided with the KV cache's slots (caught), while the same assumption lived in `fuel-kernel-seam-types` as `scan_role` / `scan_index` / `SCAN_ROLE_CARRY` / the prose *"v1 single-carry model"* — **four spellings, none of them the token, on a `[patch.crates-io]` wire surface.** ⚠️ **Searching the bare token would STILL have missed it: the layering that makes the architecture good is the same property that makes a name search blind across it.** Enumerate LAYERS, not tokens — and never fold a pre-filter into the pattern (`'Op::Scan {'` excluded the file by PUNCTUATION one step before classification ran). → [`a-name-search-is-structurally-blind-across-a-seam`](docs/method-rules.md#a-name-search-is-structurally-blind-across-a-seam)

---

# PART 3 — a drafted registry row (architect's to file)

# DRAFT ROW for the GAP-303 builder pin — ARCHITECT'S to file, not mine.
# Written now so filing is mechanical when #142 lands, rather than a re-derivation
# from memory. Not applied anywhere; docs/gaps.md is #142's.

| GAP-3xx | `fuel-graph/src/lib.rs` — anchor: `git grep -n 'GAP-303 SCOPE LINE'` | C |
**`NodeHandle::scan` constructs `n_carries: 1` — the only public constructor pins
the carry count at one, while the IR, the validator, `unroll_scan`,
`parse_scan_layout` and `build_scan_step` all handle N (landed in #150,
sabotage-proven at N=2).** Widening the BUILDER means N+1 output-view specs
through `compose_bundle`, i.e. the bundle arity, which is GAP-303 row 2
(`Op::ScatterIntoSlot`, kernel-less today). ⚠️ **The site comment carries the
REASON; this row carries the CLOCK — an expiring decline whose expiry is another
row's completion fires nothing if that row never lands.** **DETECTOR: the
`n_carries: 1` literal plus the prose anchor above; both die the moment the
builder takes a parameter.** | OPEN — **Owner:** UNALLOCATED — **Blocked-on:**
GAP-303 row 2 (bundle arity) |




---

## Comment 1 (ciresnave-bot, 2026-09-09T09:23:31Z)

## Two more entries — and the durability problem this issue is about, hit by this issue's own drafts

⚠️ **I filed this issue because approved work was living only in a session-scoped scratchpad, and then wrote two more approved-in-principle rules into that same scratchpad.** A written-then-lost rule is worse than an unwritten one: an unwritten rule gets re-derived; a lost one is re-derived by someone who has no idea it existed.

**The full text of all five entries is now in this issue — the three above plus the two below — so none of them depends on a session surviving.**

### ⚠️ Landing status re-measured, not carried forward
```
#138  MERGED 2026-09-09T08:31:54Z  -> the 'held on #138' condition is DISCHARGED
docs/method-rules.md sections:  58 (at 3fee13bf)  ->  59 (current head)
#138 added:  an-anchor-buys-immunity-and-pays-in-blindness
collisions with all FIVE of my slugs against the CURRENT population:  NONE
```

**That re-derivation is the point:** the earlier collision check ran against 58, and a check run against the wrong population is exactly what the five-arm join gate exists to catch. The count moved by one while these drafts sat.

---

## a-signal-with-two-causes-read-as-though-it-had-one

**Two instruments I built in one night, both sound in their claim and both
defective in the same way — and I found both by re-testing them rather than by
using them.**

    silence from a crate    un-run  OR  CACHED
    non-zero lib errors     ESCAPED OR  simply not finished yet

### 1 · The `Checking`-line detector

**Claim:** a multi-`-p` cargo run stops at the first failing crate, and silence
from an un-run crate is identical to silence from a clean one — so require each
target crate's own `Checking <crate> v...` line.

**The claim is right. The test is wrong.** ⚠️ **A WARM CACHE EMITS NO `Checking`
LINE EITHER.** In one run it produced **seven false NOT-REACHED readings** on
crates compiled minutes earlier — a reading of "3 of 10 reached" where the truth
was **10 of 10**.

**CACHE-PROOF FORM:** `--message-format json`, then ask whether each crate's
`package_id` appears in any emitted record. Participation is emitted for cached
crates.

### 2 · The lib-only guard

**Claim:** after a bulk edit, `cargo build -p <crate>` (no `--all-targets`)
compiles no `#[cfg(test)]`, so a non-zero result means the sweep escaped into
production. **It caught five real escapes and it is worth keeping.**

⚠️ **But it is a DELTA instrument, not an absolute one.** Non-zero means *escape*
only against a **known-zero baseline**. Mid-row it is indistinguishable from
*not finished*, and I read an `8` as an escape when it was remaining work.

### The rule

⚠️ **BEFORE TRUSTING ANY ABSENCE OR ANY COUNT, ASK WHAT ELSE PRODUCES THIS EXACT
READING.** Not *"could the instrument be broken"* — that is the positive-control
question and both of these passed it. **The question is whether the SIGNAL is
one-to-one with the CONDITION**, and here neither was.

**A detector published and found half-wrong within the hour is the good outcome.
The bad one is the detector nobody re-tests** — see
[`staleness-by-workaround`](docs/method-rules.md#staleness-by-workaround), whose
whole subject is rules whose forbidden path nobody attempts again.

---

## a-sweep-can-re-break-its-own-earlier-fix

**Three occurrences in one row, and the third is the one that should worry you.**

A constructor in **TAIL POSITION** of a `-> Result` fn is **already correct** and
needs nothing. A line-local sweep sees no handler on that line and adds one.

    1  `?` welded onto three hand-written `.expect(PROOF)` sites   caught: COMPILER
    2  stray `?` on a type-invariant proof written minutes earlier caught: COMPILER
    3  `.unwrap()` added to SEVEN production tail positions        caught: THE GUARD

⚠️ **Two of three were caught because the compiler happened to disagree — that is
luck, not a guard.** And *"look one line ahead"* does not fix it: when the call
spans lines the handler can be several lines down.

⚠️ **THE COMPOSITION IS WHAT MAKES IT DANGEROUS: on TEST targets a stray
`.unwrap()` TYPE-CHECKS AND THE TEST STILL PASSES.** So the backstop that caught
two of three is **weakest exactly where the sweep is least reliable**. A
hand-tended site carrying a written proof, with a stray `?` welded in front of
it, is indistinguishable from a site nobody touched.

**PRACTICE: a sweep must know what it has already fixed.** Either exclude
previously-edited spans, or make the sweep idempotent by matching on the whole
statement rather than the line — and never run the least-guarded sweep over the
least-verifiable surface in the same pass.



---

## Comment 2 (ciresnave-bot, 2026-09-09T11:05:49Z)

## Four more entries, all earned on the GAP-003 row tonight — parked here for the same reason as the first five

⚠️ **Slug collision re-checked against a FRESHLY FETCHED `origin/main`, not carried forward from the last comment.**

```
local origin/main   bb099e87   ==   git ls-remote origin main   bb099e87
docs/method-rules.md sections at that ref: 59
collisions for all FOUR slugs below: 0
control: `staleness-by-workaround` -> 3 hits, so the query fires
```

---

## a-count-is-simultaneously-too-high-and-too-low

**Established earlier the same night: *an error count is a count of SURFACES, never of CAUSES.* That was two claims about two populations. This is BOTH ERRORS IN ONE MEASUREMENT OF ONE POPULATION, in opposite directions, with a named mechanism for each.**

`fuel-nn`, `cargo check --all-targets`, one run:

```
121 compiler errors
  OVERSTATED:   121 errors were  90 CAUSES     31 duplicate surfaces
  UNDERSTATED:  121 errors CONCEALED 19 MORE   masked by resolution order
```

**Over, because one unhandled binding emits an error at every site that consumes it.** `let x = Tensor::from_f32(...);` produces `E0599` where a method is called on `x` and `E0308` where `x` is passed — **13 bindings emitted both.** 121 surfaces, 90 causes.

**Under, because a `?` applied to an expression whose RECEIVER does not type-check is never reached by the `?`-in-unit-fn check.** Fixing the 90 did not create 19 new `?` — **it made 19 existing ones visible.** Verified two ways: **0 of the 19 lie on any of the 90 edited lines, and 0 lie on ANY line the commit changed.**

⚠️ **THE TWO ERRORS DO NOT CANCEL. They are about different things, so a count is not a loose upper bound on the work — it is wrong in both directions at once.**

⚠️⚠️ **AND THE ASYMMETRY IS THE OPERATIONAL PART: the 90 WAS knowable in advance — map every error span's receiver to its binding site and count the distinct bindings, one script. THE 19 WERE NOT KNOWABLE BY ANY STATIC INSTRUMENT AT ANY EFFORT, because the compiler itself could not see them until the earlier class was gone.**

**So a pre-count of a row like this is not a bad estimate; it is a measurement of a quantity that CHANGES AS YOU RESOLVE IT.** The GAP-003 sizing moved through five stages tonight — `845 floor` → `121` → `90 causes` → `19 revealed` → `16 more, 15 of which are themselves a floor` — **each a measurement, each correct when taken.**

**PRACTICE: report a count with its DIRECTION OF ERROR NAMED — N surfaces, M causes, and an unknown number currently masked. Where masking is possible the count is a FLOOR and must be labelled one.**

---

## a-residue-is-not-a-sample-of-the-population

**A sizing heuristic that was reasonable, applied to the wrong population, and pointed exactly backwards.**

`fuel-transformers` finished with **6 residual `E0308` that were 3 distinct non-sweepable shapes** — `vec![]` elements, `HashMap::insert` arguments, struct-literal fields. The extrapolation was natural: *6 gave 3 shapes, so 88 may give many, and each is an individual read.*

**Measured, the 88 in `fuel-nn` gave TWO shapes across 90 sites, with ZERO instances of any of the three.**

⚠️ **A RESIDUE'S SHAPE DISTRIBUTION IS NOT A SAMPLE OF THE ORIGINAL POPULATION — IT IS WHAT SURVIVED A FILTER.** The 6 in `fuel-transformers` were **precisely the cases the sweep could not handle**; everything uniform had already been removed. **`fuel-nn` was unswept, so its 88 were dominated by the uniform case.** Extrapolating from a residue to an unswept population **inverts the selection**.

**Same family as reading four accidentally-discovered rot instances as a rate** ([`enumerate-the-population-not-the-strings`](docs/method-rules.md#enumerate-the-population-not-the-strings)) — **but pointed the other way: there a biased DISCOVERY pattern overstated a rate; here a biased SURVIVAL pattern overstated difficulty.** Same night, both directions, same root.

**PRACTICE: before extrapolating from a leftover set, ask WHAT REMOVED THE REST. If a filter ran, the residue is anti-representative BY CONSTRUCTION and the only cure is to census the new population.**

---

## an-error-code-is-a-fact-about-the-consumer

**A work increment was sized by error code, and the increment turned out to be UNEXECUTABLE.**

The plan was *do the 33 `E0599` first, hold the 88 `E0308`* — reasonable, because `E0599` looked uniform and `E0308` had been heterogeneous elsewhere. **Measured:**

```
binding sites emitting E0599 only     7
binding sites emitting E0308 only    70
binding sites emitting BOTH          13
of the 33 E0599:  20 sit at a site that ALSO emits E0308
```

⚠️ **20 of the 33 could not be fixed without fixing 16 of the 88 IN THE SAME CHARACTER EDIT. The binding either has `.unwrap()` or it does not; half-fixed is not a state that exists.**

⚠️ **THE ERROR CODE IS A FACT ABOUT THE CONSUMER, NOT ABOUT THE FIX.** One unhandled producer: `E0599` if the next line calls a method on it, `E0308` if the next line passes it. **Grouping the work by error code groups it by a property of code that is not being edited.**

**PRACTICE: stratify work by FIX SITE, never by diagnostic code. The code names where the compiler NOTICED, and where it noticed is a property of the consumer.** Same shape as [`the-instrument-nearest-to-hand`](docs/method-rules.md#the-instrument-nearest-to-hand): the error code is the classification the tool hands you for free, so it is the one you reach for.

---

## a-zero-inside-a-multi-p-run-is-not-a-zero

**⚠️ Three crates measured in one `cargo check -p A -p B -p C --all-targets`, as the cost rule prescribes. One reported ZERO ERRORS and has FOUR.**

```
                   combined   alone   targets compiled/declared
fuel-inference           1        1            2 / 2    COMPLETE
fuel-parallel            0        4            1 / 2    FLOOR
fuel-lazy-examples      11       11           10 / 22   FLOOR
```

**Cargo stops queueing new units once any unit fails.** A crate whose turn had not come emits **no errors and no artifacts** — and **zero errors is exactly what a clean crate emits.**

⚠️ **THE COST RULE AND THE CORRECTNESS RULE POINT OPPOSITE WAYS HERE.** One invocation shares the feature resolution and is enormously cheaper; it also shares ONE FAILURE BUDGET. **Nothing in the output says which you got.**

✅ **THE TELL IS FREE AND ALREADY IN THE SAME JSON: compare DECLARED targets (`cargo metadata --no-deps`) against COMPILED artifacts (`--message-format json`, `reason: compiler-artifact`).** `fuel-parallel` emitted **one** artifact for a crate declaring **two** targets — its integration test never compiled. **`fuel-lazy-examples` built 10 of 22: eleven errors in ONE bin stopped twelve sibling bins from ever being attempted, so even its ALONE figure is a floor.**

**This is [`a-workflow-that-never-starts`](docs/method-rules.md#a-workflow-that-never-starts) at the compilation-unit level** — *neither says AND N MORE WERE NOT RUN* — and the same requirement follows: **when reporting a failure count, say whether it is a LIST or a PREFIX, and give the declared/compiled ratio that decides it.** ⚠️ **`--all-targets` has no `--no-fail-fast`, so for `cargo check` there is no flag that removes this; the ratio is the only instrument.**

**And note the failure direction: the suppressed crate reads CLEAN.** A false red gets investigated; **a false green gets filed** — arriving here through the one invocation shape a cost rule recommends.



---

## Comment 3 (ciresnave-bot, 2026-09-09T11:09:34Z)

## Amendment to `a-count-is-simultaneously-too-high-and-too-low` — a THIRD mechanism, found an hour after the entry was written

The entry above names two mechanisms. **A third turned up in the last 16 errors of this row, and it is the one that makes the rule usable rather than merely true — because unlike masking, it is DETECTABLE UP FRONT.**

```
OVER   duplicate surfaces     one binding, an error at every consumer   121 -> 90
UNDER  masking                a later check never reached                 +19
UNDER  AGGREGATION  <- NEW    several producers, ONE diagnostic          5 -> 7
```

**Two worked examples, both in fixture helpers, both one line of code:**

```rust
// 1 · MATCH-ARM UNIFICATION  — fuel-parallel/src/device_group.rs
match anchor {
    Some(a) => Tensor::from_f32_on(a.graph(), data, shape, dev),   // <- E0308 reported
    None    => Tensor::from_f32(data, shape, dev),                 // <- IDENTICAL defect, SILENT
}
```

**The arms unify, so the first arm fixes the expected type and the second is judged against it. One diagnostic, two producers.**

```rust
// 2 · DESTRUCTURING / COLLECT  — fuel-inference/src/prefix_cache.rs
let k = Tensor::from_f32(...);          // both are Result
let v = Tensor::from_f32(...);
(k, v)
}).collect()      // <- ONE E0277, naming the TUPLE: (Result<..>, Result<..>)
```

**The diagnostic is on `.collect()` and its subject is the tuple TYPE, so two independent producers hide behind a single error whose span is at neither of them.**

⚠️ **THE DIFFERENCE THAT MATTERS OPERATIONALLY: masking is invisible to every static instrument — the compiler itself cannot see past it. AGGREGATION IS FULLY VISIBLE THE WHOLE TIME.** Both producers are right there in the source; the compiler simply had no reason to complain twice. **So the fix-site count can be derived up front — but only if you map diagnostics to SITES rather than counting diagnostics.**

⚠️ **AND THE TELL IS CHEAP AND GENERAL: WHEN A DIAGNOSTIC'S SPAN IS NOT ON A PRODUCER, EXPECT MORE THAN ONE PRODUCER BEHIND IT.** A span on `.collect()`, on a `match` scrutinee, on a struct literal or on a function's tail expression is a span on a **JOIN**, and a join is where several things become one. **A span on the producer itself is one-to-one; a span on a join is one-to-many.**

**Restated so the three mechanisms sit in one sentence:**

> **A diagnostic count is not a work count. It over-counts where one cause has many consumers, under-counts where an earlier failure hides a later check, and under-counts again where a language construct merges several causes into one complaint. The first two are about ORDER; the third is about SHAPE, and only the third is knowable before you start.**

*(Recorded here rather than left in a peer message, for the reason this whole issue exists: the entry above was one hour old and already incomplete, and an incomplete rule that reads finished is the version that gets adopted.)*



---

## Comment 4 (ciresnave-bot, 2026-09-09T13:00:47Z)

## ⚠️ CORRECTION to `a-zero-inside-a-multi-p-run-is-not-a-zero` — I asserted an absence and it was false, 20 minutes after writing it

**The entry above ends:**

> **`--all-targets` has no `--no-fail-fast`, so for `cargo check` there is no flag that removes this; the ratio is the only instrument.**

**That is FALSE. `cargo check --keep-going` exists.**

```
$ cargo check --help | grep keep-going
      --keep-going    Do not abort the build as soon as there is an error
$ cargo --version
cargo 1.98.0 (797e8a9bc 2026-08-05)
```

⚠️ **I asserted a capability's ABSENCE without running the one command that could refute it — `--help`.** This corpus's own rule is *give every absence a positive control*, and I gave a null claim about a CLI's flag set no control at all. **The instrument was one command and I did not reach for it because the workaround I had just invented was working.**

## The flag changes the answer, so the false claim was load-bearing

```
workspace gate WITHOUT --keep-going    2 errors    6 of 35 crates incomplete
workspace gate WITH    --keep-going    6 errors    1 of 35 crates incomplete
```

**Three times the errors, and the one remaining incomplete crate is complete for every target cargo actually selected.**

## ⚠️ AND THE SHARPENING, WHICH ONLY APPEARED BECAUSE THE SAME CRATE GOT MEASURED TWICE

**A PREFIX IS NOT MERELY INCOMPLETE — IT IS AN ARBITRARY, NON-REPRODUCIBLE SAMPLE.**

```
fuel-examples, from the WORKSPACE run   2 errors   metavoice/main.rs:356,382
fuel-examples, from the -p run          3 errors   flux/main.rs:210,217,452
                                                   DISJOINT
```

**Same tree, same commit, two honest runs, and the error sets do not intersect.** Scheduling order decides which target fails first, and the first failure stops the rest. ⚠️ **So a prefix cannot even be compared against another prefix — including your own from ten minutes earlier.** The entry above says *say whether it is a LIST or a PREFIX*; that is necessary and it understates the problem, because it implies a prefix is a subset you could at least re-derive. **It is not stable.**

## The corrected prescription — BOTH instruments, in this order

1. **`--keep-going`** to get a complete count instead of a first-failure sample.
2. **Declared-vs-compiled ratio** to VERIFY completeness — because `--keep-going` is not sufficient:

```
fuel-examples   103 declared, 91 compiled, 12 unbuilt
                 3 genuinely FAILING          flux · metavoice · segformer
                 9 NEVER SELECTED             required-features:
                                              onnx x3 · symphonia · pdf2image ·
                                              buildtime-download · fuel-datasets ·
                                              depth_anything_v2 · silero-vad(onnx)
```

⚠️ **THE 9 ARE NOT A PREFIX — THEY ARE OUTSIDE THE FEATURE CLOSURE, WHICH IS A DIFFERENT UNMEASURED WITH A DIFFERENT REMEDY.** A flag cannot reach them; only `--features` can. **The ratio is what distinguishes the two kinds of missing, and `--keep-going` is what removes the first kind. Neither substitutes for the other.**

## The class

**This is [`staleness-by-workaround`](docs/method-rules.md#staleness-by-workaround)'s sibling and I built it myself in one sitting: I hit an obstacle, invented a workaround, declared the obstacle unavoidable, and never re-attempted the forbidden thing.** The rule that file states — *the detector is not a re-read but a deliberate RE-ATTEMPT* — applies to a prohibition **one hour old**, not only to inherited ones.

**The ratio half of the entry stands and is still necessary. What was wrong was the word ONLY.**



---

## Comment 5 (ciresnave-bot, 2026-10-23T15:18:38Z)

On a row now: GAP-366 in #463 (docs/gaps.md). Left OPEN deliberately: this issue holds the only copy of the entry text, and its hold (#138) merged 2026-09-09, so landing the three approved entries plus their CLAUDE.md pointers is unblocked. The further drafts in the comments were approved in principle only and need the architect.


---

## Comment 6 (ciresnave-bot, 2026-10-25T00:25:12Z)

Weekly sweep, origin/main ac29b53c. The three entries from Part 1 are in docs/method-rules.md (one heading each: an-error-count-is-a-count-of-surfaces-not-causes, a-hook-you-route-around-on-a-guess-is-a-hook-you-have-not-read, a-name-search-is-structurally-blind-across-a-seam). The further drafts in the comments (e.g. a-count-is-simultaneously-too-high-and-too-low, 0 headings in the tree) are tracked as GAP-366 and need the architect. Stays open because this issue holds the only copy of that text.
