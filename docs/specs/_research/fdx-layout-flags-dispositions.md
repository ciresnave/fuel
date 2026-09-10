<!-- Landed by the Fuel architect 2026-09-09 from Fuel 2's measurement.
     `docs/` is architect-owned; the AUTHORSHIP DIVISION at the foot of this
     file is load-bearing and must survive any edit: the measurements, the
     discriminator and the scope find are Fuel 2's; the three rulings and
     their reasons are the architect's.
     HOMED HERE RATHER THAN IN THE SPEC ON PURPOSE: the bit1 §9.2 clause and
     the bits-2-31 must-be-zero rule are RULED and NOT WRITTEN. This file is
     the ruling record; `docs/specs/dlpack-extension.md` is where the
     obligation has to land before any reader of the spec can find it.
     That outstanding item is tracked at GAP-286. -->

> **On the line numbers in this file.** Every `:NNN` citation is paired with
> the text it points at, quoted. That is deliberate: a doc-citation sweep of this
> repo measured **line numbers 67% defective against 0% for paths and symbols**,
> so a bare number here would rot while reading as precise. **Follow the quoted
> string, not the number** — and if the two disagree, the string is the citation
> and the number is the casualty.

# `FDXBlockTable.layout_flags` — per-bit disposition, measured from the struct

**Measured 2026-09-09 at `origin/main` `62ac2991` by Fuel 2.** Derived from the
struct and the spec text, **not from the architect's ruling** — the ruling said
"splits three ways" and did not say which bit goes where. The split below was
reached independently and does agree that there are three outcomes.

## The discriminator, taken from the spec's own vocabulary

The spec distinguishes three grades, and it uses them consistently:

| grade | how the spec marks it | example | disposition |
|---|---|---|---|
| **AUTHORITATIVE** | "sole … authority", or a numbered `INVARIANT (Vnn) … MUST` | `sub_byte_bit_order` is "the **sole packing authority** within FDX" (:645); `unmapped_sentinel` carries `INVARIANT (V18) … MUST` (:1819) | **VALIDATE** |
| **ADVISORY** | the literal word *advisory*, with authority named elsewhere | `/* Advisory / fast-reject; the per-extent kind byte is authoritative */` (:784); `sym_scope` "advisory scope hint" (:1390) | **SPEC-EXEMPT** |
| **SILENT** | no MUST, no V-number, no general rule | see bits 2–31 below | **ROUTE-TO-SPEC** |

**This is the load-bearing control:** the spec contains **50 `MUST`s**, and the
field *immediately above* `layout_flags` in the same struct carries an explicit
`INVARIANT (V18) … MUST`. **So silence about `layout_flags` is meaningful
silence, not a stylistic omission.** A spec that never says MUST would license
no such inference.

## The declaration, which is the entire normative text

`fuel-ir/src/dlpack/sidecar.rs:266`, and `docs/specs/dlpack-extension.md:1826`
carries the **same sentence byte-for-byte**:

```rust
/// Layout flags: bit0 = ids sorted within a row (advisory); bit1 = table is
/// shared/read-only across the call. 0 = no claims.
pub layout_flags: u32,
```

`layout_flags` occurs **twice** in the whole spec: this declaration and one
worked example setting it to `0` (:2688). **No V-number, no MUST, nowhere.**

## Per-bit table

| bit | spec text | disposition | reason |
|---|---|---|---|
| **0** | "ids sorted within a row **(advisory)**" | **SPEC-EXEMPT** | Explicitly advisory, in the spec's established sense. The authoritative source is the table contents themselves. Same class as `FDXTiling` ("Optional.") — **and the same call GAP-286 already made when it RETRACTED its `HAS_TILING`-is-a-defect claim.** Validating it would assert an obligation the spec declines to create. |
| **1** | "table is shared/read-only across the call" | **ROUTE-TO-SPEC** | **Not** marked advisory, so it is not bit 0's case; but it constrains **consumer runtime behaviour**, not sidecar structure, so a structural validator cannot decide it. Its natural home is **§9.2 Consumer policy**, which is where every other "a consumer MUST/MUST NOT" clause lives. See the interaction below — this is the bit with real content. |
| **2–31** | *nothing* | **ROUTE-TO-SPEC** | The spec creates **no obligation in either direction**, and there is **no general reserved-bits rule** (the three `MUST be 0` hits are all V7 about `FDXExtent.sym_id`, a different field). This is a **forward-compatibility** decision, not a validation one: *must-be-zero* lets a v1 reader reject a v2 sidecar; *ignore-unknown* lets v2 add bits safely. **Guessing picks a versioning policy by accident.** |

## ⚠️ bit1 and `FDX_FLAG_READ_ONLY` have DIFFERENT SCOPES, and the difference is the point

*(Read "redundant" carefully in this document: the two flags are not redundant
as MECHANISMS — one is the tensor, one is a buffer within it. A particular
STATE can still be redundant, and one is: see the bit1 ruling below.)*

There is already a read-only mechanism, and it is **MUST-level and built**:

```
FDX_FLAG_READ_ONLY = 1 << 6        codes.rs:39, in the flag list at :57, in tests
spec: "mirror DLPACK_FLAG_BITMASK_READ_ONLY into FDX_FLAG_READ_ONLY; a read-only
       tensor's buffers MUST NOT be written by the consumer."           (:2390)
```

**It is TENSOR-WIDE — "a read-only tensor's *buffers*".** `layout_flags` bit1 is
**PER-BUFFER**: it speaks only for the block-table buffer.

⚠️ **AND THE PAGED-DECODE CASE NEEDS EXACTLY THAT GRANULARITY.** In a paged KV
cache the **block table is read during the call and the pool IS written**. So
`FDX_FLAG_READ_ONLY` is *wrong* for that tensor — it would forbid the write the
kernel must perform — and bit1 is the **only** way to say "the table specifically
is not written." **The two flags are not duplicates; one is the tensor and one is
a buffer within it.**

**The question that therefore needs ruling, and it is a real one:** what does it
mean when they **disagree** — `FDX_FLAG_READ_ONLY` clear but bit1 set (the normal
paged case, and presumably fine), versus `FDX_FLAG_READ_ONLY` set but bit1 clear
(a tensor-wide read-only claim with a table that disclaims it)? **The spec is
silent, and two flags whose relationship is undefined is how a claim gets read
two ways.**

## What is actually true of this field today

```
occurrences of layout_flags, ALL of them:
  fuel-ir/src/dlpack/sidecar.rs:266        declaration (Rust)
  fuel-ir/include/fuel_dlpack_ext.h:394    declaration (C mirror)
  fuel-ir/src/dlpack/validate/tests.rs:130,1148   written as 0
  fuel-memory/src/dlpack_view.rs:394              written as 0
  docs/specs/*.md x2                        declaration + worked example

READS, anywhere in the tree:  ZERO
```

**No named constants exist for these bits** — `bit0`/`bit1` are named only in a
doc comment, whereas siblings get real ones (`FDX_FLAG_HAS_GATHER`,
`FDX_BLOCK_UNMAPPED`). **A bit with no constant cannot be set by name**, which is
part of why nothing sets one.

**And no consumer of a sortedness claim exists** — no kernel takes a sorted-ids
fast path. **So bit0's hazard is LATENT, not live:** the day someone adds a
binary-search path keyed on bit0, a false bit0 becomes **silently wrong results**.
That is worth a note wherever the bit is documented, but it is not an argument for
validating it — **the spec still declines to create the obligation, and a
validator that enforced it would be encoding a claim the spec does not make.**

## Rulings (Fuel architect, 2026-09-09) — recorded with their reasons

All three bits are now ruled. **The dispositions below match the table above; the
REASONS are the architect's and are recorded because a ruling without its reason
gets re-litigated.**

### bit0 — **SPEC-EXEMPT. Ruled.**

Accepted on the GAP-286 `HAS_TILING` precedent. ⚠️ **The latent-hazard note stands
and does NOT change the disposition:** a validator arm would encode an obligation
the spec declines to make — **a guard whose SENTENCE is false with correct
behaviour behind it**, which this project treats as a defect in its own right.

**The note belongs at the bit; the TRIGGER belongs at the future consumer.** The
binary search keyed on bit0 is what would create the obligation, so whoever writes
it inherits the duty to demand the bit be validated. **An expiry tied to an event
that may never occur needs its detector at the event, not at the field.**

### bit1 — **becomes a MUST in §9.2. Ruled. And it does NOT conflict with `FDX_FLAG_READ_ONLY`.**

⚠️ **A SINGLE BIT CANNOT CARRY THREE STATES.** bit1 has *asserted-true* and
*not-asserted*. **It has no *asserted-false*.** Therefore:

- **bit1 CLEAR means NO CLAIM. It never means "this buffer may be written."**
- **`FDX_FLAG_READ_ONLY` set + bit1 clear is therefore REDUNDANCY, not
  contradiction** — the tensor-wide claim binds and the narrower flag adds nothing.
- **Consumer rule: HONOUR THE STRONGER OF THE TWO, and NEVER infer permission from
  the absence of the weaker.**

**The spec must SAY that.** The moment one reader treats a clear bit as a positive
licence to write, two distinct facts — *not claimed* and *claimed writable* —
have collapsed onto one encoding, and the corruption is **a write the descriptor
never authorised**. The fix is not a third flag; it is a written rule that the
clear state is silent.

### bits 2–31 — **MUST BE ZERO. Ruled, on REVERSIBILITY rather than safety.**

**MUST-BE-ZERO -> IGNORE-UNKNOWN is a compatible relaxation a later version can
make freely. IGNORE-UNKNOWN -> MUST-BE-ZERO is a breaking change that invalidates
every reader already shipped. Rule in the direction that can be undone.**

The cost of strictness today is measurably nil — **zero reads tree-wide, three
writes all literal `0`, and no named constants, so no producer can set these bits
by name.** The cost of permissiveness is paid later and cannot be recovered.

⚠️ **And strictness costs no expressiveness, because the escape hatch already
exists:** `MEANING_REQUIRES_EXT` is the mechanism for *"the base descriptor alone
does not carry this tensor's meaning."* A future load-bearing bit rides that, or a
version bump — **not a reader's willingness to ignore what it does not
understand.** In a format whose registry row exists because SILENT-WRONG is the
worst outcome, ignore-unknown is the option that manufactures it.

## Scope of this document

**All three bits are ruled above.** What this document does NOT cover:

- **The other 24 never-read fields.** This is `layout_flags` only. The
  discriminator in the first section is intended to generalise to them; **it has
  not been applied to them here, and a method that works on one field is a
  hypothesis about the rest.**
- **Whether the §9.2 clause for bit1 and the must-be-zero rule for 2–31 have been
  WRITTEN.** As of this document they are rulings, not spec text. **A ruling
  recorded only in a table is not an obligation any reader of the spec can find.**
- **Any validator arm.** None of these three dispositions is implemented.
  Each VALIDATE-or-MUST outcome is a behaviour change on a published wire format
  and needs its own born-red.

**Division of authorship, so neither half inherits the other's confidence:** the
measurements, the discriminator, and the scope find are Fuel 2's; the three
rulings and their reasons are the Fuel architect's. **The tallies elsewhere in the
GAP-286 amendment are mine and were certified structurally, not re-run.**

## ⚠️ ARCHITECT'S AMENDMENT, 2026-09-09 — A RESERVED-BITS CONVENTION *DOES* EXIST, AND THE bits-2-31 RULING NOW RESTS ON THE SPEC'S OWN PRECEDENT

**This section was written after the rulings above, by the architect, while preparing to write the bits-2-31 clause into `docs/specs/dlpack-extension.md`. It does not change the ruling. It changes what the ruling is grounded in, from my judgement to the document's own authoritative table — which is a much stronger footing.**

### The finding

The measurement above says *"there is no general reserved-bits rule (the `MUST be 0` hits are all V7 about a different field)."* **The `MUST be 0` half is correct. The conclusion is not — the convention exists and is spelled differently:**

```
:785  /* bits 9..63 reserved (0). */
:807  | 9..63 | (reserved, 0) | next addition takes bit 9 from THIS table |
```

**Line 807 sits inside the block the spec itself calls the *"Authoritative flag-bit allocation table (single owner — this is the only place a bit is assigned)"*.** So it is not incidental prose; it is the normative statement, in the one place the spec designates as normative for bits.

### ⚠️ WHY A CORRECT MEASUREMENT MISSED IT — DISPERSAL, ONE CONVENTION IN TWO SPELLINGS

    query `MUST be 0`        -> 2 hits, both `cap_kind` (V7).  The query WORKS.
    the actual convention    -> "reserved (0)" and "(reserved, 0)"
    a `MUST be 0` grep is STRUCTURALLY BLIND to a table cell reading "(reserved, 0)"

**The positive control passes and the query still asks the wrong question** — it looks for an obligation phrased as a MUST, and the spec phrases this one as an allocation-table annotation. **Same defect class as a kv-head denominator spelled seven ways: one construct, several names, and the failure is a CLEAN, PLAUSIBLE, CONFIDENT NULL.** A collision gives you hits to sift and you notice; dispersal gives you a well-formed "no such rule" and nothing looks wrong.

### ⚠️ AND I NEARLY RETRACTED THE RULING ON THE STRENGTH OF ONE CLAUSE

§9.2 already says: *"A consumer that recognizes the version but not a **set flag bit it does not understand** MUST NOT proceed as if the tensor were standard if `FDX_FLAG_MEANING_REQUIRES_EXT` is set."* **I read that as *ignore-unknown*, concluded the spec's posture contradicted my must-be-zero ruling, and began drafting a retraction. Reading the ADJACENT allocation table refuted my own correction.**

**They are not alternatives. The spec uses BOTH, for different questions:**

| mechanism | governs | site |
|---|---|---|
| **reserved (0)** | what a producer may write **at the current version** | `:785`, `:807` |
| **`MEANING_REQUIRES_EXT`** | what an OLDER consumer does when a NEWER version has allocated a bit — version skew | §9.2 |
| **`struct_bytes` size-prefix** | trailing fields an older reader has never heard of (P8) | `:835` |

**Three mechanisms, three questions.** A ruling that picks one and ignores the others is under-specified; **the bits-2-31 disposition needs the first AND the second, exactly as the `flags` field already has them.**

### What this changes about writing the clause

**Writing `bits 2..31 reserved (0)` into the spec is now APPLYING THE HOUSE CONVENTION to a field that was written without it — not creating a new obligation.** That matters for the audience: the spec binds **Fuel, Baracuda, Vulkane and any external DLPack consumer**, and asking them to follow, for `layout_flags`, the rule they already follow for `flags` is a categorically smaller ask than asking them to adopt a policy invented here.

⚠️ **It also means the version-skew half must be written in the same change, or the clause is half a policy:** an older consumer meeting a `layout_flags` bit a newer version allocated is governed by `MEANING_REQUIRES_EXT`, exactly as for `flags`. **Writing "reserved (0)" alone would leave the skew case undefined and invite the reading I nearly filed.**

**STILL NOT WRITTEN.** This amendment names the precedent and the required shape; the spec edit is a cross-project-visible change to a document whose stated audience includes two sibling projects, and it goes out as a propose-first ask, not a unilateral edit. Tracked at GAP-286.
