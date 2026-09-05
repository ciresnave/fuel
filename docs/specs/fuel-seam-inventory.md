# Fuel seam inventory — everything KISS may regulate

**Status:** MEASURED INVENTORY, 2026-09-05. **Every number here names the construct that
produced it and the ref it was taken at.** This document exists to be *refuted*, not
believed: each row is a measurement someone else can re-run.

**Baseline ref: `fe4fb8cd5dc81a0f89dea978b8d39c373f8b0858`.** Measurements taken at that ref
unless stated otherwise.

**Why this exists.** CireSnave ratified an FKC ⇄ KISS-Contract convergence in four gated
phases — Phase 1 is *functional parity*, and nothing in phases 2–4 begins until it passes —
and directed the Fuel architect to give the KISS architect **"a comprehensive list of
everything in a Fuel seam that may be regulated by KISS."** *Fuel will become KISS
compliant;* the open question is whether KISS is complete enough to be complied with.

> ⚠️ **THE SCOPE IS EVERY SEAM, NOT FKC.** The convergence was initially scoped against
> `kernel-contract-format.md` alone, by both sides. **FDX (`dlpack-extension.md`) is
> larger by bytes and was unscoped by everyone.** A parity gate that passes on FKC alone
> passes while the larger half of the seam is unexamined.

---

## 0. How to read a row

Four verdicts were agreed with the KISS architect, plus a fifth this inventory forced:

| verdict | meaning |
|---|---|
| **EQUIVALENT** | KISS mandates the same thing |
| **RENAMED** | same capability, different spelling — record **both** names |
| **ABSENT / NEEDED** | KISS must gain it → a filed KISS issue, **blocks the gate** |
| **ABSENT / DECLINED** | KISS deliberately does not carry it → **a stated reason**, does not block |
| ⚠️ **PRESENT-BUT-INERT** | Fuel *writes* it and *reads nothing*. The obligation is **accept-and-preserve**, not interpret |

**An unwritten decline is indistinguishable from an oversight** — that is what the DECLINED
row is for. And the portfolio PM's third state applies throughout: where Fuel's side is a
convention living in code with no written rule, that is **FUEL-SIDE UNWRITTEN**, *not* a
blank. **A blank cell reads as agreement and actually means "divergence unknown."**

---

## 1. The specs — seven live documents

| spec | bytes | subject |
|---|---:|---|
| `dlpack-extension.md` (**FDX**) | 199,888 | tensor interchange / the kernel-boundary tensor projection |
| `kernel-contract-format.md` (**FKC**) | 178,659 | the per-kernel contract |
| `kernel-seam-interop.md` | 46,791 | seam interop |
| `fkc-fusion-patterns.md` | 42,322 | fusion recipe / `pattern:` grammar |
| `storage-encoding.md` | 37,749 | `DType` logical + `SType`/`Encoding` physical |
| `runtime-fused-op-registration.md` | 19,440 | Tier-2 runtime fused-op registration |
| `fkc-kiss-convergence.md` | 14,386 | **a prior 2026-08-07 analysis, GAP-035…039** |

**Bytes, not clauses.** Normative-clause counts are not available for most of these — see §5.

> ⚠️ **`fkc-kiss-convergence.md` already exists and nobody was using it.** It records that
> `dims(...)` / `with_dim(...)` were **authored by Fuel and adopted into KISS-Ops §6.20** —
> so at least one row runs **the other direction**, and *"does KISS have it"* is the wrong
> question there. It is a month old and its claims are **not** re-verified here.

---

## 2. FKC — the as-built field ledger

**92 fields typed across 18 structs; 114 contract files; 1,184 ` ```fkc ` blocks.**

| class | count | definition |
|---|---:|---|
| LIVE | 76 | written > 0 **and** read > 0 |
| ⚠️ **INERT** | **7** | written > 0, **read = 0** |
| UNWRITTEN | 6 | typed, in **no** contract |
| OPAQUE | 11 | typed as a serde `Value` (cuts across the others) |

**The seven INERT — written into essentially every contract, consumed by nothing:**

| field | struct | written | type |
|---|---|---:|---|
| `layout_guarantee` | `OutputDesc` | 114/114 | `Option<String>` |
| `aliasing` | `OutputDesc` | 114/114 | `Option<String>` |
| `fast_paths` | `CapsBlock` | 114/114 | `Option<serde_yaml_ng::Value>` |
| `in_place` | `CapsBlock` | 114/114 | `Option<bool>` |
| `alignment_bytes` | `CapsBlock` | 113/114 | `Option<u64>` |
| `access_granularity_bits` | `CapsBlock` | 113/114 | `Option<u64>` |
| `substrate` | `TensorDesc` | 1/114 | `Option<String>` |

> ⚠️ **An opaque type is one route to inert, not the only one.** Four of the seven are
> ordinary typed fields. **A scan for `serde Value` finds one of seven.**

**Measured on the KISS side:** `layout_guarantee` and `access_granularity_bits` have **zero
occurrences anywhere in KISS's 12-file spec corpus** — with `blurb` (also 114/114), these
are **confirmed migration-data-loss rows.**

**Stated limits — this ledger is a floor.**

- A `#[serde(rename)]` defeats the WRITTEN column. **One found:** `return_` → `return`,
  present in 114/114, counted as 0. Two renames exist in `schema.rs`, so **at most one more.**
- **READ counts files containing a field access, not that the value reaches a decision.**
  It is a **FLOOR on "referenced" and a CEILING on "consumed"**; doc comments count.
- Field-name collisions *within* the schema are merged (`backend`, `name`).

---

## 3. FDX — the larger half, and the one nobody scoped

**179 public items** — `codes.rs` 101, `validate.rs` 28, `abi.rs` 19, `sidecar.rs` 16,
`convert.rs` 15, `header_check.rs` 0 — in `fuel-ir/src/dlpack/`, plus
`fuel-memory/src/dlpack_view.rs`.

**The regulated substance, by spec section:**

| § | capability |
|---|---|
| 3 | the base-DLTensor **honesty invariant** (marked *load-bearing*) |
| 6 | field-by-field semantics |
| 7 | buffer references, the call surface, the buffer table |
| 8 | validation — build-time / boundary-time, `Result`-returning |
| 9 | producer / consumer policies |
| 10 | the two boundaries |
| 11 | ownership, lifetime, alignment, **stream** semantics (cross-runtime) |
| 12 | capability negotiation via `BackendProbe` / `Capability` |
| 14 | versioning rules |
| 15 | interop & backward-compatibility rules |
| 16 | DLPack conformance checklist (FDX exports) |

> ⚠️ **KISS's measured absences land on three coherent sections, not scattered fields.**
> `device_type`, `byte_offset`, `data_ptr`, `zero_point`, `__dlpack__`, `capsule`,
> `managed tensor` — **0 occurrences across KISS's 12 files** — are exactly §7's
> buffer/call surface, §11's ownership-and-lifetime, and §12's negotiation vocabulary.
> **Three sections is a better argument for a work item than seven field names.**

**Nine capability flags:** `HAS_DTYPE_EXT` · `HAS_QUANT` · `HAS_SYMBOLIC` · `HAS_TILING` ·
`IS_BUNDLE` · **`MEANING_REQUIRES_EXT`** · `READ_ONLY` · `HAS_GATHER` · `HAS_AFFINE_EXTENT`.

> ⚠️ **`MEANING_REQUIRES_EXT` is the formal statement of §3's honesty invariant** — the bit
> saying *the base DLTensor alone does not carry this tensor's meaning*. A consumer that
> ignores it **reads a tensor wrongly while believing it succeeded.** If KISS has no
> analogue that is **a missing safety obligation, not a missing field**, and it ranks above
> every vocabulary row.

**Symbolic/affine extents** (`FDX_EXTENT_SCALAR/RANGE/AFFINE`, `FDX_AFFINE_MAX_TERMS = 4`)
are the tensor-side counterpart of the `dims()`/`with_dim()` work Fuel contributed to
KISS-Ops §6.20. **The shape-expression half went to KISS and the extent-encoding half did
not** — a `contributed-BY-Fuel` **fidelity** question, not an equivalence one.

---

## 4. Seam types — published interop surface

**66 public items.** `fuel-kernel-seam-types` 40 · `fuel-kernel-seam-announce` 18 ·
`fuel-kernel-seam` 8.

> ⚠️ **These are the only Fuel crates in `[patch.crates-io]`.** A capability living in a
> published seam type is regulated **whether or not any spec mentions it.**

`fuel-kernel-seam`: `JitRequest` · `JitBudget` · `JitResponse` · `SynthArtifact` ·
`ArtifactKind` · `LinkEntry` · `Synthesizer`.

`fuel-kernel-seam-types`: `OpTag` · `OpAttrs` · `PatternNode` · `to_canonical_bytes` ·
`matmul_roles` · `const_bits_narrow` · `advisory_band_reference_cases` · `shape_expr.rs`.

**`PatternNode` is a closed four-variant schema** — `Op` / `Bind` / `SeeThrough` / `Any` —
**with no `Const` variant.** Constants ride as an **attribute** (`OpAttrs.const_bits`) on the
consuming op, with `OpTag::Const` as the leaf token.

> ⚠️ **This falsifies "a closed node schema cannot express `const(bits)`."** It can, if the
> bits ride in the op's attribute record rather than in a node variant.

**The four acked leaf arms** (commit `50e71dd7`, 2026-07-23 — *"the KISS editor acked
2026-07-23 (**RULING RECORD — four-leaf-arm ack**, clean, no amendments)"*):

```
Const           -> u64(bits)   MBZ narrow-dtype rule: storage bits LOW-order, upper
                               bits zero; NaN payload carried VERBATIM, never quieted
RuntimeScalar   -> u32(slot_index)
ReducedCount    -> i64(axis)   fold-lockstep, minus keepdim
ScanPlaceholder -> u8(role: 0=carry, 1=elem) ++ u32(index)
```

> ⚠️ **HONEST SCOPE, from the commit itself: wire tokens only.** `jit::op_to_tag` emits none
> of the four; `runtime_fused::tag_to_op` **declines all four as honest misses**. This is a
> **design reference with pinned semantics and no producer** — *not* a working round-trip.
>
> ⚠️ **AND A NAME TRAP:** `OpTag::Const` is the KISS **scalar** literal leaf, *deliberately*
> not wired to `Op::Const`, a constant **tensor** leaf. **Same name, different concept** — a
> specification that says `const(bits)` without saying scalar-vs-tensor splits its readers.

---

## 5. Behaviours — what a field ledger cannot see

**A field-only ledger passes a gate it should not.**

| behaviour | where | status |
|---|---|---|
| **auto-registration** | `fkc/register.rs`, `parse.rs`, `lower.rs` | **LIVE**, ~58 families / 3 backends |
| **version gate** | `fkc/validate.rs`, `dlpack/validate.rs` | **PRESENT and ENFORCED**, born-red tested |
| the **verified ledger** | `fkc/verify/ledger.rs`, `.fkc-verified-ledger.json` | provenance of what has been checked |
| the **inventory** | `docs/kernel-contracts/_inventory/` | per-family census |
| cost compilation | `fkc/cost_compile.rs`, `cost_expr.rs` | evaluable cost expressions |
| revision hashing | `fkc/revhash.rs` | |
| per-backend link/invoke | `{cpu,cuda,vulkan}_link.rs`, `verify/invoker_*` | |
| bit-stability & ULP | `verify/bit_stability.rs`, `verify/ulp.rs` | |
| accept-coverage | `verify/accept_coverage.rs` | |
| return-shape check | `fkc/return_check.rs` | |
| precision declaration | `fkc/precision.rs` | |
| audit-flip | `fkc/contract_audit_flip.rs` | |
| runtime fused-op registration | `docs/specs/runtime-fused-op-registration.md` | Tier-2, cost-gated |

### 5.1 The version gate — two independent Fuel implementations agree

```
FDX   validate.rs   if version == 0 || version > FDX_VERSION_MAX  -> Err   (a RANGE gate)
FKC   validate.rs   if fkc_version > FKC_VERSION_MAX              -> Err   (a MAX gate)
KISS  §6.1-0008     MUST reject any value other than exactly 1
```

**All 114 contracts are `fkc_version: 1`.** Migrated onto KISS-Contract v1, **the day KISS
bumps to v2 a conforming v2 reader MUST reject all 114.** Fuel's MAX gate makes that a
non-event; exact-match makes it a rewrite.

> **Two Fuel boundary specs, written for different subjects, independently chose
> accept-up-to-MAX.** FDX's is the stronger of the two — it also rejects `version == 0`, so
> it is a bounded range rather than open-below.

### 5.2 Auto-registration — *"zero glue"* is true per-kernel, not literally

```
include_str!(corpus .fkc.md) -> import_bundle_str(contract, &LinkRegistry)
  -> validate_file -> lower_file (entry_point -> KernelRef)
  -> ImportedProvider::register_into(KernelBindingTable, FusedKernelRegistry)
  -> duplicate-detection finalize gate
```

**A provider must still supply:** the authored `.fkc.md`; a `LinkRegistry` impl (the
`entry_point` → fn-pointer symbol table); a per-family `register_*_from_contract` fn; and a
call to it in startup construction. **Fuel infers:** front-matter provider defaults
inherited by kernels that omit them; cost Judge-bootstrapped from a sentinel `CostFn`;
dispatch key, caps, precision and return-contract derived from the block.

**Genuinely not done, named by `fkc/mod.rs` itself:** the full `V-FKC-*` validator set
(partial) and the CI lint. **Carry both as Fuel-side incomplete, not as capabilities.**

**Blind region:** CPU/Vulkan/CUDA + fused are verified wired; **reference-oracle, metal,
quantized, mkl-aocl and the `dispatch/` corpus dir were not mapped to production imports.**
The README's *"every kernel"* is unproven for those; **~58 families is a floor.**

---

## 6. Open, and stated so a blank is not read as a measurement

- **No kernel count is given.** 114 files · 1,184 ` ```fkc ` blocks · README says *"~390"* ·
  the KISS architect said *"~520"*. **A block is not a kernel, and no construct yielding
  either 390 or 520 has been found.** ⚠️ **Do not adopt any of these numbers.**
- **Clause numbering.** FKC has 5 tokens matching an `N.N-NNNN` clause ID; **FDX has zero.**
  KISS-Contract numbers every normative clause. **A parity ledger cannot cite FDX by clause
  at all** — rows must cite headings or byte ranges, neither of which survives an edit.
  **This is a parity prerequisite, not a style question.** See GAP-283.
- **Not measured:** what each spec describes that no longer exists, and what exists that no
  spec describes. Needs a real diff; **no estimate is offered.**
- **Not measured:** FDX's provider-MUST vs Fuel-infers split (done for FKC only).
- **The seam crates' 66 items** are enumerated by name and count, **not yet by capability.**

---

## 7. Provenance — where the design record actually lives

Fuel holds the chain KISS does not:

- `docs/outreach/fuel-recipe-grammar-kiss-design-input.md` (2026-07-21) — *"Fuel's
  authoritative, project-agnostic design input for consolidating the fused-op recipe grammar
  into KISS."* **Flags Q4 (canonical serialization) and Q5 (`Op::Scan`) ★ MUST-CARRY —
  "must land in the KISS section verbatim rather than be re-litigated."**
- `docs/outreach/baracuda-recipe-grammar-codesign-reply{,-2,-3}.md` — the conversational record.
- `docs/recipe-signature-reference.md` — the implementation reference.
- `docs/specs/kernel-seam-interop.md` §7.3.2 — the leaf field-order table.

> ⚠️ **The four-leaf ack was a KISS editor ruling that KISS did not record.** Fuel's commit
> message preserves its title, its date, and all four byte encodings. **When a cross-project
> agreement is reached, the record must land in the standard's own forge — a ruling that
> exists only in the consumer's commit message is a design the standard has lost.**
> Same class as GAP-283: something was established, the record did not carry it, and the
> artefact readers actually read never changed.
