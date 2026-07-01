# FKC fusion-patterns: rev 2 → rev 4 delta (for Baracuda)

**Status: DRAFT — not sent.** Part of the Profile v1 circulation bundle (see
`kernel-seam-interop.md` §8). Baracuda last reviewed **rev 2** and raised must-fixes; Profile v1 pins
**rev 4**. This doc lets you confirm your findings were resolved **without diffing** — it states each
rev-2 finding and its exact resolution, then the rest of rev 3 and rev 4. Read it alongside the full
`fkc-fusion-patterns.md` (rev 4, included in the bundle); your generator re-verifies against rev 4 (fast).

**Headline: all three of your blocking items were resolved in rev 3, and carried unchanged into rev 4.**
Rev 4 changed *nothing* in the pattern grammar your `derive_pattern` targets — it only reconciled the spec
to the 2026-06-20 adaptive-fusion decision (prose + cross-refs). So a generator that conforms to rev 3
conforms to rev 4.

---

## 1. Your rev-2 blocking findings — each resolved (rev 3)

### A1 — `self.axis == input(0).rank - 1` used `-` arithmetic, which §5's grammar forbids (blocked every norm pattern)

**Resolved.** Axes and `dim[i]` indices are now compared in **normalized negative-from-end form**:
`self.axis == -1` means "the last axis at any rank," so **no `rank - k` arithmetic is needed** (and §5
still forbids it). The RmsNorm/norm `MeanDim` axis guard is now `self.axis == -1`.

- Spec anchors: `fkc-fusion-patterns.md` §5 (the normalization rule, ~line 369–370: *"op attributes
  (`self.axis`) and `dim[i]` indices are compared in normalized negative-from-end form … `self.axis == -1`
  means the last axis at any rank, so no `rank - k` arithmetic is needed"*); §8.2 worked example (the
  RmsNormLastDim pattern, axis guard `self.axis == -1`, ~line 475); resolution note ~line 550.

### A2 — the FusedLinear bias guard read `operand(0)` from a `bind` leaf (which has no operands)

**Resolved.** A `bind` leaf has no operands, so the bias-length guard now reads from the **bound input by
index**: `rank == 1 and dim[0] == input(1).dim[-1]` (where `b` is `bind: 1`, the matmul's RHS). Both §8
examples now type-check.

- Spec anchors: §8.1 FusedLinear worked example (`guard: { shape: "rank == 1 and dim[0] == input(1).dim[-1]"
  }`, ~line 447); the `input(i)` phasing accessor in §5; resolution note ~lines 552–553.

### E1 / B1 — commutative-operand canonicalization (the one that gates you) — now NORMATIVE

**Resolved, normatively** — this is exactly your condition ("must be stated normatively"). §3a.2a now
states: **before matching, Fuel canonicalizes the operands of commutative ops** (`Add`, `Mul`, `Maximum`,
`Minimum`) by a stable sort key (the same canonicalization `structure_key` uses), and a pattern's
`operands:` for a commutative op is matched **against that canonical order**. So your `derive_pattern` emits
**one** operand ordering and Fuel matches it regardless of how the graph was written — no 2ᵏ blow-up, no
need for you to enumerate orderings. Non-commutative ops (`Sub`, `Div`, `MatMul`, …) stay strictly ordered.

- Spec anchors: §3a.2a (NORMATIVE block, ~lines 248–256); the §11 resolution restates it as *"B1 —
  commutativity (BLOCKING) — RESOLVED normatively (§3a.2a)"* (~line 560) and the per-point answer *"(blocking)
  Commutative canonicalization? **Yes**, normatively — §3a.2a"* (~line 586).

**What this asks of your generator:** emit a single canonical operand ordering for commutative ops (sorted
by the stable key §3a.2a names); do **not** emit multiple orderings. That is the whole contract on your side.

---

## 2. The rest of rev 3 (beyond your three items)

Rev 3 also (none of these should affect a conforming generator, but for completeness):

- Fixed the two §8 type-check bugs (A1/A2 above).
- Added the **`input(i)` phasing rule** (an accessor that reads a bound input by index — what A2's fix uses).
- Unified guard/extract auto-skip.
- Extended the dtype list and **resolved `Bool` → `U8`** (Bool rides the `U8` honesty stand-in).
- **Pinned the `Gelu` / `GeluErf` flavors** (so a pattern doesn't silently match the wrong gelu).
- Re-sequenced multi-output handling + an import-time never-match lint ahead of cosmetic deferrals.
- Per-point resolutions are in §11.

Still deferred (unchanged, flagged so you know they're known): `chunk`/`split` → `Slice` canonicalization +
a `Split`/`Chunk` op for gated activations (SwiGLU/GeGLU) — §9-deferred. Ship those activations' pattern
once that lands; everything else is rev-3-ready.

---

## 3. Rev 4 (2026-06-20) — adaptive-fusion reconciliation only

Rev 4 made **no change to the pattern grammar, guard/extract languages, or `§3a.2a` canonicalization** —
the surface your generator targets is rev-3-stable. It only reconciled the spec to the adaptive-runtime-
fusion decision (`10-decisions-log §2026-06-20`):

- Re-scoped the one-line `FusedOpRegistry` *"frozen thereafter"* quote in §1 into the three-way split
  (primitive `Op` enum + untrusted rules stay closed; **Tier-2 trusted, Fuel-orchestrated runtime fused-op
  registration is now a goal**), and named **implementing this spec's declarative `PatternKind::Declarative`
  engine as the prerequisite/mechanism** for that Tier-2 path. (This is *why* your declarative `pattern:`
  matters beyond the build-time case — it's also how a JIT-synthesized fusion registers at runtime.)
- Pinned the missing-fusion telemetry sequencing (closed-world `FusionMissRecord` v1 first; open-world
  co-occurrence deferred).
- The **recipe principle is unchanged** and remains the canonical statement: every fused op carries
  `decompose` **and** `pattern` (both mandatory); `decompose` is total + never-panic + primitive→self.

That's the whole delta. Net for you: **re-verify your `derive_pattern` output against rev 4 (it's the
rev-3 grammar), confirm A1/A2/E1 read as resolved above, and we're clear to ratify.**
