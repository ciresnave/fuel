# Convergence-C: Shape-Expression Oracle — Design

**Date:** 2026-07-20
**Branch:** `convergence-c-shape-oracle` (isolated worktree `C:\Projects\fuel-convergence-c`, off `78260895`)
**Status:** approved (design + key decisions), pending spec review → implementation plan

## Goal

Make Fuel's FKC return-contract layer able to express, evaluate, and canonically
serialize the full KISS shape-expression vocabulary (KISS-Ops §6.20 +
KISS-Contract §6.4-0011, merged to KISS `main @ 3bd6d2d`), so a fused contract's
declared output shape is cross-checked against the op's real shape rule — the
shape-side companion to the §6.4-0006 value oracle.

## Background — current state

- `fuel-dispatch/src/fkc/return_check.rs:29` — `eval_shape_rule(rule: &str, combo, section)`
  understands exactly one rule, `same_as(role)`; every richer rule returns
  `Ok(None)` (not-evaluable → skip, never a false reject). It is invoked by
  `cross_check_fused_section` to compare a contract's declared `shape_rule`
  against the real registered `FusedOpEntry::shape_rule` fn at each probe combo.
- `OutputDesc.shape_rule` (`fkc/schema.rs:229`) is `Option<String>` — an authoring
  DSL string (`same_as(lhs)`, `fixed(D)`, `from_params(...)`), **already a Fuel FKC
  field** and already evaluated (gap-closure `b1c33f91`). (Note: correct the stale
  `ROADMAP.md:128` "parsed-but-unevaluated" line — the premise error the merged
  RFC §9 called out — as part of C-1.)
- The KISS conformance reference `conformance/src/shape_expr.rs` is a **typed AST**:
  `Dim { Extent{operand:u8, axis:u8} | Const(i64) | Param(u8) | Add|Sub|Mul|Div(Box,Box) }`
  and `ShapeExpr { SameAs{operand:u8} }`, with a §6.19 canonical wire encoder, a
  typed-decline decoder that round-trips, a gap-propagating evaluator, `resolve_axis`
  (`LAST = 0xFF`), floor-division, plus the role/index-woven `reduce_shape` /
  `gather_shape` / `matmul_shape` and `shape_consistent` (§6.4-0011).

**The gap:** Fuel can express/evaluate only `same_as`. Conformance needs the full
`DimExpr` vocab, the canonical wire serializer (byte-matching the KISS goldens),
the two shape-rule kinds, and the registry decomposes migrated onto it.

## Approach

A new module **`fuel-dispatch/src/fkc/shape_expr.rs`** — Fuel's own typed
`ShapeExpr` / `Dim` mirroring the KISS reference, with `encode`/`decode`/`eval`,
**verified byte-for-byte against the KISS golden vectors as Fuel-side fixtures.**
This is the freeze-gate philosophy applied to shapes: an *independent* Fuel impl
that byte-matches the reference — not a dependency on the KISS repo or the future
`fuel-kiss-ref-backend` adapter. `eval_shape_rule` gains a parser: `shape_rule`
string → typed AST → eval, extending (not replacing) the `same_as` fast path.

### Key decisions (approved)

1. **Typed AST, not more string-matching** — the only way to produce the wire
   format + byte-match goldens. Mirrors the KISS reference structure.
2. **New module `fkc/shape_expr.rs`** — FKC owns return-contract validation;
   goldens as fixtures. Independent-impl-now is the conformance-correct + freeze-gate
   consistent path; `fuel-kiss-ref-backend` can delegate later.
3. **Authoring surface keeps role NAMES** (`div(extent(data, last), const(2))`),
   resolved to **positional** operand indices at AST-construction time
   (§6.4-0009 wire form is positional). The typed AST stores `operand: u8`
   positions exactly like the KISS reference, so `encode()` is byte-identical.
   Role→position uses the canonical operand order (contract `accept.inputs` order).
4. **Two shape-rule kinds:**
   - **EXPRESSION** — `SameAs` + `DimExpr` (this module's core). Wired into
     `eval_shape_rule`'s expression path.
   - **ROLE/INDEX-WOVEN** (§6.20-0008) — `reduce_shape` / `gather_shape` /
     `matmul_shape`, keyed off the op's role structure (matmul reuses the pinned
     M/N/K role vector). **Not** forced through the expression evaluator — a
     distinct shape-rule variant.

### Wire format (§6.19 canonical — the byte contract to reproduce)

- Tags (one byte, `0x00` reserved → `ZeroTag`): `SameAs=0x01`, `Extent=0x02`,
  `Const=0x03`, `Param=0x04`, `Add=0x05`, `Sub=0x06`, `Mul=0x07`, `Div=0x08`.
  Reserved & **rejected** by a core reader: `Reduce=0x09`, `WithDim=0x0A`,
  `Dims=0x0B`.
- Leaf layouts: `Extent = [tag, operand:u8, axis:u8]`; `Const = [tag, i64-LE]`;
  `Param = [tag, field:u8]`; `SameAs = [tag, operand:u8]`.
- Binary node: `[tag, u16-LE len(childA), childA, u16-LE len(childB), childB]`
  (definite-length children, §6.19-0010).
- `axis`: non-negative index **or** `LAST = 0xFF` (`u8`; MAX_RANK=8, concrete axes
  `0..7`). `LAST` resolves to `rank-1` at eval. A concrete axis `>= rank`, or `LAST`
  on rank-0, is a typed decline (`AxisOutOfRange`). **`0xFF` (u8 single-axis) is a
  DISTINCT field from §6.19-0020's `0xFFFE` (u16 axis-set mask) — never unify them.**
- `÷` = floor division toward −∞; `÷0` → `DivideByZero` decline.
- Symbolic/data-dependent extent (`SYMBOLIC = i64::MIN`) → surfaced `Gap`
  (propagates through binary ops); never a decline, never a panic (§6.20-0004).
- Every malformed input is a **typed decline**, never a panic: `ZeroTag`,
  `ReservedTag`, `TruncatedBlob`, `TrailingBytes`, `AxisOutOfRange`,
  `OperandOutOfRange`, `ParamOutOfRange`, `DivideByZero`.
- **Golden anchor** (must byte-match): `Div(Extent(0, last), Const(2))`
  = `08 03 00 02 00 FF 09 00 03 02 00 00 00 00 00 00 00`.

## Increments (each independently testable; commit each)

### C-1 — EXPRESSION core (foundational, byte-match-verifiable)
Deliverables:
- Typed `ShapeExpr` / `Dim` in `fkc/shape_expr.rs` (positional operands).
- `encode` (§6.19 canonical) + `decode` (typed decline, round-trips
  `decode(encode(x)) == Ok(x)`).
- `eval` against operand shapes (+ param values when threadable), gap-propagating.
- **Byte-match tests**: the golden anchor + the 9 KISS `conformance/tests/shape_expr.rs`
  clause-cited vectors, vendored as Fuel-side fixtures.
- Wire into `eval_shape_rule`: parse the `shape_rule` string (role names) → typed
  AST (positions via canonical operand order) → eval; keep the `same_as` fast path.
  `Param` evaluates when the contract's param values are threadable, else surfaces
  `Gap` (never a false reject).
- Docs: fix `ROADMAP.md:128`; add a superseded/resolution banner to
  `docs/outreach/baracuda-shape-oracle-rfc-ask.md` (its (a)/(b) asks are resolved
  by the `3bd6d2d` merge).

### C-2 — ROLE/INDEX-WOVEN kind (§6.20-0008)
- `reduce_shape` / `gather_shape` / `matmul_shape` as a distinct shape-rule variant
  keyed off role structure (matmul reuses the pinned `{Batch=0,FreeM=1,FreeN=2,ContractedK=3}`
  role vector). Explicitly NOT routed through the expression evaluator.
- The `same_as(data)`-for-a-gather bug the oracle catches (§6.20-0008) gets a test.

### C-3 — migrate the registry decomposes
- Migrate the elementwise/reduce/offset registry decomposes onto the core vocab;
  gather/matmul onto the role-woven kind.
- **Consistency gate**: read the ratified `docs/outreach/kiss-conformance-architecture-fuel-ratify.md`
  §4 (recipe = pattern = §6.13 independence rule) *before* this increment — once
  Fuel's recipe grammar unifies with §6.13, a fused-kernel↔kiss-ref §6.13 diff
  recategorizes; the migration must stay consistent with the recorded position.

## Testing strategy

- TDD per increment (failing test first, observed red, then green).
- The **byte-match fixtures** (golden anchor + 9 KISS vectors) are the conformance
  gate: Fuel's independent encoder reproducing the KISS bytes exactly is what makes
  the shape oracle a genuine second implementation.
- Never-panic: every decline path is a typed error asserted by a test, never a panic.
- Build discipline (CLAUDE.md): `-p fuel-dispatch` only, never workspace-wide; one
  cargo invocation at a time.

## Non-goals / out of scope

- No dependency on the KISS repo or `kiss-ref` (the `fuel-kiss-ref-backend` adapter
  is future integration work; Fuel builds its own conformant impl now).
- No emit of the reserved `Reduce`/`WithDim`/`Dims` tags (reader rejects them).
- No `f32s`/gem/precision key work (that is the sk3 track, separate).
- No RFC edits — Fuel builds *against* the merged `3bd6d2d` standard; any future
  RFC change syncs KISS `763thguz`/`2fpuz4dg` first.

## Ops

- Isolated worktree (`C:\Projects\fuel-convergence-c`) off `78260895` — three Fuel
  sessions share one `.git/index`; isolation is mandatory.
- Execution via subagent-driven-development given the size (especially C-3).
- Baracuda (`3s56q9w4`) offered a byte-level pre-check of the DimExpr §6.19 codec
  once C-1 lands on the branch.
