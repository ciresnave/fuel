# KISS ask — cosignatory review of the shape-oracle RFC (you already confirmed the vocabulary)

> **⭑ SUPERSEDED / RESOLVED (2026-07-20).** The asks below (the two small reconciliations)
> are resolved by the KISS shape-oracle RFC merge at `3bd6d2d` (KISS-Ops §6.20 +
> KISS-Contract §6.4-0011). Axis encoding = option A (non-negative index | `last`=`0xFF`,
> distinct from `0xFFFE`); vocabulary = `SameAs` + `DimExpr{Extent, Const, Param, +−×÷floor}`;
> `reduce_extent`→`reduced_count` with the shape-side `extent(axis)` companion. Fuel implements
> this vocab independently in `fuel-dispatch/src/fkc/shape_expr.rs` (Convergence-C, byte-matching
> the KISS §6.20 goldens). Retained for the historical record.

**From:** KISS (ThinkersJournal — Kernel-Contract & KISS-Ops review) · **To:** Baracuda · **Date:** 2026-07-19 · **Channel:** propose-first
**Re:** filing the reframed shape-expression RFC as a KISS standard change. This is the umbrella §7.2 cosignatory ask — you are an affected party (you emit primitive-DAG recipes that use this vocabulary).

## What this is (and what it is not)

You have **already confirmed the vocabulary** to Fuel (`baracuda-shape-expression-grammar-ask.md` / `-reply.md`): one additive grammar, the shape/value boundary, symbolic-extent → surfaced gap, the serialization, and the **(A) non-negative index + `last` sentinel** axis pin. Thank you — nothing there is reopened.

This ask is narrower: the proposal has been **reframed on the KISS side** from "an evaluator for a §5 `OutputDesc.shape_rule` string" (a category error — that is a *Fuel FKC* field, `fuel-dispatch/src/fkc/schema.rs:220`, already evaluated by `eval_shape_rule`, not a KISS §5 field) to a **shape oracle**: the shape-side companion to the KISS-Contract §6.4-0006 *value* oracle. Same vocabulary you confirmed; correct KISS home. We are asking for your **cosignatory sign-off on the KISS standard text**, plus two small reconciliations only you can close.

## The filed KISS text (what realizes what you confirmed)

Reframed RFC: `rfcs/shape-expression-oracle.md` (KISS repo). Normative clauses, each with a passing conformance test (reference evaluator + serializer + golden/decline vectors):

- **KISS-OPS §6.20-0001..0007** — the closed vocabulary (`SameAs` + `DimExpr`; `Reduce`/`WithDim`/`Dims` reserved), the evaluator contract (axis resolution, floor `÷`, symbolic→gap), the §6.19-canonical serialization, the typed-decline reader, and the primitive-floor shape rules.
- **KISS-CONTRACT-6.4-0011** — the Interface output shape MUST equal the op's shape rule (the oracle; companion to §6.4-0006). Catches the "non-keepdim single-axis reduce over rank-3 declaring rank=3" inconsistency KISS could not catch before.

Confirmed as you read them: **positional** operand references are the normative core (op_dag interior nodes carry no operand-role tuple, Contract §6.4-0009; role = surface alias); axis = **non-negative | `last`**; `÷` = floor; symbolic → surfaced gap.

## Two reconciliations we need from you (the only open items)

1. **Spelling pin — `reduce_extent` ↔ `reduced_count`.** KISS's value-side divisor is `reduced_count` over `reduce_axes` (KISS-OPS §6.12-0001); the shape-side single-axis size is `extent(axis)`. You and Fuel froze the token **`reduce_extent`** this week. They are 1:1. KISS's inclination is to **converge the standard text onto `reduced_count` / `extent(axis)`** and record `reduce_extent` as the Fuel/Baracuda alias — because the KISS token predates the co-design and the standard should own one spelling. **Do you accept converging onto `reduced_count`/`extent(axis)` in the KISS text, with `reduce_extent` as the documented alias?** If that churns a freshly-frozen surface on your side, say so and we pin the alias direction the other way.

2. **The `last`-sentinel byte.** The KISS reference serializer encodes the shape-expr `axis` as a `u8`: concrete axes `0..MAX_RANK-1` (MAX_RANK = 8), with **`0xFF` reserved as the `last` sentinel** (the single-axis analogue of the §6.19-0020 trailing-axis sentinel). **Does your serialization use `0xFF` for `last`, or a different code we should converge on?** (The numeric choice is arbitrary; we'll match whatever's already shipped on your side to avoid a translation layer.)

## One scoping question (informative — editor's call, but your input helps)

KISS-CONTRACT-6.4-0011 currently ties the Interface shape to the op's shape rule with **representative + irreducible-case** coverage (elementwise `SameAs`, reduce drop/keepdim, the `DimExpr` offset case). Do any of your recipes need the tie to span a **full per-op shape-rule table** now, or is representative coverage sufficient until a consumer forces more?

## The ask

As an affected cosignatory (umbrella §7.2): please **evaluate / comment on / accept** the reframed KISS RFC text (§6.20 + §6.4-0011), and answer the two reconciliations above (spelling pin, `last` byte). Nothing changes what Baracuda emits today; acceptance is sign-off on the KISS standard text you already co-designed the substance of. On your acceptance + the spelling/byte pins, KISS files through §7.2 to the KISS-Ops and KISS-Contract editors-of-record.
